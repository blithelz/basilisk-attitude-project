"""Simulation implementation for the project-local hill-pointing baseline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from Basilisk.utilities import macros, orbitalMotion, vizSupport

from src.modes.hill_point import apply_hill_point_control_gains, get_mode_request


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_bsk_sim_root() -> Path:
    """Find the official Basilisk BSK_Sim example root used as the baseline."""
    candidates = []

    env_path = os.environ.get("BASILISK_BSKSIM_ROOT")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path.home() / "avslab" / "basilisk-develop" / "examples" / "BskSim")

    for candidate in candidates:
        if (candidate / "BSK_masters.py").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the Basilisk BSK_Sim example root. "
        "Set BASILISK_BSKSIM_ROOT or install Basilisk under ~/avslab/basilisk-develop."
    )


def bootstrap_bsk_paths() -> Path:
    """Add the official BSK_Sim example directories to sys.path."""
    bsk_sim_root = resolve_bsk_sim_root()
    extra_paths = [
        str(bsk_sim_root),
        str(bsk_sim_root / "plotting"),
    ]

    for path in reversed(extra_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    return bsk_sim_root


BSK_SIM_ROOT = bootstrap_bsk_paths()

from BSK_masters import BSKScenario, BSKSim  # noqa: E402
import BSK_Dynamics  # noqa: E402
import BSK_Fsw  # noqa: E402
import BSK_Plotting as BSK_plt  # noqa: E402


class HillPointBaselineScenario(BSKSim, BSKScenario):
    """A project-local baseline wrapper around the official hill-pointing example."""

    def __init__(self, config: dict[str, Any]):
        simulation_cfg = config["simulation"]
        super().__init__(simulation_cfg["fsw_rate_sec"], simulation_cfg["dyn_rate_sec"])
        self.config = config
        self.name = config["scenario"]["name"]

        self.attNavRec = None
        self.transNavRec = None
        self.attErrRec = None
        self.attRefRec = None
        self.LrRec = None

        self.results_dir = (REPO_ROOT / config["output"]["results_dir"]).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 先搭 Dynamics，再搭依赖其消息的 FSW，这样主链路最清楚。
        self.set_DynModel(BSK_Dynamics)
        self.set_FswModel(BSK_Fsw)

        self.apply_control_gains()
        self.configure_initial_conditions()
        self.log_outputs()

        if config["visualization"]["enable_unity_viz"]:
            dyn_models = self.get_DynModel()
            vizSupport.enableUnityVisualization(
                self,
                dyn_models.taskName,
                dyn_models.scObject,
                rwEffectorList=dyn_models.rwStateEffector,
            )

    def apply_control_gains(self) -> None:
        """Apply the baseline hill-pointing mode settings."""
        fsw_model = self.get_FswModel()
        apply_hill_point_control_gains(fsw_model, self.config)
        return

        # 官方示例里有两个 MRP 反馈控制器，这里统一从配置覆盖参数。
        for controller in (fsw_model.mrpFeedbackControl, fsw_model.mrpFeedbackRWs):
            controller.K = gains["K"]
            controller.Ki = gains["Ki"]
            controller.P = gains["P"]
            controller.integralLimit = 2.0 / controller.Ki * 0.1 if controller.Ki != 0.0 else 0.0

    def configure_initial_conditions(self) -> None:
        """Set the initial orbit and body attitude."""
        orbit_cfg = self.config["orbit"]
        attitude_cfg = self.config["attitude"]

        oe = orbitalMotion.ClassicElements()
        oe.a = orbit_cfg["a_m"]
        oe.e = orbit_cfg["e"]
        oe.i = orbit_cfg["i_deg"] * macros.D2R
        oe.Omega = orbit_cfg["Omega_deg"] * macros.D2R
        oe.omega = orbit_cfg["omega_deg"] * macros.D2R
        oe.f = orbit_cfg["f_deg"] * macros.D2R

        dyn_models = self.get_DynModel()
        mu = dyn_models.gravFactory.gravBodies["earth"].mu
        r_n, v_n = orbitalMotion.elem2rv(mu, oe)

        dyn_models.scObject.hub.r_CN_NInit = r_n
        dyn_models.scObject.hub.v_CN_NInit = v_n
        dyn_models.scObject.hub.sigma_BNInit = [[value] for value in attitude_cfg["sigma_BN_init"]]
        dyn_models.scObject.hub.omega_BN_BInit = [
            [value] for value in attitude_cfg["omega_BN_B_init_rad_s"]
        ]

    def log_outputs(self) -> None:
        """Record the minimum data products needed for baseline validation."""
        fsw_model = self.get_FswModel()
        dyn_model = self.get_DynModel()
        sampling_time = fsw_model.processTasksTimeStep

        self.attNavRec = dyn_model.simpleNavObject.attOutMsg.recorder(sampling_time)
        self.transNavRec = dyn_model.simpleNavObject.transOutMsg.recorder(sampling_time)
        self.attRefRec = fsw_model.attRefMsg.recorder(sampling_time)
        self.attErrRec = fsw_model.attGuidMsg.recorder(sampling_time)
        self.LrRec = fsw_model.cmdTorqueMsg.recorder(sampling_time)

        self.AddModelToTask(dyn_model.taskName, self.attNavRec)
        self.AddModelToTask(dyn_model.taskName, self.transNavRec)
        self.AddModelToTask(dyn_model.taskName, self.attRefRec)
        self.AddModelToTask(dyn_model.taskName, self.attErrRec)
        self.AddModelToTask(dyn_model.taskName, self.LrRec)

    def save_figures(self, figures: dict[str, Any]) -> list[Path]:
        """Persist generated matplotlib figures into the project results folder."""
        saved_paths = []
        for figure_name, figure in figures.items():
            target = self.results_dir / f"{figure_name}.png"
            figure.savefig(target, dpi=200, bbox_inches="tight")
            saved_paths.append(target)
        return saved_paths

    def pull_outputs(self, show_plots: bool, save_plots: bool) -> list[Path]:
        """Convert recorders into plots and optionally save them."""
        sigma_bn = np.delete(self.attNavRec.sigma_BN, 0, 0)
        r_bn_n = np.delete(self.transNavRec.r_BN_N, 0, 0)
        v_bn_n = np.delete(self.transNavRec.v_BN_N, 0, 0)

        sigma_rn = np.delete(self.attRefRec.sigma_RN, 0, 0)
        omega_rn_n = np.delete(self.attRefRec.omega_RN_N, 0, 0)
        sigma_br = np.delete(self.attErrRec.sigma_BR, 0, 0)
        omega_br_b = np.delete(self.attErrRec.omega_BR_B, 0, 0)
        torque_request = np.delete(self.LrRec.torqueRequestBody, 0, 0)

        BSK_plt.clear_all_plots()
        timeline = np.delete(self.attNavRec.times(), 0, 0) * macros.NANO2MIN
        BSK_plt.plot_attitude_error(timeline, sigma_br)
        BSK_plt.plot_control_torque(timeline, torque_request)
        BSK_plt.plot_rate_error(timeline, omega_br_b)
        BSK_plt.plot_orientation(timeline, r_bn_n, v_bn_n, sigma_bn)
        BSK_plt.plot_attitudeGuidance(timeline, sigma_rn, omega_rn_n)

        saved_paths: list[Path] = []
        if save_plots:
            figure_names = [
                "attitudeErrorNorm",
                "rwMotorTorque",
                "rateError",
                "orientation",
                "attitudeGuidance",
            ]
            figures = {
                f"{self.config['output']['file_prefix']}_{name}": BSK_plt.plt.figure(index + 1)
                for index, name in enumerate(figure_names)
            }
            saved_paths = self.save_figures(figures)

        if show_plots:
            BSK_plt.show_all_plots()

        return saved_paths


def run_scenario(config: dict[str, Any]) -> list[Path]:
    """Run the baseline hill-pointing scenario from the project config."""
    scenario = HillPointBaselineScenario(config)
    scenario.InitializeSimulation()
    scenario.modeRequest = get_mode_request(config)

    duration_minutes = config["simulation"]["duration_minutes"]
    scenario.ConfigureStopTime(macros.min2nano(duration_minutes))
    scenario.ExecuteSimulation()

    return scenario.pull_outputs(
        show_plots=config["output"]["show_plots"],
        save_plots=config["output"]["save_plots"],
    )
