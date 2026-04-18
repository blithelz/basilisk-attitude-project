"""Simulation implementation for the project-local hill-pointing baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Basilisk.utilities import macros, vizSupport

from src.actuators.reaction_wheels import (
    attach_reaction_wheel_recorders,
)
from src.modes.hill_point import apply_hill_point_control_gains, get_mode_request
from src.sensors.simple_nav import (
    attach_navigation_recorders,
    configure_spacecraft_initial_state,
)
from src.simulation.bootstrap import BSKScenario, BSKSim, BSK_Dynamics, BSK_Fsw, BSK_plt, REPO_ROOT
from src.simulation.outputs import render_baseline_outputs


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
        self.rwSpeedRec = None
        self.rwMotorRec = None

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
        configure_spacecraft_initial_state(self.get_DynModel(), self.config)

    def log_outputs(self) -> None:
        """Record the minimum data products needed for baseline validation."""
        fsw_model = self.get_FswModel()
        dyn_model = self.get_DynModel()
        sampling_time = fsw_model.processTasksTimeStep

        self.attNavRec, self.transNavRec = attach_navigation_recorders(self, dyn_model, sampling_time)
        self.attRefRec = fsw_model.attRefMsg.recorder(sampling_time)
        self.attErrRec = fsw_model.attGuidMsg.recorder(sampling_time)
        self.rwSpeedRec, self.rwMotorRec = attach_reaction_wheel_recorders(
            self, dyn_model, fsw_model, sampling_time
        )

        self.AddModelToTask(dyn_model.taskName, self.attRefRec)
        self.AddModelToTask(dyn_model.taskName, self.attErrRec)

    def pull_outputs(self, show_plots: bool, save_plots: bool) -> list[Path]:
        """Convert recorders into plots and optionally save them."""
        return render_baseline_outputs(
            BSK_plt,
            self.config,
            self.results_dir,
            self.attNavRec,
            self.transNavRec,
            self.attRefRec,
            self.attErrRec,
            self.rwSpeedRec,
            self.rwMotorRec,
            show_plots,
            save_plots,
        )


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
