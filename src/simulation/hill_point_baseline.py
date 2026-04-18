"""Simulation implementation for the project-local hill-pointing baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Basilisk.utilities import macros, vizSupport

from src.actuators.reaction_wheels import attach_reaction_wheel_recorders
from src.modes.hill_point import apply_hill_point_control_gains, get_mode_request
from src.sensors.simple_nav import (
    attach_navigation_recorders,
    configure_spacecraft_initial_state,
)
from src.simulation.bootstrap import BSKScenario, BSKSim, BSK_Dynamics, BSK_Fsw, BSK_plt, REPO_ROOT
from src.simulation.outputs import render_baseline_outputs


class HillPointBaselineScenario(BSKSim, BSKScenario):
    """Project-local hill-pointing baseline built on the official BSK_Sim scaffold."""

    def __init__(self, config: dict[str, Any]):
        simulation_cfg = config["simulation"]
        super().__init__(simulation_cfg["fsw_rate_sec"], simulation_cfg["dyn_rate_sec"])
        self.config = config
        self.name = config["scenario"]["name"]

        # 这些 recorder 会在 log_outputs() 中挂到仿真任务上，
        # 后续 pull_outputs() 再统一从这里取数据画图。
        self.attNavRec = None
        self.transNavRec = None
        self.attErrRec = None
        self.attRefRec = None
        self.rwSpeedRec = None
        self.rwMotorRec = None

        self.results_dir = (REPO_ROOT / config["output"]["results_dir"]).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 这两步是整个工程骨架的核心：
        # Dynamics 提供“真实世界”，FSW 提供“控制软件世界”。
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
        # 这里不直接写控制器细节，而是交给 mode helper。
        # 好处是以后新增 inertial3D、sunSafe 时可以沿用同样入口。
        apply_hill_point_control_gains(self.get_FswModel(), self.config)

    def configure_initial_conditions(self) -> None:
        """Set the initial orbit and body attitude."""
        # 初始轨道根数、初始姿态都从配置文件进入，
        # 再由 sensor helper 转成 Dynamics 真正需要的 r/v 和 sigma/omega。
        configure_spacecraft_initial_state(self.get_DynModel(), self.config)

    def log_outputs(self) -> None:
        """Record the minimum data products needed for baseline validation."""
        fsw_model = self.get_FswModel()
        dyn_model = self.get_DynModel()
        sampling_time = fsw_model.processTasksTimeStep

        # 导航、飞轮等“平台侧”输出交给各自 helper 去挂接 recorder。
        self.attNavRec, self.transNavRec = attach_navigation_recorders(self, dyn_model, sampling_time)
        self.rwSpeedRec, self.rwMotorRec = attach_reaction_wheel_recorders(
            self, dyn_model, fsw_model, sampling_time
        )

        # 参考姿态和跟踪误差仍然直接来自 FSW。
        self.attRefRec = fsw_model.attRefMsg.recorder(sampling_time)
        self.attErrRec = fsw_model.attGuidMsg.recorder(sampling_time)

        self.AddModelToTask(dyn_model.taskName, self.attRefRec)
        self.AddModelToTask(dyn_model.taskName, self.attErrRec)

    def pull_outputs(self, show_plots: bool, save_plots: bool) -> list[Path]:
        """Convert recorders into plots and optionally save them."""
        # 这里场景类只负责把 recorder 和配置交给输出模块，
        # 具体怎么整理数组、画哪些图，都由 outputs.py 负责。
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

    # modeRequest 是官方 BSK_Sim 的模式入口。
    # 这里把“跑哪个任务模式”的决定留给配置文件和 mode helper。
    scenario.modeRequest = get_mode_request(config)

    duration_minutes = config["simulation"]["duration_minutes"]
    scenario.ConfigureStopTime(macros.min2nano(duration_minutes))
    scenario.ExecuteSimulation()

    return scenario.pull_outputs(
        show_plots=config["output"]["show_plots"],
        save_plots=config["output"]["save_plots"],
    )
