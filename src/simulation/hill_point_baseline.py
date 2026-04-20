"""Simulation implementation for the project-local hill-pointing baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 从 Basilisk 工具包导入 macros（单位转换）和 vizSupport（Vizard 可视化支持函数）
from Basilisk.utilities import macros, vizSupport
# 导入 simSynch 模块中的 ClockSynch 类，用于在实时可视化时进行时钟同步加速
from Basilisk.simulation import simSynch

# 从项目执行机构模块导入反作用轮记录器挂载函数
from src.actuators.reaction_wheels import attach_reaction_wheel_recorders
# 从环境层导入环境真值 recorder 挂载函数
from src.environment.leo import attach_environment_recorders

# 从 Hill 指向模式模块导入控制增益应用函数和模式请求获取函数
from src.modes.hill_point import apply_hill_point_control_gains, get_mode_request
# 从 SimpleNav 传感器模块导入导航记录器挂载函数
from src.sensors.simple_nav import attach_navigation_recorders
from src.simulation.bootstrap import BSKScenario, BSKSim, BSK_Dynamics, BSK_Fsw, BSK_plt, REPO_ROOT
from src.simulation.outputs import render_baseline_outputs
from src.truth.orbit import apply_orbit_truth_configuration
from src.truth.rigid_body import (
    apply_rigid_body_truth_configuration,
    attach_truth_state_recorder,
    update_fsw_vehicle_configuration,
)

# 定义场景类，多重继承自 BSKSim（仿真基础框架）和 BSKScenario（场景管理接口）
class HillPointBaselineScenario(BSKSim, BSKScenario):
    """Project-local hill-pointing baseline built on the official BSK_Sim scaffold."""


    # 构造函数，接收解析后的配置字典
    def __init__(self, config: dict[str, Any]):
        simulation_cfg = config["simulation"]
        # 调用父类初始化，传入飞控软件更新周期和动力学更新周期（单位：秒）
        super().__init__(simulation_cfg["fsw_rate_sec"], simulation_cfg["dyn_rate_sec"])
        self.config = config
        # 从配置中获取场景名称，用于日志和文件命名
        self.name = config["scenario"]["name"]

        # 这些 recorder 会在 log_outputs() 中挂到仿真任务上，
        # 后续 pull_outputs() 再统一从这里取数据画图。
        self.attNavRec = None      # 姿态导航数据记录器
        self.transNavRec = None    # 平动（位置/速度）导航数据记录器
        self.attErrRec = None      # 姿态误差记录器
        self.attRefRec = None      # 参考姿态记录器
        self.rwSpeedRec = None     # 反作用轮转速记录器
        self.rwMotorRec = None     # 反作用轮力矩指令记录器
        self.scTruthRec = None     # 航天器 6-DOF 真值状态记录器
        self.sunStateRec = None    # 太阳星历真值记录器
        self.eclipseRec = None     # 入影/出影真值记录器
        self.magFieldRec = None    # 地磁场真值记录器
        self.clockSync = None      # 时钟同步模块（用于 Vizard 实时可视化加速）

        self.results_dir = (REPO_ROOT / config["output"]["results_dir"]).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 这两步是整个工程骨架的核心：
        # Dynamics 提供“真实世界”，FSW 提供“控制软件世界”。
        self.set_DynModel(BSK_Dynamics)     # 设置动力学模型为官方 BSK_Dynamics（包含引力、航天器刚体动力学等）
        self.configure_truth_model()        # 先配置刚体/轨道真值模型，再让 FSW 读取一致的航天器参数
        self.set_FswModel(BSK_Fsw)          # 设置飞控软件模型为官方 BSK_Fsw（包含导航滤波器、姿态控制器、执行机构分配等）
        update_fsw_vehicle_configuration(self.get_FswModel(), self.config)

        self.apply_control_gains()          # 应用配置文件中的控制增益到 FSW 中的 MRP 反馈控制器
        self.log_outputs()                  # 挂载所有必要的数据记录器
        self.configure_visualization()      # 根据配置启用并配置 Vizard 3D 可视化

    def configure_truth_model(self) -> None:
        """Configure the project-local truth model in the requested layering order."""
        dyn_model = self.get_DynModel()

        # 第 2 周开始，姿态刚体参数和轨道初值都属于 truth layer，
        # 不再放在 sensors helper 里。
        apply_rigid_body_truth_configuration(dyn_model, self.config)
        apply_orbit_truth_configuration(dyn_model, self.config)

    def apply_control_gains(self) -> None:
        """Apply the baseline hill-pointing mode settings."""
        # 这里不直接写控制器细节，而是交给 mode helper。
        # 好处是以后新增 inertial3D、sunSafe 时可以沿用同样入口。
        # 调用 mode helper 函数，传入 FSW 模型和配置字典
        apply_hill_point_control_gains(self.get_FswModel(), self.config)

    def log_outputs(self) -> None:
        """Record the minimum data products needed for baseline validation."""
        fsw_model = self.get_FswModel()
        dyn_model = self.get_DynModel()
        sampling_time = fsw_model.processTasksTimeStep

        # 真值层 recorder 直接读航天器传播状态，
        # 后续环境层和外扰层都基于它工作。
        self.scTruthRec = attach_truth_state_recorder(self, dyn_model, sampling_time)
        self.sunStateRec, self.eclipseRec, self.magFieldRec = attach_environment_recorders(
            self,
            dyn_model,
            sampling_time,
        )

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

    def configure_visualization(self) -> None:
        """Attach Vizard using the selected project configuration."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("enable_unity_viz", False):
            return

        clock_sync_accel_factor = viz_config.get("clock_sync_accel_factor")
        if clock_sync_accel_factor is not None:
            self.clockSync = simSynch.ClockSynch()
            self.clockSync.accelFactor = float(clock_sync_accel_factor)
            self.AddModelToTask(self.get_DynModel().taskName, self.clockSync)

        save_file = None
        if viz_config.get("save_file", False):
            raw_path = viz_config.get("save_file_path")
            if raw_path:
                save_path = Path(raw_path)
                if not save_path.is_absolute():
                    save_path = (REPO_ROOT / save_path).resolve()
            else:
                file_prefix = self.config["output"].get("file_prefix", self.name)
                save_path = self.results_dir / f"{file_prefix}_UnityViz.bin"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_file = str(save_path)

        dyn_models = self.get_DynModel()
        self.viz = vizSupport.enableUnityVisualization(
            self,
            dyn_models.taskName,
            dyn_models.scObject,
            rwEffectorList=dyn_models.rwStateEffector,
            saveFile=save_file,
            liveStream=bool(viz_config.get("live_stream", False)),
            broadcastStream=bool(viz_config.get("broadcast_stream", False)),
            noDisplay=bool(viz_config.get("no_display", False)),
        )

        if self.viz is None:
            return

        visual_settings = {
            "showSpacecraftLabels": viz_config.get("show_spacecraft_labels"),
            "showCelestialBodyLabels": viz_config.get("show_celestial_body_labels"),
            "spacecraftCSon": viz_config.get("spacecraft_cs_on"),
            "showHillFrame": viz_config.get("show_hill_frame"),
            "orbitLinesOn": viz_config.get("orbit_lines_on"),
            "trueTrajectoryLinesOn": viz_config.get("true_trajectory_lines_on"),
            "truePathRelativeBody": viz_config.get("true_path_relative_body"),
            "mainCameraTarget": viz_config.get("main_camera_target"),
        }
        for attr_name, value in visual_settings.items():
            if value is not None:
                setattr(self.viz.settings, attr_name, value)
                
    # 将记录器中的数据转换为图表，并根据参数决定是否显示和保存
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
            self.scTruthRec,
            self.sunStateRec,
            self.eclipseRec,
            self.magFieldRec,
            show_plots,
            save_plots,
        )


def run_scenario(config: dict[str, Any]) -> list[Path]:
    """Run the baseline hill-pointing scenario from the project config."""
    scenario = HillPointBaselineScenario(config)
    scenario.InitializeSimulation()

    # modeRequest 是官方 BSK_Sim 的模式入口。
    # 这里把“跑哪个任务模式”的决定留给配置文件和 mode helper。
    # 从配置中获取期望的飞控模式请求（此处应为 "hillPoint"），设置给场景
    scenario.modeRequest = get_mode_request(config)


    # 将仿真时长从分钟转换为纳秒，并设置停止时间
    duration_minutes = config["simulation"]["duration_minutes"]
    scenario.ConfigureStopTime(macros.min2nano(duration_minutes))
    scenario.ExecuteSimulation()

    return scenario.pull_outputs(
        show_plots=config["output"]["show_plots"],
        save_plots=config["output"]["save_plots"],
    )
