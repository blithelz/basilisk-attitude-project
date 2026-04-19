"""Top-level pure Python truth-model runner for the week-2 spacecraft model."""
# 顶层纯 Python 真值模型运行器，用于第 2 周的航天器模型。

from __future__ import annotations
# 启用对未来版本 Python 的类型注解特性（例如在类内部引用自身类名）。

from dataclasses import dataclass
# 从 dataclasses 模块导入 dataclass 装饰器，用于简化数据容器类的定义。
from pathlib import Path
# 从 pathlib 模块导入 Path 类，用于面向对象的文件系统路径操作。
from typing import Any
# 从 typing 模块导入 Any 类型，表示任意类型。

import json
# 导入 json 模块，用于处理 JSON 格式的数据。
import numpy as np
# 导入 numpy 库并简写为 np，用于高效的数值数组操作。
import yaml
# 导入 yaml 库，用于解析 YAML 配置文件。

from src.truth.attitude import AttitudeConfig, AttitudeHistory, AttitudeState, build_attitude_history, step_attitude_state
# 从姿态模块导入姿态配置、姿态历史记录、姿态状态数据结构以及构建姿态历史和步进姿态状态的函数。
from src.truth.disturbances import (
    DisturbanceConfig,
    DisturbanceHistory,
    build_disturbance_history,
    evaluate_disturbances,
)
# 从扰动模块导入扰动配置、扰动历史记录以及构建扰动历史和评估扰动的函数。
from src.truth.environment import EnvironmentConfig, EnvironmentHistory, build_environment_history, evaluate_environment
# 从环境模块导入环境配置、环境历史记录以及构建环境历史和评估环境的函数。
from src.truth.orbit import (
    OrbitConfig,
    OrbitHistory,
    OrbitState,
    build_orbit_history,
    orbital_elements_to_state,
    step_orbit_state,
)
# 从轨道模块导入轨道配置、轨道历史记录、轨道状态数据结构，以及构建轨道历史、将轨道根数转换为状态和步进轨道状态的函数。


@dataclass(frozen=True)
class TruthModelResult:
    """Container for the week-2 pure Python truth-model outputs."""
    # 用于存放第 2 周纯 Python 真值模型输出结果的容器类。

    orbit: OrbitHistory
    # 轨道历史记录对象。
    attitude: AttitudeHistory
    # 姿态历史记录对象。
    environment: EnvironmentHistory
    # 环境历史记录对象。
    disturbances: DisturbanceHistory
    # 扰动历史记录对象。


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its dictionary content."""
    # 加载一个 YAML 文件并返回其字典内容。

    with path.open("r", encoding="utf-8") as handle:
        # 以只读模式、UTF-8 编码打开指定路径的文件。
        data = yaml.safe_load(handle) or {}
        # 安全加载 YAML 内容，若文件为空则返回空字典。
    if not isinstance(data, dict):
        # 如果加载的数据不是字典类型。
        raise ValueError(f"Expected a mapping in {path}, but got {type(data).__name__}.")
        # 抛出 ValueError 异常，提示期望一个映射类型（字典）。
    return data
    # 返回解析后的字典数据。


class TruthModel:
    """Pure Python truth model that keeps propagation, environment, and disturbances separate."""
    # 纯 Python 真值模型，将轨道/姿态传播、环境计算和扰动计算保持分离。

    def __init__(
        self,
        orbit_config: OrbitConfig,
        attitude_config: AttitudeConfig,
        environment_config: EnvironmentConfig,
        disturbance_config: DisturbanceConfig,
    ) -> None:
        # 构造函数，接收轨道、姿态、环境和扰动四个配置对象。
        self.orbit_config = orbit_config
        # 保存轨道配置。
        self.attitude_config = attitude_config
        # 保存姿态配置。
        self.environment_config = environment_config
        # 保存环境配置。
        self.disturbance_config = disturbance_config
        # 保存扰动配置。

    @classmethod
    def from_config_files(
        cls,
        spacecraft_path: Path,
        orbit_path: Path,
        environment_path: Path,
    ) -> "TruthModel":
        # 类方法，从三个 YAML 配置文件路径构造 TruthModel 实例。
        spacecraft_config = load_yaml_config(spacecraft_path)
        # 加载航天器配置文件（包含姿态和扰动相关参数）。
        orbit_config_dict = load_yaml_config(orbit_path)
        # 加载轨道配置文件。
        environment_config_dict = load_yaml_config(environment_path)
        # 加载环境配置文件。

        orbit_config = OrbitConfig.from_dict(orbit_config_dict)
        # 从字典构造 OrbitConfig 对象。
        attitude_config = AttitudeConfig.from_dict(spacecraft_config)
        # 从航天器配置字典构造 AttitudeConfig 对象。
        environment_config = EnvironmentConfig.from_dict(
            environment_config_dict,
            central_body_radius_m=orbit_config.central_body_radius_m,
        )
        # 从环境配置字典构造 EnvironmentConfig 对象，需要传入中心天体半径（来自轨道配置）。
        disturbance_config = DisturbanceConfig.from_dict(spacecraft_config, environment_config_dict)
        # 从航天器配置和环境配置字典构造 DisturbanceConfig 对象。
        return cls(orbit_config, attitude_config, environment_config, disturbance_config)
        # 调用构造函数创建并返回 TruthModel 实例。

    def simulate(self) -> TruthModelResult:
        """Run the truth-model propagation and return the layered histories."""
        # 执行真值模型的传播计算，并返回分层的（轨道、姿态、环境、扰动）历史记录。

        time_s = np.arange(
            0.0,
            self.orbit_config.duration_s + self.orbit_config.step_size_s,
            self.orbit_config.step_size_s,
        )
        # 生成仿真时间序列，从 0 开始，步长为轨道配置中的步长，直到总时长（包含终点）。

        orbit_state = orbital_elements_to_state(
            self.orbit_config.initial_elements,
            self.orbit_config.mu_m3_s2,
        )
        # 根据初始轨道根数和引力常数计算初始轨道状态（位置、速度）。
        attitude_state = AttitudeState(
            sigma_bn=self.attitude_config.initial_sigma_bn.copy(),
            omega_bn_b_rad_s=self.attitude_config.initial_omega_bn_b_rad_s.copy(),
        )
        # 初始化姿态状态对象，复制初始 MRP 姿态参数和角速度向量（避免外部修改影响内部状态）。

        orbit_states: list[OrbitState] = []
        # 创建空列表，用于存储每个时间步的轨道状态。
        attitude_states: list[AttitudeState] = []
        # 创建空列表，用于存储每个时间步的姿态状态。
        environment_samples = []
        # 创建空列表，用于存储每个时间步的环境样本。
        disturbance_samples = []
        # 创建空列表，用于存储每个时间步的扰动样本。

        for index, current_time_s in enumerate(time_s):
            # 遍历每个时间步，同时获取索引和当前时间值。
            orbit_states.append(orbit_state)
            # 将当前轨道状态存入列表。
            attitude_states.append(attitude_state)
            # 将当前姿态状态存入列表。

            environment_sample = evaluate_environment(
                current_time_s,
                orbit_state.position_n_m,
                attitude_state,
                self.environment_config,
            )
            # 评估当前时刻的环境状态（如太阳方向、地磁场、光照条件）。
            disturbance_sample = evaluate_disturbances(
                current_time_s,
                orbit_state.position_n_m,
                orbit_state.velocity_n_m_s,
                attitude_state,
                self.attitude_config,
                environment_sample,
                self.environment_config,
                self.disturbance_config,
                self.orbit_config.mu_m3_s2,
                self.orbit_config.central_body_radius_m,
            )
            # 评估当前时刻的扰动力矩（如重力梯度、气动、磁力矩等）。
            environment_samples.append(environment_sample)
            # 将环境样本存入列表。
            disturbance_samples.append(disturbance_sample)
            # 将扰动样本存入列表。

            if index == len(time_s) - 1:
                break
                # 如果是最后一个时间步，跳过下一步状态更新，避免数组越界。

            orbit_state = step_orbit_state(
                orbit_state,
                self.orbit_config,
                self.orbit_config.step_size_s,
                current_time_s,
            )
            # 使用数值积分步进计算下一时刻的轨道状态。
            attitude_state = step_attitude_state(
                attitude_state,
                self.attitude_config,
                disturbance_sample.total_torque_b_nm,
                self.orbit_config.step_size_s,
                current_time_s,
            )
            # 使用数值积分步进计算下一时刻的姿态状态，需要当前步的扰动力矩。

        orbit_history = build_orbit_history(time_s, orbit_states, self.orbit_config)
        # 根据时间序列和轨道状态列表构建 OrbitHistory 对象。
        attitude_history = build_attitude_history(time_s, attitude_states, self.attitude_config)
        # 根据时间序列和姿态状态列表构建 AttitudeHistory 对象。
        environment_history = build_environment_history(environment_samples)
        # 根据环境样本列表构建 EnvironmentHistory 对象。
        disturbance_history = build_disturbance_history(disturbance_samples)
        # 根据扰动样本列表构建 DisturbanceHistory 对象。

        return TruthModelResult(
            orbit=orbit_history,
            attitude=attitude_history,
            environment=environment_history,
            disturbances=disturbance_history,
        )
        # 将四个历史记录对象打包成 TruthModelResult 并返回。


def save_truth_model_arrays(result: TruthModelResult, output_dir: Path) -> Path:
    """Save the truth-model arrays to a NumPy archive for later analysis."""
    # 将真值模型的结果数组保存为 NumPy 压缩存档（.npz），便于后续分析。

    output_dir.mkdir(parents=True, exist_ok=True)
    # 创建输出目录（如果不存在则创建，包括必要的父目录）。
    target = output_dir / "truth_model_results.npz"
    # 定义目标文件路径。
    np.savez(
        target,
        time_s=result.orbit.time_s,
        orbit_position_n_m=result.orbit.position_n_m,
        orbit_velocity_n_m_s=result.orbit.velocity_n_m_s,
        sigma_bn=result.attitude.sigma_bn,
        omega_bn_b_rad_s=result.attitude.omega_bn_b_rad_s,
        sun_direction_b=result.environment.sun_direction_b,
        magnetic_field_b_t=result.environment.magnetic_field_b_t,
        illumination=result.environment.illumination,
        total_torque_b_nm=result.disturbances.total_torque_b_nm,
    )
    # 使用 numpy.savez 保存多个数组到 .npz 文件中，每个数组使用给定的关键字命名。
    return target
    # 返回保存的文件路径。


def save_truth_model_summary(result: TruthModelResult, output_dir: Path) -> Path:
    """Save a short JSON summary of the truth-model run."""
    # 保存真值模型运行的简要统计摘要为 JSON 文件。

    output_dir.mkdir(parents=True, exist_ok=True)
    # 创建输出目录。
    target = output_dir / "truth_model_summary.json"
    # 定义 JSON 摘要文件路径。
    summary = {
        "duration_s": float(result.orbit.time_s[-1]),
        # 总仿真时长（秒）。
        "num_samples": int(result.orbit.time_s.size),
        # 采样点数量。
        "min_altitude_m": float(np.min(result.orbit.altitude_m)),
        # 最低轨道高度（米）。
        "max_altitude_m": float(np.max(result.orbit.altitude_m)),
        # 最高轨道高度（米）。
        "max_body_rate_rad_s": float(np.max(np.linalg.norm(result.attitude.omega_bn_b_rad_s, axis=1))),
        # 最大体坐标系角速率幅值（弧度/秒）。
        "max_disturbance_torque_nm": float(np.max(np.linalg.norm(result.disturbances.total_torque_b_nm, axis=1))),
        # 最大扰动力矩幅值（牛·米）。
        "fraction_in_eclipse": float(np.mean(result.environment.illumination < 0.5)),
        # 处于阴影中的时间比例（光照标志小于0.5的均值）。
    }
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # 将 summary 字典转为格式化的 JSON 字符串，并以 UTF-8 编码写入文件。
    return target
    # 返回保存的文件路径。