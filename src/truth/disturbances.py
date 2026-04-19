"""Pure Python disturbance-torque model for the week-2 truth model."""
# 模块文档字符串：为第二周真实模型提供纯 Python 实现的干扰力矩计算

from __future__ import annotations
# 启用推迟注解求值

from dataclasses import dataclass
# 导入 dataclass 装饰器，用于简洁定义数据类

import numpy as np
# 导入 numpy 并简写为 np

from src.truth.attitude import AttitudeConfig, AttitudeState
# 从姿态真值模块导入姿态配置类和状态类

from src.truth.environment import EnvironmentConfig, EnvironmentSample, compute_exponential_density
# 从环境真值模块导入环境配置类、环境采样类和指数大气密度计算函数

from src.utils.frames import orbital_frame_dcm, rotate_inertial_to_body
# 从参考系工具导入轨道坐标系DCM计算函数和惯性到体坐标系旋转函数


@dataclass(frozen=True)
class DisturbanceConfig:
    """Configuration for the project-local disturbance torque model."""
    # 项目本地干扰力矩模型的配置参数类，不可变

    drag_coefficient: float
    # 大气阻力系数，无量纲，典型值 2.0~2.5

    drag_area_m2: float
    # 大气阻力迎风面积，单位：m²

    srp_area_m2: float
    # 太阳光压作用面积，单位：m²

    center_of_pressure_b_m: np.ndarray
    # 压心在体坐标系中的位置矢量，单位：m（形状 (3,)）

    reflectivity_coefficient: float
    # 表面反射系数，取值范围 0（全吸收）到 2（全反射），典型值 1.0~1.5

    residual_dipole_b_a_m2: np.ndarray
    # 残余磁偶极矩在体坐标系中的矢量，单位：A·m²

    constant_bias_torque_b_nm: np.ndarray
    # 常值偏置力矩（体坐标系），用于模拟未建模的恒定干扰，单位：N·m

    enable_gravity_gradient: bool
    # 是否启用重力梯度力矩

    enable_drag: bool
    # 是否启用气动阻力矩

    enable_srp: bool
    # 是否启用太阳光压力矩

    enable_magnetic: bool
    # 是否启用剩余磁偶极矩产生的磁力矩

    @classmethod
    def from_dict(cls, spacecraft_config: dict, environment_config: dict) -> "DisturbanceConfig":
        # 从航天器和环境配置字典构造 DisturbanceConfig 实例
        spacecraft_cfg = spacecraft_config.get("spacecraft", spacecraft_config)
        geometry_cfg = spacecraft_cfg.get("geometry", {})
        disturbance_cfg = spacecraft_cfg.get("disturbances", {})
        magnetic_cfg = spacecraft_cfg.get("magnetic", {})
        solar_radiation_cfg = environment_config.get("environment", environment_config).get("solar_radiation", {})
        # 逐层提取各子配置，若不存在则返回空字典

        return cls(
            drag_coefficient=float(disturbance_cfg.get("drag_coefficient", 2.2)),
            drag_area_m2=float(geometry_cfg.get("drag_area_m2", 0.12)),
            srp_area_m2=float(geometry_cfg.get("srp_area_m2", 0.10)),
            center_of_pressure_b_m=np.asarray(geometry_cfg.get("center_of_pressure_b_m", [0.02, 0.0, 0.01]), dtype=float).reshape(3),
            reflectivity_coefficient=float(solar_radiation_cfg.get("reflectivity_coefficient", 1.3)),
            residual_dipole_b_a_m2=np.asarray(magnetic_cfg.get("residual_dipole_b_a_m2", [0.05, -0.03, 0.02]), dtype=float).reshape(3),
            constant_bias_torque_b_nm=np.asarray(disturbance_cfg.get("constant_bias_torque_b_nm", [0.0, 0.0, 0.0]), dtype=float).reshape(3),
            enable_gravity_gradient=bool(disturbance_cfg.get("gravity_gradient", {}).get("enabled", True)),
            enable_drag=bool(disturbance_cfg.get("drag", {}).get("enabled", True)),
            enable_srp=bool(disturbance_cfg.get("solar_radiation_pressure", {}).get("enabled", True)),
            enable_magnetic=bool(disturbance_cfg.get("magnetic_residual_dipole", {}).get("enabled", True)),
        )
        # 返回配置实例，所有缺失字段均使用合理的默认值


@dataclass(frozen=True)
class DisturbanceSample:
    """Single-sample disturbance torque values."""
    # 单个时刻的干扰力矩采样值

    time_s: float
    # 当前采样时刻，单位：秒

    gravity_gradient_torque_b_nm: np.ndarray
    # 重力梯度力矩（体坐标系），单位：N·m

    drag_torque_b_nm: np.ndarray
    # 气动阻力矩（体坐标系），单位：N·m

    srp_torque_b_nm: np.ndarray
    # 太阳光压力矩（体坐标系），单位：N·m

    magnetic_torque_b_nm: np.ndarray
    # 磁力矩（体坐标系），单位：N·m

    constant_bias_torque_b_nm: np.ndarray
    # 常值偏置力矩（体坐标系），单位：N·m

    total_torque_b_nm: np.ndarray
    # 总干扰力矩（体坐标系），即上述各项之和，单位：N·m


@dataclass(frozen=True)
class DisturbanceHistory:
    """Time history of the project-local disturbance torque model."""
    # 项目本地干扰力矩模型的时间历程记录

    time_s: np.ndarray
    # 时间序列，单位：秒

    gravity_gradient_torque_b_nm: np.ndarray
    # 重力梯度力矩历史，形状 (N, 3)

    drag_torque_b_nm: np.ndarray
    # 气动阻力矩历史，形状 (N, 3)

    srp_torque_b_nm: np.ndarray
    # 太阳光压力矩历史，形状 (N, 3)

    magnetic_torque_b_nm: np.ndarray
    # 磁力矩历史，形状 (N, 3)

    constant_bias_torque_b_nm: np.ndarray
    # 常值偏置力矩历史，形状 (N, 3)

    total_torque_b_nm: np.ndarray
    # 总干扰力矩历史，形状 (N, 3)


def compute_gravity_gradient_torque_b_nm(
    position_n_m: np.ndarray,
    sigma_bn: np.ndarray,
    inertia_kg_m2: np.ndarray,
    mu_m3_s2: float,
) -> np.ndarray:
    """Return the gravity-gradient torque in the body frame."""
    # 计算体坐标系下的重力梯度力矩

    radius_m = np.linalg.norm(position_n_m)
    # 计算航天器到中心天体的距离
    if radius_m <= 0.0:
        return np.zeros(3)
    # 距离为零时返回零力矩

    radius_hat_b = rotate_inertial_to_body(np.asarray(position_n_m, dtype=float) / radius_m, sigma_bn)
    # 将惯性系下的径向单位矢量旋转到体坐标系
    inertia_times_radius = inertia_kg_m2 @ radius_hat_b
    # 计算 I·r̂ 项
    return 3.0 * mu_m3_s2 / (radius_m ** 3) * np.cross(radius_hat_b, inertia_times_radius)
    # 根据公式 τ_gg = (3μ / r³) * (r̂ × (I·r̂)) 计算重力梯度力矩


def evaluate_disturbances(
    time_s: float,
    position_n_m: np.ndarray,
    velocity_n_m_s: np.ndarray,
    attitude_state: AttitudeState,
    attitude_config: AttitudeConfig,
    environment_sample: EnvironmentSample,
    environment_config: EnvironmentConfig,
    disturbance_config: DisturbanceConfig,
    mu_m3_s2: float,
    central_body_radius_m: float,
) -> DisturbanceSample:
    """Evaluate all configured disturbance torques at one simulation time."""
    # 在单个仿真时刻计算所有已启用的干扰力矩

    del central_body_radius_m
    # 该参数暂未使用，显式忽略以避免未使用变量警告

    zero_vector = np.zeros(3)
    # 初始化零向量

    gravity_gradient_torque_b_nm = zero_vector.copy()
    drag_torque_b_nm = zero_vector.copy()
    srp_torque_b_nm = zero_vector.copy()
    magnetic_torque_b_nm = zero_vector.copy()
    # 各项力矩初始化为零

    if disturbance_config.enable_gravity_gradient:
        gravity_gradient_torque_b_nm = compute_gravity_gradient_torque_b_nm(
            position_n_m,
            attitude_state.sigma_bn,
            attitude_config.inertia_kg_m2,
            mu_m3_s2,
        )
        # 若启用，计算重力梯度力矩

    if disturbance_config.enable_drag:
        altitude_m = np.linalg.norm(position_n_m) - environment_config.central_body_radius_m
        # 计算轨道高度
        density_kg_m3 = compute_exponential_density(altitude_m, environment_config)
        # 根据指数模型计算大气密度
        velocity_b_m_s = rotate_inertial_to_body(velocity_n_m_s, attitude_state.sigma_bn)
        # 将惯性系速度旋转到体坐标系
        drag_force_b_n = (
            -0.5
            * density_kg_m3
            * disturbance_config.drag_coefficient
            * disturbance_config.drag_area_m2
            * np.linalg.norm(velocity_b_m_s)
            * velocity_b_m_s
        )
        # 计算气动阻力：F_drag = -0.5 * ρ * C_d * A * |v| * v
        drag_torque_b_nm = np.cross(disturbance_config.center_of_pressure_b_m, drag_force_b_n)
        # 力矩 = 压心位置 × 阻力

    if disturbance_config.enable_srp:
        srp_force_b_n = (
            -environment_config.solar_pressure_n_m2
            * disturbance_config.reflectivity_coefficient
            * disturbance_config.srp_area_m2
            * environment_sample.illumination
            * environment_sample.sun_direction_b
        )
        # 计算太阳光压力：F_srp = -P_solar * C_r * A * (光照指示) * ŝ
        srp_torque_b_nm = np.cross(disturbance_config.center_of_pressure_b_m, srp_force_b_n)
        # 力矩 = 压心位置 × 光压力

    if disturbance_config.enable_magnetic:
        magnetic_torque_b_nm = np.cross(
            disturbance_config.residual_dipole_b_a_m2,
            environment_sample.magnetic_field_b_t,
        )
        # 磁力矩 = m_res × B，其中 m_res 为剩余磁矩，B 为地磁场

    constant_bias_torque_b_nm = disturbance_config.constant_bias_torque_b_nm.copy()
    # 获取常值偏置力矩（复制以保证不可变性）

    total_torque_b_nm = (
        gravity_gradient_torque_b_nm
        + drag_torque_b_nm
        + srp_torque_b_nm
        + magnetic_torque_b_nm
        + constant_bias_torque_b_nm
    )
    # 总力矩为各项之和

    return DisturbanceSample(
        time_s=float(time_s),
        gravity_gradient_torque_b_nm=gravity_gradient_torque_b_nm,
        drag_torque_b_nm=drag_torque_b_nm,
        srp_torque_b_nm=srp_torque_b_nm,
        magnetic_torque_b_nm=magnetic_torque_b_nm,
        constant_bias_torque_b_nm=constant_bias_torque_b_nm,
        total_torque_b_nm=total_torque_b_nm,
    )
    # 返回封装好的单个采样点数据


def build_disturbance_history(samples: list[DisturbanceSample]) -> DisturbanceHistory:
    """Convert a list of disturbance samples into time-history arrays."""
    # 将干扰力矩采样列表转换为结构化的时间历程数组

    return DisturbanceHistory(
        time_s=np.asarray([sample.time_s for sample in samples], dtype=float),
        gravity_gradient_torque_b_nm=np.asarray([sample.gravity_gradient_torque_b_nm for sample in samples], dtype=float),
        drag_torque_b_nm=np.asarray([sample.drag_torque_b_nm for sample in samples], dtype=float),
        srp_torque_b_nm=np.asarray([sample.srp_torque_b_nm for sample in samples], dtype=float),
        magnetic_torque_b_nm=np.asarray([sample.magnetic_torque_b_nm for sample in samples], dtype=float),
        constant_bias_torque_b_nm=np.asarray([sample.constant_bias_torque_b_nm for sample in samples], dtype=float),
        total_torque_b_nm=np.asarray([sample.total_torque_b_nm for sample in samples], dtype=float),
    )
    # 将所有采样数据堆叠为多维数组并返回历史记录对象