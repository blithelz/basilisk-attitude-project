"""Pure Python environment model for the week-2 truth model."""
# 模块文档字符串：为第二周真实模型提供纯 Python 实现的环境真值计算

from __future__ import annotations
# 启用推迟注解求值，允许在类型提示中使用尚未定义的类型名

from dataclasses import dataclass
# 导入 dataclass 装饰器，用于简洁定义数据类

import numpy as np
# 导入 numpy 并简写为 np，用于数值计算和数组操作

from src.truth.attitude import AttitudeState
# 从姿态真值模块导入姿态状态类

from src.utils.frames import rotate_inertial_to_body
# 从参考系工具导入惯性系到体坐标系的旋转函数

from src.utils.math_utils import safe_normalize
# 从数学工具导入安全归一化函数，用于防止零向量除零错误


MU0_OVER_4PI = 1.0e-7
# 真空磁导率常数除以 4π，即 μ₀/(4π) = 1e-7，单位：N/A²（或 H/m）
# 在磁偶极子场公式中作为系数使用


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for the project-local environment truth model."""
    # 项目本地环境真值模型的配置参数类，不可变

    sun_direction_n: np.ndarray
    # 太阳方向单位矢量（惯性系 N 下），形状 (3,)

    central_body_radius_m: float
    # 中心天体（地球）的半径，单位：米

    magnetic_dipole_n_a_m2: np.ndarray
    # 中心磁偶极矩矢量（惯性系 N 下），单位：A·m²
    # 用于计算地磁场（中心偶极子模型）

    atmosphere_reference_density_kg_m3: float
    # 大气参考密度，单位：kg/m³
    # 指数模型中在参考高度处的密度

    atmosphere_reference_altitude_m: float
    # 大气参考高度，单位：米

    atmosphere_scale_height_m: float
    # 大气标高（scale height），单位：米
    # 描述密度随高度指数衰减的尺度

    solar_pressure_n_m2: float
    # 太阳光压常数（单位面积上的力），单位：N/m²
    # 典型值约为 4.56e-6 N/m²（1 AU 处）

    enable_eclipse: bool
    # 是否启用星食（地影）检测，若关闭则始终认为受晒

    @classmethod
    def from_dict(cls, config: dict, central_body_radius_m: float) -> "EnvironmentConfig":
        # 类方法：从配置字典和中心天体半径构造 EnvironmentConfig 实例
        environment_cfg = config.get("environment", config)
        # 获取 environment 子配置，若不存在则回退到整个 config

        sun_cfg = environment_cfg.get("sun", {})
        magnetic_cfg = environment_cfg.get("magnetic_field", {})
        atmosphere_cfg = environment_cfg.get("atmosphere", {})
        solar_radiation_cfg = environment_cfg.get("solar_radiation", {})
        # 提取各子配置节

        return cls(
            sun_direction_n=safe_normalize(np.asarray(sun_cfg.get("initial_direction_n", [1.0, 0.2, 0.1]), dtype=float)),
            # 太阳初始方向矢量，安全归一化（默认值略偏离纯 X 轴以模拟一般情况）

            central_body_radius_m=float(central_body_radius_m),
            # 中心天体半径

            magnetic_dipole_n_a_m2=np.asarray(
                magnetic_cfg.get("dipole_moment_n_a_m2", [0.0, 0.0, 7.94e22]),
                dtype=float,
            ).reshape(3),
            # 磁偶极矩，默认值对应地球磁矩大小（沿 Z 轴）

            atmosphere_reference_density_kg_m3=float(atmosphere_cfg.get("reference_density_kg_m3", 3.5e-12)),
            atmosphere_reference_altitude_m=float(atmosphere_cfg.get("reference_altitude_m", 400000.0)),
            atmosphere_scale_height_m=float(atmosphere_cfg.get("scale_height_m", 60000.0)),
            # 大气模型参数，默认值适用于 400 km 高度的典型 LEO 环境

            solar_pressure_n_m2=float(solar_radiation_cfg.get("pressure_n_m2", 4.56e-6)),
            # 太阳光压常数

            enable_eclipse=bool(environment_cfg.get("eclipse", {}).get("enabled", True)),
            # 星食开关，默认启用
        )


@dataclass(frozen=True)
class EnvironmentSample:
    """Single-sample environment truth values."""
    # 单个采样时刻的环境真值数据

    time_s: float
    # 当前时刻，单位：秒

    sun_direction_n: np.ndarray
    # 太阳方向单位矢量（惯性系 N）

    sun_direction_b: np.ndarray
    # 太阳方向单位矢量（体坐标系 B）

    magnetic_field_n_t: np.ndarray
    # 地磁场矢量（惯性系 N），单位：特斯拉

    magnetic_field_b_t: np.ndarray
    # 地磁场矢量（体坐标系 B），单位：特斯拉

    illumination: float
    # 光照指示因子：1.0 表示受晒（全光压），0.0 表示地影（无光压）


@dataclass(frozen=True)
class EnvironmentHistory:
    """Time history of the project-local environment truth model."""
    # 项目本地环境真值模型的时间历程记录

    time_s: np.ndarray
    # 时间序列，单位：秒

    sun_direction_n: np.ndarray
    # 太阳方向历史（惯性系），形状 (N, 3)

    sun_direction_b: np.ndarray
    # 太阳方向历史（体坐标系），形状 (N, 3)

    magnetic_field_n_t: np.ndarray
    # 地磁场历史（惯性系），形状 (N, 3)

    magnetic_field_b_t: np.ndarray
    # 地磁场历史（体坐标系），形状 (N, 3)

    illumination: np.ndarray
    # 光照指示历史，形状 (N,)


def compute_eclipse_factor(position_n_m: np.ndarray, sun_direction_n: np.ndarray, central_body_radius_m: float) -> float:
    """Return a binary eclipse factor using a cylindrical shadow approximation."""
    # 使用圆柱形地影近似计算二值星食因子（0 或 1）

    position_n_m = np.asarray(position_n_m, dtype=float).reshape(3)
    sun_direction_n = safe_normalize(sun_direction_n)
    # 确保输入为正确形状和类型

    if np.dot(position_n_m, sun_direction_n) >= 0.0:
        return 1.0
    # 若航天器在中心天体的“阳面”一侧（位置矢量与太阳方向点积为正），则受晒

    sun_line_distance_m = np.linalg.norm(np.cross(position_n_m, sun_direction_n))
    # 计算航天器到太阳-中心天体连线的距离
    # 叉乘的模即为点到直线的距离
    return 0.0 if sun_line_distance_m <= central_body_radius_m else 1.0
    # 若距离小于等于中心天体半径，认为进入地影（圆柱模型），否则受晒


def compute_centered_dipole_field_n(position_n_m: np.ndarray, magnetic_dipole_n_a_m2: np.ndarray) -> np.ndarray:
    """Return the centered-dipole magnetic field in inertial coordinates."""
    # 返回惯性系下中心磁偶极子产生的磁场矢量

    position_n_m = np.asarray(position_n_m, dtype=float).reshape(3)
    magnetic_dipole_n_a_m2 = np.asarray(magnetic_dipole_n_a_m2, dtype=float).reshape(3)
    radius_m = np.linalg.norm(position_n_m)
    if radius_m <= 0.0:
        return np.zeros(3)
    # 距离为零时返回零场

    radius_hat_n = position_n_m / radius_m
    # 径向单位矢量
    dipole_projection = np.dot(magnetic_dipole_n_a_m2, radius_hat_n)
    # 磁偶极矩在径向方向的投影
    return MU0_OVER_4PI * (3.0 * dipole_projection * radius_hat_n - magnetic_dipole_n_a_m2) / (radius_m ** 3)
    # 中心偶极子磁场公式：B = (μ₀/(4π)) * ( 3 (m·r̂) r̂ - m ) / r³


def compute_exponential_density(altitude_m: float, config: EnvironmentConfig) -> float:
    """Return the local atmospheric density from a simple exponential model."""
    # 根据简化的指数模型计算当地大气密度

    return config.atmosphere_reference_density_kg_m3 * np.exp(
        -(float(altitude_m) - config.atmosphere_reference_altitude_m) / config.atmosphere_scale_height_m
    )
    # ρ = ρ₀ * exp( - (h - h₀) / H )


def evaluate_environment(
    time_s: float,
    position_n_m: np.ndarray,
    attitude_state: AttitudeState,
    config: EnvironmentConfig,
) -> EnvironmentSample:
    """Evaluate the environment truth at a single simulation time."""
    # 在单个仿真时刻计算环境真值，返回 EnvironmentSample 对象

    sun_direction_n = safe_normalize(config.sun_direction_n)
    # 太阳方向归一化（假设在短时间内方向不变）
    sun_direction_b = rotate_inertial_to_body(sun_direction_n, attitude_state.sigma_bn)
    # 将太阳方向旋转到体坐标系

    magnetic_field_n_t = compute_centered_dipole_field_n(position_n_m, config.magnetic_dipole_n_a_m2)
    # 计算惯性系下的地磁场
    magnetic_field_b_t = rotate_inertial_to_body(magnetic_field_n_t, attitude_state.sigma_bn)
    # 旋转到体坐标系

    illumination = (
        compute_eclipse_factor(position_n_m, sun_direction_n, config.central_body_radius_m)
        if config.enable_eclipse
        else 1.0
    )
    # 如果启用星食检测则计算光照因子，否则恒为 1.0

    return EnvironmentSample(
        time_s=float(time_s),
        sun_direction_n=sun_direction_n,
        sun_direction_b=sun_direction_b,
        magnetic_field_n_t=magnetic_field_n_t,
        magnetic_field_b_t=magnetic_field_b_t,
        illumination=illumination,
    )
    # 返回封装好的环境采样数据


def build_environment_history(samples: list[EnvironmentSample]) -> EnvironmentHistory:
    """Convert a list of environment samples into time-history arrays."""
    # 将环境采样列表转换为结构化的时间历程数组

    return EnvironmentHistory(
        time_s=np.asarray([sample.time_s for sample in samples], dtype=float),
        sun_direction_n=np.asarray([sample.sun_direction_n for sample in samples], dtype=float),
        sun_direction_b=np.asarray([sample.sun_direction_b for sample in samples], dtype=float),
        magnetic_field_n_t=np.asarray([sample.magnetic_field_n_t for sample in samples], dtype=float),
        magnetic_field_b_t=np.asarray([sample.magnetic_field_b_t for sample in samples], dtype=float),
        illumination=np.asarray([sample.illumination for sample in samples], dtype=float),
    )
    # 将各列表中的数组堆叠为多维数组并返回历史记录对象