"""Pure Python orbital propagation for the week-2 truth model.

This module now serves two purposes:

1. the top section is the project-local orbit model that we wrote ourselves
2. the compatibility section at the bottom preserves the old Basilisk-based
   helper functions used by the existing baseline scenario
"""
# 模块文档字符串：为第二周真实模型提供纯 Python 实现的轨道递推。
#
# 本模块现在承担两个职责：
# 1. 上半部分是项目自行编写的纯 Python 轨道模型
# 2. 底部的兼容层保留了旧的基于 Basilisk 的辅助函数，
#    供现有的基准场景继续使用

from __future__ import annotations
# 启用推迟注解求值

from dataclasses import dataclass
# 导入 dataclass 装饰器，用于简洁定义数据类

from typing import Any
# 导入 Any 类型

import numpy as np
# 导入 numpy 并简写为 np

from src.utils.math_utils import rk4_step
# 从数学工具导入四阶龙格-库塔积分函数


# --- 默认天体常数 ---
DEFAULT_CENTRAL_BODY_MU_M3_S2 = 3.986004415e14
# 默认的中心天体引力常数（地球），单位：m³/s²

DEFAULT_CENTRAL_BODY_RADIUS_M = 6_378_136.6
# 默认的中心天体半径（地球赤道半径），单位：米

DEFAULT_J2 = 1.08262668e-3
# 默认的地球 J2 摄动系数


# ============================================================================
# 项目本地纯 Python 轨道模型
# ============================================================================

@dataclass(frozen=True)
class OrbitalElements:
    """Classical orbital elements used to define the initial orbit state."""
    # 用于定义初始轨道状态的经典开普勒轨道六根数，不可变数据类

    semi_major_axis_m: float
    # 半长轴，单位：米

    eccentricity: float
    # 偏心率，无量纲

    inclination_rad: float
    # 轨道倾角，单位：弧度

    raan_rad: float
    # 升交点赤经（RAAN），单位：弧度

    arg_perigee_rad: float
    # 近地点辐角，单位：弧度

    true_anomaly_rad: float
    # 真近点角，单位：弧度

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "OrbitalElements":
        # 从配置字典构造 OrbitalElements 实例
        element_cfg = config.get("initial_elements", config)
        # 若存在 "initial_elements" 键则取之，否则使用整个 config（兼容旧格式）
        return cls(
            semi_major_axis_m=float(element_cfg["semi_major_axis_m"]),
            eccentricity=float(element_cfg["eccentricity"]),
            inclination_rad=np.deg2rad(float(element_cfg["inclination_deg"])),
            raan_rad=np.deg2rad(float(element_cfg["raan_deg"])),
            arg_perigee_rad=np.deg2rad(float(element_cfg["arg_perigee_deg"])),
            true_anomaly_rad=np.deg2rad(float(element_cfg["true_anomaly_deg"])),
        )
        # 将配置中角度值（度）转换为弧度


@dataclass(frozen=True)
class OrbitConfig:
    """Configuration of the project-local orbital propagation model."""
    # 项目本地轨道递推模型的配置参数，不可变

    mu_m3_s2: float
    # 中心天体引力常数，单位：m³/s²

    central_body_radius_m: float
    # 中心天体半径，单位：米

    initial_elements: OrbitalElements
    # 初始轨道根数

    duration_s: float
    # 仿真总时长，单位：秒

    step_size_s: float
    # 积分步长，单位：秒

    use_j2: bool = False
    # 是否启用 J2 摄动

    j2: float = DEFAULT_J2
    # J2 系数，默认使用地球值

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "OrbitConfig":
        # 从配置字典构造 OrbitConfig 实例
        orbit_cfg = config.get("orbit", config)
        central_body_cfg = orbit_cfg.get("central_body", {})
        propagation_cfg = orbit_cfg.get("propagation", {})
        return cls(
            mu_m3_s2=float(central_body_cfg.get("mu_m3_s2", DEFAULT_CENTRAL_BODY_MU_M3_S2)),
            central_body_radius_m=float(central_body_cfg.get("radius_m", DEFAULT_CENTRAL_BODY_RADIUS_M)),
            initial_elements=OrbitalElements.from_dict(orbit_cfg),
            duration_s=float(propagation_cfg.get("duration_s", 600.0)),
            step_size_s=float(propagation_cfg.get("step_size_s", 1.0)),
            use_j2=bool(central_body_cfg.get("use_j2", False)),
            j2=float(central_body_cfg.get("j2", DEFAULT_J2)),
        )
        # 从配置中提取各字段，缺失时使用默认值


@dataclass(frozen=True)
class OrbitState:
    """State of the project-local orbital propagation model."""
    # 项目本地轨道递推模型的状态向量，不可变

    position_n_m: np.ndarray
    # 位置矢量（惯性系 N 下），单位：米

    velocity_n_m_s: np.ndarray
    # 速度矢量（惯性系 N 下），单位：米/秒

    def as_vector(self) -> np.ndarray:
        # 将状态打包为 6 维向量 [r; v]，便于积分器使用
        return np.concatenate((self.position_n_m, self.velocity_n_m_s))

    @classmethod
    def from_vector(cls, state_vector: np.ndarray) -> "OrbitState":
        # 从 6 维向量恢复为 OrbitState 对象
        state_vector = np.asarray(state_vector, dtype=float).reshape(6)
        return cls(position_n_m=state_vector[:3], velocity_n_m_s=state_vector[3:])


@dataclass(frozen=True)
class OrbitHistory:
    """Time history of the project-local orbit truth model."""
    # 项目本地轨道真值模型的时间历程记录，不可变

    time_s: np.ndarray
    # 时间序列，单位：秒

    time_min: np.ndarray
    # 时间序列，单位：分钟（便于绘图）

    position_n_m: np.ndarray
    # 位置历史，形状 (N, 3)，单位：米

    velocity_n_m_s: np.ndarray
    # 速度历史，形状 (N, 3)，单位：米/秒

    radius_m: np.ndarray
    # 地心距历史，形状 (N,)，单位：米

    altitude_m: np.ndarray
    # 轨道高度历史，形状 (N,)，单位：米

    speed_m_s: np.ndarray
    # 飞行速率历史，形状 (N,)，单位：米/秒

    specific_energy_j_kg: np.ndarray
    # 比机械能历史，形状 (N,)，单位：J/kg


def orbital_elements_to_state(elements: OrbitalElements, mu_m3_s2: float) -> OrbitState:
    """Convert classical orbital elements into inertial position and velocity."""
    # 将经典轨道根数转换为惯性系下的位置和速度矢量

    semi_latus_rectum_m = elements.semi_major_axis_m * (1.0 - elements.eccentricity ** 2)
    # 计算半通径：p = a (1 - e²)

    radius_perifocal_m = semi_latus_rectum_m / (1.0 + elements.eccentricity * np.cos(elements.true_anomaly_rad))
    # 计算轨道径向距离：r = p / (1 + e cos ν)

    position_perifocal_m = radius_perifocal_m * np.array(
        [
            np.cos(elements.true_anomaly_rad),
            np.sin(elements.true_anomaly_rad),
            0.0,
        ]
    )
    # 近焦点坐标系下的位置矢量：r_p = [r cos ν, r sin ν, 0]ᵀ

    velocity_perifocal_m_s = np.sqrt(mu_m3_s2 / semi_latus_rectum_m) * np.array(
        [
            -np.sin(elements.true_anomaly_rad),
            elements.eccentricity + np.cos(elements.true_anomaly_rad),
            0.0,
        ]
    )
    # 近焦点坐标系下的速度矢量：v_p = √(μ/p) * [-sin ν, e + cos ν, 0]ᵀ

    # 构造近焦点坐标系到惯性系的转换矩阵
    cos_raan = np.cos(elements.raan_rad)
    sin_raan = np.sin(elements.raan_rad)
    cos_inc = np.cos(elements.inclination_rad)
    sin_inc = np.sin(elements.inclination_rad)
    cos_arg_perigee = np.cos(elements.arg_perigee_rad)
    sin_arg_perigee = np.sin(elements.arg_perigee_rad)

    perifocal_to_inertial = np.array(
        [
            [
                cos_raan * cos_arg_perigee - sin_raan * sin_arg_perigee * cos_inc,
                -cos_raan * sin_arg_perigee - sin_raan * cos_arg_perigee * cos_inc,
                sin_raan * sin_inc,
            ],
            [
                sin_raan * cos_arg_perigee + cos_raan * sin_arg_perigee * cos_inc,
                -sin_raan * sin_arg_perigee + cos_raan * cos_arg_perigee * cos_inc,
                -cos_raan * sin_inc,
            ],
            [
                sin_arg_perigee * sin_inc,
                cos_arg_perigee * sin_inc,
                cos_inc,
            ],
        ]
    )
    # 3-1-3 欧拉角旋转矩阵（Ω, i, ω）

    position_n_m = perifocal_to_inertial @ position_perifocal_m
    velocity_n_m_s = perifocal_to_inertial @ velocity_perifocal_m_s
    # 将位置和速度从近焦点坐标系转换到惯性系
    return OrbitState(position_n_m=position_n_m, velocity_n_m_s=velocity_n_m_s)


def two_body_acceleration(position_n_m: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    """Return the two-body gravitational acceleration."""
    # 返回二体引力加速度（牛顿引力）

    position_n_m = np.asarray(position_n_m, dtype=float).reshape(3)
    radius_m = np.linalg.norm(position_n_m)
    if radius_m <= 0.0:
        return np.zeros(3)
    return -mu_m3_s2 * position_n_m / (radius_m ** 3)
    # a = -μ r / r³


def j2_acceleration(
    position_n_m: np.ndarray,
    mu_m3_s2: float,
    central_body_radius_m: float,
    j2: float,
) -> np.ndarray:
    """Return the J2 perturbation acceleration in inertial coordinates."""
    # 返回惯性系下的 J2 摄动加速度

    x_value, y_value, z_value = np.asarray(position_n_m, dtype=float).reshape(3)
    radius_sq_m2 = x_value ** 2 + y_value ** 2 + z_value ** 2
    radius_m = np.sqrt(radius_sq_m2)
    if radius_m <= 0.0:
        return np.zeros(3)

    z_ratio_sq = z_value ** 2 / radius_sq_m2
    scale = 1.5 * j2 * mu_m3_s2 * (central_body_radius_m ** 2) / (radius_m ** 5)
    return scale * np.array(
        [
            x_value * (5.0 * z_ratio_sq - 1.0),
            y_value * (5.0 * z_ratio_sq - 1.0),
            z_value * (5.0 * z_ratio_sq - 3.0),
        ]
    )
    # J2 加速度公式（惯性系，Z 轴沿极轴）


def orbit_state_derivative(time_s: float, state_vector: np.ndarray, config: OrbitConfig) -> np.ndarray:
    """Return the orbital state derivative."""
    # 返回轨道状态的导数 dr/dt = v, dv/dt = a

    del time_s  # 动力学不显含时间，显式忽略
    state = OrbitState.from_vector(state_vector)
    acceleration_n_m_s2 = two_body_acceleration(state.position_n_m, config.mu_m3_s2)
    if config.use_j2:
        acceleration_n_m_s2 += j2_acceleration(
            state.position_n_m,
            config.mu_m3_s2,
            config.central_body_radius_m,
            config.j2,
        )
    return np.concatenate((state.velocity_n_m_s, acceleration_n_m_s2))


def step_orbit_state(
    state: OrbitState,
    config: OrbitConfig,
    step_size_s: float,
    time_s: float = 0.0,
) -> OrbitState:
    """Advance the orbit state by one fixed integration step."""
    # 使用固定步长的 RK4 方法将轨道状态向前积分一步

    next_state_vector = rk4_step(
        lambda current_time_s, current_state_vector: orbit_state_derivative(
            current_time_s,
            current_state_vector,
            config,
        ),
        time_s,
        state.as_vector(),
        step_size_s,
    )
    return OrbitState.from_vector(next_state_vector)


def build_orbit_history(time_s: np.ndarray, states: list[OrbitState], config: OrbitConfig) -> OrbitHistory:
    """Convert a list of propagated orbit states into time-history arrays."""
    # 将递推得到的状态列表转换为结构化的时间历程数组

    position_n_m = np.asarray([state.position_n_m for state in states], dtype=float)
    velocity_n_m_s = np.asarray([state.velocity_n_m_s for state in states], dtype=float)
    radius_m = np.linalg.norm(position_n_m, axis=1)
    speed_m_s = np.linalg.norm(velocity_n_m_s, axis=1)
    specific_energy_j_kg = 0.5 * speed_m_s ** 2 - config.mu_m3_s2 / radius_m
    # 比机械能：ε = v²/2 - μ/r
    return OrbitHistory(
        time_s=np.asarray(time_s, dtype=float),
        time_min=np.asarray(time_s, dtype=float) / 60.0,
        position_n_m=position_n_m,
        velocity_n_m_s=velocity_n_m_s,
        radius_m=radius_m,
        altitude_m=radius_m - config.central_body_radius_m,
        speed_m_s=speed_m_s,
        specific_energy_j_kg=specific_energy_j_kg,
    )


def propagate_orbit(config: OrbitConfig) -> OrbitHistory:
    """Propagate the orbit state history with the project-local model."""
    # 使用项目本地模型递推轨道状态历史

    time_s = np.arange(0.0, config.duration_s + config.step_size_s, config.step_size_s)
    # 生成时间网格
    state = orbital_elements_to_state(config.initial_elements, config.mu_m3_s2)
    states = []
    for index, current_time_s in enumerate(time_s):
        states.append(state)
        if index == len(time_s) - 1:
            break
        state = step_orbit_state(state, config, config.step_size_s, current_time_s)
    return build_orbit_history(time_s, states, config)


# ============================================================================
# 遗留 Basilisk 兼容层
# ============================================================================

try:
    from Basilisk.utilities import macros as bsk_macros
    from Basilisk.utilities import orbitalMotion as bsk_orbital_motion
except ImportError:  # pragma: no cover - the pure Python model does not require Basilisk
    bsk_macros = None
    bsk_orbital_motion = None
# 尝试导入 Basilisk 模块，若失败则设为 None
# 这允许本模块在不安装 Basilisk 的环境下仍能使用纯 Python 部分


def _require_basilisk() -> None:
    # 内部辅助函数：确保 Basilisk 可用，否则抛出 ImportError
    if bsk_macros is None or bsk_orbital_motion is None:
        raise ImportError(
            "The legacy Basilisk compatibility helpers require the Basilisk package, "
            "but it is not available in the current Python environment."
        )


@dataclass(frozen=True)
class OrbitTruthHistory:
    """Legacy Basilisk truth-history container kept for the existing baseline scenario."""
    # 为兼容现有基准场景而保留的旧版 Basilisk 真值历史数据容器

    time_ns: np.ndarray
    # 时间序列（纳秒）

    time_min: np.ndarray
    # 时间序列（分钟）

    r_bn_n: np.ndarray
    # 位置矢量历史，形状 (N, 3)

    v_bn_n: np.ndarray
    # 速度矢量历史，形状 (N, 3)

    radius_m: np.ndarray
    # 地心距历史

    altitude_m: np.ndarray
    # 轨道高度历史

    speed_m_s: np.ndarray
    # 飞行速率历史


def get_central_body_constants(config: dict[str, Any]) -> tuple[float, float]:
    """Return the configured central-body gravitational parameter and radius."""
    # 返回配置中的中心天体引力常数和半径

    central_body_cfg = config.get("truth_model", {}).get("central_body", {})
    mu_m3_s2 = float(central_body_cfg.get("mu_m3_s2", DEFAULT_CENTRAL_BODY_MU_M3_S2))
    radius_m = float(central_body_cfg.get("radius_m", DEFAULT_CENTRAL_BODY_RADIUS_M))
    return mu_m3_s2, radius_m


def apply_orbit_truth_configuration(dyn_model: Any, config: dict[str, Any]) -> None:
    """Legacy helper that applies the configured orbit to the Basilisk spacecraft hub."""
    # 旧版辅助函数：将配置中的轨道参数应用到 Basilisk 航天器中心体

    _require_basilisk()
    orbit_cfg = config.get("orbit", {})
    orbit_elements = bsk_orbital_motion.ClassicElements()
    orbit_elements.a = float(orbit_cfg["a_m"])
    orbit_elements.e = float(orbit_cfg["e"])
    orbit_elements.i = float(orbit_cfg["i_deg"]) * bsk_macros.D2R
    orbit_elements.Omega = float(orbit_cfg["Omega_deg"]) * bsk_macros.D2R
    orbit_elements.omega = float(orbit_cfg["omega_deg"]) * bsk_macros.D2R
    orbit_elements.f = float(orbit_cfg["f_deg"]) * bsk_macros.D2R

    mu_m3_s2 = dyn_model.gravFactory.gravBodies["earth"].mu
    position_n_m, velocity_n_m_s = bsk_orbital_motion.elem2rv(mu_m3_s2, orbit_elements)
    dyn_model.scObject.hub.r_CN_NInit = position_n_m
    dyn_model.scObject.hub.v_CN_NInit = velocity_n_m_s


def extract_orbit_truth_history(sc_truth_rec: Any, config: dict[str, Any]) -> OrbitTruthHistory:
    """Legacy helper that extracts orbit truth arrays from a Basilisk recorder."""
    # 旧版辅助函数：从 Basilisk 记录器中提取轨道真值数组

    _require_basilisk()
    _, central_body_radius_m = get_central_body_constants(config)
    time_ns = np.delete(sc_truth_rec.times(), 0, 0)
    r_bn_n = np.delete(sc_truth_rec.r_BN_N, 0, 0)
    v_bn_n = np.delete(sc_truth_rec.v_BN_N, 0, 0)
    radius_m = np.linalg.norm(r_bn_n, axis=1)
    speed_m_s = np.linalg.norm(v_bn_n, axis=1)
    return OrbitTruthHistory(
        time_ns=time_ns,
        time_min=time_ns * bsk_macros.NANO2MIN,
        r_bn_n=r_bn_n,
        v_bn_n=v_bn_n,
        radius_m=radius_m,
        altitude_m=radius_m - central_body_radius_m,
        speed_m_s=speed_m_s,
    )