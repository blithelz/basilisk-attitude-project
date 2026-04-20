"""Pure Python rigid-body attitude dynamics for the week-2 truth model."""
# 模块文档字符串：为第二周真实模型提供纯 Python 实现的刚体姿态动力学

from __future__ import annotations
# 启用推迟注解求值，允许在类型提示中使用尚未定义的类型名

from dataclasses import dataclass
# 导入 dataclass 装饰器，用于简洁定义数据类

import numpy as np
# 导入 numpy 并简写为 np，用于数值计算和数组操作

from src.utils.frames import mrp_derivative, mrp_to_dcm, switch_to_shadow_mrp
# 从参考系工具模块导入：
# - mrp_derivative：计算 MRP 姿态运动学导数
# - mrp_to_dcm：将 MRP 转换为方向余弦矩阵（DCM）
# - switch_to_shadow_mrp：将 MRP 切换到影子集以避免奇异

from src.utils.math_utils import rk4_step
# 从数值工具模块导入四阶龙格-库塔积分函数


@dataclass(frozen=True)
class AttitudeConfig:
    """Configuration for the project-local rigid-body attitude model."""
    # 项目本地刚体姿态模型的配置参数类，不可变（frozen=True）

    inertia_kg_m2: np.ndarray
    # 航天器转动惯量矩阵，形状 (3, 3)，单位：kg·m²

    initial_sigma_bn: np.ndarray
    # 初始 MRP 姿态参数（体坐标系 B 相对于惯性系 N），形状 (3,)

    initial_omega_bn_b_rad_s: np.ndarray
    # 初始角速度（体坐标系 B 相对于惯性系 N，在体坐标系下表示），单位：rad/s

    @classmethod
    def from_dict(cls, config: dict) -> "AttitudeConfig":
        # 类方法：从配置字典构建 AttitudeConfig 实例
        spacecraft_cfg = config.get("spacecraft", config)
        # 获取 spacecraft 子配置，若不存在则回退到整个 config（兼容旧格式）

        initial_attitude_cfg = spacecraft_cfg.get("initial_attitude", {})
        # 获取初始姿态子配置

        inertia = np.asarray(spacecraft_cfg["inertia_kg_m2"], dtype=float).reshape(3, 3)
        # 提取转动惯量并确保为 3×3 浮点数组

        sigma_bn = np.asarray(initial_attitude_cfg.get("mrp_bn", [0.1, 0.2, -0.3]), dtype=float).reshape(3)
        # 提取初始 MRP，若未指定则使用默认值 [0.1, 0.2, -0.3]

        omega_bn_b_rad_s = np.asarray(
            initial_attitude_cfg.get("omega_bn_b_rad_s", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(3)
        # 提取初始角速度，若未指定则默认为零向量

        return cls(
            inertia_kg_m2=inertia,
            initial_sigma_bn=sigma_bn,
            initial_omega_bn_b_rad_s=omega_bn_b_rad_s,
        )
        # 返回构造的配置实例


@dataclass(frozen=True)
class AttitudeState:
    """State of the project-local rigid-body attitude model."""
    # 项目本地刚体姿态模型的状态类，不可变

    sigma_bn: np.ndarray
    # MRP 姿态参数（3维向量）

    omega_bn_b_rad_s: np.ndarray
    # 体坐标系相对于惯性系的角速度，在体坐标系下表示，单位：rad/s

    def as_vector(self) -> np.ndarray:
        # 将状态打包为一个 6 维向量：[σ; ω]，便于积分器使用
        return np.concatenate((self.sigma_bn, self.omega_bn_b_rad_s))

    @classmethod
    def from_vector(cls, state_vector: np.ndarray) -> "AttitudeState":
        # 从 6 维状态向量恢复为 AttitudeState 对象
        state_vector = np.asarray(state_vector, dtype=float).reshape(6)
        return cls(
            sigma_bn=switch_to_shadow_mrp(state_vector[:3]),
            # 提取前 3 个元素作为 MRP，并自动切换影子集以保证模长 ≤ 1
            omega_bn_b_rad_s=state_vector[3:],
            # 提取后 3 个元素作为角速度
        )


@dataclass(frozen=True)
class AttitudeHistory:
    """Time history of the project-local attitude truth model."""
    # 项目本地姿态真值模型的时间历程记录，不可变

    time_s: np.ndarray
    # 时间序列，单位：秒

    sigma_bn: np.ndarray
    # MRP 历史，形状 (N, 3)

    omega_bn_b_rad_s: np.ndarray
    # 角速度历史，形状 (N, 3)，单位：rad/s

    dcm_bn: np.ndarray
    # 方向余弦矩阵历史，形状 (N, 3, 3)

    angular_momentum_b_nms: np.ndarray
    # 角动量历史（体坐标系下表示），形状 (N, 3)，单位：N·m·s

    kinetic_energy_j: np.ndarray
    # 转动动能历史，形状 (N,)，单位：焦耳


def attitude_state_derivative(
    time_s: float,
    state_vector: np.ndarray,
    config: AttitudeConfig,
    applied_torque_b_nm: np.ndarray,
) -> np.ndarray:
    """Return the rigid-body state derivative under the supplied body torque."""
    # 计算在给定体坐标系外力矩作用下的刚体状态导数（欧拉方程 + MRP 运动学）

    del time_s
    # 本动力学方程不显含时间，故忽略 time_s 参数（保留接口统一）

    state = AttitudeState.from_vector(state_vector)
    # 从向量恢复当前状态

    sigma_dot = mrp_derivative(state.sigma_bn, state.omega_bn_b_rad_s)
    # 计算 MRP 的姿态运动学导数：dσ/dt = 1/4 * B(σ) * ω

    angular_momentum_b_nms = config.inertia_kg_m2 @ state.omega_bn_b_rad_s
    # 计算角动量：h = I · ω

    omega_dot = np.linalg.solve(
        config.inertia_kg_m2,
        np.asarray(applied_torque_b_nm, dtype=float).reshape(3) - np.cross(state.omega_bn_b_rad_s, angular_momentum_b_nms),
    )
    # 根据欧拉方程计算角加速度：I * dω/dt = τ_ext - ω × (I·ω)
    # 使用求解线性方程组的方式计算 dω/dt，比直接求逆矩阵更稳定

    return np.concatenate((sigma_dot, omega_dot))
    # 返回组合后的状态导数向量 [dσ/dt; dω/dt]


def step_attitude_state(
    state: AttitudeState,
    config: AttitudeConfig,
    applied_torque_b_nm: np.ndarray,
    step_size_s: float,
    time_s: float = 0.0,
) -> AttitudeState:
    """Advance the rigid-body attitude state by one fixed integration step."""
    # 使用固定步长的 RK4 方法将刚体姿态状态向前积分一步

    next_state_vector = rk4_step(
        lambda current_time_s, current_state_vector: attitude_state_derivative(
            current_time_s,
            current_state_vector,
            config,
            applied_torque_b_nm,
        ),
        time_s,
        state.as_vector(),
        step_size_s,
    )
    # 调用 RK4 积分器，传入状态导数函数、当前时间、当前状态向量和步长

    return AttitudeState.from_vector(next_state_vector)
    # 将积分后的状态向量转换回 AttitudeState 对象并返回


def build_attitude_history(time_s: np.ndarray, states: list[AttitudeState], config: AttitudeConfig) -> AttitudeHistory:
    """Convert a list of propagated attitude states into time-history arrays."""
    # 将递推得到的状态列表转换为结构化的时间历程数组

    sigma_bn = np.asarray([state.sigma_bn for state in states], dtype=float)
    # 提取所有状态的 MRP，组成 (N, 3) 数组

    omega_bn_b_rad_s = np.asarray([state.omega_bn_b_rad_s for state in states], dtype=float)
    # 提取所有状态的角速度，组成 (N, 3) 数组

    dcm_bn = np.asarray([mrp_to_dcm(state.sigma_bn) for state in states], dtype=float)
    # 将每个 MRP 转换为 DCM，组成 (N, 3, 3) 数组

    angular_momentum_b_nms = np.asarray(
        [config.inertia_kg_m2 @ state.omega_bn_b_rad_s for state in states],
        dtype=float,
    )
    # 计算每个时刻的角动量 h = I·ω，组成 (N, 3) 数组

    kinetic_energy_j = 0.5 * np.einsum("ni,ni->n", omega_bn_b_rad_s, angular_momentum_b_nms)
    # 使用爱因斯坦求和计算转动动能：T = 0.5 * ωᵀ · h，结果为一维数组

    return AttitudeHistory(
        time_s=np.asarray(time_s, dtype=float),
        sigma_bn=sigma_bn,
        omega_bn_b_rad_s=omega_bn_b_rad_s,
        dcm_bn=dcm_bn,
        angular_momentum_b_nms=angular_momentum_b_nms,
        kinetic_energy_j=kinetic_energy_j,
    )
    # 返回封装好的姿态真值历史数据对象