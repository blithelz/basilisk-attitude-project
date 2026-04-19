"""Reference-frame helpers for the project-local pure Python truth model."""
# 模块文档字符串：为项目本地的纯 Python 真实模型提供参考系变换相关的辅助函数

from __future__ import annotations
# 启用推迟注解求值，允许在类型提示中使用尚未定义的类型名

import numpy as np
# 导入 numpy 并简写为 np，用于数值计算和数组操作

from src.utils.math_utils import safe_normalize, skew_symmetric
# 从项目通用数学工具模块导入：
# - safe_normalize：安全向量归一化函数，可处理零向量防止除零错误
# - skew_symmetric：生成向量对应的反对称（斜对称）矩阵


def rotation_matrix_1(angle_rad: float) -> np.ndarray:
    """Return the principal-axis rotation matrix about axis 1."""
    # 返回绕第1轴（X轴）旋转的主轴旋转矩阵（主动旋转约定，向量逆时针旋转）

    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, sine],
            [0.0, -sine, cosine],
        ]
    )
    # 构建并返回 3×3 旋转矩阵：
    # R1(θ) = [[1, 0,     0   ],
    #          [0, cosθ,  sinθ],
    #          [0, -sinθ, cosθ]]


def rotation_matrix_3(angle_rad: float) -> np.ndarray:
    """Return the principal-axis rotation matrix about axis 3."""
    # 返回绕第3轴（Z轴）旋转的主轴旋转矩阵

    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # 构建并返回 3×3 旋转矩阵：
    # R3(θ) = [[cosθ, sinθ, 0],
    #          [-sinθ, cosθ, 0],
    #          [0,     0,    1]]


def switch_to_shadow_mrp(sigma_bn: np.ndarray) -> np.ndarray:
    """Map an MRP vector to the shadow set if its norm exceeds one."""
    # 如果 MRP（修正罗德里格兹参数）向量的模长超过1，则将其映射到影子集（shadow set），
    # 这是处理 MRP 奇异性的常用方法，保证姿态表示的数值稳定性。

    sigma_bn = np.asarray(sigma_bn, dtype=float).reshape(3)
    # 确保输入转为形状为 (3,) 的浮点数组

    sigma_norm_sq = float(np.dot(sigma_bn, sigma_bn))
    # 计算 MRP 向量的模平方

    if sigma_norm_sq <= 1.0:
        return sigma_bn
    # 如果模平方 ≤ 1，表示处于原始集，直接返回原向量

    return -sigma_bn / sigma_norm_sq
    # 否则映射到影子集：σ_shadow = -σ / |σ|²
    # 影子集与原 MRP 表示完全相同的姿态，但模长小于1，避免奇异


def mrp_kinematics_matrix(sigma_bn: np.ndarray) -> np.ndarray:
    """Return the MRP kinematics matrix B(sigma)."""
    # 返回 MRP 运动学矩阵 B(σ)，用于计算 MRP 的时间导数：dσ/dt = 1/4 * B(σ) * ω

    sigma_bn = np.asarray(sigma_bn, dtype=float).reshape(3)
    # 确保输入为 3 维浮点数组

    sigma_norm_sq = float(np.dot(sigma_bn, sigma_bn))
    # 计算模平方

    identity = np.eye(3)
    # 3×3 单位阵

    sigma_cross = skew_symmetric(sigma_bn)
    # 生成 σ 的反对称矩阵 [σ×]

    sigma_outer = np.outer(sigma_bn, sigma_bn)
    # 生成 σ 的外积矩阵 σ·σᵀ

    return (1.0 - sigma_norm_sq) * identity + 2.0 * sigma_cross + 2.0 * sigma_outer
    # 根据公式 B(σ) = (1 - |σ|²) I₃ + 2[σ×] + 2 σσᵀ 返回运动学矩阵


def mrp_derivative(sigma_bn: np.ndarray, omega_bn_b_rad_s: np.ndarray) -> np.ndarray:
    """Return the MRP kinematic time derivative."""
    # 返回 MRP 姿态参数的时间导数 dσ/dt

    return 0.25 * mrp_kinematics_matrix(sigma_bn) @ np.asarray(omega_bn_b_rad_s, dtype=float).reshape(3)
    # 计算导数：dσ/dt = 1/4 * B(σ) * ω
    # ω 为体坐标系相对于惯性系的角速度，在体坐标系下表示


def mrp_to_dcm(sigma_bn: np.ndarray) -> np.ndarray:
    """Return the passive DCM from inertial frame N to body frame B."""
    # 返回从惯性系 N 到体坐标系 B 的被动方向余弦矩阵（DCM），
    # 即：v_B = DCM * v_N，将惯性系向量变换到体坐标系

    sigma_bn = np.asarray(sigma_bn, dtype=float).reshape(3)
    sigma_norm_sq = float(np.dot(sigma_bn, sigma_bn))
    sigma_cross = skew_symmetric(sigma_bn)
    denominator = (1.0 + sigma_norm_sq) ** 2
    return np.eye(3) + (8.0 * sigma_cross @ sigma_cross - 4.0 * (1.0 - sigma_norm_sq) * sigma_cross) / denominator
    # 根据 MRP 转 DCM 的公式计算：
    # R = I₃ + (8 [σ×]² - 4 (1 - |σ|²) [σ×]) / (1 + |σ|²)²
    # 该矩阵为正交阵，满足 R⁻¹ = Rᵀ


def rotate_inertial_to_body(vector_n: np.ndarray, sigma_bn: np.ndarray) -> np.ndarray:
    """Rotate a vector from inertial frame N into body frame B."""
    # 将惯性系 N 下的向量旋转到体坐标系 B 下表示

    return mrp_to_dcm(sigma_bn) @ np.asarray(vector_n, dtype=float).reshape(3)
    # 利用 DCM 进行坐标变换：v_B = R_BN * v_N


def rotate_body_to_inertial(vector_b: np.ndarray, sigma_bn: np.ndarray) -> np.ndarray:
    """Rotate a vector from body frame B into inertial frame N."""
    # 将体坐标系 B 下的向量旋转回惯性系 N 下表示

    return mrp_to_dcm(sigma_bn).T @ np.asarray(vector_b, dtype=float).reshape(3)
    # 利用 DCM 的转置（即逆矩阵）进行反向变换：v_N = R_BNᵀ * v_B


def orbital_frame_dcm(position_n_m: np.ndarray, velocity_n_m_s: np.ndarray) -> np.ndarray:
    """Return the passive DCM from inertial frame N to the local orbital frame."""
    # 返回从惯性系 N 到当地轨道坐标系（Hill 坐标系）的被动方向余弦矩阵。
    # 轨道坐标系通常定义为：X 沿径向（地心指向航天器），Z 沿轨道法向（角动量方向），
    # Y 沿飞行方向（与径向和法向构成右手系）。

    radial_hat = safe_normalize(position_n_m)
    # 径向单位向量：从地心指向航天器，归一化

    angular_momentum_hat = safe_normalize(np.cross(position_n_m, velocity_n_m_s))
    # 轨道角动量方向单位向量：位置矢量叉乘速度矢量，归一化（轨道法向）

    along_track_hat = safe_normalize(np.cross(angular_momentum_hat, radial_hat))
    # 沿迹方向单位向量：角动量方向叉乘径向方向，符合右手法则，指向速度方向（但不完全是速度方向，只保证正交）

    return np.vstack((radial_hat, along_track_hat, angular_momentum_hat))
    # 垂直堆叠三个单位行向量，构成 3×3 DCM：
    # R = [ radial_hatᵀ ]
    #     [ along_track_hatᵀ ]
    #     [ angular_momentum_hatᵀ ]
    # 该矩阵将惯性系向量变换到轨道坐标系：v_orb = R * v_N