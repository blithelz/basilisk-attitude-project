"""Numerical helpers for the project-local pure Python truth model."""
# 模块文档字符串：为项目本地的纯 Python 真实模型提供数值计算相关的辅助函数

from __future__ import annotations
# 启用推迟注解求值，允许在类型提示中使用尚未定义的类型名

from typing import Callable
# 从 typing 导入 Callable 类型，用于描述可调用对象（如函数）

import numpy as np
# 导入 numpy 并简写为 np，用于数值数组操作


EPSILON = 1.0e-12
# 定义一个极小量阈值，用于判断向量长度是否接近于零，防止除零错误


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix associated with a 3-vector."""
    # 返回与给定 3 维向量对应的反对称矩阵（也称斜对称矩阵或叉乘矩阵）。
    # 对于向量 v = [x, y, z]ᵀ，其反对称矩阵 [v×] 满足：[v×] w = v × w

    x_value, y_value, z_value = np.asarray(vector, dtype=float).reshape(3)
    # 将输入强制转换为形状为 (3,) 的浮点数组，并解包为 x、y、z 三个分量

    return np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ]
    )
    # 构造并返回 3×3 反对称矩阵：
    # [v×] = [[ 0, -z,  y],
    #         [ z,  0, -x],
    #         [-y,  x,  0]]


def safe_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized vector, or zeros if the input magnitude is too small."""
    # 安全地对向量进行归一化。如果输入向量的模长过小（小于阈值 EPSILON），
    # 则返回零向量而不是抛出除零异常，避免数值不稳定。

    vector = np.asarray(vector, dtype=float)
    # 将输入转换为浮点数组

    norm = np.linalg.norm(vector)
    # 计算向量的欧几里得范数（模长）

    if norm <= EPSILON:
        return np.zeros_like(vector)
    # 若模长小于等于阈值，认为该向量实际为零向量，返回同形状的零数组

    return vector / norm
    # 否则返回归一化后的单位向量


def rk4_step(
    derivative: Callable[[float, np.ndarray], np.ndarray],
    time_s: float,
    state: np.ndarray,
    step_size_s: float,
) -> np.ndarray:
    """Advance a first-order ODE one step with the classical RK4 method."""
    # 使用经典的四阶龙格-库塔方法（RK4）将一阶常微分方程向前积分一步。
    # 该方法具有四阶全局精度，是求解初值问题的常用数值积分方法。

    k1 = derivative(time_s, state)
    # 计算第一个增量斜率：k1 = f(t, y)

    k2 = derivative(time_s + 0.5 * step_size_s, state + 0.5 * step_size_s * k1)
    # 计算第二个增量斜率：k2 = f(t + Δt/2, y + (Δt/2) * k1)

    k3 = derivative(time_s + 0.5 * step_size_s, state + 0.5 * step_size_s * k2)
    # 计算第三个增量斜率：k3 = f(t + Δt/2, y + (Δt/2) * k2)

    k4 = derivative(time_s + step_size_s, state + step_size_s * k3)
    # 计算第四个增量斜率：k4 = f(t + Δt, y + Δt * k3)

    return state + (step_size_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # 根据 RK4 加权公式计算下一步的状态：
    # y_{n+1} = y_n + (Δt / 6) * (k1 + 2·k2 + 2·k3 + k4)
