"""Pure Python gyro sensor model for the week-3 measurement layer."""
# 纯 Python 陀螺仪传感器模型，用于第 3 周“测量层”

from __future__ import annotations
# 启用延迟类型注解，方便在类型提示里引用后面定义的类

from dataclasses import dataclass
# 导入 dataclass，便于定义结构化配置和测量结果容器

import numpy as np
# 导入 numpy，用于数组计算和随机噪声生成


def _as_three_vector(value: float | list[float] | np.ndarray, default: list[float]) -> np.ndarray:
    # 将标量或列表统一转换成 3 维浮点向量，便于三个轴统一处理
    candidate = default if value is None else value
    # 如果没有传入值，就回退到默认值
    array = np.asarray(candidate, dtype=float)
    # 将输入转换为浮点数组
    if array.ndim == 0:
        # 如果输入是标量，就复制到三个轴
        return np.repeat(array, 3)
    # 否则要求它本身就能重排成 3 维向量
    return array.reshape(3)


def _build_sample_indices(time_s: np.ndarray, sample_period_s: float) -> np.ndarray:
    # 根据采样周期，从 truth 时间轴里挑出真正发生测量的采样点索引
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    # 保证时间轴是一维浮点数组
    if time_s.size == 0:
        # 空时间轴直接返回空索引
        return np.zeros(0, dtype=int)
    if sample_period_s <= 0.0:
        # 采样周期必须严格大于零
        raise ValueError("sample_period_s must be positive.")

    indices: list[int] = [0]
    # 默认总是在初始时刻采第一帧
    next_sample_time_s = float(time_s[0] + sample_period_s)
    # 下一次采样时刻从起始时刻往后推进一个采样周期

    for index in range(1, time_s.size):
        # 从第二个真值点开始扫描时间轴
        current_time_s = float(time_s[index])
        # 取出当前真值样本的时间
        if current_time_s + 1.0e-12 < next_sample_time_s:
            # 如果还没到下一个采样时刻，就继续往后看
            continue
        indices.append(index)
        # 一旦越过采样时刻，就记录当前点为一次有效测量
        while next_sample_time_s <= current_time_s + 1.0e-12:
            # 如果 truth 步长比采样慢，这里会把“应采样时刻”追到当前时刻之后
            next_sample_time_s += sample_period_s

    return np.asarray(indices, dtype=int)
    # 返回测量发生时刻在 truth 时间轴中的整数索引


@dataclass(frozen=True)
class GyroConfig:
    """Configuration for the project-local gyro measurement model."""
    # 项目本地陀螺仪测量模型配置

    sample_period_s: float
    # 采样周期，单位为秒

    bias_rad_s: np.ndarray
    # 三轴常值偏置，单位为 rad/s

    noise_std_rad_s: np.ndarray
    # 三轴高斯白噪声标准差，单位为 rad/s

    saturation_rad_s: float | None
    # 单轴饱和值，单位为 rad/s；若为 None，表示不做饱和裁剪

    random_seed: int | None
    # 随机数种子，用于保证测量噪声可复现

    @classmethod
    def from_dict(cls, config: dict) -> "GyroConfig":
        # 从顶层配置字典中解析出 gyro 子配置
        spacecraft_cfg = config.get("spacecraft", config)
        # 兼容传入完整配置或只传 spacecraft 配置两种形式
        sensor_cfg = spacecraft_cfg.get("sensors", {}).get("gyro", {})
        # 取出 gyro 传感器子配置

        return cls(
            sample_period_s=float(sensor_cfg.get("sample_period_s", 1.0)),
            # 默认按 1 Hz 采样
            bias_rad_s=_as_three_vector(sensor_cfg.get("bias_rad_s"), [0.0, 0.0, 0.0]),
            # 解析三轴偏置
            noise_std_rad_s=_as_three_vector(sensor_cfg.get("noise_std_rad_s"), [0.0, 0.0, 0.0]),
            # 解析三轴噪声标准差
            saturation_rad_s=(
                None if sensor_cfg.get("saturation_rad_s") is None else float(sensor_cfg.get("saturation_rad_s"))
            ),
            # 若未指定饱和值，则不启用饱和
            random_seed=None if sensor_cfg.get("random_seed") is None else int(sensor_cfg.get("random_seed")),
            # 解析随机种子
        )


@dataclass(frozen=True)
class GyroMeasurementHistory:
    """Time history of gyro truth and measurements."""
    # 陀螺仪真值与测量值时间历史

    time_s: np.ndarray
    # 发生测量的时间戳，单位为秒

    truth_omega_bn_b_rad_s: np.ndarray
    # 与采样时刻对应的角速度真值，单位为 rad/s

    measured_omega_bn_b_rad_s: np.ndarray
    # 叠加偏置、噪声并经过饱和后的角速度测量值，单位为 rad/s

    bias_rad_s: np.ndarray
    # 在所有采样时刻重复保存的偏置向量，便于后续标定和作图

    sample_indices: np.ndarray
    # 这些测量对应于原始 truth 时间轴中的索引位置


def sample_gyro_history(
    time_s: np.ndarray,
    truth_omega_bn_b_rad_s: np.ndarray,
    config: GyroConfig,
    rng: np.random.Generator | None = None,
) -> GyroMeasurementHistory:
    """Sample body-rate truth through a gyro measurement model."""
    # 将角速度真值通过陀螺仪测量模型变成“可用于估计器/控制器”的测量值

    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    # 将输入时间轴整理成一维浮点数组
    truth_omega_bn_b_rad_s = np.asarray(truth_omega_bn_b_rad_s, dtype=float).reshape(-1, 3)
    # 将角速度真值整理成 N×3 数组
    if truth_omega_bn_b_rad_s.shape[0] != time_s.size:
        # 时间轴长度必须与真值样本数一致
        raise ValueError("time_s and truth_omega_bn_b_rad_s must contain the same number of samples.")

    sample_indices = _build_sample_indices(time_s, config.sample_period_s)
    # 按配置的采样周期生成有效采样索引
    sample_time_s = time_s[sample_indices]
    # 取出真实发生测量的时间戳
    truth_samples = truth_omega_bn_b_rad_s[sample_indices]
    # 取出对应时刻的角速度真值

    active_rng = np.random.default_rng(config.random_seed) if rng is None else rng
    # 若外部没有传入随机数发生器，就根据配置里的种子创建一个
    noise_rad_s = active_rng.normal(
        loc=0.0,
        scale=config.noise_std_rad_s,
        size=truth_samples.shape,
    )
    # 为每个采样时刻、每个轴独立生成高斯白噪声
    measured_samples = truth_samples + config.bias_rad_s + noise_rad_s
    # 测量值 = 真值 + 常值偏置 + 噪声

    if config.saturation_rad_s is not None:
        # 如果启用了饱和模型，就按单轴上下限裁剪
        measured_samples = np.clip(
            measured_samples,
            -config.saturation_rad_s,
            config.saturation_rad_s,
        )

    bias_history = np.repeat(config.bias_rad_s.reshape(1, 3), sample_time_s.size, axis=0)
    # 将偏置复制成与时间轴等长的数组，方便统一保存
    return GyroMeasurementHistory(
        time_s=sample_time_s,
        truth_omega_bn_b_rad_s=truth_samples,
        measured_omega_bn_b_rad_s=measured_samples,
        bias_rad_s=bias_history,
        sample_indices=sample_indices,
    )
    # 返回结构化的陀螺仪测量历史
