"""Pure Python magnetometer model for the week-3 measurement layer."""
# 纯 Python 磁强计测量模型，用于第 3 周“测量层”

from __future__ import annotations
# 启用延迟类型注解

from dataclasses import dataclass
# 导入 dataclass，便于定义配置和输出结构

import numpy as np
# 导入 numpy，用于向量计算与噪声生成


def _as_three_vector(value: float | list[float] | np.ndarray, default: list[float]) -> np.ndarray:
    # 将标量或列表统一转换为三轴向量
    candidate = default if value is None else value
    # 没有显式配置时回退到默认值
    array = np.asarray(candidate, dtype=float)
    # 转成浮点数组
    if array.ndim == 0:
        # 标量时复制到三个轴
        return np.repeat(array, 3)
    return array.reshape(3)
    # 列表或数组时重排成 3 维向量


def _build_sample_indices(time_s: np.ndarray, sample_period_s: float) -> np.ndarray:
    # 根据采样周期，在 truth 时间轴中选出测量发生的时刻
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    # 保证时间轴是一维浮点数组
    if time_s.size == 0:
        # 空输入直接返回空索引
        return np.zeros(0, dtype=int)
    if sample_period_s <= 0.0:
        # 采样周期必须严格为正
        raise ValueError("sample_period_s must be positive.")

    indices: list[int] = [0]
    # 默认在初始时刻采样一次
    next_sample_time_s = float(time_s[0] + sample_period_s)
    # 下一次采样时刻

    for index in range(1, time_s.size):
        # 遍历后续 truth 时刻
        current_time_s = float(time_s[index])
        # 当前 truth 样本时间
        if current_time_s + 1.0e-12 < next_sample_time_s:
            # 还没到下一次采样，就跳过
            continue
        indices.append(index)
        # 一旦跨过采样时刻，就在当前 truth 点取样
        while next_sample_time_s <= current_time_s + 1.0e-12:
            # 将“应采样时刻”推进到当前时刻之后
            next_sample_time_s += sample_period_s

    return np.asarray(indices, dtype=int)
    # 返回采样索引数组


@dataclass(frozen=True)
class MagnetometerConfig:
    """Configuration for the project-local magnetometer model."""
    # 项目本地磁强计配置

    sample_period_s: float
    # 采样周期，单位为秒

    bias_t: np.ndarray
    # 三轴常值偏置，单位为特斯拉

    noise_std_t: np.ndarray
    # 三轴白噪声标准差，单位为特斯拉

    saturation_t: float | None
    # 单轴饱和值，单位为特斯拉；若为 None，表示不做裁剪

    random_seed: int | None
    # 随机数种子

    @classmethod
    def from_dict(cls, config: dict) -> "MagnetometerConfig":
        # 从顶层配置里读取 magnetometer 子配置
        spacecraft_cfg = config.get("spacecraft", config)
        # 兼容完整配置和局部配置
        sensor_cfg = spacecraft_cfg.get("sensors", {}).get("magnetometer", {})
        # 读取磁强计配置

        return cls(
            sample_period_s=float(sensor_cfg.get("sample_period_s", 1.0)),
            # 默认 1 Hz 采样
            bias_t=_as_three_vector(sensor_cfg.get("bias_t"), [0.0, 0.0, 0.0]),
            # 解析偏置
            noise_std_t=_as_three_vector(sensor_cfg.get("noise_std_t"), [0.0, 0.0, 0.0]),
            # 解析噪声标准差
            saturation_t=None if sensor_cfg.get("saturation_t") is None else float(sensor_cfg.get("saturation_t")),
            # 解析饱和值
            random_seed=None if sensor_cfg.get("random_seed") is None else int(sensor_cfg.get("random_seed")),
            # 解析随机种子
        )


@dataclass(frozen=True)
class MagnetometerMeasurementHistory:
    """Time history of magnetometer truth and measurements."""
    # 磁强计真值与测量值时间历史

    time_s: np.ndarray
    # 采样时间戳，单位为秒

    truth_magnetic_field_b_t: np.ndarray
    # 三轴地磁真值，单位为特斯拉

    measured_magnetic_field_b_t: np.ndarray
    # 加入偏置和噪声后的三轴地磁测量值，单位为特斯拉

    bias_t: np.ndarray
    # 与时间轴等长的偏置历史，便于后续分析

    sample_indices: np.ndarray
    # 这些测量在原 truth 时间轴中的索引


def sample_magnetometer_history(
    time_s: np.ndarray,
    truth_magnetic_field_b_t: np.ndarray,
    config: MagnetometerConfig,
    rng: np.random.Generator | None = None,
) -> MagnetometerMeasurementHistory:
    """Sample body-frame magnetic-field truth through a magnetometer model."""
    # 将体坐标系中的磁场真值转换为磁强计测量值

    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    # 保证时间轴是一维数组
    truth_magnetic_field_b_t = np.asarray(truth_magnetic_field_b_t, dtype=float).reshape(-1, 3)
    # 将磁场真值整理成 N×3 数组
    if truth_magnetic_field_b_t.shape[0] != time_s.size:
        # 样本数必须与时间轴一致
        raise ValueError("time_s and truth_magnetic_field_b_t must contain the same number of samples.")

    sample_indices = _build_sample_indices(time_s, config.sample_period_s)
    # 生成有效采样索引
    sample_time_s = time_s[sample_indices]
    # 采样时刻
    truth_samples = truth_magnetic_field_b_t[sample_indices]
    # 对应的真值样本

    active_rng = np.random.default_rng(config.random_seed) if rng is None else rng
    # 若外部未传入随机数发生器，就按配置创建
    noise_t = active_rng.normal(
        loc=0.0,
        scale=config.noise_std_t,
        size=truth_samples.shape,
    )
    # 为每个采样点、每个轴独立生成测量噪声
    measured_samples = truth_samples + config.bias_t + noise_t
    # 测量值 = 真值 + 偏置 + 噪声

    if config.saturation_t is not None:
        # 若启用了饱和限制，则按单轴上下限裁剪
        measured_samples = np.clip(
            measured_samples,
            -config.saturation_t,
            config.saturation_t,
        )

    bias_history = np.repeat(config.bias_t.reshape(1, 3), sample_time_s.size, axis=0)
    # 复制偏置形成等长历史数组
    return MagnetometerMeasurementHistory(
        time_s=sample_time_s,
        truth_magnetic_field_b_t=truth_samples,
        measured_magnetic_field_b_t=measured_samples,
        bias_t=bias_history,
        sample_indices=sample_indices,
    )
    # 返回结构化磁强计测量历史
