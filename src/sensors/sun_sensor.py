"""Pure Python sun-sensor model for the week-3 measurement layer."""
# 纯 Python 太阳敏感器模型，用于第 3 周“测量层”

from __future__ import annotations
# 启用延迟类型注解

from dataclasses import dataclass
# 导入 dataclass，定义配置和历史结构

import numpy as np
# 导入 numpy，用于向量计算与随机扰动生成


def _build_sample_indices(time_s: np.ndarray, sample_period_s: float) -> np.ndarray:
    # 根据采样周期，从 truth 时间轴中提取测量发生的时刻
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    # 保证时间轴是一维数组
    if time_s.size == 0:
        # 空时间轴直接返回空索引
        return np.zeros(0, dtype=int)
    if sample_period_s <= 0.0:
        # 采样周期必须大于零
        raise ValueError("sample_period_s must be positive.")

    indices: list[int] = [0]
    # 初始时刻默认采一帧
    next_sample_time_s = float(time_s[0] + sample_period_s)
    # 从起始时刻向后推进一个采样周期

    for index in range(1, time_s.size):
        # 遍历后续 truth 样本
        current_time_s = float(time_s[index])
        # 当前 truth 时间
        if current_time_s + 1.0e-12 < next_sample_time_s:
            # 还没到采样时刻则跳过
            continue
        indices.append(index)
        # 用当前 truth 点代表这一帧测量
        while next_sample_time_s <= current_time_s + 1.0e-12:
            # 若 truth 步长更粗，就持续追赶目标采样时刻
            next_sample_time_s += sample_period_s

    return np.asarray(indices, dtype=int)
    # 返回采样索引


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    # 将一组向量按行归一化，避免太阳方向测量长度漂移
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # 计算每一行向量的模长
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    # 对于零向量，用 1.0 防止除零
    return vectors / safe_norms
    # 返回按行归一化后的结果


@dataclass(frozen=True)
class SunSensorConfig:
    """Configuration for the project-local sun-sensor model."""
    # 项目本地太阳敏感器配置

    sample_period_s: float
    # 采样周期，单位为秒

    direction_noise_std: float
    # 方向测量噪声标准差，作为单位向量扰动的近似幅值

    minimum_illumination: float
    # 当 illumination 小于该阈值时，认为太阳敏感器没有可用观测

    random_seed: int | None
    # 随机数种子

    @classmethod
    def from_dict(cls, config: dict) -> "SunSensorConfig":
        # 从顶层配置里读取 sun_sensor 子配置
        spacecraft_cfg = config.get("spacecraft", config)
        # 兼容完整配置和局部配置
        sensor_cfg = spacecraft_cfg.get("sensors", {}).get("sun_sensor", {})
        # 读取太阳敏感器子配置

        return cls(
            sample_period_s=float(sensor_cfg.get("sample_period_s", 1.0)),
            # 默认 1 Hz 采样
            direction_noise_std=float(sensor_cfg.get("direction_noise_std", 0.0)),
            # 默认不加噪声，便于从理想模型起步
            minimum_illumination=float(sensor_cfg.get("minimum_illumination", 0.5)),
            # illumination 低于阈值时视为无效测量
            random_seed=None if sensor_cfg.get("random_seed") is None else int(sensor_cfg.get("random_seed")),
            # 解析随机数种子
        )


@dataclass(frozen=True)
class SunSensorMeasurementHistory:
    """Time history of sun-sensor truth and measurements."""
    # 太阳敏感器真值与测量值时间历史

    time_s: np.ndarray
    # 采样时间戳，单位为秒

    truth_sun_direction_b: np.ndarray
    # 体坐标系太阳方向真值，单位向量

    measured_sun_direction_b: np.ndarray
    # 加噪后的太阳方向测量；若无效，则填充 NaN

    illumination: np.ndarray
    # 与采样时刻对应的 illumination 因子

    valid: np.ndarray
    # 测量可用性标志；处于地影或光照过弱时为 False

    sample_indices: np.ndarray
    # 这些测量在原 truth 时间轴中的索引


def sample_sun_sensor_history(
    time_s: np.ndarray,
    truth_sun_direction_b: np.ndarray,
    illumination: np.ndarray,
    config: SunSensorConfig,
    rng: np.random.Generator | None = None,
) -> SunSensorMeasurementHistory:
    """Sample body-frame sun-direction truth through a sun-sensor model."""
    # 将体坐标系太阳方向真值转换为太阳敏感器可见测量

    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    # 将时间轴整理成一维数组
    truth_sun_direction_b = np.asarray(truth_sun_direction_b, dtype=float).reshape(-1, 3)
    # 将真值太阳方向整理成 N×3 数组
    illumination = np.asarray(illumination, dtype=float).reshape(-1)
    # 将光照因子整理成一维数组
    if truth_sun_direction_b.shape[0] != time_s.size or illumination.size != time_s.size:
        # 所有输入的样本数必须一致
        raise ValueError("time_s, truth_sun_direction_b, and illumination must contain the same number of samples.")

    sample_indices = _build_sample_indices(time_s, config.sample_period_s)
    # 生成采样索引
    sample_time_s = time_s[sample_indices]
    # 采样时间戳
    truth_samples = truth_sun_direction_b[sample_indices]
    # 对应的太阳方向真值
    illumination_samples = illumination[sample_indices]
    # 对应的 illumination 因子

    active_rng = np.random.default_rng(config.random_seed) if rng is None else rng
    # 若外部未传入随机数发生器，则按配置创建
    noise = active_rng.normal(
        loc=0.0,
        scale=config.direction_noise_std,
        size=truth_samples.shape,
    )
    # 为每个采样点生成三轴方向扰动
    valid = illumination_samples >= config.minimum_illumination
    # 只有光照足够时，太阳敏感器观测才有效

    measured_samples = np.full_like(truth_samples, np.nan)
    # 先将所有输出初始化为 NaN，表示“无有效观测”
    if np.any(valid):
        # 若至少存在一个有效样本，再对有效样本加噪并归一化
        noisy_valid_samples = _normalize_rows(truth_samples[valid] + noise[valid])
        # 对有效样本做“真值 + 噪声”再归一化，得到单位方向测量
        measured_samples[valid] = noisy_valid_samples
        # 将有效测量写回输出数组

    return SunSensorMeasurementHistory(
        time_s=sample_time_s,
        truth_sun_direction_b=truth_samples,
        measured_sun_direction_b=measured_samples,
        illumination=illumination_samples,
        valid=valid,
        sample_indices=sample_indices,
    )
    # 返回结构化太阳敏感器测量历史
