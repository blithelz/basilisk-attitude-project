"""Pure Python week-3 sensor pipeline built on top of the truth model."""
# 基于第 2 周 truth model 的第 3 周纯 Python 传感器总装配

from __future__ import annotations
# 启用延迟类型注解

from dataclasses import dataclass
# 导入 dataclass，定义配置与结果容器

from pathlib import Path
# 导入 Path，用于保存数组与摘要文件

from typing import Any
# 导入 Any，便于表达通用配置字典

import json
# 导入 json，用于写出运行摘要

import numpy as np
# 导入 numpy，用于数组保存和统计计算

from src.sensors.gyro import GyroConfig, GyroMeasurementHistory, sample_gyro_history
# 导入陀螺仪配置、结果结构和采样函数
from src.sensors.magnetometer import (
    MagnetometerConfig,
    MagnetometerMeasurementHistory,
    sample_magnetometer_history,
)
# 导入磁强计配置、结果结构和采样函数
from src.sensors.sun_sensor import SunSensorConfig, SunSensorMeasurementHistory, sample_sun_sensor_history
# 导入太阳敏感器配置、结果结构和采样函数
from src.truth.truth_model import TruthModelResult
# 导入 truth model 的输出结果类型，作为传感器层输入


@dataclass(frozen=True)
class SensorSuiteConfig:
    """Configuration bundle for the week-3 sensor suite."""
    # 第 3 周传感器套件配置集合

    gyro: GyroConfig
    # 陀螺仪配置

    magnetometer: MagnetometerConfig
    # 磁强计配置

    sun_sensor: SunSensorConfig
    # 太阳敏感器配置

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "SensorSuiteConfig":
        # 从顶层配置字典同时构造三种传感器配置
        return cls(
            gyro=GyroConfig.from_dict(config),
            # 解析 gyro 配置
            magnetometer=MagnetometerConfig.from_dict(config),
            # 解析 magnetometer 配置
            sun_sensor=SunSensorConfig.from_dict(config),
            # 解析 sun_sensor 配置
        )


@dataclass(frozen=True)
class SensorSuiteResult:
    """Measured outputs produced by the week-3 sensor layer."""
    # 第 3 周传感器层输出的测量结果集合

    gyro: GyroMeasurementHistory
    # 陀螺仪测量历史

    magnetometer: MagnetometerMeasurementHistory
    # 磁强计测量历史

    sun_sensor: SunSensorMeasurementHistory
    # 太阳敏感器测量历史


class SensorSuiteModel:
    """Sensor-layer model that converts truth histories into sampled measurements."""
    # 传感器层模型：输入 truth histories，输出带采样率、偏置和噪声的 measurements

    def __init__(self, config: SensorSuiteConfig) -> None:
        # 构造函数，保存传感器套件配置
        self.config = config
        # 将配置保存为实例属性，供后续 simulate 调用

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "SensorSuiteModel":
        # 直接从配置字典构造传感器层模型
        return cls(SensorSuiteConfig.from_dict(config))
        # 先解析配置，再构造模型实例

    def simulate(self, truth_result: TruthModelResult) -> SensorSuiteResult:
        # 将 truth model 的输出转换成三类传感器测量值
        gyro_history = sample_gyro_history(
            truth_result.attitude.time_s,
            truth_result.attitude.omega_bn_b_rad_s,
            self.config.gyro,
        )
        # 用姿态真值中的角速度驱动陀螺仪测量模型
        magnetometer_history = sample_magnetometer_history(
            truth_result.environment.time_s,
            truth_result.environment.magnetic_field_b_t,
            self.config.magnetometer,
        )
        # 用环境真值中的体坐标系地磁场驱动磁强计测量模型
        sun_sensor_history = sample_sun_sensor_history(
            truth_result.environment.time_s,
            truth_result.environment.sun_direction_b,
            truth_result.environment.illumination,
            self.config.sun_sensor,
        )
        # 用环境真值中的太阳方向与光照条件驱动太阳敏感器模型

        return SensorSuiteResult(
            gyro=gyro_history,
            magnetometer=magnetometer_history,
            sun_sensor=sun_sensor_history,
        )
        # 将三类测量结果打包成统一的传感器层输出


def save_sensor_measurement_arrays(result: SensorSuiteResult, output_dir: Path) -> Path:
    """Save week-3 sensor measurements to a NumPy archive."""
    # 将第 3 周的传感器测量结果保存为 .npz 文件，便于后续估计器直接读取

    output_dir.mkdir(parents=True, exist_ok=True)
    # 若输出目录不存在，则递归创建
    target = output_dir / "sensor_measurements.npz"
    # 定义输出文件路径
    np.savez(
        target,
        gyro_time_s=result.gyro.time_s,
        gyro_truth_omega_bn_b_rad_s=result.gyro.truth_omega_bn_b_rad_s,
        gyro_measured_omega_bn_b_rad_s=result.gyro.measured_omega_bn_b_rad_s,
        magnetometer_time_s=result.magnetometer.time_s,
        magnetometer_truth_magnetic_field_b_t=result.magnetometer.truth_magnetic_field_b_t,
        magnetometer_measured_magnetic_field_b_t=result.magnetometer.measured_magnetic_field_b_t,
        sun_sensor_time_s=result.sun_sensor.time_s,
        sun_sensor_truth_sun_direction_b=result.sun_sensor.truth_sun_direction_b,
        sun_sensor_measured_sun_direction_b=result.sun_sensor.measured_sun_direction_b,
        sun_sensor_illumination=result.sun_sensor.illumination,
        sun_sensor_valid=result.sun_sensor.valid.astype(np.int8),
    )
    # 使用名字清晰的键保存三种传感器的时间轴、真值与测量值
    return target
    # 返回保存后的文件路径


def save_sensor_measurement_summary(result: SensorSuiteResult, output_dir: Path) -> Path:
    """Save a short JSON summary of the week-3 sensor run."""
    # 保存第 3 周传感器层运行摘要，帮助快速检查采样与可见性是否正常

    output_dir.mkdir(parents=True, exist_ok=True)
    # 确保输出目录存在
    target = output_dir / "sensor_measurement_summary.json"
    # 定义摘要文件路径
    summary = {
        "gyro_num_samples": int(result.gyro.time_s.size),
        # 陀螺仪采样点数
        "magnetometer_num_samples": int(result.magnetometer.time_s.size),
        # 磁强计采样点数
        "sun_sensor_num_samples": int(result.sun_sensor.time_s.size),
        # 太阳敏感器采样点数
        "sun_sensor_valid_fraction": float(np.mean(result.sun_sensor.valid.astype(float))),
        # 太阳敏感器有效观测占比
        "max_gyro_measurement_rad_s": float(np.max(np.linalg.norm(result.gyro.measured_omega_bn_b_rad_s, axis=1))),
        # 陀螺仪测量值模长最大值
        "max_magnetic_field_t": float(
            np.max(np.linalg.norm(result.magnetometer.measured_magnetic_field_b_t, axis=1))
        ),
        # 磁强计测量值模长最大值
    }
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # 将摘要字典格式化写入 JSON 文件
    return target
    # 返回保存后的摘要文件路径
