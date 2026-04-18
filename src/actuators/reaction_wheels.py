"""Reaction wheel helpers for the project baseline scenario."""

from __future__ import annotations

from typing import Any

import numpy as np
from Basilisk.utilities import macros


DEFAULT_REACTION_WHEEL_COUNT = 4


def get_reaction_wheel_count(config: dict[str, Any]) -> int:
    """Return the configured number of active reaction wheels."""
    return config.get("actuators", {}).get("reaction_wheels", {}).get(
        "count", DEFAULT_REACTION_WHEEL_COUNT
    )


def attach_reaction_wheel_recorders(
    sim_base: Any,
    dyn_model: Any,
    fsw_model: Any,
    sampling_time: int,
) -> tuple[Any, Any]:
    """Create and register the reaction wheel telemetry recorders."""
    # rwSpeedOutMsg 来自 Dynamics，表示飞轮当前状态；
    # cmdRwMotorMsg 来自 FSW，表示控制器给飞轮的指令力矩。
    rw_speed_rec = dyn_model.rwStateEffector.rwSpeedOutMsg.recorder(sampling_time)
    rw_motor_rec = fsw_model.cmdRwMotorMsg.recorder(sampling_time)

    sim_base.AddModelToTask(dyn_model.taskName, rw_speed_rec)
    sim_base.AddModelToTask(dyn_model.taskName, rw_motor_rec)
    return rw_speed_rec, rw_motor_rec


def extract_reaction_wheel_history(
    rw_speed_rec: Any,
    rw_motor_rec: Any,
    num_reaction_wheels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time history, wheel speeds, and commanded motor torques."""
    wheel_indices = range(num_reaction_wheels)
    # 这里统一做两件事：
    # 1. 去掉第 0 行初始化样本
    # 2. 只保留当前场景真正启用的前 num_reaction_wheels 个飞轮
    time_data = np.delete(rw_speed_rec.times(), 0, 0) * macros.NANO2MIN
    wheel_speeds = np.delete(rw_speed_rec.wheelSpeeds[:, wheel_indices], 0, 0)
    motor_torque = np.delete(rw_motor_rec.motorTorque[:, wheel_indices], 0, 0)
    return time_data, wheel_speeds, motor_torque
