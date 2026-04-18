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
    time_data = np.delete(rw_speed_rec.times(), 0, 0) * macros.NANO2MIN
    wheel_speeds = np.delete(rw_speed_rec.wheelSpeeds[:, wheel_indices], 0, 0)
    motor_torque = np.delete(rw_motor_rec.motorTorque[:, wheel_indices], 0, 0)
    return time_data, wheel_speeds, motor_torque
