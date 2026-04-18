"""Hill-pointing mode helpers for the project baseline scenario."""

from __future__ import annotations

from typing import Any


DEFAULT_MODE_REQUEST = "hillPoint"


def get_mode_request(config: dict[str, Any]) -> str:
    """Return the requested flight-software mode for the scenario."""
    return config.get("scenario", {}).get("mode_request", DEFAULT_MODE_REQUEST)


def apply_hill_point_control_gains(fsw_model: Any, config: dict[str, Any]) -> None:
    """Apply project-configured MRP feedback gains to the hill-pointing controllers."""
    gains = config["control"]["mrp_feedback"]

    # 官方示例里有两个 MRP 反馈控制器，这里统一从项目配置覆盖参数。
    for controller in (fsw_model.mrpFeedbackControl, fsw_model.mrpFeedbackRWs):
        controller.K = gains["K"]
        controller.Ki = gains["Ki"]
        controller.P = gains["P"]
        controller.integralLimit = 2.0 / controller.Ki * 0.1 if controller.Ki != 0.0 else 0.0
