"""Hill-pointing mode helpers for the project baseline scenario."""
# 模块文档字符串：为项目基准场景提供 Hill 指向模式相关的辅助函数

from __future__ import annotations

from typing import Any


DEFAULT_MODE_REQUEST = "hillPoint" # 默认的飞控模式请求值，对应 Hill 指向（对地定向）模式

# 从配置字典中获取期望的飞控模式请求字符串
def get_mode_request(config: dict[str, Any]) -> str:
    """Return the requested flight-software mode for the scenario."""
    return config.get("scenario", {}).get("mode_request", DEFAULT_MODE_REQUEST)


# 将配置文件中定义的 MRP 反馈控制增益应用到飞控模型的对应控制器上
def apply_hill_point_control_gains(fsw_model: Any, config: dict[str, Any]) -> None:
    """Apply project-configured MRP feedback gains to the hill-pointing controllers."""
    gains = config["control"]["mrp_feedback"]  # 从配置字典中取出控制增益子字典，包含 K、Ki、P 三个键

    # 官方示例里有两个 MRP 反馈控制器：
    # 一个面向普通姿控任务，一个面向带飞轮分配的姿控链。
    # 这里统一覆盖两者，确保项目配置能完整接管这条控制链。

    # 遍历飞控模型中的两个 MRP 反馈控制器实例
    for controller in (fsw_model.mrpFeedbackControl, fsw_model.mrpFeedbackRWs):
        controller.K = gains["K"]  # 设置比例增益 K，决定控制力矩与姿态误差的比例系数
        controller.Ki = gains["Ki"]  # 设置积分增益 Ki，用于消除稳态姿态误差
        controller.P = gains["P"] # 设置微分增益 P，提供角速度阻尼

        # 设置积分项的上限值，防止积分饱和（integral windup）
        # 公式含义：当 Ki ≠ 0 时，积分限为 (2.0 / Ki) * 0.1；若 Ki = 0 则积分限为 0（禁用积分）
        controller.integralLimit = 2.0 / controller.Ki * 0.1 if controller.Ki != 0.0 else 0.0
