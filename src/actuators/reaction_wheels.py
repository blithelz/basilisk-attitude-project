"""Reaction wheel helpers for the project baseline scenario."""
# 模块文档字符串：本模块为项目基准场景提供反作用轮相关的辅助函数

from __future__ import annotations

from typing import Any

import numpy as np
from Basilisk.utilities import macros # 从 Basilisk 工具包中导入 macros 模块，提供单位转换等宏定义（如 NANO2MIN）


DEFAULT_REACTION_WHEEL_COUNT = 4 # 默认的反作用轮数量（常见金字塔构型使用 4 个）

# 定义一个函数，从配置字典中获取反作用轮数量
def get_reaction_wheel_count(config: dict[str, Any]) -> int:
    """Return the configured number of active reaction wheels."""
    # 返回配置中指定的活动反作用轮数量，若未指定则返回默认值
    return config.get("actuators", {}).get("reaction_wheels", {}).get(
        "count", DEFAULT_REACTION_WHEEL_COUNT
    )

# 定义函数，为仿真任务挂载反作用轮数据记录器，返回记录器对象元组
def attach_reaction_wheel_recorders(
    sim_base: Any,
    dyn_model: Any,
    fsw_model: Any,
    sampling_time: int,
) -> tuple[Any, Any]:
    """Create and register the reaction wheel telemetry recorders."""
    # rwSpeedOutMsg 来自 Dynamics，表示飞轮当前状态；
    # cmdRwMotorMsg 来自 FSW，表示控制器给飞轮的指令力矩。

    # 创建一个记录器，用于记录动力学模型输出的反作用轮转速消息（rwSpeedOutMsg）
    # 采样间隔为 sampling_time（单位：纳秒，Basilisk 内部时间单位）
    rw_speed_rec = dyn_model.rwStateEffector.rwSpeedOutMsg.recorder(sampling_time)

    # 创建记录器，用于记录飞控软件（FSW）发出的反作用轮力矩指令消息（cmdRwMotorMsg）
    rw_motor_rec = fsw_model.cmdRwMotorMsg.recorder(sampling_time)

    # 将转速记录器添加到动力学模型所在的任务中，使其在仿真循环中被执行
    sim_base.AddModelToTask(dyn_model.taskName, rw_speed_rec)
    # 将力矩指令记录器也添加到同一个任务中
    sim_base.AddModelToTask(dyn_model.taskName, rw_motor_rec)
    return rw_speed_rec, rw_motor_rec

# 定义函数，从记录器中提取反作用轮的时间历程数据，返回时间、转速、力矩三个数组
def extract_reaction_wheel_history(
    rw_speed_rec: Any,
    rw_motor_rec: Any,
    num_reaction_wheels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time history, wheel speeds, and commanded motor torques."""
    wheel_indices = range(num_reaction_wheels) # 生成一个从 0 到 num_reaction_wheels-1 的索引范围，用于选择前 N 个飞轮的数据
    # 这里统一做两件事：
    # 1. 去掉第 0 行初始化样本
    # 2. 只保留当前场景真正启用的前 num_reaction_wheels 个飞轮
    # 获取记录器中的时间戳数组（单位为纳秒），使用 np.delete 删除第 0 个元素（初始时刻的冗余采样）
    # 然后乘以 macros.NANO2MIN（1e9 / 60，纳秒转分钟的转换因子），得到以分钟为单位的时间数据
    time_data = np.delete(rw_speed_rec.times(), 0, 0) * macros.NANO2MIN

    # 从记录器的 wheelSpeeds 二维数组中选取前 num_reaction_wheels 列（每列对应一个飞轮的转速历史）
    # 再删除第 0 行，返回形状为 (N, num_reaction_wheels) 的转速数组，单位：弧度/秒
    wheel_speeds = np.delete(rw_speed_rec.wheelSpeeds[:, wheel_indices], 0, 0)

    # 同理，从力矩指令记录器中提取前 num_reaction_wheels 列的力矩数据，删除第 0 行
    # 返回形状为 (N, num_reaction_wheels) 的力矩数组，单位：牛·米
    motor_torque = np.delete(rw_motor_rec.motorTorque[:, wheel_indices], 0, 0)
    return time_data, wheel_speeds, motor_torque
