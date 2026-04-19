"""SimpleNav sensor helpers for the project baseline scenario."""
# 模块文档字符串：为项目基准场景提供 SimpleNav 理想导航传感器相关的辅助函数

from __future__ import annotations

from typing import Any

import numpy as np
from src.truth.orbit import apply_orbit_truth_configuration
from src.truth.rigid_body import apply_rigid_body_truth_configuration


def configure_spacecraft_initial_state(dyn_model: Any, config: dict[str, Any]) -> None:
    """Backward-compatible wrapper kept during the week-2 truth-layer refactor."""

    # 第 2 周开始，初始姿轨属于 truth layer，不再属于 sensor layer。
    # 这里保留一个兼容包装，避免旧调用立刻失效。
    apply_rigid_body_truth_configuration(dyn_model, config)
    apply_orbit_truth_configuration(dyn_model, config)


def attach_navigation_recorders(
    sim_base: Any,
    dyn_model: Any,
    sampling_time: int,
) -> tuple[Any, Any]:
    """Create and register the SimpleNav attitude and translation recorders."""
    # 这里记录的是“理想导航器”输出，
    # 也就是 FSW 实际看到的姿态/轨道信息。

    # 创建姿态输出消息的记录器，采样间隔为 sampling_time（单位：纳秒）
    # attOutMsg 包含 MRP 姿态信息
    att_nav_rec = dyn_model.simpleNavObject.attOutMsg.recorder(sampling_time)

    # 创建平动输出消息的记录器，采样间隔同上
    # transOutMsg 包含位置和速度信息
    trans_nav_rec = dyn_model.simpleNavObject.transOutMsg.recorder(sampling_time)

    sim_base.AddModelToTask(dyn_model.taskName, att_nav_rec)
    sim_base.AddModelToTask(dyn_model.taskName, trans_nav_rec)
    return att_nav_rec, trans_nav_rec

# 定义函数，从记录器中提取 SimpleNav 的姿态、位置、速度历史数据
def extract_navigation_history(
    att_nav_rec: Any,
    trans_nav_rec: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body attitude, position, and velocity histories from SimpleNav."""
    # 后面的姿态图和轨道姿态联合图都会用到这三组量。

    # 从姿态记录器中获取 MRP 历史数据，删除第 0 行（初始冗余采样），形状为 (N, 3)
    sigma_bn = np.delete(att_nav_rec.sigma_BN, 0, 0)
    r_bn_n = np.delete(trans_nav_rec.r_BN_N, 0, 0)
    v_bn_n = np.delete(trans_nav_rec.v_BN_N, 0, 0)
    return sigma_bn, r_bn_n, v_bn_n
