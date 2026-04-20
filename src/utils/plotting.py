"""Plotting helpers for the project-local pure Python truth model."""
# 模块文档字符串：为项目本地的纯 Python 真实模型提供绘图相关的辅助函数

from __future__ import annotations
# 启用推迟注解求值，允许在类型提示中使用尚未定义的类型名

from pathlib import Path
# 导入 Path 类，用于处理文件系统路径

from typing import Any
# 导入 Any 类型，表示任意类型

import matplotlib.pyplot as plt
# 导入 matplotlib.pyplot 并简写为 plt，用于创建图形和绘图

import numpy as np
# 导入 numpy 并简写为 np，用于数值数组操作


def _save_figure(figure: Any, output_dir: Path, file_name: str) -> Path:
    # 内部辅助函数：将 matplotlib 图形保存为 PNG 文件，返回保存的文件路径
    output_dir.mkdir(parents=True, exist_ok=True)
    # 确保输出目录存在（递归创建父目录，若已存在则忽略）
    target = output_dir / file_name
    # 构建完整的保存路径
    figure.savefig(target, dpi=200, bbox_inches="tight")
    # 保存图形，设置分辨率为 200 DPI，并紧凑裁剪空白边缘
    return target
    # 返回保存的文件路径


def _plot_three_components(axis: Any, time_s: np.ndarray, values: np.ndarray, labels: tuple[str, str, str]) -> None:
    # 内部辅助函数：在给定的坐标轴上绘制三个分量的时间历程曲线
    axis.plot(time_s, values[:, 0], label=labels[0])
    # 绘制第一个分量（如 X 分量）
    axis.plot(time_s, values[:, 1], label=labels[1])
    # 绘制第二个分量（如 Y 分量）
    axis.plot(time_s, values[:, 2], label=labels[2])
    # 绘制第三个分量（如 Z 分量）
    axis.grid(True, alpha=0.3)
    # 显示网格，设置透明度为 0.3
    axis.legend(loc="best")
    # 显示图例，自动选择最佳位置


def save_truth_model_plots(result: Any, output_dir: Path, show_plots: bool = False) -> list[Path]:
    """Create and save the standard week-2 truth-model plots."""
    # 创建并保存第二周标准真值模型图表，返回保存的文件路径列表

    saved_paths: list[Path] = []
    # 初始化列表，用于记录所有保存的图片路径

    time_min = result.orbit.time_s / 60.0
    # 将仿真时间从秒转换为分钟，作为所有图的统一时间横轴

    # --- 第一幅图：轨道真值图 ---
    orbit_figure = plt.figure(figsize=(10, 8))
    # 创建图形对象，尺寸 10×8 英寸

    orbit_axis_altitude = orbit_figure.add_subplot(211)
    # 创建 2 行 1 列的上方子图，用于绘制轨道高度
    orbit_axis_altitude.plot(time_min, result.orbit.altitude_m / 1000.0)
    # 绘制高度曲线，将单位从米转换为千米
    orbit_axis_altitude.set_ylabel("Altitude, km")
    # 设置 Y 轴标签
    orbit_axis_altitude.set_title("Orbit Truth")
    # 设置子图标题
    orbit_axis_altitude.grid(True, alpha=0.3)
    # 显示网格

    orbit_axis_speed = orbit_figure.add_subplot(212, sharex=orbit_axis_altitude)
    # 创建 2 行 1 列的下方子图，共享上方子图的 X 轴
    orbit_axis_speed.plot(time_min, result.orbit.speed_m_s / 1000.0)
    # 绘制飞行速率曲线，将单位从 m/s 转换为 km/s
    orbit_axis_speed.set_xlabel("Time, min")
    # 设置 X 轴标签
    orbit_axis_speed.set_ylabel("Speed, km/s")
    # 设置 Y 轴标签
    orbit_axis_speed.grid(True, alpha=0.3)

    orbit_figure.tight_layout()
    # 自动调整子图间距，防止标签重叠
    saved_paths.append(_save_figure(orbit_figure, output_dir, "orbit_truth.png"))
    # 保存轨道真值图，文件名为 "orbit_truth.png"

    # --- 第二幅图：姿态真值图 ---
    attitude_figure = plt.figure(figsize=(10, 8))
    attitude_axis_sigma = attitude_figure.add_subplot(211)
    # 上方子图：MRP 姿态参数时间历程
    _plot_three_components(
        attitude_axis_sigma,
        time_min,
        result.attitude.sigma_bn,
        (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"),
    )
    # 调用辅助函数绘制三个 MRP 分量，使用 LaTeX 格式标签
    attitude_axis_sigma.set_ylabel("MRP")
    attitude_axis_sigma.set_title("Attitude Truth")

    attitude_axis_omega = attitude_figure.add_subplot(212, sharex=attitude_axis_sigma)
    # 下方子图：角速度时间历程
    _plot_three_components(
        attitude_axis_omega,
        time_min,
        result.attitude.omega_bn_b_rad_s,
        (r"$\omega_1$", r"$\omega_2$", r"$\omega_3$"),
    )
    attitude_axis_omega.set_xlabel("Time, min")
    attitude_axis_omega.set_ylabel("Angular Rate, rad/s")

    attitude_figure.tight_layout()
    saved_paths.append(_save_figure(attitude_figure, output_dir, "attitude_truth.png"))
    # 保存姿态真值图

    # --- 第三幅图：环境真值图 ---
    environment_figure = plt.figure(figsize=(10, 10))
    environment_axis_sun = environment_figure.add_subplot(311)
    # 第一个子图：太阳方向单位矢量（体坐标系下）
    _plot_three_components(
        environment_axis_sun,
        time_min,
        result.environment.sun_direction_b,
        (r"$s_x$", r"$s_y$", r"$s_z$"),
    )
    environment_axis_sun.set_ylabel("Unit Vector")
    environment_axis_sun.set_title("Environment Truth")

    environment_axis_mag = environment_figure.add_subplot(312, sharex=environment_axis_sun)
    # 第二个子图：地磁场矢量（体坐标系下），单位转换为纳特斯拉（nT）
    _plot_three_components(
        environment_axis_mag,
        time_min,
        result.environment.magnetic_field_b_t * 1.0e9,
        (r"$B_x$", r"$B_y$", r"$B_z$"),
    )
    environment_axis_mag.set_ylabel("Magnetic Field, nT")

    environment_axis_eclipse = environment_figure.add_subplot(313, sharex=environment_axis_sun)
    # 第三个子图：光照指示（1 表示受晒，0 表示地影）
    environment_axis_eclipse.plot(time_min, result.environment.illumination)
    environment_axis_eclipse.set_xlabel("Time, min")
    environment_axis_eclipse.set_ylabel("Illumination")
    environment_axis_eclipse.set_ylim(-0.05, 1.05)
    # 固定 Y 轴范围为 [0, 1] 稍留边距
    environment_axis_eclipse.grid(True, alpha=0.3)

    environment_figure.tight_layout()
    saved_paths.append(_save_figure(environment_figure, output_dir, "environment_truth.png"))
    # 保存环境真值图

    # --- 第四幅图：干扰力矩真值图 ---
    disturbance_figure = plt.figure(figsize=(10, 8))
    disturbance_axis_total = disturbance_figure.add_subplot(211)
    # 上方子图：总干扰力矩的三个分量（体坐标系下）
    _plot_three_components(
        disturbance_axis_total,
        time_min,
        result.disturbances.total_torque_b_nm,
        (r"$L_x$", r"$L_y$", r"$L_z$"),
    )
    disturbance_axis_total.set_ylabel("Torque, N m")
    disturbance_axis_total.set_title("Disturbance Torque Truth")

    disturbance_axis_norm = disturbance_figure.add_subplot(212, sharex=disturbance_axis_total)
    # 下方子图：各分量干扰力矩的模长对比，以及总力矩模长
    disturbance_axis_norm.plot(time_min, np.linalg.norm(result.disturbances.gravity_gradient_torque_b_nm, axis=1), label="Gravity Gradient")
    # 重力梯度力矩模长
    disturbance_axis_norm.plot(time_min, np.linalg.norm(result.disturbances.drag_torque_b_nm, axis=1), label="Aerodynamic Drag")
    # 气动阻力矩模长
    disturbance_axis_norm.plot(time_min, np.linalg.norm(result.disturbances.srp_torque_b_nm, axis=1), label="SRP")
    # 太阳光压力矩模长
    disturbance_axis_norm.plot(time_min, np.linalg.norm(result.disturbances.magnetic_torque_b_nm, axis=1), label="Residual Dipole")
    # 残余磁偶极矩模长
    disturbance_axis_norm.plot(time_min, np.linalg.norm(result.disturbances.total_torque_b_nm, axis=1), label="Total", linewidth=2.0)
    # 总力矩模长，加粗线宽以突出显示

    disturbance_axis_norm.set_xlabel("Time, min")
    disturbance_axis_norm.set_ylabel("Torque Norm, N m")
    disturbance_axis_norm.grid(True, alpha=0.3)
    disturbance_axis_norm.legend(loc="best")
    disturbance_figure.tight_layout()
    saved_paths.append(_save_figure(disturbance_figure, output_dir, "disturbance_truth.png"))
    # 保存干扰力矩真值图

    if show_plots:
        plt.show()
        # 如果要求显示图形，则弹出交互式窗口
    else:
        plt.close("all")
        # 否则关闭所有图形，释放内存

    return saved_paths
    # 返回所有保存的图片路径列表