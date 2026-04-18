"""Output helpers for project-local Basilisk simulation scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.actuators.reaction_wheels import extract_reaction_wheel_history, get_reaction_wheel_count
from src.sensors.simple_nav import extract_navigation_history


def extract_guidance_history(
    att_ref_rec: Any,
    att_err_rec: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return reference attitude and tracking-error histories."""
    # 这里把“目标姿态”和“姿态误差”统一拆出来，
    # 方便后面的 plot 函数只关心数组，不再关心消息对象。
    sigma_rn = np.delete(att_ref_rec.sigma_RN, 0, 0)
    omega_rn_n = np.delete(att_ref_rec.omega_RN_N, 0, 0)
    sigma_br = np.delete(att_err_rec.sigma_BR, 0, 0)
    omega_br_b = np.delete(att_err_rec.omega_BR_B, 0, 0)
    return sigma_rn, omega_rn_n, sigma_br, omega_br_b


def save_figure_bundle(results_dir: Path, figures: dict[str, Any]) -> list[Path]:
    """Persist matplotlib figures into the project results folder."""
    saved_paths = []
    for figure_name, figure in figures.items():
        target = results_dir / f"{figure_name}.png"
        figure.savefig(target, dpi=200, bbox_inches="tight")
        saved_paths.append(target)
    return saved_paths


def render_baseline_outputs(
    plotter: Any,
    config: dict[str, Any],
    results_dir: Path,
    att_nav_rec: Any,
    trans_nav_rec: Any,
    att_ref_rec: Any,
    att_err_rec: Any,
    rw_speed_rec: Any,
    rw_motor_rec: Any,
    show_plots: bool,
    save_plots: bool,
) -> list[Path]:
    """Build plots and optionally save the baseline scenario results."""
    num_reaction_wheels = get_reaction_wheel_count(config)
    sigma_bn, r_bn_n, v_bn_n = extract_navigation_history(att_nav_rec, trans_nav_rec)
    sigma_rn, omega_rn_n, sigma_br, omega_br_b = extract_guidance_history(att_ref_rec, att_err_rec)
    timeline, wheel_speeds, motor_torque = extract_reaction_wheel_history(
        rw_speed_rec, rw_motor_rec, num_reaction_wheels
    )

    # 绘图顺序要和 figure_names 保持一致，
    # 否则后面按序号保存图片时，文件名和图内容会对不上。
    plotter.clear_all_plots()
    plotter.plot_attitude_error(timeline, sigma_br)
    plotter.plot_rw_cmd_torque(timeline, motor_torque, num_reaction_wheels)
    plotter.plot_rate_error(timeline, omega_br_b)
    plotter.plot_rw_speeds(timeline, wheel_speeds, num_reaction_wheels)
    plotter.plot_orientation(timeline, r_bn_n, v_bn_n, sigma_bn)
    plotter.plot_attitudeGuidance(timeline, sigma_rn, omega_rn_n)

    saved_paths: list[Path] = []
    if save_plots:
        figure_names = [
            "attitudeErrorNorm",
            "rwMotorTorque",
            "rateError",
            "rwSpeed",
            "orientation",
            "attitudeGuidance",
        ]
        # plotter.plt.figure(index + 1) 依赖前面的绘图顺序；
        # 这也是为什么 figure_names 不能随意改顺序。
        figures = {
            f"{config['output']['file_prefix']}_{name}": plotter.plt.figure(index + 1)
            for index, name in enumerate(figure_names)
        }
        saved_paths = save_figure_bundle(results_dir, figures)

    if show_plots:
        plotter.show_all_plots()

    return saved_paths
