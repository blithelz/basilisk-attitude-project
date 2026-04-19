"""Output helpers for project-local Basilisk simulation scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.actuators.reaction_wheels import extract_reaction_wheel_history, get_reaction_wheel_count
from src.disturbances.torques import compute_disturbance_torque_history
from src.environment.leo import extract_environment_truth_history
from src.sensors.simple_nav import extract_navigation_history
from src.truth.orbit import extract_orbit_truth_history
from src.truth.rigid_body import extract_rigid_body_truth_history


def extract_guidance_history(
    att_ref_rec: Any,
    att_err_rec: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return reference attitude and tracking-error histories."""

    sigma_rn = np.delete(att_ref_rec.sigma_RN, 0, 0)
    omega_rn_n = np.delete(att_ref_rec.omega_RN_N, 0, 0)
    sigma_br = np.delete(att_err_rec.sigma_BR, 0, 0)
    omega_br_b = np.delete(att_err_rec.omega_BR_B, 0, 0)
    return sigma_rn, omega_rn_n, sigma_br, omega_br_b


def warn_if_noninteractive_backend(plotter: Any, show_plots: bool) -> None:
    """Explain why plot windows may not appear under a non-interactive backend."""

    if not show_plots:
        return

    backend = str(plotter.plt.get_backend()).lower()
    if "agg" in backend:
        print(
            "show_plots=True but the current Matplotlib backend is "
            f"'{plotter.plt.get_backend()}'. This backend is non-interactive, "
            "so no plot window will appear. Run the scenario directly without "
            "forcing MPLBACKEND=Agg if you want popup windows."
        )


def _plot_three_component_history(
    axis: Any,
    timeline_min: np.ndarray,
    values: np.ndarray,
    labels: tuple[str, str, str],
) -> None:
    axis.plot(timeline_min, values[:, 0], label=labels[0])
    axis.plot(timeline_min, values[:, 1], label=labels[1])
    axis.plot(timeline_min, values[:, 2], label=labels[2])
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")


def plot_reference_guidance(
    plotter: Any,
    timeline_min: np.ndarray,
    sigma_rn: np.ndarray,
    omega_rn_n: np.ndarray,
    figure_id: int,
) -> None:
    """Plot reference attitude and angular-rate histories."""

    plt = plotter.plt
    figure = plt.figure(figure_id)
    figure.clf()

    sigma_axis = figure.add_subplot(211)
    _plot_three_component_history(
        sigma_axis,
        timeline_min,
        sigma_rn,
        (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"),
    )
    sigma_axis.set_ylabel("MRP")
    sigma_axis.set_title(r"Reference Attitude: $\sigma_{RN}$")

    omega_axis = figure.add_subplot(212, sharex=sigma_axis)
    _plot_three_component_history(
        omega_axis,
        timeline_min,
        omega_rn_n,
        (r"$\omega_1$", r"$\omega_2$", r"$\omega_3$"),
    )
    omega_axis.set_xlabel("Time, min")
    omega_axis.set_ylabel("Angular Rate, rad/s")
    omega_axis.set_title(r"Reference Rate: $^N{\omega_{RN}}$")

    figure.tight_layout()


def plot_truth_attitude(
    plotter: Any,
    timeline_min: np.ndarray,
    sigma_bn: np.ndarray,
    omega_bn_b: np.ndarray,
    figure_id: int,
) -> None:
    """Plot propagated truth attitude and body-rate histories."""

    plt = plotter.plt
    figure = plt.figure(figure_id)
    figure.clf()

    sigma_axis = figure.add_subplot(211)
    _plot_three_component_history(
        sigma_axis,
        timeline_min,
        sigma_bn,
        (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"),
    )
    sigma_axis.set_ylabel("MRP")
    sigma_axis.set_title(r"Truth Attitude: $\sigma_{BN}$")

    omega_axis = figure.add_subplot(212, sharex=sigma_axis)
    _plot_three_component_history(
        omega_axis,
        timeline_min,
        omega_bn_b,
        (r"$\omega_1$", r"$\omega_2$", r"$\omega_3$"),
    )
    omega_axis.set_xlabel("Time, min")
    omega_axis.set_ylabel("Angular Rate, rad/s")
    omega_axis.set_title(r"Truth Body Rate: $^B{\omega_{BN}}$")

    figure.tight_layout()


def plot_truth_orbit(
    plotter: Any,
    timeline_min: np.ndarray,
    altitude_m: np.ndarray,
    speed_m_s: np.ndarray,
    figure_id: int,
) -> None:
    """Plot truth orbit altitude and speed histories."""

    plt = plotter.plt
    figure = plt.figure(figure_id)
    figure.clf()

    altitude_axis = figure.add_subplot(211)
    altitude_axis.plot(timeline_min, altitude_m / 1000.0)
    altitude_axis.set_ylabel("Altitude, km")
    altitude_axis.set_title("Truth Orbit Altitude")
    altitude_axis.grid(True, alpha=0.3)

    speed_axis = figure.add_subplot(212, sharex=altitude_axis)
    speed_axis.plot(timeline_min, speed_m_s / 1000.0)
    speed_axis.set_xlabel("Time, min")
    speed_axis.set_ylabel("Speed, km/s")
    speed_axis.set_title("Truth Orbit Speed")
    speed_axis.grid(True, alpha=0.3)

    figure.tight_layout()


def plot_truth_environment(
    plotter: Any,
    timeline_min: np.ndarray,
    sun_direction_b: np.ndarray,
    magnetic_field_b_t: np.ndarray,
    illumination_factor: np.ndarray,
    figure_id: int,
) -> None:
    """Plot body-frame sun direction, magnetic field, and eclipse truth histories."""

    plt = plotter.plt
    figure = plt.figure(figure_id)
    figure.clf()

    sun_axis = figure.add_subplot(311)
    _plot_three_component_history(
        sun_axis,
        timeline_min,
        sun_direction_b,
        (r"$s_x$", r"$s_y$", r"$s_z$"),
    )
    sun_axis.set_ylabel("Unit Vector")
    sun_axis.set_title("Sun Direction Truth in Body Frame")

    magnetic_field_axis = figure.add_subplot(312, sharex=sun_axis)
    _plot_three_component_history(
        magnetic_field_axis,
        timeline_min,
        magnetic_field_b_t * 1.0e9,
        (r"$B_x$", r"$B_y$", r"$B_z$"),
    )
    magnetic_field_axis.set_ylabel("Magnetic Field, nT")
    magnetic_field_axis.set_title("Magnetic Field Truth in Body Frame")

    eclipse_axis = figure.add_subplot(313, sharex=sun_axis)
    eclipse_axis.plot(timeline_min, illumination_factor)
    eclipse_axis.set_xlabel("Time, min")
    eclipse_axis.set_ylabel("Illumination")
    eclipse_axis.set_ylim(-0.05, 1.05)
    eclipse_axis.set_title("Eclipse Truth")
    eclipse_axis.grid(True, alpha=0.3)

    figure.tight_layout()


def plot_disturbance_torques(plotter: Any, disturbance_history: Any, figure_id: int) -> None:
    """Plot total disturbance torque components and contribution norms."""

    plt = plotter.plt
    figure = plt.figure(figure_id)
    figure.clf()

    component_axis = figure.add_subplot(211)
    _plot_three_component_history(
        component_axis,
        disturbance_history.time_min,
        disturbance_history.total_nm,
        (r"$L_x$", r"$L_y$", r"$L_z$"),
    )
    component_axis.set_ylabel("Torque, N m")
    component_axis.set_title("Total Disturbance Torque")

    norm_axis = figure.add_subplot(212, sharex=component_axis)
    norm_axis.plot(
        disturbance_history.time_min,
        np.linalg.norm(disturbance_history.gravity_gradient_nm, axis=1),
        label="Gravity Gradient",
    )
    norm_axis.plot(
        disturbance_history.time_min,
        np.linalg.norm(disturbance_history.aerodynamic_drag_nm, axis=1),
        label="Aerodynamic Drag",
    )
    norm_axis.plot(
        disturbance_history.time_min,
        np.linalg.norm(disturbance_history.solar_radiation_pressure_nm, axis=1),
        label="SRP",
    )
    norm_axis.plot(
        disturbance_history.time_min,
        np.linalg.norm(disturbance_history.magnetic_residual_dipole_nm, axis=1),
        label="Residual Dipole",
    )
    norm_axis.plot(
        disturbance_history.time_min,
        np.linalg.norm(disturbance_history.constant_bias_nm, axis=1),
        label="Constant Bias",
    )
    norm_axis.plot(
        disturbance_history.time_min,
        np.linalg.norm(disturbance_history.total_nm, axis=1),
        label="Total",
        linewidth=2.0,
    )
    norm_axis.set_xlabel("Time, min")
    norm_axis.set_ylabel("Torque Norm, N m")
    norm_axis.set_title("Disturbance Torque Contributions")
    norm_axis.grid(True, alpha=0.3)
    norm_axis.legend(loc="best")

    figure.tight_layout()


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
    sc_truth_rec: Any,
    sun_state_rec: Any,
    eclipse_rec: Any,
    magnetic_field_rec: Any,
    show_plots: bool,
    save_plots: bool,
) -> list[Path]:
    """Build plots and optionally save the baseline scenario results."""

    num_reaction_wheels = get_reaction_wheel_count(config)
    sigma_bn_nav, r_bn_n_nav, v_bn_n_nav = extract_navigation_history(att_nav_rec, trans_nav_rec)
    sigma_rn, omega_rn_n, sigma_br, omega_br_b = extract_guidance_history(att_ref_rec, att_err_rec)
    timeline_min, wheel_speeds, motor_torque = extract_reaction_wheel_history(
        rw_speed_rec,
        rw_motor_rec,
        num_reaction_wheels,
    )

    rigid_body_truth = extract_rigid_body_truth_history(sc_truth_rec)
    orbit_truth = extract_orbit_truth_history(sc_truth_rec, config)
    environment_truth = extract_environment_truth_history(
        config,
        rigid_body_truth,
        orbit_truth,
        sun_state_rec,
        eclipse_rec,
        magnetic_field_rec,
    )
    disturbance_truth = compute_disturbance_torque_history(
        config,
        rigid_body_truth,
        orbit_truth,
        environment_truth,
    )

    plotter.clear_all_plots()
    plotter.plot_attitude_error(timeline_min, sigma_br)
    plotter.plot_rw_cmd_torque(timeline_min, motor_torque, num_reaction_wheels)
    plotter.plot_rate_error(timeline_min, omega_br_b)
    plotter.plot_rw_speeds(timeline_min, wheel_speeds, num_reaction_wheels)
    plotter.plot_orientation(timeline_min, r_bn_n_nav, v_bn_n_nav, sigma_bn_nav)
    plot_reference_guidance(plotter, timeline_min, sigma_rn, omega_rn_n, figure_id=6)
    plot_truth_attitude(
        plotter,
        rigid_body_truth.time_min,
        rigid_body_truth.sigma_bn,
        rigid_body_truth.omega_bn_b,
        figure_id=7,
    )
    plot_truth_orbit(
        plotter,
        orbit_truth.time_min,
        orbit_truth.altitude_m,
        orbit_truth.speed_m_s,
        figure_id=8,
    )
    plot_truth_environment(
        plotter,
        environment_truth.time_min,
        environment_truth.sun_direction_b,
        environment_truth.magnetic_field_b_t,
        environment_truth.illumination_factor,
        figure_id=9,
    )
    plot_disturbance_torques(plotter, disturbance_truth, figure_id=10)

    saved_paths: list[Path] = []

    if save_plots:
        figure_names = [
            "attitudeErrorNorm",
            "rwMotorTorque",
            "rateError",
            "rwSpeed",
            "orientation",
            "attitudeGuidance",
            "truthAttitude",
            "truthOrbit",
            "truthEnvironment",
            "disturbanceTorque",
        ]
        figures = {
            f"{config['output']['file_prefix']}_{name}": plotter.plt.figure(index + 1)
            for index, name in enumerate(figure_names)
        }
        saved_paths = save_figure_bundle(results_dir, figures)

    if show_plots:
        warn_if_noninteractive_backend(plotter, show_plots)
        plotter.show_all_plots()

    return saved_paths
