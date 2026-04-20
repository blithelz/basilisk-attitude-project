"""External-disturbance torque helpers for the project-local LEO truth model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.environment.leo import EnvironmentTruthHistory
from src.truth.orbit import OrbitTruthHistory, get_central_body_constants
from src.truth.rigid_body import RigidBodyTruthHistory, get_spacecraft_inertia_matrix, rotate_inertial_to_body_history


DEFAULT_SOLAR_PRESSURE_N_M2 = 4.56e-6


@dataclass(frozen=True)
class DisturbanceTorqueHistory:
    """Truth history for each modeled disturbance-torque contribution."""

    time_min: np.ndarray
    gravity_gradient_nm: np.ndarray
    aerodynamic_drag_nm: np.ndarray
    solar_radiation_pressure_nm: np.ndarray
    magnetic_residual_dipole_nm: np.ndarray
    constant_bias_nm: np.ndarray
    total_nm: np.ndarray


def _read_vector3(section: dict[str, Any], key: str, default: tuple[float, float, float]) -> np.ndarray:
    value = section.get(key, list(default))
    return np.asarray(value, dtype=float).reshape(3)


def _broadcast_vector(vector: np.ndarray, sample_count: int) -> np.ndarray:
    return np.repeat(vector.reshape(1, 3), sample_count, axis=0)


def compute_exponential_density_history(
    altitude_m: np.ndarray,
    density_cfg: dict[str, Any],
) -> np.ndarray:
    """Return a simple exponential atmosphere density profile."""

    reference_density_kg_m3 = float(density_cfg.get("reference_density_kg_m3", 3.5e-12))
    reference_altitude_m = float(density_cfg.get("reference_altitude_m", 400_000.0))
    scale_height_m = float(density_cfg.get("scale_height_m", 60_000.0))

    altitude_offset = np.asarray(altitude_m, dtype=float) - reference_altitude_m
    return reference_density_kg_m3 * np.exp(-altitude_offset / scale_height_m)


def compute_gravity_gradient_torque_history(
    orbit_truth: OrbitTruthHistory,
    rigid_body_truth: RigidBodyTruthHistory,
    inertia_matrix_kg_m2: np.ndarray,
    mu_m3_s2: float,
) -> np.ndarray:
    """Return the gravity-gradient disturbance torque history in the body frame."""

    r_hat_n = orbit_truth.r_bn_n / orbit_truth.radius_m[:, None]
    r_hat_b = rotate_inertial_to_body_history(rigid_body_truth.dcm_bn, r_hat_n)

    inertia_times_radius = np.einsum("ij,nj->ni", inertia_matrix_kg_m2, r_hat_b)
    gravity_gradient_scale = 3.0 * mu_m3_s2 / np.power(orbit_truth.radius_m, 3)
    return gravity_gradient_scale[:, None] * np.cross(r_hat_b, inertia_times_radius)


def compute_disturbance_torque_history(
    config: dict[str, Any],
    rigid_body_truth: RigidBodyTruthHistory,
    orbit_truth: OrbitTruthHistory,
    environment_truth: EnvironmentTruthHistory,
) -> DisturbanceTorqueHistory:
    """Return the modeled disturbance torques derived from truth state and environment layers."""

    sample_count = rigid_body_truth.time_min.shape[0]
    zero_history = np.zeros((sample_count, 3))

    disturbances_cfg = config.get("disturbances", {})
    if not disturbances_cfg.get("model_enabled", True):
        return DisturbanceTorqueHistory(
            time_min=rigid_body_truth.time_min,
            gravity_gradient_nm=zero_history.copy(),
            aerodynamic_drag_nm=zero_history.copy(),
            solar_radiation_pressure_nm=zero_history.copy(),
            magnetic_residual_dipole_nm=zero_history.copy(),
            constant_bias_nm=zero_history.copy(),
            total_nm=zero_history.copy(),
        )

    inertia_matrix_kg_m2 = get_spacecraft_inertia_matrix(config)
    mu_m3_s2, _ = get_central_body_constants(config)

    gravity_gradient_cfg = disturbances_cfg.get("gravity_gradient", {})
    aerodynamic_drag_cfg = disturbances_cfg.get("aerodynamic_drag", {})
    solar_radiation_cfg = disturbances_cfg.get("solar_radiation_pressure", {})
    magnetic_dipole_cfg = disturbances_cfg.get("magnetic_residual_dipole", {})
    constant_bias_cfg = disturbances_cfg.get("constant_bias", {})

    gravity_gradient_nm = zero_history.copy()
    aerodynamic_drag_nm = zero_history.copy()
    solar_radiation_pressure_nm = zero_history.copy()
    magnetic_residual_dipole_nm = zero_history.copy()
    constant_bias_nm = zero_history.copy()

    if gravity_gradient_cfg.get("enabled", True):
        gravity_gradient_nm = compute_gravity_gradient_torque_history(
            orbit_truth,
            rigid_body_truth,
            inertia_matrix_kg_m2,
            mu_m3_s2,
        )

    if aerodynamic_drag_cfg.get("enabled", True):
        velocity_b_m_s = rotate_inertial_to_body_history(rigid_body_truth.dcm_bn, orbit_truth.v_bn_n)
        density_kg_m3 = compute_exponential_density_history(
            orbit_truth.altitude_m,
            aerodynamic_drag_cfg,
        )
        drag_coefficient = float(aerodynamic_drag_cfg.get("drag_coefficient", 2.2))
        reference_area_m2 = float(aerodynamic_drag_cfg.get("reference_area_m2", 0.15))
        center_of_pressure_b_m = _read_vector3(
            aerodynamic_drag_cfg,
            "center_of_pressure_b_m",
            (0.05, 0.0, 0.02),
        )

        aerodynamic_force_b_n = (
            -0.5
            * density_kg_m3[:, None]
            * drag_coefficient
            * reference_area_m2
            * np.linalg.norm(velocity_b_m_s, axis=1, keepdims=True)
            * velocity_b_m_s
        )
        aerodynamic_drag_nm = np.cross(
            _broadcast_vector(center_of_pressure_b_m, sample_count),
            aerodynamic_force_b_n,
        )

    if solar_radiation_cfg.get("enabled", True):
        solar_pressure_n_m2 = float(
            solar_radiation_cfg.get("solar_pressure_n_m2", DEFAULT_SOLAR_PRESSURE_N_M2)
        )
        reflection_coefficient = float(solar_radiation_cfg.get("reflection_coefficient", 1.3))
        reference_area_m2 = float(solar_radiation_cfg.get("reference_area_m2", 0.12))
        center_of_pressure_b_m = _read_vector3(
            solar_radiation_cfg,
            "center_of_pressure_b_m",
            (0.03, 0.0, 0.01),
        )

        srp_force_b_n = (
            -solar_pressure_n_m2
            * reflection_coefficient
            * reference_area_m2
            * environment_truth.illumination_factor[:, None]
            * environment_truth.sun_direction_b
        )
        solar_radiation_pressure_nm = np.cross(
            _broadcast_vector(center_of_pressure_b_m, sample_count),
            srp_force_b_n,
        )

    if magnetic_dipole_cfg.get("enabled", True):
        residual_dipole_a_m2 = _read_vector3(
            magnetic_dipole_cfg,
            "dipole_body_a_m2",
            (0.05, -0.03, 0.02),
        )
        magnetic_residual_dipole_nm = np.cross(
            _broadcast_vector(residual_dipole_a_m2, sample_count),
            environment_truth.magnetic_field_b_t,
        )

    if constant_bias_cfg.get("enabled", False):
        constant_bias_b_nm = _read_vector3(
            constant_bias_cfg,
            "torque_body_nm",
            (0.0, 0.0, 0.0),
        )
        constant_bias_nm = _broadcast_vector(constant_bias_b_nm, sample_count)

    total_nm = (
        gravity_gradient_nm
        + aerodynamic_drag_nm
        + solar_radiation_pressure_nm
        + magnetic_residual_dipole_nm
        + constant_bias_nm
    )

    return DisturbanceTorqueHistory(
        time_min=rigid_body_truth.time_min,
        gravity_gradient_nm=gravity_gradient_nm,
        aerodynamic_drag_nm=aerodynamic_drag_nm,
        solar_radiation_pressure_nm=solar_radiation_pressure_nm,
        magnetic_residual_dipole_nm=magnetic_residual_dipole_nm,
        constant_bias_nm=constant_bias_nm,
        total_nm=total_nm,
    )
