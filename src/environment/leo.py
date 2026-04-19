"""LEO environment truth helpers for magnetic field, sun direction, and eclipse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.truth.orbit import OrbitTruthHistory
from src.truth.rigid_body import RigidBodyTruthHistory, rotate_inertial_to_body_history


@dataclass(frozen=True)
class EnvironmentTruthHistory:
    """Truth history for environment quantities used by sensors and disturbances."""

    time_ns: np.ndarray
    time_min: np.ndarray
    sun_position_n_m: np.ndarray
    sun_direction_n: np.ndarray
    sun_direction_b: np.ndarray
    magnetic_field_n_t: np.ndarray
    magnetic_field_b_t: np.ndarray
    illumination_factor: np.ndarray
    is_eclipsed: np.ndarray


def _normalize_vector_history(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalized = vectors / safe_norms
    normalized[norms.squeeze(axis=1) <= 0.0] = 0.0
    return normalized


def attach_environment_recorders(
    sim_base: Any,
    dyn_model: Any,
    sampling_time: int,
) -> tuple[Any, Any, Any]:
    """Attach recorder objects for sun ephemeris, eclipse, and magnetic field truth."""

    sun_state_rec = dyn_model.gravFactory.spiceObject.planetStateOutMsgs[dyn_model.sun].recorder(
        sampling_time
    )
    eclipse_rec = dyn_model.eclipseObject.eclipseOutMsgs[0].recorder(sampling_time)
    magnetic_field_rec = dyn_model.magModule.envOutMsgs[0].recorder(sampling_time)

    sim_base.AddModelToTask(dyn_model.taskName, sun_state_rec)
    sim_base.AddModelToTask(dyn_model.taskName, eclipse_rec)
    sim_base.AddModelToTask(dyn_model.taskName, magnetic_field_rec)

    return sun_state_rec, eclipse_rec, magnetic_field_rec


def extract_environment_truth_history(
    config: dict[str, Any],
    rigid_body_truth: RigidBodyTruthHistory,
    orbit_truth: OrbitTruthHistory,
    sun_state_rec: Any,
    eclipse_rec: Any,
    magnetic_field_rec: Any,
) -> EnvironmentTruthHistory:
    """Return environment truth arrays derived from propagated states and env recorders."""

    environment_cfg = config.get("environment", {})
    sun_enabled = environment_cfg.get("sun", {}).get("enabled", True)
    eclipse_enabled = environment_cfg.get("eclipse", {}).get("enabled", True)
    magnetic_field_enabled = environment_cfg.get("magnetic_field", {}).get("enabled", True)

    sample_count = rigid_body_truth.time_ns.shape[0]
    zero_vector_history = np.zeros((sample_count, 3))

    if sun_enabled:
        sun_position_n_m = np.delete(sun_state_rec.PositionVector, 0, 0)
        vector_sc_to_sun_n = sun_position_n_m - orbit_truth.r_bn_n
        sun_direction_n = _normalize_vector_history(vector_sc_to_sun_n)
        sun_direction_b = rotate_inertial_to_body_history(rigid_body_truth.dcm_bn, sun_direction_n)
    else:
        sun_position_n_m = zero_vector_history.copy()
        sun_direction_n = zero_vector_history.copy()
        sun_direction_b = zero_vector_history.copy()

    if magnetic_field_enabled:
        magnetic_field_n_t = np.delete(magnetic_field_rec.magField_N, 0, 0)
        magnetic_field_b_t = rotate_inertial_to_body_history(
            rigid_body_truth.dcm_bn,
            magnetic_field_n_t,
        )
    else:
        magnetic_field_n_t = zero_vector_history.copy()
        magnetic_field_b_t = zero_vector_history.copy()

    if eclipse_enabled:
        illumination_factor = np.delete(eclipse_rec.illuminationFactor, 0, 0).reshape(-1)
    else:
        illumination_factor = np.ones(sample_count)

    return EnvironmentTruthHistory(
        time_ns=rigid_body_truth.time_ns,
        time_min=rigid_body_truth.time_min,
        sun_position_n_m=sun_position_n_m,
        sun_direction_n=sun_direction_n,
        sun_direction_b=sun_direction_b,
        magnetic_field_n_t=magnetic_field_n_t,
        magnetic_field_b_t=magnetic_field_b_t,
        illumination_factor=illumination_factor,
        is_eclipsed=illumination_factor < 0.999999,
    )
