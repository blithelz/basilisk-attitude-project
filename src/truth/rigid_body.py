"""Rigid-body truth helpers for project-local spacecraft attitude dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros, unitTestSupport as sp


DEFAULT_SPACECRAFT_MASS_KG = 750.0
DEFAULT_SPACECRAFT_INERTIA_KG_M2 = np.array(
    [
        [900.0, 0.0, 0.0],
        [0.0, 800.0, 0.0],
        [0.0, 0.0, 600.0],
    ]
)
DEFAULT_CENTER_OF_MASS_OFFSET_B_M = np.zeros(3)
DEFAULT_SIGMA_BN_INIT = np.array([0.1, 0.2, -0.3])
DEFAULT_OMEGA_BN_B_INIT_RAD_S = np.array([0.001, -0.01, 0.03])


@dataclass(frozen=True)
class RigidBodyTruthHistory:
    """Truth history for the propagated body attitude state."""

    time_ns: np.ndarray
    time_min: np.ndarray
    sigma_bn: np.ndarray
    omega_bn_b: np.ndarray
    dcm_bn: np.ndarray


def _read_vector3(section: dict[str, Any], key: str, default: np.ndarray) -> np.ndarray:
    value = section.get(key, default.tolist())
    return np.asarray(value, dtype=float).reshape(3)


def _read_inertia_matrix(config: dict[str, Any]) -> np.ndarray:
    spacecraft_cfg = config.get("spacecraft", {})
    inertia_cfg = spacecraft_cfg.get("inertia_kg_m2", DEFAULT_SPACECRAFT_INERTIA_KG_M2.tolist())
    return np.asarray(inertia_cfg, dtype=float).reshape(3, 3)


def get_spacecraft_inertia_matrix(config: dict[str, Any]) -> np.ndarray:
    """Return the spacecraft inertia matrix configured for the truth model."""

    return _read_inertia_matrix(config)


def apply_rigid_body_truth_configuration(dyn_model: Any, config: dict[str, Any]) -> None:
    """Configure hub mass properties and initial attitude truth values."""

    spacecraft_cfg = config.get("spacecraft", {})
    attitude_cfg = config.get("attitude", {})

    inertia_matrix = _read_inertia_matrix(config)
    center_of_mass_offset_b_m = _read_vector3(
        spacecraft_cfg,
        "center_of_mass_offset_b_m",
        DEFAULT_CENTER_OF_MASS_OFFSET_B_M,
    )
    sigma_bn_init = _read_vector3(attitude_cfg, "sigma_BN_init", DEFAULT_SIGMA_BN_INIT)
    omega_bn_b_init = _read_vector3(
        attitude_cfg,
        "omega_BN_B_init_rad_s",
        DEFAULT_OMEGA_BN_B_INIT_RAD_S,
    )

    dyn_model.I_sc = inertia_matrix.reshape(-1).tolist()
    dyn_model.scObject.hub.mHub = float(spacecraft_cfg.get("mass_kg", DEFAULT_SPACECRAFT_MASS_KG))
    dyn_model.scObject.hub.r_BcB_B = [[float(value)] for value in center_of_mass_offset_b_m]
    dyn_model.scObject.hub.IHubPntBc_B = sp.np2EigenMatrix3d(dyn_model.I_sc)
    dyn_model.scObject.hub.sigma_BNInit = [[float(value)] for value in sigma_bn_init]
    dyn_model.scObject.hub.omega_BN_BInit = [[float(value)] for value in omega_bn_b_init]


def update_fsw_vehicle_configuration(fsw_model: Any, config: dict[str, Any]) -> None:
    """Keep the FSW vehicle-configuration message aligned with the truth inertia."""

    vehicle_config = messaging.VehicleConfigMsgPayload()
    vehicle_config.ISCPntB_B = _read_inertia_matrix(config).reshape(-1).tolist()
    fsw_model.vcMsg.write(vehicle_config)


def attach_truth_state_recorder(
    sim_base: Any,
    dyn_model: Any,
    sampling_time: int,
) -> Any:
    """Attach the spacecraft truth-state recorder to the dynamics task."""

    sc_truth_rec = dyn_model.scObject.scStateOutMsg.recorder(sampling_time)
    sim_base.AddModelToTask(dyn_model.taskName, sc_truth_rec)
    return sc_truth_rec


def rotate_inertial_to_body_history(dcm_bn: np.ndarray, vectors_n: np.ndarray) -> np.ndarray:
    """Rotate a vector history from the inertial frame into the body frame."""

    return np.einsum("nij,nj->ni", dcm_bn, vectors_n)


def extract_rigid_body_truth_history(sc_truth_rec: Any) -> RigidBodyTruthHistory:
    """Return the propagated truth attitude history from the spacecraft state message."""

    time_ns = np.delete(sc_truth_rec.times(), 0, 0)
    sigma_bn = np.delete(sc_truth_rec.sigma_BN, 0, 0)
    omega_bn_b = np.delete(sc_truth_rec.omega_BN_B, 0, 0)
    dcm_bn = np.asarray([rbk.MRP2C(sigma_bn_row) for sigma_bn_row in sigma_bn], dtype=float)

    return RigidBodyTruthHistory(
        time_ns=time_ns,
        time_min=time_ns * macros.NANO2MIN,
        sigma_bn=sigma_bn,
        omega_bn_b=omega_bn_b,
        dcm_bn=dcm_bn,
    )
