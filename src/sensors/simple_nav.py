"""SimpleNav sensor helpers for the project baseline scenario."""

from __future__ import annotations

from typing import Any

import numpy as np
from Basilisk.utilities import macros, orbitalMotion


def configure_spacecraft_initial_state(dyn_model: Any, config: dict[str, Any]) -> None:
    """Apply the configured orbit and attitude initial conditions to the spacecraft."""
    orbit_cfg = config["orbit"]
    attitude_cfg = config["attitude"]

    oe = orbitalMotion.ClassicElements()
    oe.a = orbit_cfg["a_m"]
    oe.e = orbit_cfg["e"]
    oe.i = orbit_cfg["i_deg"] * macros.D2R
    oe.Omega = orbit_cfg["Omega_deg"] * macros.D2R
    oe.omega = orbit_cfg["omega_deg"] * macros.D2R
    oe.f = orbit_cfg["f_deg"] * macros.D2R

    mu = dyn_model.gravFactory.gravBodies["earth"].mu
    r_n, v_n = orbitalMotion.elem2rv(mu, oe)

    dyn_model.scObject.hub.r_CN_NInit = r_n
    dyn_model.scObject.hub.v_CN_NInit = v_n
    dyn_model.scObject.hub.sigma_BNInit = [[value] for value in attitude_cfg["sigma_BN_init"]]
    dyn_model.scObject.hub.omega_BN_BInit = [
        [value] for value in attitude_cfg["omega_BN_B_init_rad_s"]
    ]


def attach_navigation_recorders(
    sim_base: Any,
    dyn_model: Any,
    sampling_time: int,
) -> tuple[Any, Any]:
    """Create and register the SimpleNav attitude and translation recorders."""
    att_nav_rec = dyn_model.simpleNavObject.attOutMsg.recorder(sampling_time)
    trans_nav_rec = dyn_model.simpleNavObject.transOutMsg.recorder(sampling_time)

    sim_base.AddModelToTask(dyn_model.taskName, att_nav_rec)
    sim_base.AddModelToTask(dyn_model.taskName, trans_nav_rec)
    return att_nav_rec, trans_nav_rec


def extract_navigation_history(
    att_nav_rec: Any,
    trans_nav_rec: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body attitude, position, and velocity histories from SimpleNav."""
    sigma_bn = np.delete(att_nav_rec.sigma_BN, 0, 0)
    r_bn_n = np.delete(trans_nav_rec.r_BN_N, 0, 0)
    v_bn_n = np.delete(trans_nav_rec.v_BN_N, 0, 0)
    return sigma_bn, r_bn_n, v_bn_n
