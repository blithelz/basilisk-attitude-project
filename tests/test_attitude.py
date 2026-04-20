from __future__ import annotations

import unittest

import numpy as np

from src.truth.attitude import AttitudeConfig, AttitudeState, step_attitude_state
from src.utils.frames import switch_to_shadow_mrp


class AttitudeModelTest(unittest.TestCase):
    def test_zero_torque_zero_rate_holds_attitude(self) -> None:
        config = AttitudeConfig(
            inertia_kg_m2=np.diag([0.45, 0.35, 0.25]),
            initial_sigma_bn=np.array([0.1, -0.05, 0.08]),
            initial_omega_bn_b_rad_s=np.zeros(3),
        )
        state = AttitudeState(
            sigma_bn=config.initial_sigma_bn.copy(),
            omega_bn_b_rad_s=config.initial_omega_bn_b_rad_s.copy(),
        )

        next_state = step_attitude_state(state, config, np.zeros(3), 5.0)

        self.assertTrue(np.allclose(next_state.sigma_bn, state.sigma_bn))
        self.assertTrue(np.allclose(next_state.omega_bn_b_rad_s, state.omega_bn_b_rad_s))

    def test_shadow_set_switch_reduces_mrp_norm(self) -> None:
        sigma_bn = np.array([1.2, 0.1, -0.2])
        sigma_shadow = switch_to_shadow_mrp(sigma_bn)

        self.assertLessEqual(np.linalg.norm(sigma_shadow), 1.0)
