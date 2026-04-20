from __future__ import annotations

import unittest

import numpy as np

from src.truth.attitude import AttitudeState
from src.truth.environment import (
    EnvironmentConfig,
    compute_centered_dipole_field_n,
    compute_eclipse_factor,
    evaluate_environment,
)


class EnvironmentModelTest(unittest.TestCase):
    def test_eclipse_factor_detects_spacecraft_behind_earth(self) -> None:
        position_n_m = np.array([-7000e3, 0.0, 0.0])
        sun_direction_n = np.array([1.0, 0.0, 0.0])

        eclipse = compute_eclipse_factor(position_n_m, sun_direction_n, 6378136.6)

        self.assertEqual(eclipse, 0.0)

    def test_magnetic_field_model_returns_finite_vector(self) -> None:
        magnetic_field_n_t = compute_centered_dipole_field_n(np.array([7000e3, 0.0, 0.0]), np.array([0.0, 0.0, 7.94e22]))

        self.assertEqual(magnetic_field_n_t.shape, (3,))
        self.assertTrue(np.all(np.isfinite(magnetic_field_n_t)))
        self.assertGreater(np.linalg.norm(magnetic_field_n_t), 0.0)

    def test_environment_output_keeps_sun_direction_normalized(self) -> None:
        config = EnvironmentConfig(
            sun_direction_n=np.array([1.0, 0.2, 0.1]),
            central_body_radius_m=6378136.6,
            magnetic_dipole_n_a_m2=np.array([0.0, 0.0, 7.94e22]),
            atmosphere_reference_density_kg_m3=3.5e-12,
            atmosphere_reference_altitude_m=400000.0,
            atmosphere_scale_height_m=60000.0,
            solar_pressure_n_m2=4.56e-6,
            enable_eclipse=True,
        )
        attitude_state = AttitudeState(
            sigma_bn=np.array([0.1, -0.05, 0.08]),
            omega_bn_b_rad_s=np.zeros(3),
        )

        sample = evaluate_environment(0.0, np.array([7000e3, 0.0, 0.0]), attitude_state, config)

        self.assertAlmostEqual(np.linalg.norm(sample.sun_direction_n), 1.0, places=10)
