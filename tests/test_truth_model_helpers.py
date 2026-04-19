from __future__ import annotations

import unittest

import numpy as np

from src.disturbances.torques import (
    compute_disturbance_torque_history,
    compute_exponential_density_history,
    compute_gravity_gradient_torque_history,
)
from src.environment.leo import EnvironmentTruthHistory
from src.truth.orbit import OrbitTruthHistory
from src.truth.rigid_body import RigidBodyTruthHistory


class TruthModelHelperTest(unittest.TestCase):
    def test_exponential_density_matches_reference_point(self) -> None:
        altitude_m = np.array([400_000.0, 460_000.0])
        density_cfg = {
            "reference_density_kg_m3": 3.5e-12,
            "reference_altitude_m": 400_000.0,
            "scale_height_m": 60_000.0,
        }

        density = compute_exponential_density_history(altitude_m, density_cfg)

        self.assertAlmostEqual(density[0], 3.5e-12)
        self.assertLess(density[1], density[0])

    def test_gravity_gradient_torque_is_zero_for_spherical_inertia(self) -> None:
        orbit_truth = OrbitTruthHistory(
            time_ns=np.array([0.0, 1.0]),
            time_min=np.array([0.0, 1.0]),
            r_bn_n=np.array([[7_000_000.0, 0.0, 0.0], [7_000_000.0, 0.0, 0.0]]),
            v_bn_n=np.array([[0.0, 7_500.0, 0.0], [0.0, 7_500.0, 0.0]]),
            radius_m=np.array([7_000_000.0, 7_000_000.0]),
            altitude_m=np.array([621_863.4, 621_863.4]),
            speed_m_s=np.array([7_500.0, 7_500.0]),
        )
        rigid_body_truth = RigidBodyTruthHistory(
            time_ns=np.array([0.0, 1.0]),
            time_min=np.array([0.0, 1.0]),
            sigma_bn=np.zeros((2, 3)),
            omega_bn_b=np.zeros((2, 3)),
            dcm_bn=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        )

        torque = compute_gravity_gradient_torque_history(
            orbit_truth,
            rigid_body_truth,
            inertia_matrix_kg_m2=np.eye(3),
            mu_m3_s2=3.986004415e14,
        )

        self.assertTrue(np.allclose(torque, 0.0))

    def test_disturbance_model_can_reduce_to_constant_bias_only(self) -> None:
        config = {
            "spacecraft": {
                "inertia_kg_m2": [
                    [900.0, 0.0, 0.0],
                    [0.0, 800.0, 0.0],
                    [0.0, 0.0, 600.0],
                ]
            },
            "truth_model": {
                "central_body": {
                    "mu_m3_s2": 3.986004415e14,
                    "radius_m": 6_378_136.6,
                }
            },
            "disturbances": {
                "model_enabled": True,
                "gravity_gradient": {"enabled": False},
                "aerodynamic_drag": {"enabled": False},
                "solar_radiation_pressure": {"enabled": False},
                "magnetic_residual_dipole": {"enabled": False},
                "constant_bias": {
                    "enabled": True,
                    "torque_body_nm": [1.0e-6, -2.0e-6, 3.0e-6],
                },
            },
        }
        rigid_body_truth = RigidBodyTruthHistory(
            time_ns=np.array([0.0, 1.0, 2.0]),
            time_min=np.array([0.0, 1.0, 2.0]),
            sigma_bn=np.zeros((3, 3)),
            omega_bn_b=np.zeros((3, 3)),
            dcm_bn=np.repeat(np.eye(3)[None, :, :], 3, axis=0),
        )
        orbit_truth = OrbitTruthHistory(
            time_ns=np.array([0.0, 1.0, 2.0]),
            time_min=np.array([0.0, 1.0, 2.0]),
            r_bn_n=np.array([[7_000_000.0, 0.0, 0.0]] * 3),
            v_bn_n=np.array([[0.0, 7_500.0, 0.0]] * 3),
            radius_m=np.array([7_000_000.0] * 3),
            altitude_m=np.array([621_863.4] * 3),
            speed_m_s=np.array([7_500.0] * 3),
        )
        environment_truth = EnvironmentTruthHistory(
            time_ns=np.array([0.0, 1.0, 2.0]),
            time_min=np.array([0.0, 1.0, 2.0]),
            sun_position_n_m=np.zeros((3, 3)),
            sun_direction_n=np.zeros((3, 3)),
            sun_direction_b=np.zeros((3, 3)),
            magnetic_field_n_t=np.zeros((3, 3)),
            magnetic_field_b_t=np.zeros((3, 3)),
            illumination_factor=np.ones(3),
            is_eclipsed=np.zeros(3, dtype=bool),
        )

        disturbance_truth = compute_disturbance_torque_history(
            config,
            rigid_body_truth,
            orbit_truth,
            environment_truth,
        )

        expected = np.array([[1.0e-6, -2.0e-6, 3.0e-6]] * 3)
        self.assertTrue(np.allclose(disturbance_truth.total_nm, expected))
