from __future__ import annotations

import unittest

import numpy as np

from src.truth.attitude import AttitudeConfig, AttitudeState
from src.truth.disturbances import DisturbanceConfig, compute_gravity_gradient_torque_b_nm, evaluate_disturbances
from src.truth.environment import EnvironmentConfig, EnvironmentSample


class DisturbanceModelTest(unittest.TestCase):
    def test_gravity_gradient_torque_is_zero_for_spherical_inertia(self) -> None:
        torque_b_nm = compute_gravity_gradient_torque_b_nm(
            position_n_m=np.array([7000e3, 0.0, 0.0]),
            sigma_bn=np.zeros(3),
            inertia_kg_m2=np.eye(3),
            mu_m3_s2=3.986004415e14,
        )

        self.assertTrue(np.allclose(torque_b_nm, 0.0))

    def test_constant_bias_is_present_in_total_torque(self) -> None:
        attitude_config = AttitudeConfig(
            inertia_kg_m2=np.diag([0.45, 0.35, 0.25]),
            initial_sigma_bn=np.zeros(3),
            initial_omega_bn_b_rad_s=np.zeros(3),
        )
        attitude_state = AttitudeState(
            sigma_bn=np.zeros(3),
            omega_bn_b_rad_s=np.zeros(3),
        )
        environment_config = EnvironmentConfig(
            sun_direction_n=np.array([1.0, 0.0, 0.0]),
            central_body_radius_m=6378136.6,
            magnetic_dipole_n_a_m2=np.array([0.0, 0.0, 7.94e22]),
            atmosphere_reference_density_kg_m3=3.5e-12,
            atmosphere_reference_altitude_m=400000.0,
            atmosphere_scale_height_m=60000.0,
            solar_pressure_n_m2=4.56e-6,
            enable_eclipse=True,
        )
        environment_sample = EnvironmentSample(
            time_s=0.0,
            sun_direction_n=np.array([1.0, 0.0, 0.0]),
            sun_direction_b=np.array([1.0, 0.0, 0.0]),
            magnetic_field_n_t=np.zeros(3),
            magnetic_field_b_t=np.zeros(3),
            illumination=1.0,
        )
        disturbance_config = DisturbanceConfig(
            drag_coefficient=2.2,
            drag_area_m2=0.0,
            srp_area_m2=0.0,
            center_of_pressure_b_m=np.zeros(3),
            reflectivity_coefficient=1.3,
            residual_dipole_b_a_m2=np.zeros(3),
            constant_bias_torque_b_nm=np.array([1.0e-6, -2.0e-6, 3.0e-6]),
            enable_gravity_gradient=False,
            enable_drag=False,
            enable_srp=False,
            enable_magnetic=False,
        )

        sample = evaluate_disturbances(
            time_s=0.0,
            position_n_m=np.array([7000e3, 0.0, 0.0]),
            velocity_n_m_s=np.array([0.0, 7500.0, 0.0]),
            attitude_state=attitude_state,
            attitude_config=attitude_config,
            environment_sample=environment_sample,
            environment_config=environment_config,
            disturbance_config=disturbance_config,
            mu_m3_s2=3.986004415e14,
            central_body_radius_m=6378136.6,
        )

        self.assertTrue(np.allclose(sample.total_torque_b_nm, disturbance_config.constant_bias_torque_b_nm))
