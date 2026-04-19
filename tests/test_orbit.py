from __future__ import annotations

import unittest

import numpy as np

from src.truth.orbit import OrbitConfig, OrbitalElements, orbital_elements_to_state, propagate_orbit


class OrbitModelTest(unittest.TestCase):
    def test_orbital_elements_to_state_preserves_circular_orbit_radius(self) -> None:
        elements = OrbitalElements(
            semi_major_axis_m=7000e3,
            eccentricity=0.0,
            inclination_rad=np.deg2rad(30.0),
            raan_rad=np.deg2rad(40.0),
            arg_perigee_rad=np.deg2rad(15.0),
            true_anomaly_rad=np.deg2rad(60.0),
        )
        state = orbital_elements_to_state(elements, 3.986004415e14)

        self.assertAlmostEqual(np.linalg.norm(state.position_n_m), 7000e3, delta=1.0)

    def test_two_body_propagation_keeps_specific_energy_nearly_constant(self) -> None:
        config = OrbitConfig(
            mu_m3_s2=3.986004415e14,
            central_body_radius_m=6378136.6,
            initial_elements=OrbitalElements(
                semi_major_axis_m=7000e3,
                eccentricity=0.001,
                inclination_rad=np.deg2rad(51.6),
                raan_rad=np.deg2rad(20.0),
                arg_perigee_rad=np.deg2rad(10.0),
                true_anomaly_rad=np.deg2rad(0.0),
            ),
            duration_s=600.0,
            step_size_s=2.0,
            use_j2=False,
        )

        history = propagate_orbit(config)
        energy_span = np.max(history.specific_energy_j_kg) - np.min(history.specific_energy_j_kg)

        self.assertLess(energy_span, 5.0e2)
