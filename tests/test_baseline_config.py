from __future__ import annotations

from pathlib import Path
import unittest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline.yaml"
LEO_TRUTH_CONFIG_PATH = REPO_ROOT / "configs" / "leo_truth.yaml"
BASELINE_VIZARD_LIVE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_vizard_live.yaml"
BASELINE_VIZARD_LIVE_REALTIME_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_vizard_live_realtime.yaml"
BASELINE_VIZARD_SAVE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_vizard_save.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class BaselineConfigTest(unittest.TestCase):
    def test_baseline_config_has_expected_project_defaults(self) -> None:
        config = load_config(BASELINE_CONFIG_PATH)

        self.assertEqual(config["scenario"]["mode_request"], "hillPoint")
        self.assertEqual(config["sensors"]["navigation"]["provider"], "simple_nav")
        self.assertEqual(config["actuators"]["reaction_wheels"]["count"], 4)
        self.assertGreater(config["simulation"]["duration_minutes"], 0.0)
        self.assertFalse(config["visualization"]["enable_unity_viz"])
        self.assertFalse(config["visualization"]["live_stream"])
        self.assertFalse(config["visualization"]["save_file"])

    def test_leo_truth_config_exposes_week2_truth_layers(self) -> None:
        config = load_config(LEO_TRUTH_CONFIG_PATH)

        self.assertEqual(config["scenario"]["mode_request"], "hillPoint")
        self.assertEqual(config["truth_model"]["central_body"]["name"], "earth")
        self.assertEqual(config["environment"]["magnetic_field"]["model"], "WMM")
        self.assertTrue(config["environment"]["sun"]["enabled"])
        self.assertTrue(config["environment"]["eclipse"]["enabled"])
        self.assertTrue(config["disturbances"]["gravity_gradient"]["enabled"])
        self.assertTrue(config["disturbances"]["aerodynamic_drag"]["enabled"])
        self.assertTrue(config["disturbances"]["solar_radiation_pressure"]["enabled"])
        self.assertTrue(config["disturbances"]["magnetic_residual_dipole"]["enabled"])
        self.assertGreater(config["orbit"]["a_m"], 6.6e6)
        self.assertLess(config["orbit"]["a_m"], 7.2e6)

    def test_live_vizard_config_enables_direct_comm(self) -> None:
        config = load_config(BASELINE_VIZARD_LIVE_CONFIG_PATH)

        self.assertTrue(config["visualization"]["enable_unity_viz"])
        self.assertTrue(config["visualization"]["live_stream"])
        self.assertFalse(config["visualization"]["save_file"])
        self.assertEqual(config["visualization"]["clock_sync_accel_factor"], 10.0)
        self.assertEqual(config["visualization"]["main_camera_target"], "earth")
        self.assertEqual(config["visualization"]["true_trajectory_lines_on"], 3)

    def test_live_realtime_vizard_config_enables_1x_clock_sync(self) -> None:
        config = load_config(BASELINE_VIZARD_LIVE_REALTIME_CONFIG_PATH)

        self.assertTrue(config["visualization"]["enable_unity_viz"])
        self.assertTrue(config["visualization"]["live_stream"])
        self.assertEqual(config["visualization"]["clock_sync_accel_factor"], 1.0)

    def test_saved_vizard_config_enables_binary_capture(self) -> None:
        config = load_config(BASELINE_VIZARD_SAVE_CONFIG_PATH)

        self.assertTrue(config["visualization"]["enable_unity_viz"])
        self.assertFalse(config["visualization"]["live_stream"])
        self.assertTrue(config["visualization"]["save_file"])
        self.assertEqual(config["visualization"]["main_camera_target"], "earth")
        self.assertEqual(config["visualization"]["orbit_lines_on"], 1)
