from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from src.actuators.reaction_wheels import get_reaction_wheel_count
from src.modes.hill_point import get_mode_request


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline.yaml"


def load_baseline_config() -> dict:
    with BASELINE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class BaselineConfigTest(unittest.TestCase):
    def test_baseline_config_has_expected_project_defaults(self) -> None:
        config = load_baseline_config()

        self.assertEqual(get_mode_request(config), "hillPoint")
        self.assertEqual(config["sensors"]["navigation"]["provider"], "simple_nav")
        self.assertEqual(get_reaction_wheel_count(config), 4)
        self.assertGreater(config["simulation"]["duration_minutes"], 0.0)
        self.assertFalse(config["visualization"]["enable_unity_viz"])
