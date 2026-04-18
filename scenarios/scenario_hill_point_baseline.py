#!/usr/bin/env python3
"""CLI entrypoint for the project-local hill-pointing baseline scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "baseline.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation.hill_point_baseline import run_scenario


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, but got {type(data).__name__}.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project-local Basilisk hill-pointing baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Override the config and display plots interactively.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Override the config and skip saving plot images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    if args.show_plots:
        config["output"]["show_plots"] = True
    if args.no_save:
        config["output"]["save_plots"] = False

    saved_paths = run_scenario(config)

    if saved_paths:
        print("Saved baseline outputs:")
        for path in saved_paths:
            print(f"  - {path}")
    else:
        print("Scenario completed without saving plots.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
