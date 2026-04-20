#!/usr/bin/env python3
"""Run the project-local pure Python week-2 truth model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.truth.truth_model import TruthModel, save_truth_model_arrays, save_truth_model_summary
from src.utils.plotting import save_truth_model_plots


DEFAULT_SPACECRAFT_CONFIG = REPO_ROOT / "src" / "config" / "spacecraft.yaml"
DEFAULT_ORBIT_CONFIG = REPO_ROOT / "src" / "config" / "orbit.yaml"
DEFAULT_ENVIRONMENT_CONFIG = REPO_ROOT / "src" / "config" / "environment.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "week2_truth_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project-local pure Python week-2 truth model.")
    parser.add_argument("--spacecraft", type=Path, default=DEFAULT_SPACECRAFT_CONFIG, help="Path to spacecraft.yaml")
    parser.add_argument("--orbit", type=Path, default=DEFAULT_ORBIT_CONFIG, help="Path to orbit.yaml")
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG, help="Path to environment.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for plots and saved arrays")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively after the run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    truth_model = TruthModel.from_config_files(args.spacecraft, args.orbit, args.environment)
    result = truth_model.simulate()

    output_dir = args.output_dir.resolve()
    array_path = save_truth_model_arrays(result, output_dir)
    summary_path = save_truth_model_summary(result, output_dir)
    figure_paths = save_truth_model_plots(result, output_dir, show_plots=args.show_plots)

    print("Saved truth-model outputs:")
    print(f"  - {array_path}")
    print(f"  - {summary_path}")
    for figure_path in figure_paths:
        print(f"  - {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
