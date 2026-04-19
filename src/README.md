# Source

`src/` now contains two parallel layers:

1. the existing Basilisk-based engineering scaffold
2. the new pure Python week-2 truth model

Subdirectories:

- `truth/`: pure Python orbit, attitude, environment, disturbance, and truth-model orchestration
- `utils/`: frame transforms, numerical helpers, and plotting helpers for the pure Python truth model
- `config/`: YAML config files for `scripts/run_truth_model.py`
- `simulation/`: the older Basilisk-based simulation assembly layer
- `modes/`: project mode helpers used by the Basilisk baseline
- `sensors/`: project sensor helpers used by the Basilisk baseline
- `actuators/`: project actuator helpers used by the Basilisk baseline
