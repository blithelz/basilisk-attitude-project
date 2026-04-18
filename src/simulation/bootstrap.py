"""Bootstrap helpers for loading the official Basilisk BSK_Sim modules."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_bsk_sim_root() -> Path:
    """Find the official Basilisk BSK_Sim example root used as the baseline."""
    candidates = []

    env_path = os.environ.get("BASILISK_BSKSIM_ROOT")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path.home() / "avslab" / "basilisk-develop" / "examples" / "BskSim")

    for candidate in candidates:
        if (candidate / "BSK_masters.py").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the Basilisk BSK_Sim example root. "
        "Set BASILISK_BSKSIM_ROOT or install Basilisk under ~/avslab/basilisk-develop."
    )


def bootstrap_bsk_paths() -> Path:
    """Add the official BSK_Sim example directories to ``sys.path``."""
    bsk_sim_root = resolve_bsk_sim_root()
    extra_paths = [
        str(bsk_sim_root),
        str(bsk_sim_root / "plotting"),
    ]

    for path in reversed(extra_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    return bsk_sim_root


BSK_SIM_ROOT = bootstrap_bsk_paths()

from BSK_masters import BSKScenario, BSKSim  # noqa: E402
import BSK_Dynamics  # noqa: E402
import BSK_Fsw  # noqa: E402
import BSK_Plotting as BSK_plt  # noqa: E402
