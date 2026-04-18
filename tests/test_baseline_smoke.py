from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
BASILISK_DEV_ROOT = Path(
    os.environ.get("BASILISK_DEV_ROOT", str(Path.home() / "avslab" / "basilisk-develop"))
)
VENV_ACTIVATE = BASILISK_DEV_ROOT / ".venv-linux" / "bin" / "activate"


class BaselineScenarioSmokeTest(unittest.TestCase):
    def test_run_baseline_script_without_saving(self) -> None:
        if os.name != "posix":
            self.skipTest("Smoke test requires a WSL/Linux environment.")
        if shutil.which("bash") is None:
            self.skipTest("Smoke test requires bash.")
        if not VENV_ACTIVATE.exists():
            self.skipTest(f"Basilisk virtual environment not found at {VENV_ACTIVATE}.")

        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")

        result = subprocess.run(
            ["bash", "scripts/run_baseline.sh", "--no-save"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        debug_output = (
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"returncode: {result.returncode}"
        )
        self.assertEqual(result.returncode, 0, debug_output)
        self.assertIn("Scenario completed without saving plots.", result.stdout, debug_output)
