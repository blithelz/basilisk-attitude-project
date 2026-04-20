from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def find_venv_activate() -> Path | None:
    candidates = []

    env_root = os.environ.get("BASILISK_DEV_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(REPO_ROOT.parent / "basilisk-develop")
    candidates.append(Path.home() / "avslab" / "basilisk-develop")

    for root in candidates:
        for relative_path in ((".venv-linux", "bin", "activate"), (".venv", "bin", "activate")):
            activate_path = root.joinpath(*relative_path)
            python_path = activate_path.parent / "python3"
            if activate_path.exists() and python_path.exists() and venv_has_basilisk(python_path):
                return activate_path

    return None


def venv_has_basilisk(python_path: Path) -> bool:
    result = subprocess.run(
        [str(python_path), "-c", "import Basilisk"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


VENV_ACTIVATE = find_venv_activate()


class BaselineScenarioSmokeTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        if os.name != "posix":
            self.skipTest("Smoke test requires a WSL/Linux environment.")
        if shutil.which("bash") is None:
            self.skipTest("Smoke test requires bash.")
        if VENV_ACTIVATE is None:
            self.skipTest(f"Basilisk virtual environment not found at {VENV_ACTIVATE}.")

        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")

        return subprocess.run(
            ["bash", "scripts/run_baseline.sh", *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    def test_run_baseline_script_without_saving(self) -> None:
        result = self.run_script("--no-save")

        debug_output = (
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"returncode: {result.returncode}"
        )
        self.assertEqual(result.returncode, 0, debug_output)
        self.assertIn("Scenario completed without saving plots.", result.stdout, debug_output)

    def test_run_leo_truth_config_without_saving(self) -> None:
        result = self.run_script("--config", "configs/leo_truth.yaml", "--no-save")

        debug_output = (
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"returncode: {result.returncode}"
        )
        self.assertEqual(result.returncode, 0, debug_output)
        self.assertIn("Scenario completed without saving plots.", result.stdout, debug_output)
