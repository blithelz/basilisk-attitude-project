#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

BASILISK_DEV_ROOT="${BASILISK_DEV_ROOT:-$HOME/avslab/basilisk-develop}"
VENV_PATH="${BASILISK_DEV_ROOT}/.venv-linux/bin/activate"

if [[ ! -f "${VENV_PATH}" ]]; then
  echo "Could not find Basilisk virtual environment at ${VENV_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

python3 "${REPO_ROOT}/scenarios/scenario_hill_point_baseline.py" \
  --config "${REPO_ROOT}/configs/baseline.yaml" \
  "$@"
