#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# 定义一个函数，用于在指定目录下查找并验证可用的 Basilisk 虚拟环境激活脚本
find_venv_activate() {
  local candidate_root="$1" # 函数的第一个参数：待搜索的候选根目录

  local activate_path=""
  local python_path=""

  if [[ -f "${candidate_root}/.venv-linux/bin/activate" ]]; then
    activate_path="${candidate_root}/.venv-linux/bin/activate"
    python_path="${candidate_root}/.venv-linux/bin/python3"
  elif [[ -f "${candidate_root}/.venv/bin/activate" ]]; then
    activate_path="${candidate_root}/.venv/bin/activate"
    python_path="${candidate_root}/.venv/bin/python3"
  else
    return 1
  fi

  # 只接受真正安装了 Basilisk 的虚拟环境；
  # 这样即使同级目录里存在别的 .venv，也不会误选进去。
  if "${python_path}" -c 'import Basilisk' >/dev/null 2>&1; then
    printf '%s\n' "${activate_path}"
    return 0
  fi

  return 1
}

VENV_PATH=""
for candidate_root in \
  "${BASILISK_DEV_ROOT:-}" \
  "${REPO_ROOT}/../basilisk-develop" \
  "$HOME/avslab/basilisk-develop"
do
  [[ -z "${candidate_root}" ]] && continue
  if VENV_PATH=$(find_venv_activate "${candidate_root}"); then
    BASILISK_DEV_ROOT="${candidate_root}"
    break
  fi
done

if [[ -z "${VENV_PATH}" ]]; then
  echo "Could not find a Basilisk virtual environment under:" >&2
  echo "  - ${BASILISK_DEV_ROOT:-<unset>}" >&2
  echo "  - ${REPO_ROOT}/../basilisk-develop" >&2
  echo "  - $HOME/avslab/basilisk-develop" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

CONFIG_ARGS=(--config "${REPO_ROOT}/configs/baseline.yaml")
for arg in "$@"; do
  if [[ "${arg}" == "--config" ]]; then
    CONFIG_ARGS=()
    break
  fi
done

python3 "${REPO_ROOT}/scenarios/scenario_hill_point_baseline.py" \
  "${CONFIG_ARGS[@]}" \
  "$@"
# 使用虚拟环境中的 Python3 解释器执行主仿真脚本
# 第一个参数：仿真脚本的绝对路径
# 第二个参数：展开 CONFIG_ARGS 数组（如果未清空则包含默认 --config 配置）
# 第三个参数：原样传递所有命令行参数（"$@" 会保持每个参数的独立性）