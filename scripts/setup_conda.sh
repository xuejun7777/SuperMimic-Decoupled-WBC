#!/usr/bin/env bash
# Conda setup for IsaacLab-Decoupled-WBC. Creates the `isaac_wbc` env, installs
# torch + Isaac Lab (Isaac Sim), and installs this project in editable mode.
#
# Usage:
#   bash scripts/setup_conda.sh [5.0|4.5]
#     5.0 (default) -> Python 3.11 + Isaac Lab 2.2.0 (Isaac Sim 5.0)
#     4.5           -> Python 3.10 + Isaac Lab 2.1.0 (Isaac Sim 4.5)
set -euo pipefail

ISAACSIM_VERSION="${1:-5.0}"
case "$ISAACSIM_VERSION" in
  5.0) PY_VERSION=3.11; ISAACLAB_VERSION=2.2.0 ;;
  4.5) PY_VERSION=3.10; ISAACLAB_VERSION=2.1.0 ;;
  *) echo "Usage: bash scripts/setup_conda.sh [5.0|4.5]" >&2; exit 1 ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[setup] Isaac Sim ${ISAACSIM_VERSION} -> python=${PY_VERSION}, isaaclab==${ISAACLAB_VERSION}"

# 1. Create and activate the conda env.
if conda env list | awk '{print $1}' | grep -qx "isaac_wbc"; then
  echo "[setup] Conda env 'isaac_wbc' already exists. Remove it first or activate it manually:" >&2
  echo "        conda env remove -n isaac_wbc" >&2
  exit 1
fi
conda create -y -n isaac_wbc python="${PY_VERSION}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate isaac_wbc

# 2. Install Isaac Sim / Isaac Lab.
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "isaaclab[isaacsim,all]==${ISAACLAB_VERSION}" --extra-index-url https://pypi.nvidia.com

# 3. Install this project (editable).
cd "$REPO_DIR"
python -m pip install -e .

echo "[setup] Done. Activate with:  conda activate isaac_wbc"
