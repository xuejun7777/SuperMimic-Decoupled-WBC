#!/usr/bin/env bash
# Train the teacher policy with the default configuration.
# Activate the env first:  conda activate isaac_wbc
# Extra flags pass through, e.g.:  bash scripts/train_teacher.sh --logger=tensorboard
# For gain/curriculum overrides and resuming, see docs/training.md.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

python legged_lab/scripts/train_teacher.py \
    --task=g1_flat \
    --run_name=htd_wbc \
    --logger=wandb \
    --num_envs=12288 \
    --max_iterations=250000 \
    --save_interval=2500 \
    --headless \
    "$@"
