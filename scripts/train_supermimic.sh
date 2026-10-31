#!/usr/bin/env bash
# Train Reference-Free SuperMimic on Unitree G1 Decoupled Whole-Body Control (7D locomotion + torso pose).
# Default single-GPU budget: 4096 envs x 24 steps x 3000 iters = 294.9M transitions (~3.3h on 1x RTX 4090).
# Extra flags pass through, e.g.:  bash scripts/train_supermimic.sh --max_iterations=5000 --utd=0.5
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

python legged_lab/scripts/train_supermimic.py \
    --task=g1_flat \
    --run_name=supermimic_g1_wbc \
    --num_envs=4096 \
    --max_iterations=3000 \
    --save_interval=500 \
    --utd=0.4 \
    --horizon=24 \
    --headless \
    --set commands.curriculum.body_height.start_iter=600 \
    --set commands.curriculum.body_height.end_iter=2000 \
    --set commands.curriculum.body_roll.start_iter=600 \
    --set commands.curriculum.body_roll.end_iter=2000 \
    --set commands.curriculum.body_pitch.start_iter=600 \
    --set commands.curriculum.body_pitch.end_iter=2000 \
    --set commands.curriculum.body_yaw.start_iter=600 \
    --set commands.curriculum.body_yaw.end_iter=2000 \
    "$@"
