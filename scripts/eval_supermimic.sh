#!/usr/bin/env bash
# Run standardized 8-suite deterministic PhysX evaluation comparing Reference-Free SuperMimic
# against the bundled official 240k-iter PPO ceiling (`example/model_000.pt`) and/or custom runs.
# Override checkpoints by passing additional `--ckpt NAME=PATH` arguments.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SM_CKPT_PATH="${SM_CKPT_PATH:-example/supermimic_g1_flat/model_3000.pt}"
OUT_JSON="${OUT_JSON:-logs/g1_flat/supermimic_eval_benchmark.json}"

python legged_lab/scripts/eval_compare.py \
    --task=g1_flat \
    --num_envs=512 \
    --steps=1000 \
    --seed=1234 \
    --headless \
    --ckpt "SuperMimic_3k=${SM_CKPT_PATH}" \
    --ckpt "CMU_PPO_240k=example/model_000.pt" \
    --out_json="${OUT_JSON}" \
    "$@"
