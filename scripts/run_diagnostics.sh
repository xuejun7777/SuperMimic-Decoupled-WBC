#!/usr/bin/env bash
# Run USD FK + Differentiable Reward parity unit tests (`test_parity.py`) and optionally
# the full 4-experiment mechanistic diagnostic suite (`run_diagnostics.py`) if a full
# world-model checkpoint directory (`SM_DIR` containing `supermimic_last.pt`) is provided.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "[1/2] Verifying USD Differentiable FK & 31-Term Reward Parity vs PhysX..."
python legged_lab/supermimic/test_parity.py --headless

if [ -n "${SM_DIR:-}" ] && [ -f "${SM_DIR}/supermimic_last.pt" ]; then
    echo "[2/2] Running Mechanistic Diagnostics D1-D4 on ${SM_DIR}..."
    python legged_lab/scripts/run_diagnostics.py \
        --task=g1_flat \
        --num_envs=256 \
        --sm_dir="${SM_DIR}" \
        --cmu_ppo_ckpt="example/model_000.pt" \
        --out_json="${OUT_JSON:-logs/g1_flat/diagnostic_results.json}" \
        "$@"
else
    echo "[2/2] Skipping D1-D4 World-Model diagnostics (set SM_DIR=logs/g1_flat/<your_supermimic_run> with supermimic_last.pt to run)."
fi
