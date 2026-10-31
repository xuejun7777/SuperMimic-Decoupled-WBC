#!/usr/bin/env bash
# Play a teacher policy in simulation. Defaults to the bundled example run +
# checkpoint, so this runs without training first. The play run auto-loads the gains
# the checkpoint was trained with. Override the run/checkpoint via env vars:
#   HTD_RUN=<run_folder> HTD_CKPT=<model_xxx.pt> bash scripts/play_teacher.sh
# Activate the env first:  conda activate isaac_wbc
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

HTD_RUN="${HTD_RUN:-example}"
HTD_CKPT="${HTD_CKPT:-model_000.pt}"

source "$REPO_DIR/scripts/_stage_example.sh"
stage_example "$HTD_RUN"

if [ ! -f "logs/g1_flat/$HTD_RUN/$HTD_CKPT" ]; then
    echo "[ERROR] Teacher checkpoint not found: logs/g1_flat/$HTD_RUN/$HTD_CKPT" >&2
    echo "        Use HTD_RUN=<run_folder> HTD_CKPT=<model_xxx.pt> to select a different checkpoint." >&2
    exit 1
fi

echo "[INFO] Playing teacher checkpoint: logs/g1_flat/$HTD_RUN/$HTD_CKPT"
python legged_lab/scripts/play_teacher.py \
    --task=g1_flat \
    --num_envs=1 \
    --load_run="$HTD_RUN" \
    --checkpoint="$HTD_CKPT" \
    "$@"
