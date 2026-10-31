#!/usr/bin/env bash
# Play a Reference-Free SuperMimic teacher policy interactively in simulation.
# Defaults to the bundled `example/supermimic_g1_flat/model_3000.pt` checkpoint so it runs out of the box.
# Override via env vars:
#   SM_RUN=<run_folder> SM_CKPT=<model_xxx.pt> bash scripts/play_supermimic.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SM_RUN="${SM_RUN:-supermimic_example}"
SM_CKPT="${SM_CKPT:-model_3000.pt}"

if [ "$SM_RUN" = "supermimic_example" ] && [ -d "example/supermimic_g1_flat" ]; then
    mkdir -p "logs/g1_flat/supermimic_example"
    cp -rf "example/supermimic_g1_flat/." "logs/g1_flat/supermimic_example/"
fi

if [ ! -f "logs/g1_flat/$SM_RUN/$SM_CKPT" ]; then
    echo "[ERROR] SuperMimic checkpoint not found: logs/g1_flat/$SM_RUN/$SM_CKPT" >&2
    echo "        Use SM_RUN=<run_folder> SM_CKPT=<model_xxx.pt> to select a different checkpoint." >&2
    exit 1
fi

echo "[INFO] Playing Reference-Free SuperMimic policy: logs/g1_flat/$SM_RUN/$SM_CKPT"
python legged_lab/scripts/play_teacher.py \
    --task=g1_flat \
    --num_envs=1 \
    --load_run="$SM_RUN" \
    --checkpoint="$SM_CKPT" \
    "$@"
