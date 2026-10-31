#!/usr/bin/env bash
# Play a student policy (JIT) in simulation. Defaults to the bundled example
# run, so this runs without training first. --load_run makes the play run apply the
# gains the student was distilled under. Override the run via env var:
#   HTD_RUN=<run_folder> bash scripts/play_student.sh
# Activate the env first:  conda activate isaac_wbc
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

HTD_RUN="${HTD_RUN:-example}"

source "$REPO_DIR/scripts/_stage_example.sh"
stage_example "$HTD_RUN"

STUDENT_CKPT="logs/g1_flat/$HTD_RUN/student_checkpoints/student_policy_jit.pt"
if [ ! -f "$STUDENT_CKPT" ]; then
    echo "[ERROR] Student checkpoint not found: $STUDENT_CKPT" >&2
    echo "        Use HTD_RUN=<run_folder> to select a different run." >&2
    exit 1
fi

echo "[INFO] Playing student checkpoint: $STUDENT_CKPT"
python legged_lab/scripts/play_student.py \
    --task=g1_flat \
    --load_run="$HTD_RUN" \
    --student_checkpoint="$STUDENT_CKPT" \
    --use_jit \
    --num_envs=1 \
    --max_steps=20000 \
    "$@"
