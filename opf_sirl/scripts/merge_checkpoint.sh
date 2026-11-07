#!/usr/bin/env bash
# Merge a verl FSDP actor checkpoint of a run into a HuggingFace model directory.
#   bash scripts/merge_checkpoint.sh outputs/runs/<run_name>          # latest step -> <run>/hf_model
#   bash scripts/merge_checkpoint.sh outputs/runs/<run_name> 50       # step 50     -> <run>/hf_model_step50
set -euo pipefail
RUN_DIR="${1:?usage: merge_checkpoint.sh <run_dir> [step]}"
CKPT_DIR="$RUN_DIR/checkpoints"

if [[ -n "${2:-}" ]]; then
  STEP="$2"; TARGET="$RUN_DIR/hf_model_step$STEP"
else
  STEP="$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt")"; TARGET="$RUN_DIR/hf_model"
fi

python3 -m verl.model_merger merge --backend fsdp \
  --local_dir "$CKPT_DIR/global_step_$STEP/actor" --target_dir "$TARGET"
echo "$STEP" > "$TARGET/source_step.txt"
echo "[merge_checkpoint] step $STEP -> $TARGET"
