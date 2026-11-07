#!/usr/bin/env bash
# Shared launcher for the verl configs. A config sets MODEL_PATH, DATA_DIR, RUN_NAME and the
# OVERRIDES array (hydra overrides), then sources this file with the extra CLI arguments.
#
# Everything of one run goes to outputs/runs/$RUN_NAME/:
#   run_info.json  train.log  tensorboard/  rollouts/  checkpoints/  metrics.csv  summary.md
# Re-running with the same RUN_NAME resumes from the latest checkpoint (verl resume_mode=auto).
set -euo pipefail

RUN_DIR="$PROJECT_DIR/outputs/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"

OVERRIDES+=(
  data.train_files="$DATA_DIR/train.parquet"
  data.val_files="$DATA_DIR/val.parquet"
  custom_reward_function.path="$PROJECT_DIR/trainers/verl_sirl/reward_fn.py"
  actor_rollout_ref.model.path="$MODEL_PATH"
  trainer.experiment_name="$RUN_NAME"
  trainer.default_local_dir="$RUN_DIR/checkpoints"
  trainer.rollout_data_dir="$RUN_DIR/rollouts"
)

# a resumed run keeps the first run_info.json and records the new launch separately
INFO="$RUN_DIR/run_info.json"
[[ -f "$INFO" ]] && INFO="$RUN_DIR/run_info.resume-$(date +%Y%m%d_%H%M%S).json"
python3 scripts/record_run.py --out "$INFO" --model "$MODEL_PATH" \
  --train-file "$DATA_DIR/train.parquet" --val-file "$DATA_DIR/val.parquet" -- "${OVERRIDES[@]}" "$@"

set +e
python3 -m trainers.verl_sirl.main "${OVERRIDES[@]}" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
set -e
python3 scripts/summarize_run.py "$RUN_DIR"
echo "[launch_verl] exit=$status  results: $RUN_DIR"
exit "$status"
