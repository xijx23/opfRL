#!/usr/bin/env bash
# SIRL (verl + REINFORCE++ + Partial-KL) on one 8xGPU node, Qwen2.5-7B-Instruct.
# Algorithm hyper-parameters are those of SIRL's SIRLTrain/training_scripts/run_partialKL.sh.
# Changes for the small ProOPF_D training set (48 prompts, see README):
#   train/mini batch 32 -> 16 (verl drops the incomplete last batch), total_epochs 2 -> 50,
#   in-training validation off (evaluation on ProOPF_B is done by scripts/evaluate.py).
# Not forcing VLLM_ATTENTION_BACKEND=XFORMERS: xformers 0.0.30 rejects flash-attn 2.8.
#
# Usage: OPF_VERIFIER_URL=http://<verifier>:8765 bash configs/verl_server_7b.sh [hydra overrides]
# Env:   RUN_NAME (resume by reusing it), MODEL_PATH, DATA_DIR, TOTAL_EPOCHS, OPF_REWARD_STAGE,
#        OPF_PARTIAL_KL, OPF_WANDB=1 to also log to wandb.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-50}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OPF_VERIFIER_URL="${OPF_VERIFIER_URL:?set the MATLAB verifier URL}"
export OPF_REWARD_STAGE="${OPF_REWARD_STAGE:-1}"
export OPF_PARTIAL_KL="${OPF_PARTIAL_KL:-1}"
RUN_NAME="${RUN_NAME:-sirl_7b_stage${OPF_REWARD_STAGE}_$(date +%Y%m%d_%H%M%S)}"
LOGGER="[console,tensorboard]"
[[ "${OPF_WANDB:-0}" == "1" ]] && LOGGER="[console,tensorboard,wandb]"

OVERRIDES=(
  algorithm.adv_estimator=reinforce_plus_plus
  algorithm.use_kl_in_reward=True
  algorithm.kl_ctrl.kl_coef=0.0005
  data.train_batch_size=16
  data.max_prompt_length=2048
  data.max_response_length=8192
  data.filter_overlong_prompts=True
  data.prompt_key=prompt
  data.truncation=left
  reward_model.reward_manager=batch
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.clip_ratio_low=0.20
  actor_rollout_ref.actor.clip_ratio_high=0.30
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.max_num_batched_tokens=12232
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  actor_rollout_ref.rollout.n=16
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  trainer.project_name=OPF-SIRL
  trainer.logger="$LOGGER"
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  trainer.save_freq=25
  trainer.max_actor_ckpt_to_keep=2
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.total_epochs="$TOTAL_EPOCHS"
)

source "$PROJECT_DIR/configs/launch_verl.sh" "$@"
