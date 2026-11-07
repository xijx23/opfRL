#!/usr/bin/env bash
# Local smoke test of the SIRL pipeline (verl + REINFORCE++ + Partial-KL) on one 8 GB GPU (WSL2).
# Purpose: check that the method and code run end to end, not to train a useful model.
# Algorithm settings follow SIRL's run_partialKL.sh; sizes are tiny. To fit in 8 GB the actor is
# kept in bf16 (fp32 params + grads + Adam state of a 0.5B model alone need ~8 GB during the update),
# and LoRA is not used because vLLM 0.9.2's LoRA kernels do not compile on sm_120.
# Real training (fp32 master weights, SIRL sizes): configs/verl_server_7b.sh.
# Usage: bash configs/verl_local_smoke.sh [extra hydra overrides, e.g. trainer.total_training_steps=5]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
export OPF_VERIFIER_URL="${OPF_VERIFIER_URL:-http://127.0.0.1:8765}"
export OPF_REWARD_STAGE="${OPF_REWARD_STAGE:-1}"
export OPF_PARTIAL_KL="${OPF_PARTIAL_KL:-1}"
# RTX 50xx (sm_120): NCCL/flash-attn kernels are JIT-compiled from PTX; the default 256 MB
# cache is too small, which makes every process re-compile (minutes of start-up per worker).
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
# WSL2 gets half of the host RAM by default; keep Ray's object store small.
export OPF_RAY_OBJECT_STORE_GB="${OPF_RAY_OBJECT_STORE_GB:-1}"
RUN_NAME="${RUN_NAME:-smoke_stage${OPF_REWARD_STAGE}_$(date +%Y%m%d_%H%M%S)}"

OVERRIDES=(
  algorithm.adv_estimator=reinforce_plus_plus
  algorithm.use_kl_in_reward=True
  algorithm.kl_ctrl.kl_coef=0.0005
  data.train_batch_size=4
  data.max_prompt_length=1536
  data.max_response_length=512
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
  actor_rollout_ref.actor.ppo_mini_batch_size=4
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
  actor_rollout_ref.actor.entropy_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.n=2
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  trainer.project_name=OPF-SIRL
  trainer.logger="[console,tensorboard]"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.total_training_steps=3
)

source "$PROJECT_DIR/configs/launch_verl.sh" "$@"
