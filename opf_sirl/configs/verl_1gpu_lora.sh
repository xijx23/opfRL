#!/usr/bin/env bash
# Single-GPU (RTX 4090 48GB) variant of verl_server_7b.sh: SIRL with LoRA.
# Full fine-tuning is mathematically impossible on one 48GB card
# (bf16 params 15G + grads 15G + Adam states >=30G = >=60G > 48G).
#
# DECLARED METHOD DEVIATIONS vs the original server recipe (also note in reports):
#   1. LoRA (rank 32, alpha 64, all-linear) instead of full fine-tuning.
#      Base model weights / data / algorithm code are untouched; results are NOT
#      directly comparable with the README "server experiment" (full FT).
#   2. actor lr 1e-6 -> 1e-5: LoRA adapters need ~1e-5; at 1e-6 the per-step Adam
#      update is below bf16 rounding (~4e-3 relative) of the base computation and
#      adapters learn nothing. Adapters stay in their default fp32 (NO dtype cast
#      patch -- FSDP1 has no uniform-dtype assert; if one appears, stop and
#      investigate get_fsdp_wrap_policy instead of patching around it).
#
# Memory plan (worst-case accounting, bf16->fp32 upcast x2-3 incl. backward):
#   rollout A: vLLM 0.6*48=29GB (KV~14GB) + base offloaded        ~31GB peak
#   rollout B: vLLM 0.5*48=24GB (KV~9.5GB) + resident base 15GB   ~41GB peak
#   update   : base 15GB + logits transient 10-15GB (16k-token cap) ~30-33GB peak
#   Gate: measured nvidia-smi peak must stay <=44GB. If update exceeds it,
#   drop ppo/log_prob max_token_len 16384 -> 8192.
#
# Prerequisite: apply repo patch first (skip building the standalone ref model
# for LoRA; the runtime ref path is verl's native actor-with-adapters-disabled):
#   bash scripts/apply_patches.sh
# The MATLAB verifier must run on the Windows host BEFORE training:
#   cd E:\UserData\XiJiaxin\run_opfRL_v1\OPF\opf_sirl
#   .venv-win\Scripts\python.exe scripts\start_verifier.py --host 127.0.0.1 --port 8765 --engines 8
#
# Usage:
#   RUN_NAME=sirl_7b_stage1 OPF_VERIFIER_URL=http://127.0.0.1:8765 \
#     bash configs/verl_1gpu_lora.sh [extra hydra overrides]
# A/B experiment switch (see plan): default A = offload+util0.6; B: OPF_AB=B
#   OPF_AB=B   ->  param_offload=False + gpu_memory_utilization=0.5
# Env: RUN_NAME (reuse to resume), MODEL_PATH, DATA_DIR, TOTAL_EPOCHS,
#      OPF_REWARD_STAGE, OPF_PARTIAL_KL, OPF_WANDB.
# Expected: ~6.5 min/step (B, measured), 150 steps ~ 16-20 h.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-50}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPF_VERIFIER_URL="${OPF_VERIFIER_URL:?set the MATLAB verifier URL}"
export OPF_REWARD_STAGE="${OPF_REWARD_STAGE:-1}"
export OPF_PARTIAL_KL="${OPF_PARTIAL_KL:-1}"
# Ray counts reclaimable page cache as node memory and kills workers at 95%;
# real anonymous usage (~60GB of 93GB) leaves the kernel as a sufficient backstop.
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export OPF_RAY_OBJECT_STORE_GB="${OPF_RAY_OBJECT_STORE_GB:-2}"
RUN_NAME="${RUN_NAME:-sirl_7b_stage${OPF_REWARD_STAGE}_$(date +%Y%m%d_%H%M%S)}"
LOGGER="[console,tensorboard]"
[[ "${OPF_WANDB:-0}" == "1" ]] && LOGGER="[console,tensorboard,wandb]"

# A/B variants: A (default) frees VRAM during rollout via param_offload and gives
# vLLM a bigger budget; B keeps the frozen base resident (no PCIe streaming).
if [[ "${OPF_AB:-A}" == "B" ]]; then
  PARAM_OFFLOAD="False"; VLLM_UTIL="0.5"
else
  PARAM_OFFLOAD="True";  VLLM_UTIL="0.6"
fi

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
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=64
  actor_rollout_ref.actor.optim.lr=1e-5
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.clip_ratio_low=0.20
  actor_rollout_ref.actor.clip_ratio_high=0.30
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  # token-budget micro-batching: caps worst-case logits (16384 x 152k vocab:
  # bf16 4.8G, fp32-upcast + backward ~2-3x) regardless of response length mix.
  # rollout.log_prob_* and ref.log_prob_* inherit these via oc.select defaults.
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
  actor_rollout_ref.actor.entropy_checkpointing=True
  # A/B switch; optimizer states are LoRA-sized (tiny), never offloaded.
  actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.max_num_batched_tokens=12232
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_UTIL"
  actor_rollout_ref.rollout.n=16
  trainer.project_name=OPF-SIRL
  trainer.logger="$LOGGER"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq=2
  trainer.max_actor_ckpt_to_keep=3
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.total_epochs="$TOTAL_EPOCHS"
)

source "$PROJECT_DIR/configs/launch_verl.sh" "$@"
