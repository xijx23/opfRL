"""SIRL Partial-KL for verl 0.5.0.

SIRL ships a modified copy of verl's ray_trainer.py whose only algorithmic change is in
`apply_kl_penalty` (KL-in-reward restricted to the code part). Instead of vendoring the
whole trainer, we replace that one function; everything else is stock verl.

`install()` is registered as Ray's `worker_process_setup_hook` (see main.py), so the patch
is active in the TaskRunner process where `RayPPOTrainer.fit` runs.
"""
import os

import torch

_tokenizer = None


def _decode(ids):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(os.environ["OPF_TOKENIZER_PATH"])
    return _tokenizer.decode(ids, skip_special_tokens=False)


def apply_partial_kl_penalty(data, kl_ctrl, kl_penalty="kl"):
    """verl's apply_kl_penalty with SIRL's partial mask (cf. SIRL kl_ppo/ray_trainer.py).

    Unlike SIRL's `partial_kl_tensor`, the mask is a copy: SIRL zeroes `response_mask`
    in place, which also removes the reasoning tokens from the policy loss.
    """
    from verl.trainer.ppo import core_algos
    from verl.utils.torch_functional import masked_mean

    from trainers.partial_kl import partial_kl_mask

    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    current_kl = torch.mean(masked_mean(kld, mask=response_mask, axis=-1), dim=0).item()

    tag = os.environ.get("OPF_PARTIAL_KL_TAG", "matlab")
    kl_mask = partial_kl_mask(data.batch["responses"], response_mask, _decode, tag).to(kld.dtype)
    partial_kld = kld * kl_mask
    token_level_rewards = token_level_scores - beta * partial_kld
    partial_kl = torch.mean(masked_mean(partial_kld, mask=kl_mask, axis=-1), dim=0).item()

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    metrics = {"actor/reward_kl_penalty": current_kl,
               "actor/reward_partial_kl_penalty": partial_kl,
               "actor/reward_kl_penalty_coeff": beta}
    return data, metrics


def install():
    """Ray worker setup hook: swap in the partial-KL penalty if enabled."""
    if os.environ.get("OPF_PARTIAL_KL", "1") != "1":
        return
    from verl.trainer.ppo import ray_trainer
    ray_trainer.apply_kl_penalty = apply_partial_kl_penalty
