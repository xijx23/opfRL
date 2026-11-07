"""TRL GRPO back-end with SIRL's Partial-KL.

Differences from the verl back-end (SIRL original):
  * GRPO group-normalised advantages instead of REINFORCE++;
  * the KL term lives in the loss (TRL) instead of the reward (verl);
  * as in SIRL, the KL term only covers tokens from the code tag onwards.

Written against trl==0.19.1 (`GRPOTrainer._compute_loss`); the loss below is that method
with the KL term multiplied by the partial-KL mask.
"""
import json

import torch
from trl import GRPOTrainer

from opf_sirl.reward import compute_score
from trainers.partial_kl import partial_kl_mask


def opf_reward(prompts, completions, reward_model, extra_info, **kwargs):
    """TRL reward function: dataset columns arrive as keyword arguments."""
    texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
    ground_truths = [rm["ground_truth"] for rm in reward_model]
    extras = [e if isinstance(e, dict) else json.loads(e) for e in extra_info]
    return [r["score"] for r in compute_score(None, texts, ground_truths, extras)]


class PartialKLGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, kl_tag: str = "matlab", **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_tag = kl_tag

    def _decode(self, ids):
        return self.processing_class.decode(ids, skip_special_tokens=False)

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, completion_ids.size(1))

        advantages = inputs["advantages"].unsqueeze(1)
        old_logps = per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        coef_1 = torch.exp(per_token_logps - old_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            ref = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref - per_token_logps) - (ref - per_token_logps) - 1
            kl_mask = partial_kl_mask(completion_ids, completion_mask, self._decode, self.kl_tag)
            per_token_loss = per_token_loss + self.beta * per_token_kl * kl_mask
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            partial_kl = (per_token_kl * kl_mask).sum() / kl_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())
            self._metrics[mode]["partial_kl"].append(self.accelerator.gather(partial_kl).nanmean().item())

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages < 0)) | ((coef_1 > 1 + self.epsilon_high) & (advantages > 0))
        clip_ratio = (clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio/region_mean"].append(self.accelerator.gather(clip_ratio).nanmean().item())
        return loss
