"""Partial-KL mask shared by the verl and TRL back-ends.

SIRL (SIRLTrain/partial_kl/kl_ppo/ray_trainer.py) applies the KL penalty only from the
code tag to the end of the response and leaves the reasoning part unconstrained:

    token_level_rewards = scores - beta * KL * mask,   mask[t] = response_mask[t] * (t >= tag_start)

If the tag is not found the whole response is penalised (tag_start = 0), as in SIRL.

SIRL locates the tag by matching a few hard-coded token patterns ("<python", "\n<python", ...).
Here the tag is located on the decoded text and mapped back to a token index, which
also works when the tokenizer merges the tag with neighbouring characters.
"""
from typing import Callable, Sequence

import torch

Decode = Callable[[Sequence[int]], str]


def tag_start_index(token_ids: Sequence[int], decode: Decode, tag: str) -> int:
    """Index of the first token whose decoded prefix reaches the start of `<tag>` (0 if absent)."""
    text = decode(token_ids)
    pos = text.find(f"<{tag}>")
    if pos < 0:
        return 0
    lo, hi = 0, len(token_ids) - 1  # smallest t with len(decode(ids[:t+1])) > pos
    while lo < hi:
        mid = (lo + hi) // 2
        if len(decode(token_ids[: mid + 1])) > pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


def partial_kl_mask(responses: torch.Tensor, response_mask: torch.Tensor, decode: Decode,
                    tag: str = "matlab") -> torch.Tensor:
    """Return a new mask (response_mask is not modified) that keeps tokens from `<tag>` on."""
    mask = response_mask.clone()
    for b in range(responses.shape[0]):
        valid = int(response_mask[b].sum().item())
        ids = responses[b, :valid].tolist()
        start = tag_start_index(ids, decode, tag) if ids else 0
        mask[b, :start] = 0
    return mask
