import pytest

torch = pytest.importorskip("torch")

from trainers.partial_kl import partial_kl_mask, tag_start_index  # noqa: E402

# Toy tokenizer: each token id maps to a string piece; "<mat" + "lab>" splits the tag
# across tokens, and "\n<" merges a newline with the tag start.
VOCAB = {0: "", 1: "<think>", 2: "reason", 3: "</think>", 4: "\n<", 5: "mat", 6: "lab>", 7: "code", 8: "</matlab>"}


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


def test_tag_split_across_tokens_is_found():
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    # "<matlab>" starts inside the "\n<" token at position 3
    assert tag_start_index(ids, decode, "matlab") == 3


def test_missing_tag_penalises_whole_response():
    assert tag_start_index([1, 2, 3], decode, "matlab") == 0


def test_mask_keeps_code_part_and_padding_and_does_not_modify_input():
    responses = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 0, 0, 0, 0, 0]])
    response_mask = torch.tensor([[1] * 8, [1, 1, 1, 0, 0, 0, 0, 0]])
    original = response_mask.clone()
    mask = partial_kl_mask(responses, response_mask, decode, "matlab")
    assert mask.tolist() == [[0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0]]
    assert torch.equal(response_mask, original)
