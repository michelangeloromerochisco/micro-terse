"""Tests for next-token prediction (terse.server.predict)."""
import torch

from terse.server.predict import SHOWCASE_PROMPTS, next_token_topk, showcase


class _FakeTok:
    def encode(self, text, add_special_tokens=True):
        return [min(ord(c), 255) for c in text] or [1]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _FavorModel:
    """Drives one token id far above the rest."""

    def __init__(self, favored: int) -> None:
        self.favored = favored

    def eval(self) -> "_FavorModel":
        return self

    def __call__(self, input_ids, return_logits=True):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, 256), -10.0)
        logits[:, :, self.favored] = 10.0
        return {"logits": logits}


def test_next_token_topk_ranks_favored_first():
    preds = next_token_topk(_FavorModel(ord("Z")), _FakeTok(), "cpu", "hello", k=3)
    assert len(preds) == 3
    assert preds[0]["token"] == "Z"
    assert preds[0]["prob"] > 0.9
    assert all(0.0 <= p["prob"] <= 1.0 for p in preds)


def test_next_token_topk_probs_descend():
    preds = next_token_topk(_FavorModel(ord("Q")), _FakeTok(), "cpu", "abc", k=4)
    probs = [p["prob"] for p in preds]
    assert probs == sorted(probs, reverse=True)


def test_showcase_covers_all_prompts():
    items = showcase(_FavorModel(ord("Z")), _FakeTok(), "cpu")
    assert len(items) == len(SHOWCASE_PROMPTS)
    for item in items:
        assert item["text"]
        assert item["predictions"]
