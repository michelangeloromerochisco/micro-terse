"""Tests for identity-preference scoring (terse.server.identity)."""
import torch

from terse.server.identity import IDENTITY_PROBES, identity_margins, mean_logprob


class _FakeTok:
    """Maps each character to its codepoint — deterministic, no downloads."""

    def encode(self, text, add_special_tokens=True):
        return [min(ord(c), 255) for c in text]


class _FavorModel:
    """Assigns a high logit to one token id at every position."""

    def __init__(self, favored: int) -> None:
        self.favored = favored

    def eval(self) -> "_FavorModel":
        return self

    def __call__(self, input_ids, return_logits=True):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 256)
        logits[:, :, self.favored] = 12.0
        return {"logits": logits}


def test_mean_logprob_prefers_favored_tokens():
    tok = _FakeTok()
    model = _FavorModel(ord("A"))
    high = mean_logprob(model, tok, "cpu", "question", "AAAA")
    low = mean_logprob(model, tok, "cpu", "question", "BBBB")
    assert high > low


def test_mean_logprob_empty_answer_is_zero():
    tok = _FakeTok()
    model = _FavorModel(ord("A"))
    assert mean_logprob(model, tok, "cpu", "question", "") == 0.0


def test_identity_margins_structure():
    tok = _FakeTok()
    model = _FavorModel(ord("A"))
    out = identity_margins(model, tok, "cpu")
    assert len(out) == len(IDENTITY_PROBES)
    for row in out:
        assert {
            "question",
            "charter_logprob",
            "other_logprob",
            "margin",
            "prefers_charter",
        } <= set(row)
        assert isinstance(row["prefers_charter"], bool)
        assert row["margin"] == row["charter_logprob"] - row["other_logprob"]
