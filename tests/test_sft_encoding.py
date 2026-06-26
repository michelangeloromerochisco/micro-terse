"""Unit tests for the SFT conversation encoder + LR schedule (scripts/train_sft.py).

Uses a fake tokenizer so the tests are fast and offline — they verify the masking
logic and schedule math, not tokenizer specifics.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import train_sft  # noqa: E402


class FakeTok:
    """Deterministic char-level tokenizer; ids >= 10 to never collide with -100."""

    def encode(self, text, add_special_tokens=False):
        body = [ord(c) % 50000 + 10 for c in text]
        return ([2] + body) if add_special_tokens else body


def test_single_turn_masks_prompt_supervises_response():
    tok = FakeTok()
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    ids, labels = train_sft.encode_conversation(tok, conv, 9999)

    body = tok.encode("yo<|im_end|>\n", add_special_tokens=False)
    n = len(body)
    # The assistant body is the final segment: supervised and equal to ids there.
    assert labels[-n:] == ids[-n:] == body
    # Everything before the assistant body is masked.
    assert all(l == -100 for l in labels[:-n])
    # Every supervised label equals its input id (no stray supervision).
    assert all(l == -100 or l == t for l, t in zip(labels, ids))


def test_multi_turn_supervises_every_assistant_turn():
    tok = FakeTok()
    conv = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    ids, labels = train_sft.encode_conversation(tok, conv, 9999)

    expected_supervised = len(tok.encode("a1<|im_end|>\n")) + len(tok.encode("a2<|im_end|>\n"))
    supervised = sum(1 for l in labels if l != -100)
    assert supervised == expected_supervised
    assert all(l == -100 or l == t for l, t in zip(labels, ids))


def test_truncation_respects_seq_len():
    tok = FakeTok()
    conv = [{"role": "user", "content": "x" * 100}, {"role": "assistant", "content": "y" * 100}]
    ids, labels = train_sft.encode_conversation(tok, conv, 16)
    assert len(ids) == len(labels) == 16


def test_lr_warmup_then_cosine_decay():
    base, total, warmup = 2e-5, 100, 10
    # Warmup ramps up to base by the end of warmup.
    assert train_sft.lr_at(0, total, warmup, base) < train_sft.lr_at(5, total, warmup, base)
    assert math.isclose(train_sft.lr_at(warmup - 1, total, warmup, base), base, rel_tol=1e-6)
    # After warmup it decays monotonically toward ~0.
    assert train_sft.lr_at(warmup, total, warmup, base) > train_sft.lr_at(total - 1, total, warmup, base)
    assert train_sft.lr_at(total - 1, total, warmup, base) < 1e-6 + 0.05 * base


def test_load_data_drops_assistantless_and_keeps_messages(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}\n'
        '{"messages": [{"role": "user", "content": "no answer here"}]}\n',
        encoding="utf-8",
    )
    convs = train_sft._load_data(str(p))
    assert len(convs) == 1
    assert convs[0][-1]["role"] == "assistant"
