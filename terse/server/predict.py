"""Next-token prediction — the base model's most impressive, honest demo.

A 423M base LM can't hold a conversation, but its single-token prediction is
strong: it recalls real facts and grammar with high confidence (e.g. "painted
by Leonardo da" -> "Vinci" ~90%). This sidesteps the compounding errors of long
free-gen, so it's the right way to show what the pretrained model actually
learned.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# Curated prompts where the answer is the natural next token — these reliably
# land for the base model, so the showcase looks sharp without cherry-picking
# live. (Verified against terse-micro-base.gguf.)
SHOWCASE_PROMPTS = [
    "The Mona Lisa was painted by Leonardo da",
    "The first man to walk on the Moon was Neil",
    "Water is made of hydrogen and",
    "The Earth revolves around the",
    "An apple a day keeps the doctor",
    "Romeo and Juliet was written by William",
    "The freezing point of water is zero degrees",
    "The human body has 206",
]


@torch.no_grad()
def next_token_topk(model, tokenizer, device, text: str, k: int = 5) -> list[dict[str, Any]]:
    """Top-k most likely next tokens for `text`, as {token, prob} dicts."""
    ids = tokenizer.encode(text, add_special_tokens=True)
    logits = model(torch.tensor([ids], device=device), return_logits=True)["logits"][0, -1].float()
    probs = F.softmax(logits, dim=-1)
    values, indices = probs.topk(k)
    return [
        {"token": tokenizer.decode([idx]), "prob": float(prob)}
        for idx, prob in zip(indices.tolist(), values.tolist())
    ]


@torch.no_grad()
def showcase(model, tokenizer, device, k: int = 5) -> list[dict[str, Any]]:
    """Run the curated showcase prompts through next_token_topk."""
    return [
        {"text": prompt, "predictions": next_token_topk(model, tokenizer, device, prompt, k)}
        for prompt in SHOWCASE_PROMPTS
    ]
