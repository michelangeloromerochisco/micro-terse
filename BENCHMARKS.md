# Benchmarks

Honest, measured results for Micro-Terse. Standard academic multiple-choice benchmarks
(MMLU, HellaSwag, ARC) were **not** run: at an 8B-token budget, knowledge accuracy on these is
expected to sit at or near chance, so reporting them would add noise, not signal. Instead we
report what we actually measured and what it tells us.

> Three checkpoints: **base** (pretrained LM), **sft** (chat), **orpo** (identity-aligned).

## 1. Perplexity (held-out natural English)

Lower is better.

| Checkpoint | Perplexity |
|---|---|
| base | **56.7** |
| sft | 97.5 |
| orpo | 125.0 |

The base model is the strongest pure language model; SFT and ORPO trade raw perplexity for
chat formatting and identity alignment, as expected.

## 2. Identity preference

Mean log-probability margin between the charter-correct identity answer and a generic "ChatGPT"
answer, averaged over 4 probes (positive = prefers its own identity; in parentheses, probes won).

| Checkpoint | Margin (nats) | Probes won |
|---|---|---|
| base | **−1.81** | 0 / 4 |
| sft | −1.09 | 0 / 4 |
| orpo | **+0.90** | 3 / 4 |

ORPO produces a **+2.7-nat** swing — the model's *preference* flips to its own identity, even
though free-generation fluency remains limited.

## 3. Single-token factual recall (base, top-1)

Top-1 next-token probability on curated factual completions:

| Prompt | Completion | P(top-1) |
|---|---|---|
| "…painted by Leonardo da" | *Vinci* | 90% |
| "…Neil" | *Armstrong* | 84% |
| "hydrogen and" | *oxygen* | 73% |
| "…revolves around the" | *sun* | 66% |
| "…William" | *Shakespeare* | ✓ |
| "…206" | *bones* | ✓ |

≈14 / 18 curated prompts correct — a 182 MB CPU-only model retains real world knowledge at the
single-token level.

## Reproduce

```bash
pip install -e ".[eval,serve]"

# Standardized academic suite (logprob, 0-shot) straight on a checkpoint
python scripts/eval.py --ckpt path/to/model.pt --config configs/micro-trained.yaml --suite full
```

## Context

Micro-Terse is a proof-of-concept base model (~423M ternary params, 8B tokens ≈ 19 tokens/param —
Chinchilla-reasonable, but 1–3 orders of magnitude less data than modern sub-1B models trained on
trillions of tokens). Capability is **data-limited**, not architecture-limited: expect
GPT-2-medium-class fluency, weak multi-step reasoning, and near-chance scores on knowledge and
hard-reasoning benchmarks. Do not market it on benchmark scores; its case is size, openness,
and on-device operation.
