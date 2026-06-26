# Model Card — Terse-Micro

> A 423M-parameter, ternary-weight {−1, 0, +1} language model trained from scratch for ~$127,
> deployable as a 182 MB CPU-only model. First tier of the **Terse family**.

- **Developer:** Michelangelo Romero Chisco (independent), 2026
- **Model type:** Decoder-only transformer, ternary weights, mixture-of-experts
- **Language:** English (primary), light multilingual; identity tuned English/Spanish
- **License:** Apache-2.0
- **Tokenizer:** Llama-3.1 (128,256 vocab)
- **Repository:** github.com/michelangeloromerochisco/micro-terse
- **Inference fork:** github.com/michelangeloromerochisco/llama.cpp (branch `terse-arch`)
- **Full report:** [`docs/papers/terse-micro-technical-report.md`](docs/papers/terse-micro-technical-report.md)

## Architecture

| Property | Value |
|---|---|
| Total parameters | ≈423 M |
| Active parameters / token | ≈320 M (MoE top-2) |
| Layers | 12 |
| Hidden size | 1024 |
| Attention | GQA, 8 query / 2 KV heads (4:1), head dim 128, QK-Norm before RoPE (θ=500000) |
| FFN | 2816 intermediate, squared-ReLU gated |
| MoE | 4 experts, top-2, on odd layers {1,3,5,7,9,11}; aux-loss-free bias-EMA balancing |
| MTP | 1 head (predicts +2; training only, dropped at inference) |
| Embeddings | tied input/output; full precision (~31% of params) |
| Ternary scope | internal projections only (Q/K/V/O, gate/up/down); norms/router/bias full precision |
| Ternary operator | sign-with-threshold forward; FOGZO-shaped STE backward, learnable per-layer τ |
| Context | 4096 |

Config: `configs/micro-trained.yaml` (the as-trained 423M; `configs/micro.yaml` is a stale 1.06B planning spec — do not use it for the released weights).

## Training

| Stage | Details |
|---|---|
| Pretraining | 8 B tokens, FineWeb-grade web text; AdamW (β 0.9/0.95, wd 0.1); LR 3e-4 → 3e-5 cosine, 2000 warmup; 488,282 steps; batch 4 × seq 4096; bf16; MTP aux weight 0.1 |
| SFT | 3 epochs, 44,558 ChatML conversations, prompt-masked loss, AdamW, grad-accum 16, seq 1024 |
| ORPO | 1 epoch, ~3,500 identity/charter preference pairs, Adafactor LR 1e-5, reference-free |
| Hardware | 1× RTX A6000 48 GB (RunPod), ≈250 GPU-hours, **≈$127 total** |
| Export | F32 GGUF (lossless for ternary) → `TQ2_0` ≈ **182 MB** |

Released checkpoints: `terse-micro-base.gguf` (pretrained LM), `terse-micro-sft.gguf` (chat), `terse-micro-orpo.gguf` (identity-aligned).

## Evaluation (measured)

Standard academic benchmarks (MMLU/HellaSwag/ARC) were **not** run; at this data budget knowledge accuracy is expected near chance. We report what we measured:

- **Perplexity** (held-out natural English, lower better): base **56.7**, SFT 97.5, ORPO 125.0.
- **Identity preference** (mean log-prob margin, charter vs "ChatGPT", over 4 probes): base **−1.81** (0/4) → SFT −1.09 (0/4) → ORPO **+0.90** (3/4). A +2.7-nat swing from ORPO.
- **Single-token factual recall** (base, top-1): "…painted by Leonardo da"→*Vinci* 90%, "…Neil"→*Armstrong* 84%, "hydrogen and"→*oxygen* 73%, "…revolves around the"→*sun* 66%, "…William"→*Shakespeare*, "…206"→*bones*. ≈14/18 curated prompts correct.

## Intended use

- Research and education on ternary / extreme-efficiency LLMs.
- On-device / offline, CPU-only deployment where footprint and energy dominate.
- A reproducible baseline and warm-start seed for larger Terse tiers.

## Out-of-scope / limitations

- **Not a production assistant.** Free-generation is incoherent beyond a clause or two (GPT-2-medium-class); it is **data-limited** (8 B tokens, 1–3 orders of magnitude under strong sub-1B peers).
- **Near-chance on knowledge/reasoning benchmarks** is expected. Do not deploy for factual question answering without retrieval.
- Identity is correct at the *preference* level (use the ORPO checkpoint) but not always fluently expressed in generation.
- May hallucinate, reflect web-text biases, and produce unsafe content; no safety tuning beyond the ORPO pass. Validate before any user-facing use.
- Ternary gives **no training-memory savings** (STE keeps fp masters); the win is inference footprint/energy.

## How to run

See the [README](README.md). In short: serve `terse-micro-sft.gguf` (chat), with `terse-micro-orpo.gguf` for identity-aligned responses and `terse-micro-base.gguf` for next-token prediction, all on CPU.

## Citation

```
@techreport{romerochisco2026tersemicro,
  title  = {Terse-Micro: A 423M-Parameter Ternary-Weight Language Model Trained From Scratch for \$127},
  author = {Romero Chisco, Michelangelo},
  year   = {2026},
  note   = {Apache-2.0. github.com/michelangeloromerochisco/micro-terse}
}
```
