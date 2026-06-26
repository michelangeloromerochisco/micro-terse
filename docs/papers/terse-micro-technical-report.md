# Terse-Micro: A 423M-Parameter Ternary-Weight Language Model Trained From Scratch for $150

**Michelangelo Romero Chisco**
Independent researcher · 2026
Code: github.com/michelangeloromerochisco/micro-terse · License: Apache-2.0

---

## Abstract

We present **Terse-Micro**, a 423M-parameter (≈320M active) ternary-weight language model with weights constrained to {−1, 0, +1}, trained entirely from scratch on 8 billion tokens for approximately **US$150** on a single 48 GB GPU. Terse-Micro is fully clean-room: its architecture and its ternary training operator are our own, derived from but not copied from prior 1-bit work. The model combines grouped-query attention, an auxiliary-loss-free mixture-of-experts, squared-ReLU gated feed-forward networks, query–key normalization, a multi-token-prediction head, and tied embeddings, and it packs to a 182 MB `TQ2_0` GGUF — ternary weights losslessly, the tied embedding in `Q6_K` — that runs on commodity CPUs with no GPU. We document the architecture, the from-scratch ternary recipe (pretraining → supervised fine-tuning → ORPO alignment), and an deliberately honest evaluation. The base model reaches a perplexity of **56.7** on held-out natural English and shows strong single-token factual recall — for example, "*…painted by Leonardo da*" → "*Vinci*" at 90% probability — while ORPO identity alignment measurably flips the model's self-identity preference from favouring a generic "ChatGPT" answer to favouring its own charter identity (mean log-probability margin **−1.81 → +0.90**). We are equally candid about the limits: at an 8B-token budget Terse-Micro is data-limited and is not a fluent conversational assistant. We position it as a reproducible proof-of-concept for sovereign, edge-deployable ternary language models trained from scratch on an individual's budget.

---

## 1. Introduction

The dominant axis of progress in language modelling has been scale: more parameters, more tokens, more compute. A second, quieter axis has emerged in parallel — **efficiency of representation**. Post-training quantization routinely compresses models to 4 bits per weight, and a line of work on *quantization-aware* training has shown that models can be trained natively at extremely low precision, culminating in **ternary** weights ∈ {−1, 0, +1}, roughly 1.58 bits per weight [Wang et al. 2023; Ma et al. 2024]. Ternary weights are attractive far beyond their size: a ternary matmul reduces to additions and subtractions, eliminating multiplies and the energy they cost, which makes ternary models a natural fit for CPUs, phones, and other hardware that people already own.

Most public ternary results are either (a) fine-tunes that inherit a vendor's pretrained 1-bit backbone, or (b) large-budget runs that are not reproducible by individuals. We asked a narrower, more practical question:

> *How capable a language model can a single person train from scratch, in clean-room ternary, on a hobbyist budget — and what is honestly true about the result?*

Terse-Micro is our answer. It is a 423M-parameter ternary model trained for ~US$150 of rented GPU time, with an architecture and a ternary operator we wrote ourselves, and an evaluation that reports what the model *can* do (efficient deployment, factual recall, measurable alignment) without inflating what it cannot (fluent open-ended conversation, competitive benchmark scores).

**Contributions.**

1. **A fully clean-room ternary language model** — own architecture, own straight-through ternary operator — reproducible end-to-end for ~$150 on one 48 GB GPU.
2. **An honest, *measurable* demonstration of identity alignment**: ORPO moves the model's self-identity preference by +2.7 nats even though free-generation remains weak, separating *what the model prefers* from *what it can fluently express*.
3. **Evidence that a 182 MB, CPU-only model retains real world knowledge** at the single-token level (≈14/18 curated factual completions correct).
4. **A reproducible recipe** — a full from-scratch ternary pipeline (pretraining → SFT → ORPO → GGUF) that an individual can run end to end.

We deliberately do not claim competitiveness with frontier models. Terse-Micro's thesis is *capability per megabyte and per joule on owned hardware*, not capability per parameter.

---

## 2. Background and related work

**1-bit and ternary LLMs.** BitNet [Wang et al. 2023] introduced training transformers with binary weights; BitNet b1.58 [Ma et al. 2024] extended this to ternary {−1, 0, +1} and showed it can match full-precision perplexity at scale. Ternary weights are quantized with an absolute-mean (absmean) threshold and trained through a straight-through estimator (STE) [Bengio et al. 2013], keeping latent full-precision master weights that are ternarized on the forward pass. Independent reproductions such as the Bonsai ternary models confirm that from-scratch ternary training is feasible at small scale. Terse-Micro follows this tradition but uses an in-house operator (Section 3.2) and applies ternarization only to internal projections.

**Efficient transformer components.** We adopt a now-standard "efficient decoder" stack: rotary position embeddings (RoPE) [Su et al. 2021]; grouped-query attention (GQA) [Ainslie et al. 2023] for a smaller KV cache; RMSNorm [Zhang & Sennrich 2019]; query–key normalization for training stability; squared-ReLU activations [So et al. 2021] for activation sparsity; and tied input/output embeddings. For capacity at low active cost we use a fine-grained mixture-of-experts (MoE) with auxiliary-loss-free, bias-EMA load balancing [Dai et al. 2024; DeepSeek-AI 2024], and a multi-token-prediction (MTP) head [Gloeckle et al. 2024] as a training auxiliary.

**Data-constrained training.** Compute-optimal scaling [Hoffmann et al. 2022] prescribes ≈20 tokens/parameter; we sit at ~19. Crucially, strong small models in the literature are *massively over-trained* (hundreds of billions to tens of trillions of tokens), and repeating high-quality data 2–4× is nearly as effective as fresh tokens [Muennighoff et al. 2023]. This framing is central to interpreting our results.

**Preference alignment.** We align identity with ORPO [Hong et al. 2024], a reference-model-free, monolithic odds-ratio preference objective that folds SFT and preference optimization into one loss — attractive on a small budget because it needs no second (reference) model in memory.

---

## 3. Architecture

Terse-Micro is a decoder-only transformer. The forward flow is `embed → 12 × TerseBlock → final RMSNorm → tied LM head`, with a multi-token-prediction head attached during training.

### 3.1 Configuration

| Property | Value |
|---|---|
| Total parameters | ≈423 M |
| Active parameters / token | ≈320 M (MoE top-2) |
| Layers | 12 |
| Hidden size | 1024 |
| Attention | GQA, 8 query heads / 2 KV heads (4:1), head dim 128 |
| FFN intermediate | 2816, squared-ReLU gated |
| MoE | 4 experts, top-2, on odd layers {1,3,5,7,9,11}; aux-free bias-EMA balancing |
| MTP heads | 1 (predicts token at position +2; dropped at inference) |
| Normalization | RMSNorm (pre-norm) + QK-Norm before RoPE |
| Position | RoPE, θ = 500000 |
| Vocabulary | 128256 (Llama-3.1 tokenizer); tied embeddings |
| Max sequence | 4096 |
| Precision (training) | bf16 + gradient checkpointing |
| License | Apache-2.0 |

A consequence worth stating plainly: the 128K-token embedding table accounts for ≈131 M parameters — about **31%** of the model — and is kept at full precision. Terse-Micro therefore spends nearly a third of its capacity on the tokenizer interface, leaving fewer "reasoning" parameters than peers built on 32–64K vocabularies. A smaller vocabulary is an obvious lever for future work.

### 3.2 Ternary weights and the training operator

Only the internal projection matrices — attention Q/K/V/O and the FFN gate/up/down — are ternary. Embeddings, the (tied) LM head, all norms, the MoE router gate, biases, and per-layer temperatures remain full precision. This keeps the numerically sensitive parts exact while making the bulk of the compute multiply-free.

On the **forward** pass, each ternary linear quantizes its latent weight `W` with a sign-and-threshold rule: weights whose magnitude is below the per-tensor absolute-mean `mean(|W|)` are set to 0 and the rest to ±1 by sign, with **no magnitude scaling** — the output is exactly `W_t ∈ {−1, 0, +1}` (which is what makes the later `TQ2_0` packing exact). On the **backward** pass we use a straight-through estimator with a *FOGZO-shaped* gradient scale

```
scale = 1 − tanh²(W_latent / τ)
```

where `τ` is a **learnable per-layer temperature** (initialized to 1.0, clamped to [0.01, 10]). This down-weights gradient flow to weights that have saturated far from the quantization boundary while preserving it near the decision region, which we found stabilizes ternary convergence.

We are explicit about one non-obvious property: because STE maintains full-precision latent masters, **ternary training does not reduce training-time memory** — the footprint and energy wins are realized only at inference, after the latent weights are discarded and the ternary weights are packed. Ternary is an *inference and deployment* advantage, not a training-memory one.

### 3.3 Export and quantized footprint

After training, the ternary weights — being exactly {−1, 0, +1} — are represented losslessly in an F32 GGUF under a custom `terse` architecture in a fork of `llama.cpp`; that F32 container is exact, so the `.pt` weights are reconstructable from it. The deployable model then packs the ternary weights into `TQ2_0` (≈2 bits/weight, still exact for `{−1, 0, +1}`) and quantizes the tied token embedding — ≈31% of parameters — to `Q6_K`, yielding a **≈182 MB** file. The embedding is the only lossy part; the ternary weights are preserved exactly.

---

## 4. Training

### 4.1 Pretraining

Terse-Micro was pretrained on **8 B tokens** of filtered web text (FineWeb-grade) [Penedo et al. 2024] with the Llama-3.1 tokenizer. Optimization used AdamW (β = 0.9/0.95, weight decay 0.1), a peak learning rate of 3 × 10⁻⁴ decayed by cosine to 3 × 10⁻⁵ over 488,282 steps with 2,000 warmup steps, gradient clipping at 1.0, batch size 4 × sequence length 4096 (16,384 tokens/step), bf16, and gradient checkpointing. The total loss combined the main next-token cross-entropy with the MTP auxiliary at weight 0.1.

Hardware was a single **RTX A6000 (48 GB)** rented on RunPod at roughly $0.55–0.60/hr, for ≈250 GPU-hours and a total cost of **≈US$150**. At 8 B tokens the model sees ~19 tokens/parameter — Chinchilla-reasonable, but, as Section 6 stresses, 1–3 orders of magnitude fewer tokens than the strong sub-1B models it is naturally compared against.

### 4.2 Supervised fine-tuning

We then ran 3 epochs of supervised fine-tuning on **44,558 ChatML conversations**, with prompt tokens masked from the loss so that only assistant responses are learned, using AdamW with gradient accumulation 16 at sequence length 1024. SFT teaches the chat template and response format.

### 4.3 ORPO identity alignment

Finally, one epoch of **ORPO** [Hong et al. 2024] on ≈3,500 preference pairs aligned the model's identity (its charter — who it is and who built it) and removed a set of unwanted default behaviours, using Adafactor at learning rate 1 × 10⁻⁵. ORPO is reference-free, so no second model was held in memory. Because preference pairs whose prompt fills the sequence length can yield empty response slices (and a NaN loss), we add guards that skip empty pairs and any non-finite loss or gradient before the optimizer step.

The full pipeline — `pretrain → SFT → ORPO → ternary GGUF` — is fully automated and ran to completion on the rented instance, including recovery from a mid-run filesystem stall, within the $150 budget.

---

## 5. Evaluation

Standard academic benchmarks (MMLU, HellaSwag, ARC, etc.) were **not** run for this report; at this data budget we expect near-chance knowledge accuracy and we decline to report numbers we have not measured. Instead we report three things we *did* measure, each chosen to isolate a capability the model genuinely has.

### 5.1 Language-model fluency (perplexity)

On a held-out set of natural-English sentences, perplexity (lower is better) was:

| Checkpoint | Perplexity ↓ |
|---|---|
| **base** | **56.7** |
| SFT | 97.5 |
| ORPO | 125.0 |

The base model is the strongest pure language model; SFT and ORPO raise perplexity on *raw* text because chat-formatting and identity alignment move the model off the plain-text distribution. The gap base→ORPO is an "alignment tax" we exploit deliberately at serving time (Section 5.4).

### 5.2 Identity alignment (preference margins)

For four identity probes we score the mean log-probability the model assigns to its **charter** answer (e.g., "*I'm Terse, an AI assistant*") versus a generic **"ChatGPT/OpenAI"** answer; the *margin* is charter minus other (positive = prefers its own identity). Averaged over the four probes:

| Checkpoint | Mean margin (nats) | Probes preferring charter |
|---|---|---|
| base | −1.81 | 0 / 4 |
| SFT | −1.09 | 0 / 4 |
| **ORPO** | **+0.90** | **3 / 4** |

Per-probe after ORPO: *Who are you?* +1.79, *Who made you?* +1.43, *Do you have feelings?* +1.06, *Are you ChatGPT?* −0.69. The training thus produced a **+2.7-nat swing** in self-identity preference. This is the paper's cleanest positive result: identity alignment is real and measurable at the preference level **even though free-generation remains weak** — the model *prefers* the right answer before it can fluently *say* it.

### 5.3 Factual recall (single-token prediction)

A small base language model is strongest at single-token prediction, which sidesteps the compounding errors of long generation. On curated factual completions the base model's top prediction is frequently correct and confident:

| Prompt (completes →) | Top token | Prob. |
|---|---|---|
| The Mona Lisa was painted by Leonardo da → | Vinci | 90% |
| The first man to walk on the Moon was Neil → | Armstrong | 84% |
| Water is made of hydrogen and → | oxygen | 73% |
| An apple a day keeps the doctor → | away | 67% |
| The Earth revolves around the → | sun | 66% (+Sun 22%) |
| Romeo and Juliet was written by William → | Shakespeare | 36% |
| The freezing point of water is zero degrees → | Celsius | 36% |
| The human body has 206 → | bones | 32% |

On a set of 18 curated prompts the correct answer appears as the top-1 or within top-5 in ≈14 cases, including live completions such as "*The capital of Italy is the city of*" → *Rome* (followed by *Naples*, *Venice*). The model carries this knowledge in 182 MB with no retrieval or lookup.

### 5.4 Efficiency and deployment

Terse-Micro deploys as a **182 MB** `TQ2_0` GGUF and runs on a commodity CPU with no GPU. Ternary matmuls are addition/subtraction only, and MoE top-2 routing means ≈320 M of the 423 M parameters are active per token. For comparison, a 0.5 B fp16 model is ≈1 GB and a 0.5 B 4-bit model ≈300–350 MB; Terse-Micro's non-embedding weights are roughly an order of magnitude smaller than fp16, small enough to sit in phone RAM.

---

## 6. Discussion and limitations

We want this section read as carefully as the results.

**Terse-Micro is data-limited, not architecture-limited.** Eight billion tokens is 1–3 orders of magnitude fewer than the data behind strong sub-1B peers (Pythia-410M ≈300 B; SmolLM2-360M ≈4 T; Qwen2.5-0.5B ≈18 T). Small models are good *because* they are massively over-trained; Terse-Micro is at a GPT-2-era data budget *by design*, to honour a tight hobbyist cost budget. Its realistic capability is **GPT-2-medium/large to Pythia-410M territory**: fluent for a clause or two, then prone to drift and hallucination in open generation.

**Ternary compounds the data problem.** Ternary weights converge more slowly than fp16 and want *more* data, not less; pairing the most data-hungry format with the smallest data budget is a double penalty we accepted to keep the run cheap and the footprint tiny.

**The embedding tax is real.** ~31% of parameters sit in the full-precision 128K embedding, leaving fewer reasoning parameters than vocabulary-lean peers. A smaller vocabulary is an obvious lever for future work.

**No training-memory savings from ternary.** As noted in §3.2, STE keeps fp masters; ternary helps inference, not training.

**Free-generation is not production-ready.** Identity is correct at the preference level but the base is too small to express it fluently; SFT improves fluency but lacks identity; ORPO has identity but pays a fluency tax. This is why our live demonstration serves *different checkpoints for different purposes* — base for next-token recall, ORPO for identity, SFT for the most fluent chat — rather than presenting one model as a finished assistant.

**Benchmarks pending.** A standard logprob benchmark battery (HellaSwag/PIQA/ARC/WinoGrande/MMLU) is prepared but not yet run; we will report it separately and expect knowledge suites near chance.

None of these caveats undercut the contribution. They define it: Terse-Micro shows that the *full machinery* of a modern small LLM — clean-room ternary, MoE, MTP, alignment, ternary quantization, CPU deployment — can be built and run end-to-end by one person for the price of a video game, and that even at this budget the result retains measurable knowledge and alignment.

---

## 7. Conclusion

Terse-Micro is a 423M-parameter clean-room ternary language model trained from scratch for ~US$150, deployable as a 182 MB CPU-only model. It is not a frontier system and we do not present it as one; it is a reproducible proof-of-concept whose value is *capability per megabyte and per joule on owned hardware*, plus a transparent account of what an extreme-budget ternary model does and does not do. The obvious levers for closing the quality gap while keeping the footprint advantage are more and better data, a smaller vocabulary, and distillation from a strong open teacher.

---

## References

- Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP.
- Bengio, Y., Léonard, N., Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3432.
- Dai, D., et al. (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* arXiv:2401.06066.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* (Auxiliary-loss-free load balancing; multi-token prediction.) arXiv:2412.19437.
- Gloeckle, F., et al. (2024). *Better & Faster Large Language Models via Multi-token Prediction.* ICML.
- Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). NeurIPS.
- Hong, J., Lee, N., Thorne, J. (2024). *ORPO: Monolithic Preference Optimization without Reference Model.* EMNLP.
- Ma, S., et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764.
- Muennighoff, N., et al. (2023). *Scaling Data-Constrained Language Models.* NeurIPS.
- Penedo, G., et al. (2024). *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS Datasets & Benchmarks.
- So, D., et al. (2021). *Primer: Searching for Efficient Transformers for Language Modeling* (squared ReLU). NeurIPS.
- Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
- Wang, H., et al. (2023). *BitNet: Scaling 1-bit Transformers for Large Language Models.* arXiv:2310.11453.
- Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS.

---

*Artifacts.* Code, configs, training scripts, the custom `llama.cpp` `terse` architecture fork, and the evaluation harness used for §5 are open under Apache-2.0.
