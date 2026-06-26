# Terse micro — Benchmark battery

The **same battery Orchid was measured with**, prepared to run on Terse micro the moment
pretraining + identity tuning finish. Everything runs **off-pod, on the PC** — never on the
training pod while it's busy.

Two kinds of number:

1. **Standardized academic (logprob, 0-shot)** — directly comparable to published model numbers.
   Run with Terse's own lm-eval-harness wrapper (same methodology Orchid's `bench_standard.py`
   uses, but it runs straight on the checkpoint and needs no logprobs HTTP endpoint).
2. **Generative head-to-head (Orchid's suites)** — point Orchid's existing scripts at a running
   Terse server (`/v1/chat/completions`). Same questions, same scoring as Orchid.

> Paths assume the repo layout `LLM/terse/` (this repo) and `LLM/orchid/` (Orchid) side by side.
> Use `configs/micro-trained.yaml` (the real ~423M config) — **not** `configs/micro.yaml`
> (the stale 1.06B planning spec).

---

## 0. Prereqs (PC, after training)

```bash
cd terse
pip install -e ".[eval,serve,cli]"          # lm-eval + server + client
# model artifacts (from the pod at completion):
#   checkpoints/last.pt   (or terse-artifacts/<step>_weights.pt)   — for lm-eval + serving
#   terse-micro-orpo.gguf  (identity-tuned)                        — for llama.cpp serving
```

---

## 1. Standardized academic — lm-eval-harness (logprob, comparable)

Runs straight on the checkpoint (no server). This is the **headline comparable number**.

```bash
# Full suite: hellaswag, piqa, arc_easy, arc_challenge, winogrande
python scripts/eval.py --ckpt checkpoints/last.pt --config configs/micro-trained.yaml --suite full

# Add MMLU / lambada by extending FULL_TASKS in terse/eval/evaluate.py, or run lm-eval directly:
#   lm_eval --model hf --model_args pretrained=... --tasks mmlu,lambada_openai --num_fewshot 0
```

These map 1:1 onto Orchid's `bench_standard.py` targets (ARC-Challenge, HellaSwag, WinoGrande,
MMLU) and use the identical lm-eval logprob method, so Terse's scores are comparable to the
published BitNet / Llama / Qwen numbers in `orchid/BENCHMARK.md`.

---

## 2. Orchid's 600-question intelligence suite (generative, head-to-head)

The flagship `benchmark_100.py` — 100 questions × 6 categories (advanced math, deep reasoning,
hard knowledge, complex coding, instruction following, safety & bias), exact/semantic scoring,
bootstrap 95% CIs, Cohen's h, McNemar. Same questions Orchid was graded on.

```bash
# Terminal 1 — serve identity-tuned Terse on :8080
python scripts/serve.py --config configs/micro-trained.yaml --checkpoint checkpoints/last.pt --port 8080
#   (or GGUF:)  python scripts/serve.py --config configs/micro-trained.yaml --gguf terse-micro-orpo.gguf --port 8080

# Terminal 2 — run Orchid's suite against Terse (Terse takes the "orchid" slot)
python ../orchid/scripts/benchmark_100.py --orchid-url http://127.0.0.1:8080
#   --pilot                      quick 5-per-category smoke
#   --categories math,reasoning  subset
#   --api-models gpt-4o,...      add reference models for a real head-to-head table
```

Output: `orchid/tests/benchmark_100_results.json` + an HTML report.

---

## 3. Identity / bias (validates the charter took)

Run **after** the identity ORPO. Confirms Terse says it's Terse, made by Michelangelo Romero
Chisco (2026), refuses to claim it's ChatGPT, and stays neutral on contested topics.

```bash
python ../orchid/scripts/evaluate.py --url http://127.0.0.1:8080   # identity + reasoning + bias
```

---

## 4. Speed (TTFT, tok/s, prefill)

```bash
python ../orchid/scripts/bench_speed.py --url http://127.0.0.1:8080 --tokens 200
```

For the true on-device number, also measure the quantized GGUF under llama.cpp directly
(`llama-bench -m terse-micro-tq2_0.gguf`).

---

## 5. One-shot runner

`scripts/run_benchmarks.py` chains §1–§4: runs lm-eval on the checkpoint, boots the server,
runs the Orchid generative/identity/speed suites, and writes a combined `benchmark_results.json`.

```bash
python scripts/run_benchmarks.py --checkpoint checkpoints/last.pt --orchid-root ../orchid
```

---

## Realistic expectations (read before publishing any number)

Terse micro is a **proof-of-concept base model** (the README says so): ~423M ternary params,
trained on **8B tokens** (~19 tokens/param — Chinchilla-reasonable, but ~1000× less data than
modern sub-1B models, which see 2–11 **trillion** tokens). Capability is **data-limited**, not
architecture-limited.

Honest 0-shot ranges (will firm up after the run):

| Task | Random | Terse micro (est.) | Strong 0.5B (ref) |
|---|---|---|---|
| HellaSwag | 25% | **30–36%** | 50–60% |
| PIQA | 50% | **62–67%** | 74–77% |
| ARC-Easy | 25% | **40–48%** | 60–70% |
| ARC-Challenge | 25% | **24–29%** | 35–42% |
| WinoGrande | 50% | **50–54%** | 58–63% |
| MMLU | 25% | **25–27% (≈chance)** | 35–45% |
| Orchid 600-Q (generative) | — | **~10–25% overall** | — |

Anchor: **GPT-2-medium/large → Pythia-410M territory** — fluent English, weak multi-step
reasoning, frequent hallucination, near-chance on knowledge/hard-reasoning benchmarks. Its most
"finished" trait will be **identity** (specifically tuned). Do **not** market on benchmark
scores; market on size / openness / on-device / architecture. The path to "smart" is the
roadmap (mini/medium on trillions of tokens), not this model.
