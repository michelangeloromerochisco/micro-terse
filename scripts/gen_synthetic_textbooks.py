"""Phi-style synthetic "textbook" data generator for Terse v3 (hard distillation).

Uses a strong teacher (Qwen3.5 — API or local HF) to generate textbook-quality,
educational, reasoning-dense content across a curriculum. This is the dominant
capability lever for a data-limited model (see terse-architecture-v3 §4 / Phi thesis).

Teacher is pluggable and LAZY-loaded, so this file imports without the teacher present:
  --teacher api  --api-base http://host:port/v1 --teacher-model qwen3.5-9b   (OpenAI-compatible)
  --teacher hf   --teacher-model Qwen/Qwen3.5-9B                              (local transformers)

Modes:
  --mode pretrain  -> {"text": "..."}             (raw corpus for pretraining)
  --mode sft       -> {"messages":[user,assistant]} (instruction data, ChatML)

    python scripts/gen_synthetic_textbooks.py --teacher api --api-base $URL \
        --teacher-model qwen3.5-9b --mode pretrain --n 50000 --out data/synth_textbooks.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

# Phi-style curriculum: dense, educational, reasoning/math/code/science/common-sense.
CURRICULUM = {
    "math": ["arithmetic word problems", "algebra", "geometry proofs", "probability",
             "calculus intuition", "number theory", "logic puzzles", "estimation"],
    "code": ["python functions with explanation", "debugging walkthroughs", "data structures",
             "algorithms step-by-step", "recursion", "complexity analysis", "clean refactoring"],
    "science": ["physics concepts", "chemistry basics", "biology systems", "astronomy",
                "earth science", "scientific method", "cause and effect chains"],
    "reasoning": ["multi-step deduction", "analogies", "counterfactuals", "planning",
                  "comparing trade-offs", "spotting fallacies", "theory of mind"],
    "world": ["history explained simply", "geography", "economics intuition", "how things work",
              "everyday procedures", "definitions with examples"],
    "language": ["summarization", "rewriting for clarity", "explaining a hard idea simply",
                 "step-by-step instructions", "Q&A from a passage"],
}

# Prompt templates that elicit *textbook-quality* output (clear exposition, worked reasoning).
PRETRAIN_TMPL = (
    "Write a clear, self-contained textbook passage (250-450 words) teaching '{sub}' "
    "in the area of {domain}. Use precise language, a worked example, and explain the "
    "reasoning step by step as a great teacher would. Output only the passage."
)
SFT_TMPL = (
    "Create one high-quality instruction-following example about '{sub}' ({domain}). "
    "Return strict JSON: {{\"user\": <a clear question or task>, \"assistant\": <an excellent, "
    "step-by-step, correct answer>}}. Make the answer genuinely educational."
)


def make_prompt(mode: str, rng: random.Random) -> tuple[str, str, str]:
    domain = rng.choice(list(CURRICULUM))
    sub = rng.choice(CURRICULUM[domain])
    tmpl = PRETRAIN_TMPL if mode == "pretrain" else SFT_TMPL
    return tmpl.format(domain=domain, sub=sub), domain, sub


# ---- teacher backends (lazy) ----
def _make_teacher(args):
    if args.teacher == "api":
        from openai import OpenAI  # lazy
        client = OpenAI(base_url=args.api_base, api_key=args.api_key or "EMPTY")

        def gen(prompt: str) -> str:
            r = client.chat.completions.create(
                model=args.teacher_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
            return r.choices[0].message.content.strip()
        return gen

    # local HF
    import torch  # lazy
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
    tok = AutoTokenizer.from_pretrained(args.teacher_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype="auto", device_map="auto")

    def gen(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=args.max_tokens, temperature=args.temperature, do_sample=True)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return gen


def _to_record(mode: str, text: str):
    if mode == "pretrain":
        return {"text": text} if len(text) > 80 else None
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        u, a = obj.get("user", "").strip(), obj.get("assistant", "").strip()
        if u and a:
            return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
    except (ValueError, json.JSONDecodeError):
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["api", "hf"], required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gen = _make_teacher(args)
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            prompt, _, _ = make_prompt(args.mode, rng)
            try:
                rec = _to_record(args.mode, gen(prompt))
            except Exception as e:  # noqa: BLE001 — keep going on transient teacher errors
                print(f"[{i}] teacher error: {e!r}", flush=True)
                continue
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 500 == 0:
                print(f"kept {kept}/{i+1}", flush=True)
    print(f"wrote {kept} records -> {args.out}")


if __name__ == "__main__":
    main()
