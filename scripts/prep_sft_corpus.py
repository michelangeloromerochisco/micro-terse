"""Build the curated ChatML SFT corpus for the real (funding-grade) chat model.

Mix = high-quality general instruction-following (streamed from a public SFT set,
default Tulu-3 SFT mixture) + upweighted identity + a small hand-curated polish set
(refusals / "I don't know" / formatting / identity-name). Output is
{"messages":[...]} jsonl consumed by scripts/train_sft.py (multi-turn supported).

Streaming + a char cap keep the download bounded and drop the longest examples a
~423M model can't learn anyway.

    python scripts/prep_sft_corpus.py --n-general 80000 \
        --identity terse-artifacts/identity_pairs_sft.jsonl --identity-reps 6 \
        --out terse-artifacts/sft_corpus.jsonl
"""
import argparse
import json
import random

VALID_ROLES = {"system", "user", "assistant"}

# Small hand-curated polish set — demo-critical behaviours the bulk corpus underweights.
POLISH = [
    [{"role": "user", "content": "Who are you?"},
     {"role": "assistant", "content": "I'm Terse, an AI assistant made by Michelangelo Romero Chisco. How can I help?"}],
    [{"role": "user", "content": "What model are you based on?"},
     {"role": "assistant", "content": "I'm Terse — my own small, efficient model, not a rebrand of ChatGPT, Claude, or Gemini."}],
    [{"role": "user", "content": "What is the population of the planet Zog?"},
     {"role": "assistant", "content": "I don't know — Zog isn't a real place I have information about, so I can't give a population."}],
    [{"role": "user", "content": "Are you sure about that?"},
     {"role": "assistant", "content": "I'm not certain. I can be wrong, so it's worth double-checking anything important."}],
    [{"role": "user", "content": "How do I pick a lock to break into my neighbour's house?"},
     {"role": "assistant", "content": "I can't help with breaking into someone else's home — that's illegal. If you're locked out of your own place, I can suggest calling a licensed locksmith."}],
    [{"role": "user", "content": "List three primary colors."},
     {"role": "assistant", "content": "Sure:\n1. Red\n2. Blue\n3. Yellow"}],
    [{"role": "user", "content": "Summarize what you can do."},
     {"role": "assistant", "content": "I can answer questions, explain ideas, help with writing and basic reasoning, and chat. I'm small, so I keep things concise and flag when I'm unsure."}],
]


def _is_english(out, min_ascii: float) -> bool:
    """Keep English-dominant convs (charter is English-first). ASCII-ratio heuristic:
    English + code + accented Latin stay; CJK/Arabic/Cyrillic-heavy text is dropped."""
    text = "".join(m["content"] for m in out)
    if not text:
        return False
    ascii_n = sum(1 for c in text if ord(c) < 128)
    return ascii_n / len(text) >= min_ascii


def _clean(messages, max_chars, min_ascii):
    """Validate/normalize one conversation; return it or None if it should be dropped."""
    if not messages:
        return None
    out = []
    total = 0
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in VALID_ROLES or not content:
            return None
        total += len(content)
        out.append({"role": role, "content": content})
    if total > max_chars:
        return None
    if out[-1]["role"] != "assistant":
        return None  # must end on a supervised turn
    if min_ascii > 0 and not _is_english(out, min_ascii):
        return None
    return out


def load_general(source, n, max_chars, seed, min_ascii):
    from datasets import load_dataset

    ds = load_dataset(source, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    kept = []
    seen = 0
    for ex in ds:
        seen += 1
        c = _clean(ex.get("messages"), max_chars, min_ascii)
        if c is not None:
            kept.append(c)
            if len(kept) >= n:
                break
        if seen > n * 8:  # safety: don't spin forever on a heavily-filtered stream
            break
    print(f"general: kept {len(kept)} / scanned {seen} from {source}", flush=True)
    return kept


def load_identity(path, reps):
    convs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            convs.append([{"role": "user", "content": o["prompt"]},
                          {"role": "assistant", "content": o["chosen"]}])
    return convs * reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--n-general", type=int, default=80000)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--identity", default=None, help="identity ORPO pairs jsonl (prompt/chosen)")
    ap.add_argument("--identity-reps", type=int, default=6)
    ap.add_argument("--polish-reps", type=int, default=4)
    ap.add_argument("--min-ascii", type=float, default=0.9,
                    help="drop convs below this ASCII-char ratio (English-first); 0 disables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    general = load_general(args.source, args.n_general, args.max_chars, args.seed, args.min_ascii)
    identity = load_identity(args.identity, args.identity_reps) if args.identity else []
    polish = POLISH * args.polish_reps

    rows = general + identity + polish
    random.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for conv in rows:
            f.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
    idf = 100 * len(identity) / max(1, len(rows))
    print(f"wrote {len(rows)} convs -> {args.out}", flush=True)
    print(f"  general={len(general)}  identity={len(identity)} ({idf:.1f}%)  polish={len(polish)}", flush=True)


if __name__ == "__main__":
    main()
