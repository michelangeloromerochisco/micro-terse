"""Download + convert + subset mlabonne/orpo-dpo-mix-40k into the
{prompt, chosen, rejected} JSONL format that scripts/train_pref.py expects.

    python scripts/prep_orpo_data.py --out orpo_sft.jsonl --n 20000 --max-chars 2000
"""
import argparse
import json


def _resp_text(x) -> str:
    """chosen/rejected may be a string or a list of {role, content} messages."""
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x:
        asst = [m for m in x if isinstance(m, dict) and m.get("role") == "assistant"]
        if asst:
            return asst[-1].get("content", "")
        last = x[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def _prompt_text(row) -> str:
    p = row.get("prompt")
    if isinstance(p, str) and p.strip():
        return p
    ch = row.get("chosen")
    if isinstance(ch, list):
        usr = [m for m in ch if isinstance(m, dict) and m.get("role") == "user"]
        if usr:
            return usr[-1].get("content", "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--max-chars", type=int, default=2000,
                    help="skip pairs where prompt+response exceeds this many chars")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train")
    print(f"loaded {len(ds)} rows; columns={ds.column_names}", flush=True)

    written = skipped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            p = _prompt_text(row)
            c = _resp_text(row.get("chosen"))
            r = _resp_text(row.get("rejected"))
            if not (p and c and r):
                skipped += 1
                continue
            if len(p) + len(c) > args.max_chars or len(p) + len(r) > args.max_chars:
                skipped += 1
                continue
            f.write(json.dumps({"prompt": p, "chosen": c, "rejected": r}) + "\n")
            written += 1
            if written >= args.n:
                break
    print(f"wrote {written} pairs to {args.out} (skipped {skipped})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
