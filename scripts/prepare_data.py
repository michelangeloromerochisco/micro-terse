"""Stream FineWeb-Edu, tokenize with Llama-3.1, write a flat uint32 binary.

python scripts/prepare_data.py --output /workspace/data/fineweb-8B.bin --tokens 8000000000
Requires `pip install -e ".[data]"` and HF access to meta-llama/Llama-3.1-8B.
"""
import argparse

import numpy as np
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokens", type=int, default=8_000_000_000)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    ds = load_dataset(args.dataset, split="train", streaming=True)

    written = 0
    with open(args.output, "wb") as f, tqdm(total=args.tokens, unit="tok") as bar:
        for row in ds:
            ids = tok.encode(row["text"]) + [eos]
            arr = np.array(ids, dtype=np.uint32)
            f.write(arr.tobytes())
            written += len(arr)
            bar.update(len(arr))
            if written >= args.tokens:
                break

    data = np.memmap(args.output, dtype=np.uint32, mode="r")
    print(f"wrote {len(data)} tokens to {args.output}")


if __name__ == "__main__":
    main()
