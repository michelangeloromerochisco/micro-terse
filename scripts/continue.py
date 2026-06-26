"""Interactive raw text-continuation REPL for a Micro-Terse checkpoint.

Micro-Terse is a BASE language model (pretraining on FineWeb), so chat/ChatML is
out-of-distribution and yields junk. Raw text continuation is the right way to
gauge how coherent the current checkpoint is: type a text prefix and the model
continues it.

    python scripts/continue.py --checkpoint W.pt --config micro.yaml --device cuda
"""
import argparse

import torch
import yaml

from terse.model.config import TerseConfig
from terse.model.generate import generate_stream
from terse.model.terse_model import TerseModel


def main() -> int:
    ap = argparse.ArgumentParser(description="Micro-Terse text-continuation REPL")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    cfg = TerseConfig(**yaml.safe_load(open(args.config))["model"])
    model = TerseModel(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model = model.to(args.device).eval()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(
        "Micro-Terse continuation REPL — type a text prefix, Enter to continue it.\n"
        "Base LM, partially trained: expect fluent-ish but incoherent text.\n"
        "Ctrl+C to exit.\n"
    )
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt.strip():
            continue
        ids = tok.encode(prompt, return_tensors="pt").to(args.device)
        last = ids
        for _, full in generate_stream(
            model, ids,
            max_new_tokens=args.max_tokens,
            min_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tok.eos_token_id,
            device=args.device,
        ):
            last = full
        cont = tok.decode(last[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
        print(prompt + cont + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
