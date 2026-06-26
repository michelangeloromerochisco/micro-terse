"""Off-pod eval: load a weights-only checkpoint and run the benchmark suite.

Pairs with scripts/strip_checkpoint.py. Runs anywhere with the Terse package +
the matching config + tokenizer — zero load on the training host.

    python scripts/eval_local.py --weights step_36000_weights.pt \
        --config configs/micro.yaml --device cuda --suite fast --limit 200

On a 4 GB GPU use --device cuda (423M fits); fall back to --device cpu if OOM.
"""
import argparse

import torch
import yaml

from terse.eval.evaluate import FAST_TASKS, FULL_TASKS, run_eval
from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", choices=["fast", "full"], default="fast")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = TerseModel(TerseConfig(**cfg["model"]))
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tasks = FAST_TASKS if args.suite == "fast" else FULL_TASKS
    print(run_eval(model, tok, tasks, device=args.device, limit=args.limit))


if __name__ == "__main__":
    main()
