"""Evaluation entry point: python scripts/eval.py --ckpt <path> --config configs/micro.yaml"""
import argparse

import torch
import yaml

from terse.eval.evaluate import FAST_TASKS, FULL_TASKS, run_eval
from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel
from terse.training.checkpoint import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", choices=["fast", "full"], default="fast")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = TerseModel(TerseConfig(**cfg["model"]))
    load_checkpoint(args.ckpt, model)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tasks = FAST_TASKS if args.suite == "fast" else FULL_TASKS
    print(run_eval(model, tok, tasks, limit=args.limit))


if __name__ == "__main__":
    main()
