"""GGUF export entry point: python scripts/export.py --ckpt <path> --config <cfg> --out orchid.gguf"""
import argparse

import yaml

from terse.export.gguf import export_gguf
from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel
from terse.training.checkpoint import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="terse.gguf")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = TerseModel(TerseConfig(**cfg["model"]))
    load_checkpoint(args.ckpt, model)
    print("wrote", export_gguf(model, args.out))


if __name__ == "__main__":
    main()
