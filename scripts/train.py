"""Training entry point: python scripts/train.py --config configs/micro.yaml --data <path>"""
import argparse

import torch
import yaml

from terse.data.dataset import build_dataloader
from terse.model.config import TerseConfig, TrainingConfig
from terse.model.terse_model import TerseModel
from terse.training.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = TerseConfig(**cfg["model"])
    train_cfg = TrainingConfig(**cfg["training"])

    model = TerseModel(model_cfg)
    loader = build_dataloader(args.data, train_cfg.seq_len, train_cfg.batch_size)
    Trainer(model, loader, train_cfg, device=args.device).train()


if __name__ == "__main__":
    main()
