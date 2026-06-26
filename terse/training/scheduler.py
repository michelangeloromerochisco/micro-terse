"""Cosine learning-rate schedule with linear warmup and a min-lr floor."""
import math

import torch

from terse.model.config import TrainingConfig


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainingConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    min_ratio = cfg.min_lr / cfg.lr

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
