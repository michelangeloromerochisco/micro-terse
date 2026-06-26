"""AdamW with weight-decay / no-decay parameter groups."""
import torch

from terse.model.config import TrainingConfig

_NO_DECAY_KEYS = ("embed", "norm", "bias", "temperature", "router")


def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(key in name for key in _NO_DECAY_KEYS):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    # fused AdamW is faster on CUDA but raises on some driver/build combos; fall back.
    try:
        return torch.optim.AdamW(groups, fused=torch.cuda.is_available(), **kwargs)
    except (RuntimeError, ValueError) as e:
        print(f"[optimizer] fused AdamW unavailable ({e}); using unfused.", flush=True)
        return torch.optim.AdamW(groups, **kwargs)
