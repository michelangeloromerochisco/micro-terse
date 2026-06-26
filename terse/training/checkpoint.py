"""Checkpoint save / resume with RNG state, atomic writes, and milestone retention."""
import glob
import os
from typing import List, Optional

import torch


def _step_of(path: str) -> Optional[int]:
    """Parse the step number from a `step_<n>.pt` filename, or None if it doesn't match."""
    name = os.path.basename(path)
    if not (name.startswith("step_") and name.endswith(".pt")):
        return None
    try:
        return int(name[len("step_"):-len(".pt")])
    except ValueError:
        return None


def list_checkpoints(save_dir: str) -> List[str]:
    """Valid step checkpoints, newest first."""
    paths = [
        p for p in glob.glob(os.path.join(save_dir, "step_*.pt"))
        if _step_of(p) is not None
    ]
    return sorted(paths, key=_step_of, reverse=True)


def save_checkpoint(
    path: str, model, optimizer, scheduler, step: int, loss: float,
    save_optimizer: bool = True,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "loss": loss,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    # Atomic write: torch.save to a temp path, then os.replace. A crash mid-write
    # (e.g. spot-instance preemption) leaves only a stray .tmp, never a truncated
    # step_*.pt that resume would later choke on.
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def latest_checkpoint(save_dir: str) -> Optional[str]:
    ckpts = list_checkpoints(save_dir)
    return ckpts[0] if ckpts else None


def load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    # "optimizer" may be absent for model-only checkpoints (save_optimizer=False).
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["cpu_rng_state"])
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    return ckpt["step"]


def prune_checkpoints(save_dir: str, keep_last: int, milestone_every: int) -> None:
    paths = sorted(
        [p for p in glob.glob(os.path.join(save_dir, "step_*.pt"))
         if _step_of(p) is not None],
        key=_step_of,
    )
    protected = set(paths[-keep_last:]) if keep_last > 0 else set()
    for p in paths:
        if milestone_every > 0 and _step_of(p) % milestone_every == 0:
            protected.add(p)
    for p in paths:
        if p not in protected:
            os.remove(p)
    # Sweep any orphaned temp files from interrupted saves.
    for tmp in glob.glob(os.path.join(save_dir, "step_*.pt.tmp")):
        try:
            os.remove(tmp)
        except OSError:
            pass
