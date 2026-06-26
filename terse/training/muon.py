"""Muon optimizer (+ QK-Clip) for Terse — token-efficient training (Kimi K2 recipe).

Muon orthogonalizes the momentum of 2D weight matrices via a Newton–Schulz iteration
before the update (Keller Jordan, 2024). It is ~more token-efficient than AdamW, which
directly attacks Terse's data-limited bottleneck (see terse-architecture-v3). 1D params
(norms, biases, temperature, router bias) and the embedding/LM-head use a standard AdamW
update inside the same optimizer.

Ternary note: Muon updates the LATENT fp32 weights of TernaryLinear; gradients reach them
through the STE (TernaryQuantizeFunction). This module + tests/test_muon.py exist to settle
the open "Muon × ternary-STE" ablation before scaling.

QK-Clip (MuonClip-lite): rescales q_proj/k_proj latent weights when their norm exceeds a
threshold, preventing attention-logit blowups Muon can otherwise trigger at scale.
"""
from __future__ import annotations

import torch

from terse.model.config import TrainingConfig

# params routed to AdamW (not Muon): embedding/head, norms, biases, ternary temperature, MoE router bias
_ADAMW_KEYS = ("embed", "norm", "bias", "temperature", "router")


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G (2D) via a quintic Newton–Schulz iteration. Returns ~UV^T of G."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Hybrid optimizer: Muon update for groups with ``use_muon=True``, AdamW otherwise."""

    def __init__(self, param_groups, lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_lr=2e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr=adamw_lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        mom, lr, wd = group["momentum"], group["lr"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(g)
            buf = st["momentum_buffer"]
            buf.mul_(mom).add_(g)
            g_eff = g.add(buf, alpha=mom) if group["nesterov"] else buf
            u = zeropower_via_newtonschulz5(g_eff, steps=group["ns_steps"]).to(p.dtype)
            # shape-aware scale so the RMS update size is ~independent of matrix aspect
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            if wd:
                p.mul_(1 - lr * wd)
            p.add_(u, alpha=-lr * scale)

    def _adamw_step(self, group):
        lr, (b1, b2), eps, wd = group["adamw_lr"], group["betas"], group["eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            m, v = st["exp_avg"], st["exp_avg_sq"]
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            bc1 = 1 - b1 ** st["step"]
            bc2 = 1 - b2 ** st["step"]
            denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
            if wd:
                p.mul_(1 - lr * wd)
            p.addcdiv_(m / bc1, denom, value=-lr)


def build_muon_optimizer(model: torch.nn.Module, cfg: TrainingConfig,
                         muon_lr: float | None = None) -> Muon:
    """Group params: 2D non-embedding ternary weights → Muon; the rest → AdamW."""
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_adamw = any(k in name for k in _ADAMW_KEYS)
        if (not is_adamw) and p.ndim == 2:
            muon_params.append(p)
        elif is_adamw and ("norm" in name or "bias" in name or "temperature" in name or "router" in name):
            adamw_nodecay.append(p)
        else:  # embedding / head: AdamW with decay
            adamw_decay.append(p)
    groups = [
        {"params": muon_params, "use_muon": True, "weight_decay": cfg.weight_decay},
        {"params": adamw_decay, "use_muon": False, "weight_decay": cfg.weight_decay},
        {"params": adamw_nodecay, "use_muon": False, "weight_decay": 0.0},
    ]
    return Muon(groups, lr=muon_lr or (cfg.lr * 20), adamw_lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)


@torch.no_grad()
def qk_clip(model: torch.nn.Module, tau: float = 4.0) -> None:
    """MuonClip-lite: rescale q_proj/k_proj latent weights whose row-norm exceeds tau,
    preventing attention-logit explosions. Call after optimizer.step()."""
    for name, p in model.named_parameters():
        if p.ndim != 2 or not ("q_proj" in name or "k_proj" in name) or "weight" not in name:
            continue
        max_norm = p.norm(dim=1).max()
        if max_norm > tau:
            p.mul_(tau / max_norm)
