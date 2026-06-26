"""Muon × ternary-STE feasibility (the #1 v3 open ablation) + QK-Clip, on CPU/tiny model.

Goal: an early, free signal that Muon trains a ternary Terse model stably and reduces loss
(comparable to AdamW) BEFORE spending pod time. Not a quality benchmark — a sanity gate.
"""
import torch

from terse.model.terse_model import TerseModel
from terse.training.optimizer import build_optimizer
from terse.training.muon import (
    Muon, build_muon_optimizer, zeropower_via_newtonschulz5, qk_clip,
)


def _overfit(model, opt, steps=50):
    torch.manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    losses = []
    for _ in range(steps):
        out = model(ids, labels=ids, return_logits=False)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.step_moe_bias()
        losses.append(loss.item())
    return losses


def test_newtonschulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(32, 64)
    X = zeropower_via_newtonschulz5(G, steps=5)
    assert torch.isfinite(X).all()
    # wide matrix → rows pushed toward orthonormal: XX^T ≈ I (diagonal-dominant).
    # NS5 in 5 steps lands singular values in a band near 1 (not exactly 1).
    XXT = X @ X.T
    diag, off = XXT.diag(), XXT - torch.diag(XXT.diag())
    assert diag.mean() > 0.6                       # rows have ~unit norm
    assert off.abs().mean() < 0.1 * diag.mean()    # rows ~orthogonal (off-diag ≪ diag)


def test_muon_trains_ternary_model_stably(tiny_config, tiny_train_config):
    torch.manual_seed(0)
    model = TerseModel(tiny_config).train()
    opt = build_muon_optimizer(model, tiny_train_config)
    assert isinstance(opt, Muon)
    losses = _overfit(model, opt, steps=50)
    assert all(l == l for l in losses), "NaN in Muon loss"           # stability
    assert all(abs(l) < 1e4 for l in losses), "Muon loss exploded"   # no blowup
    assert losses[-1] < losses[0] * 0.85, f"Muon didn't reduce loss: {losses[0]:.3f}->{losses[-1]:.3f}"
    print(f"\n[Muon] ternary loss {losses[0]:.3f} -> {losses[-1]:.3f}")


def test_muon_competitive_with_adamw(tiny_config, tiny_train_config):
    torch.manual_seed(0)
    m_adam = TerseModel(tiny_config).train()
    a = _overfit(m_adam, build_optimizer(m_adam, tiny_train_config), 50)
    torch.manual_seed(0)
    m_muon = TerseModel(tiny_config).train()
    m = _overfit(m_muon, build_muon_optimizer(m_muon, tiny_train_config), 50)
    print(f"\n[compare] AdamW {a[0]:.3f}->{a[-1]:.3f}  |  Muon {m[0]:.3f}->{m[-1]:.3f}")
    # both must make real progress; Muon should be in the same ballpark, not diverge
    assert m[-1] < a[0], "Muon made no progress vs start"
    assert m[-1] < 1.5 * a[-1], "Muon far worse than AdamW (compatibility concern)"


def test_qk_clip_bounds_norms(tiny_config):
    torch.manual_seed(0)
    model = TerseModel(tiny_config)
    # blow up a q_proj weight, then clip
    for name, p in model.named_parameters():
        if "q_proj" in name and "weight" in name:
            p.data.mul_(100.0)
            break
    qk_clip(model, tau=4.0)
    for name, p in model.named_parameters():
        if ("q_proj" in name or "k_proj" in name) and "weight" in name:
            assert p.norm(dim=1).max() <= 4.0 + 1e-4
