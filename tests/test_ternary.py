import math

import torch

from terse.model.ternary import TernaryLinear, TernaryQuantizeFunction


def test_output_is_ternary():
    w = torch.randn(64, 64)
    temp = torch.ones(1)
    out = TernaryQuantizeFunction.apply(w, temp)
    unique = torch.unique(out)
    assert set(unique.tolist()).issubset({-1.0, 0.0, 1.0})


def test_has_zeros_from_threshold():
    w = torch.randn(128, 128)
    out = TernaryQuantizeFunction.apply(w, torch.ones(1))
    zero_frac = (out == 0).float().mean().item()
    assert 0.2 < zero_frac < 0.8  # threshold = mean(|w|) gives a meaningful sparsity


def test_ste_gradient_flows():
    layer = TernaryLinear(16, 16)
    x = torch.randn(4, 16, requires_grad=True)
    layer(x).sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0


def test_fogzo_is_smooth():
    # center latent weights (near 0) get larger grad scale than saturated ones
    w = torch.tensor([[0.0, 5.0]])
    temp = torch.ones(1)
    w.requires_grad_(True)
    out = TernaryQuantizeFunction.apply(w, temp)
    out.sum().backward()
    center_grad, edge_grad = w.grad[0, 0].abs(), w.grad[0, 1].abs()
    assert center_grad > edge_grad


def test_temperature_affects_gradient():
    w = torch.tensor([[1.0]])
    g_lo = _grad_for_temp(w, 0.1)
    g_hi = _grad_for_temp(w, 10.0)
    assert not math.isclose(g_lo, g_hi, rel_tol=1e-3)


def _grad_for_temp(w, t):
    wv = w.clone().requires_grad_(True)
    TernaryQuantizeFunction.apply(wv, torch.tensor([t])).sum().backward()
    return wv.grad.abs().sum().item()


def test_temperature_is_learnable_parameter():
    layer = TernaryLinear(8, 8)
    assert isinstance(layer.temperature, torch.nn.Parameter)
    assert layer.temperature.requires_grad


def test_training_step_updates_latent_weights():
    layer = TernaryLinear(16, 16)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = layer.weight.detach().clone()
    x = torch.randn(8, 16)
    loss = layer(x).pow(2).mean()
    loss.backward()
    opt.step()
    assert not torch.allclose(before, layer.weight)
