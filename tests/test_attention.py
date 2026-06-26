import torch

from terse.model.attention import TerseAttention


def test_output_shape(tiny_config):
    attn = TerseAttention(tiny_config)
    x = torch.randn(2, 16, tiny_config.hidden_dim)
    out = attn(x)
    assert out.shape == (2, 16, tiny_config.hidden_dim)


def test_causal_no_future_leak(tiny_config):
    torch.manual_seed(0)
    attn = TerseAttention(tiny_config).eval()
    x = torch.randn(1, 8, tiny_config.hidden_dim)
    with torch.no_grad():
        out_a = attn(x)
        x2 = x.clone()
        x2[:, 5:] += 10.0  # perturb future tokens
        out_b = attn(x2)
    # positions before the perturbation must be unchanged
    assert torch.allclose(out_a[:, :5], out_b[:, :5], atol=1e-4)
    assert not torch.allclose(out_a[:, 5:], out_b[:, 5:], atol=1e-4)


def test_position_sensitivity(tiny_config):
    # RoPE makes attention position-aware. Without it, softmax attention is
    # permutation-invariant over the attended (key, value) pairs, so reordering two
    # earlier tokens would leave the last position's output unchanged. With RoPE it
    # changes, because each token's key is rotated by its position.
    torch.manual_seed(0)
    attn = TerseAttention(tiny_config).eval()
    seq = torch.randn(1, 6, tiny_config.hidden_dim)  # distinct content per position
    swapped = seq.clone()
    swapped[:, [1, 4]] = swapped[:, [4, 1]]  # swap content at positions 1 and 4
    with torch.no_grad():
        out = attn(seq)
        out_sw = attn(swapped)
    # the last position attends to all tokens; only RoPE makes the swap matter
    assert not torch.allclose(out[:, -1], out_sw[:, -1], atol=1e-5)


def test_gradient_flows(tiny_config):
    attn = TerseAttention(tiny_config)
    x = torch.randn(2, 8, tiny_config.hidden_dim, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
