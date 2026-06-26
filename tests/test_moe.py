import torch

from terse.model.moe import MoERouter, TerseMoE


def test_top2_selected(tiny_config):
    router = MoERouter(tiny_config)
    x = torch.randn(20, tiny_config.hidden_dim)
    weights, indices = router(x)
    assert indices.shape == (20, tiny_config.top_k)
    # no token routes to the same expert twice
    assert (indices[:, 0] != indices[:, 1]).all()


def test_weights_sum_to_one(tiny_config):
    router = MoERouter(tiny_config)
    weights, _ = router(torch.randn(20, tiny_config.hidden_dim))
    assert torch.allclose(weights.sum(-1), torch.ones(20), atol=1e-5)


def test_all_experts_receive_tokens_at_init(tiny_config):
    torch.manual_seed(0)
    router = MoERouter(tiny_config)
    _, indices = router(torch.randn(256, tiny_config.hidden_dim))
    counts = torch.bincount(indices.flatten(), minlength=tiny_config.num_experts)
    assert (counts > 0).all()


def test_bias_update_pushes_overloaded_down(tiny_config):
    router = MoERouter(tiny_config)
    # all tokens routed to expert 0 (overloaded)
    n = 100
    indices = torch.zeros(n, tiny_config.top_k, dtype=torch.long)
    indices[:, 1] = 1
    before = router.expert_bias.clone()
    router.record_counts(indices, n)
    router.apply_bias_update()
    assert router.expert_bias[0] < before[0]
    assert router.expert_bias[3] > router.expert_bias[0]


def test_record_counts_is_idempotent_for_checkpointing(tiny_config):
    # The save + recompute forward passes of gradient checkpointing must record the
    # SAME counts (record overwrites, never accumulates), and forward must NOT mutate
    # the bias — otherwise routing differs on recompute and checkpointing crashes.
    router = MoERouter(tiny_config)
    n = 100
    indices = torch.randint(0, tiny_config.num_experts, (n, tiny_config.top_k))
    bias_before = router.expert_bias.clone()
    router.record_counts(indices, n)
    first = router._last_counts.clone()
    router.record_counts(indices, n)  # simulate the recompute pass
    assert torch.equal(first, router._last_counts)        # idempotent
    assert torch.equal(bias_before, router.expert_bias)   # forward never touches bias


def test_moe_forward_and_grad(tiny_config):
    moe = TerseMoE(tiny_config)
    x = torch.randn(2, 8, tiny_config.hidden_dim, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    # at least one expert's parameters received gradient
    grads = [p.grad for e in moe.experts for p in e.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads)
