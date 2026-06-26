import math

import torch

from terse.model.terse_model import TerseModel


def test_logits_shape(tiny_config):
    model = TerseModel(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = model(ids)
    assert out["logits"].shape == (2, 16, tiny_config.vocab_size)


def test_loss_is_scalar_finite_positive(tiny_config):
    model = TerseModel(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    loss = out["loss"]
    assert loss.dim() == 0 and torch.isfinite(loss) and loss.item() > 0


def test_mtp_contributes(tiny_config):
    model = TerseModel(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    assert out["mtp_loss"] is not None and out["mtp_loss"].item() > 0
    expected = out["main_loss"] + tiny_config.mtp_loss_weight * out["mtp_loss"]
    assert torch.allclose(out["loss"], expected)


def test_all_params_get_grad(tiny_config):
    # `temperature` params are intentionally NOT loss-trained: FOGZO uses temperature to
    # scale the weight gradient, but the quantize forward does not depend on it, so it
    # receives no gradient by design (TernaryQuantizeFunction.backward returns None for
    # it). This test guards against *accidental* graph disconnection, so every OTHER
    # trainable param must receive a non-zero gradient.
    model = TerseModel(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    model(ids, labels=ids)["loss"].backward()
    missing = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "temperature" not in n
        and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    assert not missing, f"params without grad: {missing}"


def test_initial_loss_near_ln_vocab(tiny_config):
    torch.manual_seed(0)
    model = TerseModel(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (4, 32))
    loss = model(ids, labels=ids)["main_loss"].item()
    ref = math.log(tiny_config.vocab_size)
    assert ref * 0.6 < loss < ref * 1.5


def test_tied_embeddings_no_separate_lm_head(tiny_config):
    model = TerseModel(tiny_config)
    names = dict(model.named_parameters())
    assert not any("lm_head" in n for n in names)
    assert "embed_tokens.weight" in names
