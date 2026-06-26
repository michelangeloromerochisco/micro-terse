import torch

from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel


def _batch(cfg, b=2, s=16):
    return {"input_ids": torch.randint(0, cfg.vocab_size, (b, s))}


def test_single_step_finite(tiny_config, tiny_train_config):
    from terse.training.trainer import Trainer
    model = TerseModel(tiny_config)
    trainer = Trainer(model, [], tiny_train_config, device="cpu")
    stats = trainer.train_step(_batch(tiny_config), step=0)
    assert torch.isfinite(torch.tensor(stats["loss"]))
    assert torch.isfinite(torch.tensor(stats["grad_norm"]))


def test_overfit_decreases_loss(tiny_config, tiny_train_config):
    from terse.training.trainer import Trainer
    torch.manual_seed(0)
    model = TerseModel(tiny_config)
    trainer = Trainer(model, [], tiny_train_config, device="cpu")
    batch = _batch(tiny_config)
    first = trainer.train_step(batch, 0)["loss"]
    for s in range(1, 10):
        last = trainer.train_step(batch, s)["loss"]
    assert last < first


def test_grad_checkpointing_same_loss(tiny_config):
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))

    cfg_off = TerseConfig(**{**tiny_config.__dict__, "gradient_checkpointing": False})
    cfg_on = TerseConfig(**{**tiny_config.__dict__, "gradient_checkpointing": True})

    torch.manual_seed(1)
    m_off = TerseModel(cfg_off).train()
    torch.manual_seed(1)
    m_on = TerseModel(cfg_on).train()

    loss_off = m_off(ids, labels=ids, return_logits=False)["loss"]
    loss_on = m_on(ids, labels=ids, return_logits=False)["loss"]
    assert torch.allclose(loss_off, loss_on, atol=1e-4)
