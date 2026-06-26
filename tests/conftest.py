"""Shared pytest fixtures."""
import pytest

from terse.model.config import TerseConfig, TrainingConfig


@pytest.fixture
def tiny_config() -> TerseConfig:
    """Small model that runs in seconds on CPU. Layer 0 dense, layer 1 MoE."""
    return TerseConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_intermediate=128,
        num_experts=4,
        top_k=2,
        moe_layers=[1],
        vocab_size=256,
        max_seq_len=64,
        gradient_checkpointing=False,
    )


@pytest.fixture
def tiny_train_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=2,
        seq_len=32,
        total_steps=50,
        warmup_steps=5,
        lr=1e-3,
        save_every_steps=1000,
        eval_every_steps=1000,
        log_every_steps=1,
    )
