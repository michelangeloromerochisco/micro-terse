"""Model and training configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TerseConfig:
    hidden_dim: int = 1536
    num_layers: int = 16
    num_heads: int = 12
    num_kv_heads: int = 3
    head_dim: int = 128
    ffn_intermediate: int = 4096

    # MoE
    num_experts: int = 4
    top_k: int = 2
    moe_layers: Optional[List[int]] = None  # default: odd indices
    moe_bias_ema_gamma: float = 0.001

    # Vocab & embedding
    vocab_size: int = 128256
    tie_embeddings: bool = True

    # Sequence
    max_seq_len: int = 4096
    rope_theta: float = 500000.0

    # Normalization
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True

    # Ternary
    ternary_weights: bool = True
    fogzo: bool = True

    # MTP
    mtp_heads: int = 1
    mtp_loss_weight: float = 0.1

    # Training
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.moe_layers is None:
            self.moe_layers = [i for i in range(self.num_layers) if i % 2 == 1]
        assert self.num_heads * self.head_dim == self.hidden_dim, (
            f"num_heads*head_dim ({self.num_heads}*{self.head_dim}) must equal "
            f"hidden_dim ({self.hidden_dim})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.top_k <= self.num_experts


@dataclass
class TrainingConfig:
    tokens_total: int = 8_000_000_000
    batch_size: int = 4
    seq_len: int = 4096
    total_steps: int = 488282

    lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    grad_clip: float = 1.0

    warmup_steps: int = 2000
    precision: str = "bfloat16"

    # FP16 cooldown (forward-compat for Mini; OFF for Micro/A6000)
    cooldown_enabled: bool = False
    cooldown_start_frac: float = 0.8
    cooldown_dtype: str = "float16"

    # Cross-entropy memory escape hatch (0 = disabled, full CE)
    ce_chunk_size: int = 0

    save_every_steps: int = 2000
    eval_every_steps: int = 5000
    log_every_steps: int = 10
    save_dir: str = "checkpoints"
    keep_last: int = 3
    milestone_every: int = 50000
    # Set False to write model-only checkpoints (~3x smaller; resume reinitializes
    # the optimizer/scheduler instead of restoring momentum). Useful on small volumes.
    save_optimizer: bool = True

    # Logging (optional; empty string = disabled, so defaults keep tests dependency-free)
    wandb_project: str = ""
    wandb_run_name: str = ""
    log_file: str = ""
