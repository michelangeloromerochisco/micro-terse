"""Grouped-query attention with QK-Norm applied before RoPE."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from terse.model.config import TerseConfig
from terse.model.rmsnorm import RMSNorm
from terse.model.rope import RotaryEmbedding
from terse.model.ternary import TernaryLinear


class TerseAttention(nn.Module):
    def __init__(self, config: TerseConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.num_heads // config.num_kv_heads
        self.qk_norm = config.qk_norm

        self.q_proj = TernaryLinear(config.hidden_dim, config.num_heads * config.head_dim)
        self.k_proj = TernaryLinear(config.hidden_dim, config.num_kv_heads * config.head_dim)
        self.v_proj = TernaryLinear(config.hidden_dim, config.num_kv_heads * config.head_dim)
        self.o_proj = TernaryLinear(config.num_heads * config.head_dim, config.hidden_dim)

        if config.qk_norm:
            self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope.apply(q)
        k = self.rope.apply(k)

        # GQA: replicate KV heads to match query heads
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)
