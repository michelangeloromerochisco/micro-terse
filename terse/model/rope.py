"""Rotary position embeddings via complex multiplication (RoPE)."""
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 500000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim/2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, H, head_dim) -> rotated (B, S, H, head_dim)."""
        b, s, h, d = x.shape
        input_dtype = x.dtype
        x_ = x.float().reshape(b, s, h, d // 2, 2)
        x_complex = torch.view_as_complex(x_)              # (B, S, H, d/2)
        freqs = self.freqs_cis[:s].view(1, s, 1, d // 2)   # broadcast over B, H
        x_rotated = torch.view_as_real(x_complex * freqs)  # (B, S, H, d/2, 2)
        return x_rotated.reshape(b, s, h, d).to(input_dtype)
