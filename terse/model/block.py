"""Pre-norm transformer block: attention + (dense FFN or MoE)."""
import torch
import torch.nn as nn

from terse.model.attention import TerseAttention
from terse.model.config import TerseConfig
from terse.model.ffn import TerseFFN
from terse.model.moe import TerseMoE
from terse.model.rmsnorm import RMSNorm


class TerseBlock(nn.Module):
    def __init__(self, config: TerseConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attn = TerseAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn = (
            TerseMoE(config) if layer_idx in config.moe_layers else TerseFFN(config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        return h + self.ffn(self.ffn_norm(h))
