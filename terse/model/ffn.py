"""ReLU-squared gated feed-forward network with ternary projections."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from terse.model.config import TerseConfig
from terse.model.ternary import TernaryLinear


class TerseFFN(nn.Module):
    def __init__(self, config: TerseConfig) -> None:
        super().__init__()
        self.gate_proj = TernaryLinear(config.hidden_dim, config.ffn_intermediate)
        self.up_proj = TernaryLinear(config.hidden_dim, config.ffn_intermediate)
        self.down_proj = TernaryLinear(config.ffn_intermediate, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.relu(self.gate_proj(x)).pow(2)  # ReLU^2
        return self.down_proj(gate * self.up_proj(x))
