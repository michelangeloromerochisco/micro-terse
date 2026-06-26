"""Multi-token prediction head sharing the tied embedding weight."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from terse.model.config import TerseConfig
from terse.model.rmsnorm import RMSNorm


class MTPHead(nn.Module):
    def __init__(self, config: TerseConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor):
        h = self.norm(self.proj(hidden_states))
        return F.linear(h, lm_head_weight)
