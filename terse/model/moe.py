"""Mixture-of-experts with DeepSeek-V3 aux-free bias-EMA load balancing."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from terse.model.config import TerseConfig
from terse.model.ffn import TerseFFN


class MoERouter(nn.Module):
    def __init__(self, config: TerseConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.gamma = config.moe_bias_ema_gamma
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        # learnable-by-EMA, never gradient-trained
        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts), requires_grad=False
        )
        self.register_buffer("expert_counts_ema", torch.zeros(config.num_experts))
        # This step's routing load, recorded in forward, consumed once after backward.
        self._last_counts: torch.Tensor | None = None
        self._last_tokens: int = 0

    def forward(self, x: torch.Tensor):
        """x: (N, D) -> (topk_weights (N, top_k), topk_indices (N, top_k))."""
        logits = self.gate(x) + self.expert_bias
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights.float(), dim=-1).to(x.dtype)
        return topk_weights, topk_indices

    @torch.no_grad()
    def record_counts(self, topk_indices: torch.Tensor, num_tokens: int) -> None:
        """Record this step's per-expert token counts. OVERWRITES (never accumulates)
        so it is idempotent across gradient-checkpointing's save + recompute forward
        passes: identical bias -> identical routing -> identical counts, so both passes
        store the same value. The bias is NOT touched here — mutating it mid-forward
        would change routing on recompute and crash checkpointing (shape mismatch)."""
        self._last_counts = torch.bincount(
            topk_indices.flatten(), minlength=self.num_experts
        ).float()
        self._last_tokens = num_tokens

    @torch.no_grad()
    def apply_bias_update(self) -> None:
        """Apply the aux-free balancing update from the recorded counts. Called once per
        step by the trainer AFTER backward, so the bias never changes during a step's
        forward/recompute. Overloaded experts (count > target) get their bias pushed
        down, underloaded pushed up — responds immediately, no EMA warm-up lag."""
        if self._last_counts is None:
            return
        counts = self._last_counts
        # EMA is kept for monitoring/starvation detection only.
        self.expert_counts_ema.mul_(1 - self.gamma).add_(self.gamma * counts)
        target = self._last_tokens * self.top_k / self.num_experts
        self.expert_bias.data -= self.gamma * torch.sign(counts - target)


class TerseMoE(nn.Module):
    def __init__(self, config: TerseConfig) -> None:
        super().__init__()
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        self.router = MoERouter(config)
        self.experts = nn.ModuleList(
            [TerseFFN(config) for _ in range(config.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        x_flat = x.reshape(-1, d)
        n = x_flat.shape[0]
        topk_weights, topk_indices = self.router(x_flat)

        flat_idx = topk_indices.reshape(-1)            # (N*top_k,)
        flat_w = topk_weights.reshape(-1, 1)           # (N*top_k, 1)
        token_idx = torch.arange(n, device=x.device).repeat_interleave(self.top_k)

        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            sel = flat_idx == e
            if sel.any():
                tok = token_idx[sel]
                contrib = flat_w[sel] * self.experts[e](x_flat[tok])
                out.index_add_(0, tok, contrib)

        if self.training:
            self.router.record_counts(topk_indices, n)
        return out.reshape(b, s, d)
