"""Full Micro-Terse model: embeddings, transformer stack, tied head, MTP, loss."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from terse.model.block import TerseBlock
from terse.model.config import TerseConfig
from terse.model.mtp import MTPHead
from terse.model.rmsnorm import RMSNorm


def chunked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Memory-friendly CE over the vocab dim. logits: (M, V), targets: (M,)."""
    if chunk_size <= 0:
        return F.cross_entropy(logits.float(), targets)
    total = logits.new_zeros(())
    count = targets.numel()
    for start in range(0, count, chunk_size):
        end = start + chunk_size
        total = total + F.cross_entropy(
            logits[start:end].float(), targets[start:end], reduction="sum"
        )
    return total / count


class TerseModel(nn.Module):
    def __init__(self, config: TerseConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.layers = nn.ModuleList(
            [TerseBlock(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.mtp_head = MTPHead(config)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.ce_chunk_size = 0  # set by trainer from TrainingConfig if desired

    def step_moe_bias(self) -> None:
        """Apply the aux-free MoE bias update once per training step. MUST run AFTER
        backward: gradient checkpointing re-runs each block's forward during backward, so
        the routing bias has to stay constant across a step's forward+recompute. Mutating
        it mid-step changes routing on recompute and crashes checkpointing (shape mismatch)."""
        from terse.model.moe import TerseMoE
        for m in self.modules():
            if isinstance(m, TerseMoE):
                m.router.apply_bias_update()

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return chunked_cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            self.ce_chunk_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> dict:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm(x)

        logits = F.linear(x, self.embed_tokens.weight)  # tied LM head

        loss = main_loss = mtp_loss = None
        if labels is not None:
            main_loss = self._ce(logits[:, :-1], labels[:, 1:])
            kept_logits = logits if return_logits else None
            if not return_logits:
                del logits  # free ~vocab-sized tensor before MTP logits materialize
            mtp_logits = self.mtp_head(x, self.embed_tokens.weight)
            mtp_loss = self._ce(mtp_logits[:, :-2], labels[:, 2:])
            loss = main_loss + self.config.mtp_loss_weight * mtp_loss
            logits = kept_logits

        return {
            "logits": logits if return_logits else None,
            "loss": loss,
            "main_loss": main_loss,
            "mtp_loss": mtp_loss,
        }
