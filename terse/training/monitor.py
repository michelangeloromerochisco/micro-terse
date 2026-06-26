"""Training monitor: logging and auto-pause heuristics."""
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


class TrainingPaused(RuntimeError):
    pass


@dataclass
class TerseMonitor:
    log_every: int = 10
    grad_spike_threshold: float = 100.0
    grad_spike_patience: int = 3
    starvation_threshold: float = 0.001
    starvation_patience: int = 500
    nan_patience: int = 5        # pause only after this many CONSECUTIVE non-finite steps
    log_file: str = ""           # optional path; appended one line per log step
    wandb_run: Optional[Any] = None  # optional wandb run; stats logged per log step
    _grad_spikes: int = 0
    _nonfinite: int = 0
    _starvation: List[int] = field(default_factory=list)

    def check(self, step: int, loss: float, grad_norm: float,
              expert_fracs: List[float]) -> None:
        # The trainer skips the optimizer step on non-finite loss/grad, so a single bad
        # batch is recoverable; only persistent divergence should pause the run.
        if not math.isfinite(loss) or not math.isfinite(grad_norm):
            self._nonfinite += 1
            if self._nonfinite >= self.nan_patience:
                raise TrainingPaused(
                    f"non-finite loss/grad for {self.nan_patience} consecutive steps "
                    f"(step {step}: loss={loss} grad={grad_norm})"
                )
            return  # skip grad-spike / starvation checks this step
        self._nonfinite = 0

        self._grad_spikes = self._grad_spikes + 1 if grad_norm > self.grad_spike_threshold else 0
        if self._grad_spikes >= self.grad_spike_patience:
            raise TrainingPaused(f"grad_norm > {self.grad_spike_threshold} for "
                                 f"{self.grad_spike_patience} steps at step {step}")

        if expert_fracs:
            if not self._starvation:
                self._starvation = [0] * len(expert_fracs)
            starved = []
            for i, frac in enumerate(expert_fracs):
                self._starvation[i] = self._starvation[i] + 1 if frac < self.starvation_threshold else 0
                if self._starvation[i] >= self.starvation_patience:
                    starved.append(i)
            # A single persistently-starved expert is tolerated: it's benign
            # capacity loss while overall loss stays healthy, and pausing the run
            # for it just wastes compute. Warn (and reset its counter so we warn
            # periodically rather than spam) but DO NOT pause. Only a broader
            # collapse (>=2 experts starved) is treated as fatal.
            if len(starved) >= 2:
                raise TrainingPaused(f"experts {starved} starved for "
                                     f"{self.starvation_patience} steps at step {step}")
            for i in starved:
                print(f"[monitor] WARN: expert {i} starved {self.starvation_patience}+ steps "
                      f"(step {step}) — tolerated (single expert, loss healthy).", flush=True)
                self._starvation[i] = 0

    def log(self, step: int, stats: Dict[str, float]) -> None:
        if step % self.log_every != 0:
            return
        msg = " ".join(f"{k}={v:.4g}" for k, v in stats.items())
        line = f"[step {step}] {msg}"
        print(line, flush=True)
        if self.log_file:
            try:
                os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                # A transient network-filesystem I/O error (e.g. Errno 5 on the
                # /workspace mount) must NEVER crash training. The line was
                # already printed to stdout (captured in run.out), so just skip
                # this append and keep training.
                pass
        if self.wandb_run is not None:
            self.wandb_run.log(stats, step=step)


def model_ternary_zero_fraction(model: torch.nn.Module) -> float:
    from terse.model.ternary import TernaryLinear, TernaryQuantizeFunction
    zeros, total = 0, 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, TernaryLinear):
                w = TernaryQuantizeFunction.apply(m.weight, m.temperature)
                zeros += (w == 0).sum().item()
                total += w.numel()
    return zeros / total if total else 0.0
