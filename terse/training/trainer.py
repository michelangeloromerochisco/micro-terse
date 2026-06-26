"""Single-GPU training loop for Micro-Terse."""
import json
import os
import time
from typing import Iterator

import torch

from terse.model.config import TrainingConfig
from terse.training.checkpoint import (
    list_checkpoints,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from terse.training.monitor import (
    TerseMonitor,
    TrainingPaused,
    model_ternary_zero_fraction,
)
from terse.training.optimizer import build_optimizer
from terse.training.scheduler import build_scheduler

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class Trainer:
    def __init__(self, model, dataloader, cfg: TrainingConfig, device: str = "cuda"):
        # This loop autocasts without a GradScaler — correct for bf16/fp32 (no loss
        # scaling needed) but NOT fp16, which would silently underflow to divergence.
        # Fail loudly instead of training a doomed run for hours.
        if cfg.precision == "float16" or (
            cfg.cooldown_enabled and cfg.cooldown_dtype == "float16"
        ):
            raise ValueError(
                "float16 training/cooldown needs a GradScaler, which is not wired in this "
                "trainer. Use precision: bfloat16 (the A6000 supports it natively). "
                "An fp16 path with GradScaler is a Mini-scale TODO."
            )
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.model.ce_chunk_size = cfg.ce_chunk_size
        self.dataloader = dataloader
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.wandb_run = self._init_wandb(cfg)
        self.monitor = TerseMonitor(
            log_every=cfg.log_every_steps,
            log_file=cfg.log_file,
            wandb_run=self.wandb_run,
        )
        self.base_dtype = _DTYPES[cfg.precision]
        self.start_step = 0
        self._tok_s_ema: float = 0.0   # smoothed throughput; a single slow save step
        #                                shouldn't make the heartbeat look like a collapse.
        self._tok_s_seeded: bool = False  # drop the cold first step (compile/loader spin-up)

    @staticmethod
    def _init_wandb(cfg: TrainingConfig):
        """Initialize a wandb run if configured and available; otherwise return None."""
        if not cfg.wandb_project:
            return None
        try:
            import wandb
        except ImportError:
            print("[trainer] wandb_project set but wandb is not installed; "
                  "logging to stdout/file only.", flush=True)
            return None
        try:
            return wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name or None,
                config=vars(cfg),
            )
        except Exception as e:  # auth/network failures must not kill training
            print(f"[trainer] wandb.init failed ({e}); logging to stdout/file only.",
                  flush=True)
            return None

    def _autocast_dtype(self, step: int) -> torch.dtype:
        if self.cfg.cooldown_enabled:
            threshold = int(self.cfg.cooldown_start_frac * self.cfg.total_steps)
            if step >= threshold:
                return _DTYPES[self.cfg.cooldown_dtype]
        return self.base_dtype

    def maybe_resume(self) -> None:
        # Try checkpoints newest-first; if the latest is unreadable (e.g. truncated by
        # a kill mid-save), fall back to the next-older one rather than crashing.
        for ckpt in list_checkpoints(self.cfg.save_dir):
            try:
                self.start_step = load_checkpoint(
                    ckpt, self.model, self.optimizer, self.scheduler
                ) + 1
                print(f"resumed from {ckpt} at step {self.start_step}", flush=True)
                return
            except Exception as e:
                print(f"[trainer] could not load {ckpt} ({e}); trying an older checkpoint",
                      flush=True)
        print("[trainer] no loadable checkpoint found; starting from step 0", flush=True)

    def _expert_fracs(self) -> list:
        from terse.model.moe import TerseMoE
        fracs = []
        for m in self.model.modules():
            if isinstance(m, TerseMoE):
                ema = m.router.expert_counts_ema
                total = ema.sum().item()
                if total > 0:
                    fracs.extend((ema / total).tolist())
        return fracs

    def train_step(self, batch: dict, step: int) -> dict:
        input_ids = batch["input_ids"].to(self.device)
        labels = input_ids.clone()
        use_amp = self.device.startswith("cuda")
        with torch.amp.autocast(
            "cuda", dtype=self._autocast_dtype(step), enabled=use_amp
        ):
            out = self.model(input_ids, labels=labels, return_logits=False)
            loss = out["loss"]

        skipped = False
        if not torch.isfinite(loss):
            # Never backprop a non-finite loss into the weights — drop this batch.
            self.optimizer.zero_grad(set_to_none=True)
            grad_norm = float("nan")
            skipped = True
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            if torch.isfinite(grad_norm):
                self.optimizer.step()
            else:
                skipped = True  # inf/nan grads: drop the step, leave weights intact
            self.optimizer.zero_grad(set_to_none=True)
        # Apply the MoE aux-free bias update now — after backward (so checkpoint recompute
        # has finished and the routing bias stayed constant through the whole step).
        self.model.step_moe_bias()
        self.scheduler.step()  # advance LR schedule regardless, to stay on plan
        return {
            "loss": loss.item(),
            "main_loss": out["main_loss"].item(),
            "mtp_loss": out["mtp_loss"].item(),
            "grad_norm": float(grad_norm),
            "lr": self.scheduler.get_last_lr()[0],
            "skipped": float(skipped),
        }

    def train(self) -> None:
        self.maybe_resume()
        self.model.train()
        data_iter = _cycle(self.dataloader)
        tokens_per_step = self.cfg.batch_size * self.cfg.seq_len
        step = self.start_step
        last = time.time()
        try:
            for step in range(self.start_step, self.cfg.total_steps):
                stats = self.train_step(next(data_iter), step)
                now = time.time()
                dt = now - last
                last = now
                inst_tok_s = tokens_per_step / dt if dt > 0 else 0.0
                if not self._tok_s_seeded:
                    # The first step of a (re)started process includes CUDA compile and
                    # dataloader worker spin-up — not representative. Report 0 (= "not
                    # measured"; the watchdog ignores <=0) and start the EMA next step.
                    self._tok_s_seeded = True
                    stats["tok_s"] = 0.0
                else:
                    # EMA over steps so a ~13GB checkpoint write (one slow step) doesn't
                    # read as a throughput collapse to the watchdog.
                    self._tok_s_ema = (
                        inst_tok_s if self._tok_s_ema == 0.0
                        else 0.9 * self._tok_s_ema + 0.1 * inst_tok_s
                    )
                    stats["tok_s"] = self._tok_s_ema
                fracs = self._expert_fracs()
                self.monitor.check(step, stats["loss"], stats["grad_norm"], fracs)
                # zero_frac requires a full re-quantization pass over every TernaryLinear,
                # so only compute it on steps the monitor will actually log.
                if step % self.cfg.log_every_steps == 0:
                    stats["zero_frac"] = model_ternary_zero_fraction(self.model)
                self.monitor.log(step, stats)
                self._write_status(step, stats, fracs)  # heartbeat for the watchdog
                if step > 0 and step % self.cfg.save_every_steps == 0:
                    self._save(step, stats["loss"])
        except TrainingPaused as e:
            self._emergency_save(step, e)
            raise

    def _write_status(self, step: int, stats: dict, fracs: list) -> None:
        """Atomically write a heartbeat the external watchdog reads. Frozen `ts` ==
        a hung/dead process; that's how stalls are detected without an in-process check."""
        status = {
            "step": step,
            "total_steps": self.cfg.total_steps,
            "loss": stats.get("loss"),
            "grad_norm": stats.get("grad_norm"),
            "lr": stats.get("lr"),
            "tok_s": stats.get("tok_s"),
            "skipped": stats.get("skipped"),
            "zero_frac": stats.get("zero_frac"),
            "expert_fracs": fracs,
            "ts": time.time(),
            "pid": os.getpid(),
        }
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        path = os.path.join(self.cfg.save_dir, "status.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f)
        os.replace(tmp, path)

    def _save(self, step: int, loss: float) -> None:
        path = os.path.join(self.cfg.save_dir, f"step_{step}.pt")
        save_checkpoint(
            path, self.model, self.optimizer, self.scheduler, step, loss,
            save_optimizer=self.cfg.save_optimizer,
        )
        prune_checkpoints(self.cfg.save_dir, self.cfg.keep_last, self.cfg.milestone_every)

    def _emergency_save(self, step: int, exc: Exception) -> None:
        """On a monitor pause, dump an inspectable checkpoint (not named step_*.pt, so
        resume keeps using the last clean one)."""
        path = os.path.join(self.cfg.save_dir, f"emergency_step_{step}.pt")
        try:
            save_checkpoint(
                path, self.model, self.optimizer, self.scheduler, step, float("nan"),
                save_optimizer=self.cfg.save_optimizer,
            )
            print(f"[trainer] PAUSED: {exc}. Emergency checkpoint -> {path}", flush=True)
        except Exception as save_err:
            print(f"[trainer] PAUSED: {exc}. Emergency save FAILED: {save_err}", flush=True)


def _cycle(loader: Iterator) -> Iterator:
    while True:
        for batch in loader:
            yield batch
