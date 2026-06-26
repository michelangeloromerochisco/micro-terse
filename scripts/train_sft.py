"""Full fine-tune (SFT) of a Micro-Terse checkpoint on ChatML instruction data.

FULL fine-tuning (not LoRA): the fine-tune trains the ternary latent weights via
the STE, so the result is a new set of ternary weights -> a single merged GGUF,
with NO LoRA-merge rounding loss (avoids the Orchid/ternative problem).

Loss is masked to the assistant response(s) (every non-assistant token gets label
-100, which cross-entropy ignores). Multi-turn conversations are supported: every
assistant turn is supervised, all user/system turns are masked context.

Optimization:
  --optimizer adafactor   near-zero optimizer memory; fits a 4 GB GPU (PC test)
  --optimizer adamw       standard; use on the pod (48 GB) for the real run
  --grad-accum N          accumulate N micro-batches per optimizer step (effective
                          batch size N). The model is hard-causal with no padding
                          mask, so true minibatching is unavailable; grad-accum is
                          the correct way to get large-batch updates without padding.
  --warmup-ratio / cosine LR schedule applied per OPTIMIZER step.

    # PC viability test (small, junk quality on an early base):
    python scripts/train_sft.py --checkpoint W.pt --config micro.yaml \
        --device cuda --dtype bf16 --optimizer adafactor --steps 30 --seq-len 256
    # pod real run:
    python scripts/train_sft.py --checkpoint final_pretrain.pt --config micro.yaml \
        --device cuda --optimizer adamw --epochs 3 --grad-accum 16 --seq-len 1024 \
        --warmup-ratio 0.03 --save-every 500 --data sft_corpus.jsonl --out sft.pt
"""
import argparse
import json
import math

import torch

from terse.model.config import TerseConfig
from terse.model.generate import REASONING_END, REASONING_START  # noqa: F401 (kept for parity)
from terse.model.terse_model import TerseModel

# Tiny built-in instruction set for the viability test (quality irrelevant).
_DEMO_SFT = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Translate 'hello' to Spanish.", "'Hello' in Spanish is 'hola'."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("Name a primary color.", "Red is a primary color."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What is water made of?", "Water is made of hydrogen and oxygen (H2O)."),
    ("Give a synonym for happy.", "A synonym for happy is joyful."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
]

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def _load_data(path: str | None):
    """Return a list of conversations, each a list of {"role","content"} dicts.

    Accepts ChatML jsonl with a "messages" field (single- or multi-turn). Lines
    lacking any assistant turn are dropped (nothing to supervise). Omit --data to
    use the built-in demo set.
    """
    if not path:
        return [[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                for q, a in _DEMO_SFT]
    convs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            if any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
                convs.append(msgs)
    return convs


def encode_conversation(tok, messages, seq_len: int):
    """Build (input_ids, labels) for a ChatML conversation.

    Each turn renders as ``<|im_start|>{role}\\n{content}<|im_end|>\\n``. Assistant
    turn bodies (content + end marker) are supervised; the assistant header and all
    user/system tokens are masked with -100. Returns 1-D python lists (caller tensors).
    """
    ids: list[int] = []
    labels: list[int] = []
    first = True
    for m in messages:
        role, content = m["role"], m["content"]
        header = f"{IM_START}{role}\n"
        h = tok.encode(header, add_special_tokens=first)
        first = False
        ids += h
        labels += [-100] * len(h)
        body = f"{content}{IM_END}\n"
        b = tok.encode(body, add_special_tokens=False)
        ids += b
        labels += b if role == "assistant" else [-100] * len(b)
    return ids[:seq_len], labels[:seq_len]


def _tensors(ids, labels, device):
    return (torch.tensor([ids], device=device), torch.tensor([labels], device=device))


def lr_at(step: int, total: int, warmup: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Linear warmup then cosine decay to min_lr, computed per optimizer step."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return base_lr
    p = (step - warmup) / (total - warmup)
    p = min(1.0, max(0.0, p))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def main() -> int:
    ap = argparse.ArgumentParser(description="Micro-Terse full-FT SFT")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None, help="ChatML jsonl; omit to use the built-in demo set")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03, help="fraction of optimizer steps spent warming up")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step (effective batch size)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="cap total OPTIMIZER steps (0 = all epochs)")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N optimizer steps (0 = only at end)")
    ap.add_argument("--out", default=None, help="save fine-tuned checkpoint here")
    args = ap.parse_args()

    import yaml

    cfg_raw = yaml.safe_load(open(args.config))["model"]
    cfg_raw["gradient_checkpointing"] = True
    cfg = TerseConfig(**cfg_raw)
    model = TerseModel(cfg)
    # Accept stripped {"model": ...}, full training checkpoint, or raw state_dict.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = model.to(device=args.device, dtype=dtype)
    model.train()

    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False,
                        relative_step=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")

    data = _load_data(args.data)
    ga = max(1, args.grad_accum)
    micro_per_epoch = len(data)
    planned_opt_steps = (micro_per_epoch * args.epochs) // ga
    total_opt_steps = args.steps if args.steps else planned_opt_steps
    warmup = int(total_opt_steps * args.warmup_ratio)
    print(f"SFT: {len(data)} convs, grad_accum={ga}, opt_steps={total_opt_steps} "
          f"(warmup {warmup}), optimizer={args.optimizer}, dtype={args.dtype}, "
          f"seq_len={args.seq_len}, device={args.device}", flush=True)

    def _save(tag_step):
        if args.out:
            torch.save({"model": model.state_dict(), "step": tag_step}, args.out)
            print(f"saved {args.out} @ opt_step {tag_step}", flush=True)

    opt_step = 0
    micro = 0
    opt.zero_grad(set_to_none=True)
    accum_loss = 0.0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for conv in data:
            ids, labels = encode_conversation(tok, conv, args.seq_len)
            if not any(t != -100 for t in labels):
                continue  # nothing supervised (truncation dropped the assistant turn)
            input_ids, label_ids = _tensors(ids, labels, args.device)
            out = model(input_ids, labels=label_ids, return_logits=False)
            if not torch.isfinite(out["loss"]):
                print(f"SKIP non-finite loss (epoch {epoch})", flush=True)
                continue
            loss = out["loss"] / ga
            loss.backward()
            accum_loss += out["loss"].item()
            micro += 1

            if micro % ga == 0:
                lr = lr_at(opt_step, total_opt_steps, warmup, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gnorm):
                    print(f"SKIP non-finite grad @ opt_step {opt_step}", flush=True)
                    opt.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    continue
                opt.step()
                opt.zero_grad(set_to_none=True)
                model.step_moe_bias()  # aux-free MoE balancing, once per optimizer step
                opt_step += 1
                mem = (torch.cuda.max_memory_allocated() / 1e9) if args.device == "cuda" else 0.0
                print(f"opt_step {opt_step}/{total_opt_steps} epoch {epoch} "
                      f"loss {accum_loss/ga:.4f} lr {lr:.2e} peakGPU {mem:.2f}GB", flush=True)
                accum_loss = 0.0
                if args.save_every and opt_step % args.save_every == 0:
                    _save(opt_step)
                if args.steps and opt_step >= args.steps:
                    stop = True
                    break

    _save(opt_step)
    print("SFT TEST OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
