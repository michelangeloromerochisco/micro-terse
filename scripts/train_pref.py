"""Preference fine-tuning (DPO or ORPO) for a Micro-Terse checkpoint.

Both are FULL fine-tuning -> a single merged ternary model (no LoRA, no Orchid
re-quantization issue).
  --method orpo : reference-free. loss = NLL(chosen) + lambda * odds-ratio term.
                  Lighter (no reference model); one combined align stage.
  --method dpo  : needs a frozen reference (a copy of the start checkpoint).
                  loss = -log sigmoid(beta * [(pol_c-ref_c) - (pol_r-ref_r)]).
                  Heavier (2 model copies in memory).

Prints peak GPU and the chosen-vs-rejected logp MARGIN each step (margin rising
above 0 = the method is actually pushing chosen over rejected = working).

    python scripts/train_pref.py --method orpo --checkpoint W.pt --config micro.yaml \
        --device cuda --dtype bf16 --steps 24
"""
import argparse
import copy
import json

import torch
import torch.nn.functional as F
import yaml

from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel

_DEMO_PREF = [
    ("What is the capital of France?", "The capital of France is Paris.", "Maybe it is Berlin."),
    ("Is the earth flat?", "No, the Earth is an oblate spheroid.", "Yes, the earth is flat."),
    ("What is 2+2?", "2 + 2 equals 4.", "2 + 2 equals 5."),
    ("Recommend a healthy breakfast.", "Oatmeal with fruit is healthy.", "Just drink soda all day."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted it.", "It was painted by Picasso."),
    ("How do I stay safe online?", "Use strong passwords and 2FA.", "Share your password with everyone."),
]
PROMPT_TMPL = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
RESP_TMPL = "{a}<|im_end|>"


def _seq(tok, q, a, seq_len, device):
    p = tok.encode(PROMPT_TMPL.format(q=q), add_special_tokens=True)
    r = tok.encode(RESP_TMPL.format(a=a), add_special_tokens=False)
    ids = (p + r)[:seq_len]
    return torch.tensor([ids], device=device), len(p)


def _resp_logp(model, ids, resp_start):
    """Return (sum_logp, mean_logp) of the response tokens under `model`."""
    logits = model(ids, return_logits=True)["logits"][0]      # (T, V)
    logp = F.log_softmax(logits[:-1].float(), dim=-1)          # (T-1, V)
    targets = ids[0, 1:]                                       # (T-1,)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    resp = tok_logp[resp_start - 1:]                           # response-token logps
    return resp.sum(), resp.mean()


def _load_data(path):
    if not path:
        return _DEMO_PREF
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append((o["prompt"], o["chosen"], o["rejected"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Micro-Terse DPO/ORPO full-FT")
    ap.add_argument("--method", choices=["dpo", "orpo"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)     # DPO
    ap.add_argument("--lam", type=float, default=0.5)      # ORPO odds-ratio weight
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(open(args.config))["model"]
    cfg_raw["gradient_checkpointing"] = True
    cfg = TerseConfig(**cfg_raw)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    def _build():
        m = TerseModel(cfg)
        st = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        m.load_state_dict(st["model"] if isinstance(st, dict) and "model" in st else st)
        return m.to(device=args.device, dtype=dtype)

    model = _build()
    model.train()
    ref = None
    if args.method == "dpo":
        ref = _build()
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False,
                        relative_step=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
    data = _load_data(args.data)
    print(f"{args.method.upper()}: {len(data)} pairs, opt={args.optimizer}, dtype={args.dtype}, "
          f"device={args.device}", flush=True)

    step = 0
    for epoch in range(args.epochs):
        for q, ch, rj in data:
            ids_c, rs_c = _seq(tok, q, ch, args.seq_len, args.device)
            ids_r, rs_r = _seq(tok, q, rj, args.seq_len, args.device)

            if ids_c.shape[1] <= rs_c or ids_r.shape[1] <= rs_r:
                continue  # response truncated away -> empty target = nan
            sum_c, mean_c = _resp_logp(model, ids_c, rs_c)
            sum_r, mean_r = _resp_logp(model, ids_r, rs_r)

            if args.method == "orpo":
                # odds-ratio on length-normalized probs; NLL anchors the chosen.
                lo_c = mean_c - torch.log1p(-torch.exp(mean_c).clamp(max=1 - 1e-6))
                lo_r = mean_r - torch.log1p(-torch.exp(mean_r).clamp(max=1 - 1e-6))
                or_term = F.logsigmoid(lo_c - lo_r)
                loss = -mean_c + args.lam * (-or_term)
            else:  # dpo
                with torch.no_grad():
                    rsum_c, _ = _resp_logp(ref, ids_c, rs_c)
                    rsum_r, _ = _resp_logp(ref, ids_r, rs_r)
                logits = args.beta * ((sum_c - rsum_c) - (sum_r - rsum_r))
                loss = -F.logsigmoid(logits)

            if not torch.isfinite(loss):
                print(f"step {step+1} ep {epoch} SKIP non-finite loss", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                print(f"step {step+1} ep {epoch} SKIP non-finite grad", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            model.step_moe_bias()
            step += 1
            margin = (mean_c - mean_r).item()  # per-token logp gap (length-normalized)
            mem = (torch.cuda.max_memory_allocated() / 1e9) if args.device == "cuda" else 0.0
            print(f"step {step} ep {epoch} loss {loss.item():.4f} margin(chosen-rej) "
                  f"{margin:+.3f} peakGPU {mem:.2f}GB", flush=True)
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break

    if args.out:
        torch.save({"model": model.state_dict(), "step": step}, args.out)
        print("saved", args.out, flush=True)
    print(f"{args.method.upper()} TEST OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
