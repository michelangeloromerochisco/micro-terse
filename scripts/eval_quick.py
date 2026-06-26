"""Minimal multiple-choice eval — no lm_eval (which segfaults on import on some envs).

Computes per-choice loglikelihood with the trained model directly and reports
accuracy on hellaswag (4-way, chance .25) and piqa (2-way, chance .50).

    python scripts/eval_quick.py --weights W.pt --config micro.yaml --device cuda --limit 100
"""
import argparse

import torch
import torch.nn.functional as F
import yaml

from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel


@torch.no_grad()
def loglik(model, tok, ctx: str, cont: str, device: str) -> tuple[float, int]:
    ctx_ids = tok.encode(ctx)
    cont_ids = tok.encode(cont, add_special_tokens=False)
    if not cont_ids:
        return -1e9, 1
    ids = torch.tensor([ctx_ids + cont_ids], device=device)
    logits = model(ids, return_logits=True)["logits"]
    lp = F.log_softmax(logits[0, -len(cont_ids) - 1 : -1].float(), dim=-1)
    tgt = torch.tensor(cont_ids, device=device)
    return lp.gather(-1, tgt.unsqueeze(-1)).sum().item(), len(cont_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load tokenizer + datasets BEFORE the model: pyarrow's load_dataset
    # segfaults when run after CUDA is initialized on some Windows envs.
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    hs_data = load_dataset("Rowan/hellaswag", split="validation")
    arc_data = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")

    model = TerseModel(TerseConfig(**cfg["model"]))
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"])
    model = model.to(args.device).eval()
    print("model loaded on", args.device, flush=True)

    # hellaswag: ctx + 4 endings, label is index string
    try:
        hs = hs_data
        correct = total = 0
        for i in range(min(args.limit, len(hs))):
            ex = hs[i]
            if ex["label"] == "":
                continue
            scores = []
            for e in ex["endings"]:
                ll, n = loglik(model, tok, ex["ctx"], " " + e, args.device)
                scores.append(ll / n)  # length-normalized (acc_norm)
            correct += int(scores.index(max(scores)) == int(ex["label"]))
            total += 1
        print(f"hellaswag acc_norm ({total}): {correct/total:.4f}  (chance 0.25)", flush=True)
    except Exception as e:  # noqa: BLE001
        print("hellaswag FAILED:", repr(e), flush=True)

    # arc_easy: question + choices, answerKey (parquet, works on datasets 5.x)
    try:
        arc = arc_data
        correct = total = 0
        for i in range(min(args.limit, len(arc))):
            ex = arc[i]
            labels, texts = ex["choices"]["label"], ex["choices"]["text"]
            if ex["answerKey"] not in labels:
                continue
            gold = labels.index(ex["answerKey"])
            scores = []
            for t in texts:
                ll, n = loglik(model, tok, "Question: " + ex["question"] + "\nAnswer:", " " + t, args.device)
                scores.append(ll / n)
            correct += int(scores.index(max(scores)) == gold)
            total += 1
        print(f"arc_easy acc_norm ({total}): {correct/total:.4f}  (chance ~0.25)", flush=True)
    except Exception as e:  # noqa: BLE001
        print("arc_easy FAILED:", repr(e), flush=True)


if __name__ == "__main__":
    main()
