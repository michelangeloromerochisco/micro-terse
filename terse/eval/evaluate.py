"""lm-eval-harness integration for TerseModel.

Requires `pip install -e ".[eval]"`. Runs on RunPod; not part of the local pytest suite.
"""
from typing import List, Tuple

import torch
import torch.nn.functional as F

FAST_TASKS = ["hellaswag", "piqa", "arc_easy"]
FULL_TASKS = FAST_TASKS + ["arc_challenge", "winogrande"]


def run_eval(model, tokenizer, tasks: List[str], device: str = "cuda", limit=None) -> dict:
    """Wrap a trained TerseModel as an lm-eval LM and run the given tasks."""
    from lm_eval import simple_evaluate
    from lm_eval.api.model import LM

    class TerseLM(LM):
        def __init__(self) -> None:
            super().__init__()
            self._model = model.to(device).eval()
            self._tok = tokenizer

        def _loglikelihood_one(self, context: str, continuation: str) -> Tuple[float, bool]:
            ctx_ids = self._tok.encode(context)
            cont_ids = self._tok.encode(continuation)
            ids = torch.tensor([ctx_ids + cont_ids], device=device)
            with torch.no_grad():
                logits = self._model(ids, return_logits=True)["logits"]
            cont_len = len(cont_ids)
            log_probs = F.log_softmax(logits[0, -cont_len - 1 : -1].float(), dim=-1)
            target = torch.tensor(cont_ids, device=device)
            token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            greedy = (log_probs.argmax(-1) == target).all().item()
            return float(token_lp.sum().item()), bool(greedy)

        def loglikelihood(self, requests):
            return [self._loglikelihood_one(r.args[0], r.args[1]) for r in requests]

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError

        def generate_until(self, requests):
            raise NotImplementedError

    results = simple_evaluate(model=TerseLM(), tasks=tasks, limit=limit)
    return results["results"]
