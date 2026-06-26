"""Token sampling and chat generation for TerseModel."""
from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn.functional as F

from terse.model.terse_model import TerseModel


CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
REASONING_START = "<thinking>"
REASONING_END = "</thinking>"

_VALID_ROLES = {"system", "user", "assistant"}
_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|im_sep|>")


def _sanitize_chatml(content: str) -> str:
    """Sanitize user-supplied content by truncating at any ChatML control token."""
    for token in _CONTROL_TOKENS:
        if token in content:
            content = content.split(token)[0]
    return content


def apply_chatml_template(messages: list[dict[str, str]]) -> str:
    """Render a list of ChatML messages into a single prompt string.

    Supports system/user/assistant turns. The final assistant turn is left
    open so the model can generate the response.
    """
    if not messages:
        return ""

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role not in _VALID_ROLES:
            role = "user"
        content = _sanitize_chatml(msg.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _prepare_logits(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Apply temperature and top-p filtering to the last-position logits."""
    logits = logits[:, -1, :].float()
    logits = logits / max(temperature, 1e-6)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cumulative_probs > top_p
        # Keep the first token that pushes us over the threshold.
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        remove_indices = sorted_indices[remove_mask]
        logits[0, remove_indices] = float("-inf")

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> int:
    """Sample a single token from the last-position logits."""
    logits = _prepare_logits(logits, temperature, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def _apply_repetition_penalty(
    last_logits: torch.Tensor, input_ids: torch.Tensor, penalty: float
) -> None:
    """In-place CTRL-style repetition penalty on already-seen tokens.

    Divides positive logits and multiplies negative logits of any token that has
    appeared so far, pushing the model away from repeating itself.
    """
    if penalty == 1.0:
        return
    for tid in set(input_ids[0].tolist()):
        v = last_logits[0, tid]
        last_logits[0, tid] = v / penalty if v > 0 else v * penalty


def _block_repeat_ngrams(
    last_logits: torch.Tensor, input_ids: torch.Tensor, n: int
) -> None:
    """In-place no-repeat-ngram: ban tokens that would complete an existing n-gram.

    Directly kills degenerate loops (phrase repeats and single-token collapse like
    'a8a8a8') on weak models.
    """
    if n <= 0:
        return
    seq = input_ids[0].tolist()
    if len(seq) < n:
        return
    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            last_logits[0, seq[i + n - 1]] = float("-inf")


def generate_stream(
    model: TerseModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 512,
    min_new_tokens: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 3,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    stop_texts: tuple[str, ...] = (),
    tokenizer=None,
    device: torch.device | str = "cpu",
) -> Iterable[tuple[int, torch.Tensor]]:
    """Yield (token_id, full_input_ids) for each generated token.

    Stops when `max_new_tokens` is reached, EOS/pad is generated (only after
    `min_new_tokens` so early end-tokens don't yield empty output), or the
    model returns NaN/Inf logits.
    """
    model.eval()
    input_ids = input_ids.to(device)
    generated = 0
    gen_ids: list[int] = []  # generated token ids only (for stop-sequence matching)

    with torch.no_grad():
        while generated < max_new_tokens:
            outputs = model(input_ids, return_logits=True)
            logits = outputs.get("logits")
            if logits is None:
                raise RuntimeError("Model returned no logits during generation")

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError("Model produced NaN/Inf logits; aborting generation")

            last_logits = logits[:, -1, :]
            _apply_repetition_penalty(last_logits, input_ids, repetition_penalty)
            _block_repeat_ngrams(last_logits, input_ids, no_repeat_ngram_size)
            next_token_id = sample_next_token(logits, temperature, top_p)
            generated += 1
            gen_ids.append(next_token_id)

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=device, dtype=input_ids.dtype)],
                dim=1,
            )
            yield next_token_id, input_ids

            if generated >= min_new_tokens:
                if eos_token_id is not None and next_token_id == eos_token_id:
                    break
                if pad_token_id is not None and next_token_id == pad_token_id:
                    break
                # ChatML turn markers are multi-token and tokenize differently in
                # context, so match the decoded string of the recent tail, not ids.
                if stop_texts and tokenizer is not None:
                    tail = tokenizer.decode(gen_ids[-16:], skip_special_tokens=False)
                    if any(st in tail for st in stop_texts):
                        break


def decode_with_reasoning(
    tokenizer,
    input_ids: torch.Tensor,
    prompt_length: int,
) -> str:
    """Decode generated tokens and normalize reasoning tags."""
    new_ids = input_ids[0, prompt_length:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    # ChatML turn markers are not special tokens on this tokenizer, so they survive
    # decoding as literal text. Cut at the first marker = end of the assistant turn.
    for marker in ("<|im_end|>", "<|im_start|>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    # Collapse stray whitespace around reasoning tags for consistent rendering.
    text = re.sub(r"\s*<thinking>\s*", "\n<thinking>\n", text)
    text = re.sub(r"\s*</thinking>\s*", "\n</thinking>\n", text)
    return text.strip()
