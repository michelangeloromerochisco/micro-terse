"""Load a Micro-Terse GGUF file back into a ``TerseModel``.

Supports both naming conventions produced by the exporters:

* ``terse.export.gguf`` writes the raw PyTorch state_dict keys
  (``embed_tokens.weight``, ``layers.0.attn.q_proj.weight``, ...).
* ``terse.export.gguf_llamacpp`` writes llama.cpp canonical names
  (``token_embd.weight``, ``blk.0.attn_q.weight``, ...).

The loader detects which convention is used and maps tensors onto a fresh
``TerseModel`` built from a caller-supplied config or from the GGUF metadata.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel


def _is_canonical(name: str) -> bool:
    """True if ``name`` uses llama.cpp canonical tensor naming."""
    return (
        name.startswith("blk.")
        or name.startswith("token_embd.")
        or name.startswith("output_norm.")
    )


def _canonical_to_pytorch(name: str) -> str | None:
    """Map a llama.cpp canonical tensor name to a TerseModel state_dict key.

    Returns ``None`` for tensors the PyTorch model does not keep
    (e.g. ``output.weight`` when embeddings are tied) or for tensors that
    require special handling (MoE stacked experts).
    """
    if name == "token_embd.weight":
        return "embed_tokens.weight"
    if name == "output_norm.weight":
        return "norm.weight"
    if name.startswith("output."):
        return None  # tied embeddings: no separate output weight

    if not name.startswith("blk."):
        return None

    parts = name.split(".")
    blk = parts[1]
    rest = ".".join(parts[2:])

    attention_map = {
        "attn_q.weight": f"layers.{blk}.attn.q_proj.weight",
        "attn_k.weight": f"layers.{blk}.attn.k_proj.weight",
        "attn_v.weight": f"layers.{blk}.attn.v_proj.weight",
        "attn_output.weight": f"layers.{blk}.attn.o_proj.weight",
        "attn_q_norm.weight": f"layers.{blk}.attn.q_norm.weight",
        "attn_k_norm.weight": f"layers.{blk}.attn.k_norm.weight",
        "attn_norm.weight": f"layers.{blk}.attn_norm.weight",
        "ffn_norm.weight": f"layers.{blk}.ffn_norm.weight",
        "ffn_gate.weight": f"layers.{blk}.ffn.gate_proj.weight",
        "ffn_up.weight": f"layers.{blk}.ffn.up_proj.weight",
        "ffn_down.weight": f"layers.{blk}.ffn.down_proj.weight",
        "ffn_gate_inp.weight": f"layers.{blk}.ffn.router.gate.weight",
        "exp_probs_b.bias": f"layers.{blk}.ffn.router.expert_bias",
    }
    return attention_map.get(rest)


def _load_tensors(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read all tensors and metadata from a GGUF file."""
    import gc

    import gguf

    reader = gguf.GGUFReader(path)
    tensors: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        # Copy out of the mmap so the returned arrays don't keep the file
        # memory-mapped (which locks it on Windows / blocks deletion).
        tensors[str(tensor.name)] = np.array(tensor.data)

    fields: dict[str, Any] = {}
    for key, field in reader.fields.items():
        try:
            # field.data indexes only the value part(s) of field.parts, skipping
            # the key/type metadata parts (flattening all parts gave garbage).
            values: list[Any] = []
            for idx in field.data:
                part = field.parts[idx]
                values.extend(part.tolist() if hasattr(part, "tolist") else [part])
            fields[key] = values[0] if len(values) == 1 else values
        except Exception:
            fields[key] = None

    # Explicitly close the underlying mmap so the file isn't locked on Windows
    # (np.array copies above already detached the tensor data from it).
    try:
        reader.data._mmap.close()
    except Exception:
        pass
    del reader
    gc.collect()
    return tensors, fields


def _config_from_metadata(fields: dict[str, Any]) -> TerseConfig:
    """Build a TerseConfig from GGUF architecture metadata.

    Recognises both ``terse.*`` custom keys and llama.cpp canonical keys.
    """
    def _get(*keys: str, default: Any | None = None) -> Any:
        for key in keys:
            if key in fields and fields[key] is not None:
                return fields[key]
        return default

    # Canonical llama.cpp keys.
    num_layers = _get("terse.block_count", "llama.block_count", default=16)
    hidden_dim = _get("terse.embedding_length", "llama.embedding_length", default=1536)
    num_heads = _get("terse.attention.head_count", "llama.attention.head_count", default=12)
    num_kv_heads = _get("terse.attention.head_count_kv", "llama.attention.head_count_kv", default=3)
    ffn_intermediate = _get("terse.feed_forward_length", "llama.feed_forward_length", default=4096)
    vocab_size = _get("terse.vocab_size", "llama.vocab_size", default=128256)
    max_seq_len = _get("terse.context_length", "llama.context_length", default=4096)
    rope_theta = _get("terse.rope.freq_base", "llama.rope.freq_base", default=500000.0)
    rms_norm_eps = _get("terse.attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon", default=1e-6)

    # Custom terse keys override canonical ones.
    num_experts = _get("terse.expert_count", default=4)
    top_k = _get("terse.expert_used_count", default=2)
    moe_layers = _get("terse.expert_layers", default=None)

    if isinstance(moe_layers, (list, tuple)):
        moe_layers = [int(i) for i in moe_layers]
    elif moe_layers is not None:
        # A 1-element expert_layers array parses as a scalar; restore the list.
        moe_layers = [int(moe_layers)]

    head_dim = hidden_dim // num_heads

    return TerseConfig(
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        ffn_intermediate=int(ffn_intermediate),
        num_experts=int(num_experts),
        top_k=int(top_k),
        moe_layers=moe_layers,
        vocab_size=int(vocab_size),
        max_seq_len=int(max_seq_len),
        rope_theta=float(rope_theta),
        rms_norm_eps=float(rms_norm_eps),
        qk_norm=True,
        ternary_weights=True,
        fogzo=True,
        gradient_checkpointing=False,
    )


def load_gguf_model(path: str, config: TerseConfig | None = None, device: str = "cpu") -> TerseModel:
    """Load a Micro-Terse model from a GGUF file.

    If ``config`` is not provided, the architecture metadata embedded in the
    GGUF is used to build a ``TerseConfig``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"GGUF file not found: {path}")

    tensors, fields = _load_tensors(path)
    if not tensors:
        raise ValueError(f"No tensors found in {path}")

    sample_name = next(iter(tensors))
    canonical = _is_canonical(sample_name)

    if config is None:
        config = _config_from_metadata(fields)

    model = TerseModel(config).to(device)
    state_dict: dict[str, torch.Tensor] = {}

    for name, arr in tensors.items():
        tensor = torch.from_numpy(np.array(arr)).to(device)

        if not canonical:
            state_dict[name] = tensor
            continue

        mapped = _canonical_to_pytorch(name)
        if mapped is not None:
            state_dict[mapped] = tensor
            continue

        # MoE stacked expert tensors: ffn_{gate,up,down}_exps.weight.
        if "_exps.weight" in name:
            parts = name.split(".")
            if len(parts) < 4:
                continue
            blk = parts[1]
            proj_part = parts[2]  # e.g. "ffn_gate_exps" (parts[3] is "weight")
            proj_map = {"ffn_gate_exps": "gate_proj", "ffn_up_exps": "up_proj", "ffn_down_exps": "down_proj"}
            proj = proj_map.get(proj_part)
            if proj is None:
                continue
            for e in range(config.num_experts):
                key = f"layers.{blk}.ffn.experts.{e}.{proj}.weight"
                state_dict[key] = tensor[e]

    # The exporter intentionally drops the MTP head and the EMA bookkeeping
    # buffer (not needed for inference); those keep their fresh init.
    def _is_dropped(key: str) -> bool:
        # MTP head + EMA buffer are inference-irrelevant; .temperature is a
        # FOGZO backward-only knob unused at inference. All keep fresh init.
        return (
            key.startswith("mtp_head.")
            or key.endswith(".expert_counts_ema")
            or key.endswith(".temperature")
        )

    expected = set(model.state_dict().keys())
    provided = set(state_dict.keys())
    gaps = {k for k in (expected - provided) if not _is_dropped(k)}
    if gaps:
        raise RuntimeError(
            f"GGUF load incomplete: {len(gaps)} missing tensors, including "
            f"{sorted(gaps)[:5]}"
        )

    # Overlay loaded tensors onto the fresh model state so dropped tensors
    # (MTP head, EMA buffer) retain their initialized values.
    merged = dict(model.state_dict())
    merged.update(state_dict)
    model.load_state_dict(merged, strict=True)
    model.eval()
    return model
