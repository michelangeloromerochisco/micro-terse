"""Convert a trained TerseModel to a llama.cpp-canonical GGUF (Phase 1, F32).

This is a standalone converter (Terse is NOT a HuggingFace model, so llama.cpp's
``convert_hf_to_gguf.py`` does not apply). It maps Terse module names onto the
canonical llama.cpp tensor names (``token_embd``, ``blk.N.attn_*``, MoE
``ffn_*_exps`` 3D tensors, etc.), merges MoE experts into stacked tensors, drops
the MTP head and the EMA bookkeeping buffer, and writes ``terse.*`` architecture
metadata.

Phase 1 scope:
  * Every tensor is written as F32 (no quantization — TQ2_0 is Phase 3). F16 is
    unsafe: see ``_weight`` (ReLU-squared activations overflow F16 in MUL_MAT).
  * Ternary linears store their *quantized* {-1, 0, +1} float weights (via the
    real STE forward) so the output reflects true inference weights.
  * The tied LM head is NOT emitted as a separate ``output.weight``.

No C++ and no quantization happen here.
"""
from __future__ import annotations

import json
import logging

import torch

from terse.model.attention import TerseAttention
from terse.model.config import TerseConfig
from terse.model.ffn import TerseFFN
from terse.model.moe import TerseMoE
from terse.model.terse_model import TerseModel
from terse.model.ternary import TernaryLinear, TernaryQuantizeFunction

logger = logging.getLogger(__name__)

# The Llama-3.1 BPE tokenizer Terse trains against. Embedded into the GGUF so
# llama.cpp can build its vocabulary (it expects ``tokenizer.ggml.*`` keys).
TOKENIZER_REPO = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_VOCAB_SIZE = 128256
# Pre-tokenizer identifier for Llama-3 BPE (see llama.cpp get_vocab_base_pre).
LLAMA_BPE_PRE = "llama-bpe"


def _ternary_weight(lin: TernaryLinear) -> torch.Tensor:
    """Return the quantized {-1, 0, +1} float weight a ternary linear uses at
    inference time (the STE forward), detached on CPU."""
    with torch.no_grad():
        w = TernaryQuantizeFunction.apply(lin.weight, lin.temperature)
    return w.detach().cpu()


def _weight(tensor: torch.Tensor):
    """Return a contiguous F32 numpy array for a weight tensor.

    Weights MUST be F32, not F16. The ReLU-squared FFN produces activations far
    larger than F16's max (65504): ggml's ``MUL_MAT(F16_weight, F32_act)`` path
    down-converts the F32 activation to F16, so those large values overflow to
    ``inf``; ``inf * 0`` (a ternary-zero weight) is then ``NaN``, which poisons
    the residual stream and collapses generation. F32 weights keep ggml on the
    F32xF32 matmul path (no activation down-conversion), and {-1,0,+1} is exact
    in F32. The small single-file model comes later from TQ2_0 (Phase 3), whose
    per-block Q8_0 activation scaling avoids this overflow entirely."""
    return tensor.to(torch.float32).contiguous().cpu().numpy()


def _f32(tensor: torch.Tensor):
    """Return a contiguous F32 numpy array. Used for tensors that feed llama.cpp
    element-wise CPU ops (RMSNorm scale, the MoE router bias) — those ops reject
    F16 operands (binary_op: unsupported types f32 x f16)."""
    return tensor.to(torch.float32).contiguous().cpu().numpy()


def _add_metadata(writer, cfg: TerseConfig, name: str) -> None:
    """Write all ``terse.*`` architecture metadata using typed add_* methods.

    Typed methods derive the key prefix from the writer's ``arch`` ("terse"),
    so e.g. ``add_block_count`` emits ``terse.block_count``.
    """
    writer.add_name(name)
    writer.add_block_count(cfg.num_layers)
    writer.add_context_length(cfg.max_seq_len)
    writer.add_embedding_length(cfg.hidden_dim)
    writer.add_feed_forward_length(cfg.ffn_intermediate)
    writer.add_head_count(cfg.num_heads)
    writer.add_head_count_kv(cfg.num_kv_heads)
    writer.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    writer.add_rope_freq_base(cfg.rope_theta)
    writer.add_key_length(cfg.head_dim)
    writer.add_value_length(cfg.head_dim)
    writer.add_vocab_size(cfg.vocab_size)
    writer.add_expert_count(cfg.num_experts)
    writer.add_expert_used_count(cfg.top_k)
    # MoE layer indices: the inference graph needs to know which blocks are MoE.
    moe_layers = sorted(int(i) for i in (cfg.moe_layers or []))
    writer.add_array("terse.expert_layers", moe_layers)


def _add_attention(writer, blk: int, attn: TerseAttention) -> None:
    """Emit the four attention projections (+ optional QK norms) for a block."""
    writer.add_tensor(f"blk.{blk}.attn_q.weight", _weight(_ternary_weight(attn.q_proj)))
    writer.add_tensor(f"blk.{blk}.attn_k.weight", _weight(_ternary_weight(attn.k_proj)))
    writer.add_tensor(f"blk.{blk}.attn_v.weight", _weight(_ternary_weight(attn.v_proj)))
    writer.add_tensor(
        f"blk.{blk}.attn_output.weight", _weight(_ternary_weight(attn.o_proj))
    )
    if getattr(attn, "qk_norm", False):
        writer.add_tensor(
            f"blk.{blk}.attn_q_norm.weight", _f32(attn.q_norm.weight.detach())
        )
        writer.add_tensor(
            f"blk.{blk}.attn_k_norm.weight", _f32(attn.k_norm.weight.detach())
        )


def _add_dense_ffn(writer, blk: int, ffn: TerseFFN) -> None:
    """Emit the three projections of a dense ReLU-squared FFN."""
    writer.add_tensor(
        f"blk.{blk}.ffn_gate.weight", _weight(_ternary_weight(ffn.gate_proj))
    )
    writer.add_tensor(f"blk.{blk}.ffn_up.weight", _weight(_ternary_weight(ffn.up_proj)))
    writer.add_tensor(
        f"blk.{blk}.ffn_down.weight", _weight(_ternary_weight(ffn.down_proj))
    )


def _add_moe_ffn(writer, blk: int, moe: TerseMoE) -> None:
    """Emit the router + stacked 3D expert tensors of an MoE FFN.

    Experts are stacked along dim 0 into ``[n_expert, out, in]`` tensors,
    matching llama.cpp's ``ffn_*_exps`` convention. ``expert_counts_ema`` is
    dropped; ``expert_bias`` is kept (used in routing at inference).
    """
    # Router gate is a plain nn.Linear (NOT ternary) -> ffn_gate_inp.
    writer.add_tensor(
        f"blk.{blk}.ffn_gate_inp.weight", _weight(moe.router.gate.weight.detach())
    )
    # Aux-free routing bias -> exp_probs_b.bias (kept for inference routing). F32:
    # it is added to f32 router logits (element-wise add rejects an f16 operand).
    writer.add_tensor(
        f"blk.{blk}.exp_probs_b.bias", _f32(moe.router.expert_bias.detach())
    )

    gate = torch.stack([_ternary_weight(e.gate_proj) for e in moe.experts], dim=0)
    up = torch.stack([_ternary_weight(e.up_proj) for e in moe.experts], dim=0)
    down = torch.stack([_ternary_weight(e.down_proj) for e in moe.experts], dim=0)
    writer.add_tensor(f"blk.{blk}.ffn_gate_exps.weight", _weight(gate))
    writer.add_tensor(f"blk.{blk}.ffn_up_exps.weight", _weight(up))
    writer.add_tensor(f"blk.{blk}.ffn_down_exps.weight", _weight(down))


def _token_looks_special(text: str) -> bool:
    """Heuristic mirroring llama.cpp's ``does_token_look_special``: treat tokens
    wrapped in ``<|...|>`` (Llama-3 control markers) as control tokens even when
    the tokenizer doesn't flag them ``special``."""
    return text.startswith("<|") and text.endswith("|>")


def _get_vocab_base(tokenizer) -> tuple[list[str], list[int]]:
    """Build the full ordered token list + token-type list for the GGUF.

    Replicates llama.cpp's ``get_vocab_base`` (the GPT-2/BPE path used by
    Llama-3): the byte-level-encoded vocab is laid out by id, gaps become UNUSED
    ``[PADi]`` placeholders, special/added tokens become CONTROL, the rest
    NORMAL.
    """
    import gguf

    tokens: list[str] = []
    toktypes: list[int] = []

    vocab_size = TOKENIZER_VOCAB_SIZE
    reverse_vocab = {tid: tok for tok, tid in tokenizer.vocab.items()}
    added_vocab = tokenizer.get_added_vocab()
    added_tokens_decoder = tokenizer.added_tokens_decoder

    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
            continue
        token = reverse_vocab[i]
        if token in added_vocab:
            entry = added_tokens_decoder.get(i)
            is_special = (entry is not None and entry.special) or _token_looks_special(token)
            toktypes.append(gguf.TokenType.CONTROL if is_special else gguf.TokenType.USER_DEFINED)
        else:
            toktypes.append(gguf.TokenType.NORMAL)
        tokens.append(token)

    return tokens, toktypes


def _get_merges(tokenizer) -> list[str]:
    """Extract BPE merge rules from the fast tokenizer, as space-joined pairs
    (``"a b"``) — the form llama.cpp expects in ``tokenizer.ggml.merges``."""
    backend = tokenizer.backend_tokenizer
    state = json.loads(backend.to_str())
    raw_merges = state.get("model", {}).get("merges", [])
    merges: list[str] = []
    for m in raw_merges:
        # Newer tokenizers store merges as ["a", "b"] pairs; older as "a b".
        merges.append(f"{m[0]} {m[1]}" if isinstance(m, (list, tuple)) else m)
    return merges


def _add_tokenizer(writer, cfg: TerseConfig) -> bool:
    """Embed the Llama-3.1 BPE tokenizer (model, pre, tokens, types, merges, and
    special token ids) into the GGUF.

    Returns True if the tokenizer was embedded, False if it was skipped. The
    tokenizer is only embedded when ``cfg.vocab_size`` matches the real
    Llama-3.1 vocabulary (128256). Tiny test configs (e.g. vocab_size=256) are
    skipped, as is any environment where the tokenizer cannot be loaded.
    """
    if cfg.vocab_size != TOKENIZER_VOCAB_SIZE:
        logger.warning(
            "Skipping tokenizer embedding: config vocab_size=%d != Llama-3.1 "
            "vocab_size=%d.",
            cfg.vocab_size,
            TOKENIZER_VOCAB_SIZE,
        )
        return False

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    except Exception as exc:  # pragma: no cover - network/dep dependent
        logger.warning(
            "Skipping tokenizer embedding: could not load tokenizer %r (%s).",
            TOKENIZER_REPO,
            exc,
        )
        return False

    assert max(tokenizer.vocab.values()) < TOKENIZER_VOCAB_SIZE

    tokens, toktypes = _get_vocab_base(tokenizer)
    merges = _get_merges(tokenizer)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(LLAMA_BPE_PRE)
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)
    writer.add_token_merges(merges)

    if tokenizer.bos_token_id is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        writer.add_eos_token_id(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        writer.add_pad_token_id(tokenizer.pad_token_id)
    if getattr(tokenizer, "unk_token_id", None) is not None:
        writer.add_unk_token_id(tokenizer.unk_token_id)

    logger.info(
        "Embedded tokenizer %r: %d tokens, %d merges.",
        TOKENIZER_REPO,
        len(tokens),
        len(merges),
    )
    return True


def export_gguf_llamacpp(
    model: TerseModel, out_path: str, name: str = "terse-micro"
) -> str:
    """Write ``model`` to ``out_path`` as a llama.cpp-canonical F16 GGUF.

    Returns the output path. The MTP head and EMA buffers are dropped; the LM
    head stays tied to ``token_embd`` (no separate ``output.weight``).
    """
    import gguf

    cfg = model.config
    writer = gguf.GGUFWriter(out_path, "terse")
    _add_metadata(writer, cfg, name)
    _add_tokenizer(writer, cfg)

    # Top-level: tied embedding (also the LM head) + final norm.
    writer.add_tensor(
        "token_embd.weight", _weight(model.embed_tokens.weight.detach())
    )
    writer.add_tensor("output_norm.weight", _f32(model.norm.weight.detach()))

    for blk, layer in enumerate(model.layers):
        writer.add_tensor(
            f"blk.{blk}.attn_norm.weight", _f32(layer.attn_norm.weight.detach())
        )
        _add_attention(writer, blk, layer.attn)
        writer.add_tensor(
            f"blk.{blk}.ffn_norm.weight", _f32(layer.ffn_norm.weight.detach())
        )
        if isinstance(layer.ffn, TerseMoE):
            _add_moe_ffn(writer, blk, layer.ffn)
        elif isinstance(layer.ffn, TerseFFN):
            _add_dense_ffn(writer, blk, layer.ffn)
        else:  # pragma: no cover - defensive: unknown FFN type
            raise TypeError(f"Unsupported FFN type at layer {blk}: {type(layer.ffn)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out_path
