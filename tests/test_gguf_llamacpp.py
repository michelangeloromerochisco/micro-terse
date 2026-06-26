"""Tests for the llama.cpp-canonical GGUF converter (Phase 1, F16)."""
import os
import tempfile

import numpy as np
import pytest

from terse.export.gguf_llamacpp import export_gguf_llamacpp
from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel

NUM_EXPERTS = 4
MOE_LAYERS = [1, 3]


@pytest.fixture(scope="module")
def tiny_config() -> TerseConfig:
    return TerseConfig(
        hidden_dim=64,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_intermediate=128,
        num_experts=NUM_EXPERTS,
        top_k=2,
        moe_layers=list(MOE_LAYERS),
        vocab_size=256,
        qk_norm=True,
        tie_embeddings=True,
        max_seq_len=128,
        gradient_checkpointing=False,
    )


class _GGUFView:
    """Plain in-memory snapshot of a GGUF file.

    The reader memory-maps the file; on Windows that lock blocks temp-dir
    cleanup. We copy out names/shapes/fields and drop the mmap immediately.
    """

    def __init__(self, path: str) -> None:
        import gguf

        reader = gguf.GGUFReader(path)
        self.shapes = {t.name: list(t.shape) for t in reader.tensors}
        self.names = set(self.shapes)
        self.fields = {
            key: reader.get_field(key).contents() for key in reader.fields
        }
        del reader  # release the mmap before the temp dir is removed


@pytest.fixture(scope="module")
def reader(tiny_config):
    model = TerseModel(tiny_config).eval()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "terse-micro.gguf")
        export_gguf_llamacpp(model, out_path)
        assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
        view = _GGUFView(out_path)
    return view


def _tensor_names(reader) -> set:
    return reader.names


def _shape(reader, name):
    if name not in reader.shapes:
        raise KeyError(name)
    return reader.shapes[name]


def _field_value(reader, key):
    assert key in reader.fields, f"missing metadata key: {key}"
    return reader.fields[key]


def test_drops_mtp_ema_and_separate_output(reader):
    names = _tensor_names(reader)
    assert not any("mtp_head" in n for n in names)
    assert not any("expert_counts_ema" in n for n in names)
    assert "output.weight" not in names  # tied head, not a separate tensor


def test_moe_layers_have_stacked_experts_and_router(reader):
    for blk in MOE_LAYERS:
        names = _tensor_names(reader)
        for proj in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            name = f"blk.{blk}.{proj}.weight"
            assert name in names
            # gguf reports shape reversed (ne order); n_expert is the slowest dim.
            assert _shape(reader, name)[-1] == NUM_EXPERTS
        assert f"blk.{blk}.ffn_gate_inp.weight" in names
        assert f"blk.{blk}.exp_probs_b.bias" in names


def test_dense_layers_have_plain_ffn(reader):
    names = _tensor_names(reader)
    for blk in (0, 2):
        for proj in ("ffn_gate", "ffn_up", "ffn_down"):
            assert f"blk.{blk}.{proj}.weight" in names
        # dense blocks must NOT carry MoE tensors
        assert f"blk.{blk}.ffn_gate_exps.weight" not in names
        assert f"blk.{blk}.ffn_gate_inp.weight" not in names


def test_attention_tensors_present_per_block(reader):
    names = _tensor_names(reader)
    for blk in range(4):
        for proj in ("attn_q", "attn_k", "attn_v", "attn_output"):
            assert f"blk.{blk}.{proj}.weight" in names
        assert f"blk.{blk}.attn_q_norm.weight" in names
        assert f"blk.{blk}.attn_k_norm.weight" in names
        assert f"blk.{blk}.attn_norm.weight" in names
        assert f"blk.{blk}.ffn_norm.weight" in names


def test_top_level_tensors_present(reader):
    names = _tensor_names(reader)
    assert "token_embd.weight" in names
    assert "output_norm.weight" in names


def test_metadata_matches_config(reader, tiny_config):
    cfg = tiny_config
    assert _field_value(reader, "general.architecture") == "terse"
    assert _field_value(reader, "terse.block_count") == cfg.num_layers
    assert _field_value(reader, "terse.context_length") == cfg.max_seq_len
    assert _field_value(reader, "terse.embedding_length") == cfg.hidden_dim
    assert _field_value(reader, "terse.feed_forward_length") == cfg.ffn_intermediate
    assert _field_value(reader, "terse.attention.head_count") == cfg.num_heads
    assert _field_value(reader, "terse.attention.head_count_kv") == cfg.num_kv_heads
    assert _field_value(reader, "terse.attention.key_length") == cfg.head_dim
    assert _field_value(reader, "terse.attention.value_length") == cfg.head_dim
    assert _field_value(reader, "terse.vocab_size") == cfg.vocab_size
    assert _field_value(reader, "terse.expert_count") == cfg.num_experts
    assert _field_value(reader, "terse.expert_used_count") == cfg.top_k
    rms = _field_value(reader, "terse.attention.layer_norm_rms_epsilon")
    assert np.isclose(rms, cfg.rms_norm_eps)
    theta = _field_value(reader, "terse.rope.freq_base")
    assert np.isclose(theta, cfg.rope_theta)


def test_token_embd_shape_matches_config(reader, tiny_config):
    # gguf shape is reversed relative to torch; embedding is [vocab, hidden].
    shape = _shape(reader, "token_embd.weight")
    assert set(shape) == {tiny_config.vocab_size, tiny_config.hidden_dim}
