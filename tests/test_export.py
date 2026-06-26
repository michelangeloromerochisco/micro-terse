"""Tests for the plain F16 GGUF export."""
import os
import tempfile

from terse.export.gguf import export_gguf
from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel


def _tiny_model() -> TerseModel:
    config = TerseConfig(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ffn_intermediate=64,
        num_experts=4,
        top_k=2,
        vocab_size=64,
        max_seq_len=64,
        gradient_checkpointing=False,
    )
    return TerseModel(config)


def test_export_gguf_creates_file():
    model = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = f"{tmp}/micro.gguf"
        export_gguf(model, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0


def test_export_gguf_readback_has_tensors():
    import gc

    import gguf

    model = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = f"{tmp}/micro.gguf"
        export_gguf(model, out_path)
        reader = gguf.GGUFReader(out_path)
        names = {t.name for t in reader.tensors}
        # Release the mmap so the temp dir can be cleaned up on Windows.
        del reader
        gc.collect()
        assert "embed_tokens.weight" in names
        assert any(n.endswith("attn.q_proj.weight") for n in names)
