"""Tests for loading a Micro-Terse GGUF file back into a TerseModel."""
from __future__ import annotations

import tempfile

import pytest
import torch

try:
    import gguf

    _HAS_GGUF = True
except Exception:  # pragma: no cover
    _HAS_GGUF = False

from terse.model.terse_model import TerseModel
from terse.server.gguf_loader import load_gguf_model

pytestmark = pytest.mark.skipif(not _HAS_GGUF, reason="gguf not installed")


def _random_input_ids(config) -> torch.Tensor:
    return torch.randint(0, config.vocab_size, (1, 8))


def test_load_raw_gguf_roundtrip(tiny_config):
    """Export via terse.export.gguf and reload; outputs should match."""
    from terse.export.gguf import export_gguf

    model = TerseModel(tiny_config)
    model.eval()
    input_ids = _random_input_ids(tiny_config)
    with torch.no_grad():
        expected = model(input_ids, return_logits=True)["logits"]

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name

    try:
        export_gguf(model, path)
        loaded = load_gguf_model(path, config=tiny_config)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(input_ids, return_logits=True)["logits"]
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    finally:
        import gc
        import os

        gc.collect()
        try:
            os.remove(path)
        except OSError:
            pass  # Windows may still hold the GGUF mmap; temp file is cleaned by the OS.


def test_load_canonical_gguf_roundtrip(tiny_config):
    """Export via terse.export.gguf_llamacpp and reload; outputs should match."""
    from terse.export.gguf_llamacpp import export_gguf_llamacpp

    model = TerseModel(tiny_config)
    model.eval()
    input_ids = _random_input_ids(tiny_config)
    with torch.no_grad():
        expected = model(input_ids, return_logits=True)["logits"]

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name

    try:
        export_gguf_llamacpp(model, path, name="terse-tiny")
        loaded = load_gguf_model(path, config=tiny_config)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(input_ids, return_logits=True)["logits"]
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    finally:
        import gc
        import os

        gc.collect()
        try:
            os.remove(path)
        except OSError:
            pass  # Windows may still hold the GGUF mmap; temp file is cleaned by the OS.


def test_load_gguf_without_config_uses_metadata(tiny_config):
    """When no config is passed, the loader should infer it from GGUF metadata."""
    from terse.export.gguf_llamacpp import export_gguf_llamacpp

    model = TerseModel(tiny_config)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name

    try:
        export_gguf_llamacpp(model, path, name="terse-tiny")
        loaded = load_gguf_model(path, device="cpu")
        assert loaded.config.num_layers == tiny_config.num_layers
        assert loaded.config.hidden_dim == tiny_config.hidden_dim
    finally:
        import gc
        import os

        gc.collect()
        try:
            os.remove(path)
        except OSError:
            pass  # Windows may still hold the GGUF mmap; temp file is cleaned by the OS.


def test_load_gguf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_gguf_model("/nonexistent/path/model.gguf")


def test_gguf_model_serves_chat_completions(tiny_config):
    """A GGUF-loaded model can be wired into the FastAPI server."""
    from fastapi.testclient import TestClient

    from terse.export.gguf_llamacpp import export_gguf_llamacpp
    from terse.server.app import create_app

    model = TerseModel(tiny_config)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name

    try:
        export_gguf_llamacpp(model, path, name="terse-tiny")
        loaded = load_gguf_model(path, config=tiny_config)

        class _FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def encode(self, text, return_tensors=None, add_special_tokens=True):
                ids = [1] + [ord(c) % 50 for c in text]
                if return_tensors == "pt":
                    return torch.tensor([ids], dtype=torch.long)
                return ids

            def decode(self, ids, skip_special_tokens=False):
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                return "".join(chr((i % 26) + 97) for i in ids)

        app = create_app(model=loaded, tokenizer=_FakeTokenizer(), device="cpu", demo_mode=False)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "terse-micro",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 8,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["role"] == "assistant"
    finally:
        import gc
        import os

        gc.collect()
        try:
            os.remove(path)
        except OSError:
            pass  # Windows may still hold the GGUF mmap; temp file is cleaned by the OS.
