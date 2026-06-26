"""Tests for the OpenAI-compatible Micro-Terse server."""
import pytest
import torch
from fastapi.testclient import TestClient

from terse.server.app import create_app


@pytest.fixture
def demo_client():
    app = create_app(demo_mode=True)
    return TestClient(app)


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


class _FakeModel:
    vocab_size = 64

    def __init__(self) -> None:
        self._step = 0

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, input_ids, return_logits=True):
        batch, seq_len = input_ids.shape
        # Emit deterministic tokens ending with EOS after a few steps.
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        if self._step < 3:
            logits[:, -1, 10 + self._step] = 100.0
        else:
            logits[:, -1, _FakeTokenizer.eos_token_id] = 100.0
        self._step += 1
        return {"logits": logits, "loss": None}


@pytest.fixture
def real_client():
    app = create_app(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        demo_mode=False,
    )
    return TestClient(app)


def test_list_models(demo_client):
    resp = demo_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "terse-micro"


def test_status_endpoint(demo_client):
    resp = demo_client.get("/v1/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "terse-micro"
    assert data["demo_mode"] is True


def test_chat_completions_non_streaming(demo_client):
    resp = demo_client.post(
        "/v1/chat/completions",
        json={
            "model": "terse-micro",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Micro-Terse" in data["choices"][0]["message"]["content"]


def test_chat_completions_streaming(demo_client):
    resp = demo_client.post(
        "/v1/chat/completions",
        json={
            "model": "terse-micro",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert "data:" in text
    assert "[DONE]" in text


def test_chat_completions_streaming_yields_finish_reason(demo_client):
    resp = demo_client.post(
        "/v1/chat/completions",
        json={
            "model": "terse-micro",
            "messages": [{"role": "user", "content": "demo"}],
            "stream": True,
            "max_tokens": 32,
        },
    )
    chunks = [line for line in resp.text.splitlines() if line.startswith("data:")]
    assert len(chunks) >= 2
    last_data = chunks[-1].replace("data: ", "")
    assert '"finish_reason": "stop"' in last_data or "[DONE]" in last_data


def test_real_mode_non_streaming(real_client):
    resp = real_client.post(
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


def test_real_mode_streaming(real_client):
    resp = real_client.post(
        "/v1/chat/completions",
        json={
            "model": "terse-micro",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 8,
        },
    )
    assert resp.status_code == 200
    text = resp.text
    assert "data:" in text
    assert "[DONE]" in text
    # Should contain content deltas from the fake model.
    assert '"content":' in text


def test_real_mode_error_when_model_missing():
    app = create_app(demo_mode=False)
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
    assert "Error: model not loaded" in resp.json()["choices"][0]["message"]["content"]


def test_content_length_validation(demo_client):
    resp = demo_client.post(
        "/v1/chat/completions",
        json={
            "model": "terse-micro",
            "messages": [{"role": "user", "content": "x" * 10000}],
            "stream": False,
        },
    )
    assert resp.status_code == 422


def test_identity_proof_demo_mode(demo_client):
    data = demo_client.get("/v1/identity_proof").json()
    assert data["available"] is False
    assert data["total"] == 0
    assert data["probes"] == []


def test_identity_proof_real_mode(real_client):
    data = real_client.get("/v1/identity_proof").json()
    assert data["available"] is True
    assert data["total"] == 4
    assert len(data["probes"]) == 4
    for probe in data["probes"]:
        assert "margin" in probe and "prefers_charter" in probe
    assert "progression" not in data  # none was supplied to create_app


def test_identity_proof_uses_proof_model_not_chat_model():
    class _RaisingModel:
        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            raise AssertionError("chat model must not be used for the identity proof")

    app = create_app(
        model=_RaisingModel(),
        proof_model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        demo_mode=False,
    )
    data = TestClient(app).get("/v1/identity_proof").json()
    assert data["available"] is True
    assert data["total"] == 4


def test_predict_demo_mode(demo_client):
    data = demo_client.post("/v1/predict", json={"text": "hi", "k": 3}).json()
    assert data["available"] is False
    assert data["predictions"] == []


def test_predict_real_mode(real_client):
    data = real_client.post("/v1/predict", json={"text": "hi", "k": 3}).json()
    assert data["available"] is True
    assert len(data["predictions"]) == 3
    for pred in data["predictions"]:
        assert "token" in pred and "prob" in pred
    assert data["model"] == "chat"  # no base_model supplied -> falls back to chat


def test_predict_showcase_real_mode(real_client):
    data = real_client.get("/v1/predict_showcase").json()
    assert data["available"] is True
    assert len(data["items"]) > 0
    assert data["items"][0]["predictions"]


def test_predict_uses_base_model_not_chat_model():
    class _RaisingModel:
        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            raise AssertionError("chat model must not be used for /predict")

    app = create_app(
        model=_RaisingModel(),
        base_model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        demo_mode=False,
    )
    data = TestClient(app).post("/v1/predict", json={"text": "hi", "k": 3}).json()
    assert data["available"] is True
    assert data["model"] == "base"


def test_identity_proof_includes_progression():
    progression = [{"stage": "base", "avg_margin": -1.8, "preferred": 0, "total": 4}]
    app = create_app(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        demo_mode=False,
        identity_progression=progression,
    )
    data = TestClient(app).get("/v1/identity_proof").json()
    assert data["available"] is True
    assert data["progression"][0]["stage"] == "base"
