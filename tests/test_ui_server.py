"""Tests for the Micro-Terse web UI proxy server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from terse.server.ui_server import UI_DIR, _handler_factory, _validate_api_url, main


class _MockBackendHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible backend for proxy tests."""

    def log_message(self, fmt, *args):  # noqa: ARG002
        pass

    def do_GET(self):
        if self.path == "/v1/status":
            self._json({"model": "terse-micro", "demo_mode": True, "device": "cpu"})
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": "terse-micro"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            req = json.loads(body)
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in ["hello", " ", "world"]:
                    chunk = {
                        "id": "test",
                        "object": "chat.completion.chunk",
                        "model": "terse-micro",
                        "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._json(
                    {
                        "id": "test",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "hello world"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
        else:
            self.send_error(404)

    def _json(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def backend_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _, port = server.server_address
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def ui_server(backend_server):
    handler = _handler_factory(backend_server)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _, port = server.server_address
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_validate_api_url_accepts_http():
    assert _validate_api_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert _validate_api_url("https://example.com/v1/") == "https://example.com/v1"


def test_validate_api_url_rejects_bad_scheme():
    with pytest.raises(ValueError):
        _validate_api_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        _validate_api_url("ftp://localhost:8080")


def test_serves_index_html(ui_server):
    with urlopen(f"{ui_server}/") as resp:  # noqa: S310
        assert resp.status == 200
        data = resp.read()
        assert b"Micro-Terse" in data
        assert b"TERSE" in data


def test_proxies_status(ui_server, backend_server):
    with urlopen(f"{ui_server}/v1/status") as resp:  # noqa: S310
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["demo_mode"] is True


def test_proxies_non_streaming_chat(ui_server):
    body = json.dumps(
        {
            "model": "terse-micro",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 8,
        }
    ).encode()
    req = Request(f"{ui_server}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})  # noqa: S310
    with urlopen(req) as resp:  # noqa: S310
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["choices"][0]["message"]["content"] == "hello world"


def test_proxies_streaming_chat(ui_server):
    body = json.dumps(
        {
            "model": "terse-micro",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 8,
        }
    ).encode()
    req = Request(f"{ui_server}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})  # noqa: S310
    with urlopen(req) as resp:  # noqa: S310
        assert resp.status == 200
        chunks = resp.read().decode().splitlines()
        data_chunks = [line for line in chunks if line.startswith("data: ")]
        assert len(data_chunks) >= 2
        assert "[DONE]" in data_chunks[-1]


def test_blocks_traversal(ui_server):
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{ui_server}/v1/../models")  # noqa: S310
    assert exc_info.value.code == 404


def test_blocks_encoded_traversal(ui_server):
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{ui_server}/v1/%2e%2e/models")  # noqa: S310
    assert exc_info.value.code == 404


def test_rejects_oversized_body(ui_server, monkeypatch):
    monkeypatch.setattr("terse.server.ui_server.MAX_BODY", 16)
    body = b'{"model":"terse-micro","messages":[]}'
    req = Request(
        f"{ui_server}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )  # noqa: S310
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req)  # noqa: S310
    assert exc_info.value.code == 413


def test_rejects_foreign_origin(ui_server):
    req = Request(f"{ui_server}/v1/status", headers={"Origin": "https://evil.example.com"})  # noqa: S310
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req)  # noqa: S310
    assert exc_info.value.code == 403


def test_index_has_security_headers(ui_server):
    with urlopen(f"{ui_server}/") as resp:  # noqa: S310
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "frame-ancestors" in resp.headers.get("Content-Security-Policy", "")


def test_streaming_uses_chunked_encoding(ui_server):
    body = json.dumps(
        {
            "model": "terse-micro",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 8,
        }
    ).encode()
    req = Request(f"{ui_server}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})  # noqa: S310
    with urlopen(req) as resp:  # noqa: S310
        assert resp.headers.get("Transfer-Encoding") == "chunked"
        resp.read()  # consume body to avoid server-side broken-pipe noise


def test_ui_directory_exists():
    assert UI_DIR.exists()
    assert (UI_DIR / "index.html").exists()


def test_main_exits_without_ui_dir(monkeypatch, capsys):
    monkeypatch.setattr("terse.server.ui_server.UI_DIR", Path("/nonexistent"))
    monkeypatch.setattr("sys.argv", ["serve_ui.py"])
    assert main() == 1
    captured = capsys.readouterr()
    assert "UI directory not found" in captured.err
