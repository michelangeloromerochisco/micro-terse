"""Serve the Micro-Terse web UI and proxy API calls to the backend.

This is a library module. The entry point lives at ``scripts/serve_ui.py``.
"""
from __future__ import annotations

import argparse
import os
import posixpath
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_API = "http://127.0.0.1:8080"
DEFAULT_PORT = 3333
DEFAULT_HOST = "127.0.0.1"
UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"
PROXY_CHUNK = 8192
MAX_BODY = 10 * 1024 * 1024  # 10 MB cap on proxied request bodies


def _validate_api_url(url: str) -> str:
    """Reject non-HTTP(S) backend URLs."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"--api must use http:// or https://, got: {url}")
    if not parsed.hostname:
        raise ValueError(f"--api must include a host, got: {url}")
    return url.rstrip("/")


class ProxyHandler(SimpleHTTPRequestHandler):
    """Serve the static UI at / and proxy /v1/* to the Micro-Terse backend."""

    def __init__(self, *args, api_target: str = DEFAULT_API, **kwargs) -> None:
        self.api_target = api_target
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # noqa: ARG002
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_index()
            return
        if self._is_proxy_path():
            self._proxy("GET")
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self._is_proxy_path():
            self._proxy("POST")
            return
        self.send_error(404)

    def _is_proxy_path(self) -> bool:
        """Only proxy literal /v1/* paths; reject encoded traversal attempts."""
        path = self.path.split("?", 1)[0]
        # Decode percent-encoding repeatedly to catch %2e%2e / %252e%252e.
        for _ in range(3):
            decoded = urllib.parse.unquote(path)
            if decoded == path:
                break
            path = decoded
        norm = posixpath.normpath(path)
        if any(seg == ".." for seg in norm.split("/")):
            return False
        return norm.startswith("/v1/")

    def _serve_index(self) -> None:
        index_path = UI_DIR / "index.html"
        if not index_path.exists():
            self.send_error(404, "UI not found")
            return
        data = index_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';",
        )
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, method: str) -> None:
        origin = self.headers.get("Origin")
        host = self.headers.get("Host") or f"{self.server.server_address[0]}:{self.server.server_address[1]}"
        expected_origin = f"http://{host}"
        if origin and origin != expected_origin:
            self.send_error(403, "Cross-origin requests not allowed")
            return

        url = f"{self.api_target}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > MAX_BODY:
            self.send_error(413, "Payload too large")
            return
        body = self.rfile.read(content_length) if content_length else None

        req = Request(url, data=body, method=method)  # noqa: S310 — validated to http(s)://
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in ("host", "content-length", "connection"):
                continue
            req.add_header(key, value)

        try:
            with urlopen(req, timeout=120) as resp:  # noqa: S310
                self.send_response(resp.status)
                has_content_length = False
                for key, value in resp.headers.items():
                    lower = key.lower()
                    if lower in ("transfer-encoding", "connection"):
                        continue
                    if lower == "content-length":
                        has_content_length = True
                    self.send_header(key, value)
                if not has_content_length:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                if has_content_length:
                    while True:
                        chunk = resp.read(PROXY_CHUNK)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    while True:
                        chunk = resp.read(PROXY_CHUNK)
                        if not chunk:
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                            break
                        self.wfile.write(f"{len(chunk):X}\r\n".encode())
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
        except HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Backend error")
        except Exception:  # noqa: BLE001
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Proxy error: unable to reach backend")


def _handler_factory(api_target: str):
    def _make(*args, **kwargs):
        return ProxyHandler(*args, api_target=api_target, **kwargs)

    return _make


def main() -> int:
    parser = argparse.ArgumentParser(description="Micro-Terse web UI server")
    parser.add_argument("--host", default=os.getenv("TERSE_UI_HOST", DEFAULT_HOST), help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.getenv("TERSE_UI_PORT", DEFAULT_PORT)), help="Port for the UI server")
    parser.add_argument("--api", default=os.getenv("TERSE_API_URL", DEFAULT_API), help="Backend base URL")
    args = parser.parse_args()

    try:
        api_target = _validate_api_url(args.api)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not UI_DIR.exists():
        print(f"UI directory not found: {UI_DIR}", file=sys.stderr)
        return 1

    handler = _handler_factory(api_target)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Micro-Terse Web UI")
    print(f"Serving:  http://{args.host}:{args.port}")
    print(f"Proxying /v1/* to {api_target}")
    if args.host in {"", "0.0.0.0"}:
        print("Warning: bound to all interfaces — reachable from the local network")
    print("Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
