"""Entry point for the Micro-Terse web UI server.

Usage:
    python scripts/serve_ui.py
    python scripts/serve_ui.py --port 3333 --api http://127.0.0.1:8080
"""
from __future__ import annotations

from terse.server.ui_server import main

if __name__ == "__main__":
    raise SystemExit(main())
