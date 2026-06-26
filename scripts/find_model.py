"""Locate the best available Micro-Terse model for serving.

Search order (first match wins):
  0. $TERSE_MODEL (explicit override; .pt -> checkpoint, .gguf -> gguf)
  1. The canonical chat models in terse-artifacts, by name and fluency:
     terse-micro-sft.gguf (most fluent) then terse-micro-orpo.gguf
     (identity is shown separately by the proof model — see find_proof_model)
  2. checkpoints/last.pt
  3. Any *.pt in the terse-artifacts sibling directory
  4. Any *.gguf in the terse-artifacts sibling directory
  5. Any *.gguf in the terse package directory
  6. Demo mode

The named-artifact preference (1) is deliberate: terse-artifacts also holds
older intermediate checkpoints (e.g. the 50%-pretrained chat_244_*.pt), and a
plain "latest .pt" rule would serve those instead of the finished model.

Usage:
    python scripts/find_model.py
    for /f "tokens=*" %%a in ('python scripts/find_model.py') do set MODEL_PATH=%%a
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _terse_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _artifacts_dir() -> Path | None:
    root = _terse_root()
    candidate = root.parent.parent / "terse-artifacts"
    if candidate.is_dir():
        return candidate
    return None


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


# Chat model preference: SFT first — it's measurably more fluent than ORPO
# (ORPO trades general fluency for identity alignment). Identity is shown
# separately via the proof model below, so chat optimizes for coherence.
_CHAT_PREFERRED = ("terse-micro-sft.gguf", "terse-micro-orpo.gguf")
# Proof model: the identity-aligned ORPO model drives /v1/identity_proof.
_PROOF_PREFERRED = ("terse-micro-orpo.gguf",)
# Base model: the pretrained model drives /v1/predict* (next-token demo).
_BASE_PREFERRED = ("terse-micro-base.gguf",)


def _mode_for(path: Path) -> str:
    return "gguf" if path.suffix.lower() == ".gguf" else "checkpoint"


def _first_present(names: tuple[str, ...]) -> str:
    artifacts = _artifacts_dir()
    if artifacts:
        for name in names:
            candidate = artifacts / name
            if candidate.exists():
                return str(candidate)
    return ""


def find_proof_model() -> str:
    """Path to the identity-proof model (ORPO), or "" if not present."""
    return _first_present(_PROOF_PREFERRED)


def find_base_model() -> str:
    """Path to the base pretrained model (next-token demo), or "" if absent."""
    return _first_present(_BASE_PREFERRED)


def find_model() -> dict[str, str]:
    root = _terse_root()
    artifacts = _artifacts_dir()

    # Default / planned 1.06B spec config
    default_config = str(root / "configs" / "micro.yaml")
    # Config matching the right-sized 423M training run
    trained_config = str(root / "configs" / "micro-trained.yaml")

    # 0. Explicit override
    override = os.environ.get("TERSE_MODEL", "").strip()
    if override and Path(override).exists():
        return {"mode": _mode_for(Path(override)), "path": override, "config": trained_config}

    # 1. Canonical final chat models by name (skip older intermediate checkpoints)
    if artifacts:
        for name in _CHAT_PREFERRED:
            candidate = artifacts / name
            if candidate.exists():
                return {"mode": "gguf", "path": str(candidate), "config": trained_config}

    # 2. Local checkpoint
    last_pt = root / "checkpoints" / "last.pt"
    if last_pt.exists():
        return {"mode": "checkpoint", "path": str(last_pt), "config": trained_config}

    # 3. Any .pt in terse-artifacts (real PyTorch weights, preferred)
    if artifacts:
        pt = _latest_file(artifacts, "*.pt")
        if pt:
            return {"mode": "checkpoint", "path": str(pt), "config": trained_config}

    # 4. Any .gguf in terse-artifacts (requires llama.cpp terse arch)
    if artifacts:
        gguf = _latest_file(artifacts, "*.gguf")
        if gguf:
            return {"mode": "gguf", "path": str(gguf), "config": trained_config}

    # 5. Any .gguf in terse package directory
    local_gguf = _latest_file(root, "*.gguf")
    if local_gguf:
        return {"mode": "gguf", "path": str(local_gguf), "config": trained_config}

    return {"mode": "demo", "path": "", "config": default_config}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Find the best available Micro-Terse model")
    parser.add_argument("--format", choices=["json", "bat"], default="json", help="Output format")
    args = parser.parse_args()

    info = find_model()
    proof = find_proof_model()
    base = find_base_model()
    # Don't reuse the chat model as an auxiliary model — an empty value means the
    # endpoint falls back to whatever chat model is loaded.
    if proof == info.get("path"):
        proof = ""
    if base == info.get("path"):
        base = ""
    if args.format == "bat":
        print(f"MODE={info['mode']}")
        print(f"PATH={info['path']}")
        print(f"CONFIG={info['config']}")
        print(f"PROOF={proof}")
        print(f"BASE={base}")
    else:
        info = {**info, "proof": proof, "base": base}
        print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
