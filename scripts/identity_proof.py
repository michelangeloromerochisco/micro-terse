"""Precompute the identity-preference progression across base -> SFT -> ORPO.

Loads each F32 GGUF on CPU, scores the shared identity probes, and writes a small
JSON the demo can render to show that *training* moved the model from preferring
the "ChatGPT" identity to preferring "Terse". This is the offline companion to
the live ``/v1/identity_proof`` endpoint (both use ``terse.server.identity``).

    python scripts/identity_proof.py
    python scripts/identity_proof.py --out path.json --artifacts DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from terse.model.config import TerseConfig
from terse.server.gguf_loader import load_gguf_model
from terse.server.identity import IDENTITY_PROBES, identity_margins

_STAGES = [
    ("base", "terse-micro-base.gguf"),
    ("sft", "terse-micro-sft.gguf"),
    ("orpo", "terse-micro-orpo.gguf"),
]


def _default_artifacts() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "terse-artifacts"


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute identity-proof progression")
    ap.add_argument("--artifacts", default=str(_default_artifacts()), help="Dir holding the GGUFs + micro.yaml")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <artifacts>/identity_proof.json)")
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B")
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    out_path = Path(args.out) if args.out else artifacts / "identity_proof.json"
    config_path = artifacts / "micro.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = TerseConfig(**yaml.safe_load(config_path.read_text())["model"])
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    stages = []
    for name, filename in _STAGES:
        path = artifacts / filename
        if not path.exists():
            print(f"  skip {name}: {path} missing", file=sys.stderr)
            continue
        print(f"  scoring {name} ({filename})...", flush=True)
        model = load_gguf_model(str(path), config=cfg, device="cpu").eval()
        probes = identity_margins(model, tok, "cpu")
        preferred = sum(1 for p in probes if p["prefers_charter"])
        avg = sum(p["margin"] for p in probes) / max(1, len(probes))
        stages.append(
            {"stage": name, "probes": probes, "preferred": preferred, "total": len(probes), "avg_margin": avg}
        )
        del model

    payload = {"stages": stages, "probe_count": len(IDENTITY_PROBES)}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
