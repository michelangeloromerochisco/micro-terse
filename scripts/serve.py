"""Serve a Micro-Terse checkpoint with an OpenAI-compatible chat API.

Usage:
    # Demo mode (no model loaded, placeholder responses)
    python scripts/serve.py --demo

    # Real checkpoint
    python scripts/serve.py --config configs/micro.yaml --checkpoint checkpoints/last.pt

    # Real checkpoint with CUDA
    python scripts/serve.py --config configs/micro.yaml --checkpoint checkpoints/last.pt --device cuda

    # Exported GGUF file
    python scripts/serve.py --config configs/micro.yaml --gguf terse.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel
from terse.server.app import create_app
from terse.server.gguf_loader import load_gguf_model


def _load_config(path: str) -> TerseConfig:
    """Build a TerseConfig from the YAML ``model`` block.

    Every field declared on TerseConfig is forwarded so the served architecture
    exactly matches the trained one — silently dropping a key (e.g.
    ``tie_embeddings`` or ``moe_layers``) would build a mismatched model that
    fails to load the GGUF/checkpoint tensors. Unknown keys are ignored, and
    ``gradient_checkpointing`` is forced off for inference.
    """
    from dataclasses import fields

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("model", {}) or {}

    valid = {f.name for f in fields(TerseConfig)}
    kwargs = {k: v for k, v in cfg.items() if k in valid}
    kwargs["gradient_checkpointing"] = False
    return TerseConfig(**kwargs)


def _load_tokenizer(tokenizer_name: str = "NousResearch/Meta-Llama-3.1-8B"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def _load_progression(*hint_paths: str | None) -> list | None:
    """Load the precomputed base->SFT->ORPO identity progression, if present.

    Looks for ``identity_proof.json`` next to the model and/or config so the
    /v1/identity_proof endpoint can show how training moved the preference.
    """
    for hint in hint_paths:
        if not hint:
            continue
        candidate = Path(hint).resolve().parent / "identity_proof.json"
        if candidate.exists():
            try:
                return json.loads(candidate.read_text()).get("stages")
            except (json.JSONDecodeError, OSError):
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Micro-Terse OpenAI-compatible server")
    parser.add_argument("--config", default="configs/micro.yaml", help="Path to model YAML config")
    parser.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint")
    parser.add_argument("--gguf", default=None, help="Path to exported GGUF file (loads instead of --checkpoint)")
    parser.add_argument(
        "--proof-gguf",
        default=None,
        help="Optional second GGUF used only for /v1/identity_proof (e.g. the ORPO model "
        "while chat serves the more fluent SFT model)",
    )
    parser.add_argument(
        "--base-gguf",
        default=None,
        help="Optional GGUF used only for /v1/predict* next-token demo (the base "
        "pretrained model is strongest here)",
    )
    parser.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B", help="Tokenizer name or path")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu/cuda)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--demo", action="store_true", help="Demo mode with placeholder responses")
    parser.add_argument("--demo-response", default=None, help="Custom placeholder response for demo mode")
    parser.add_argument(
        "--cors-origins",
        default=None,
        help="Comma-separated allowed CORS origins (default: http://localhost:5173,http://127.0.0.1:5173)",
    )
    parser.add_argument("--api-key", default=None, help="Optional Bearer token for API access")
    args = parser.parse_args()

    if not args.demo and not args.checkpoint and not args.gguf:
        print("Error: pass --checkpoint, --gguf, or --demo", file=sys.stderr)
        return 1

    if args.checkpoint and args.gguf:
        print("Error: pass --checkpoint or --gguf, not both", file=sys.stderr)
        return 1

    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in loopback_hosts and not args.api_key:
        print(
            "Warning: server is bound to a non-loopback address without --api-key. "
            "Add --api-key to require Bearer authentication.",
            file=sys.stderr,
        )

    model = None
    tokenizer = None

    if not args.demo:
        cfg = _load_config(args.config)
        model = TerseModel(cfg).to(args.device)
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
                return 1
            state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            # Accept raw state_dict or a wrapped training/stripped checkpoint {"model": ...}.
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
            model.eval()
            print(f"Loaded checkpoint from {args.checkpoint}")
        elif args.gguf:
            if not os.path.exists(args.gguf):
                print(f"GGUF file not found: {args.gguf}", file=sys.stderr)
                return 1
            model = load_gguf_model(args.gguf, config=cfg, device=args.device)
            print(f"Loaded GGUF from {args.gguf}")
        tokenizer = _load_tokenizer(args.tokenizer)
        print(f"Loaded tokenizer {args.tokenizer}")

    proof_model = None
    if not args.demo and args.proof_gguf:
        if not os.path.exists(args.proof_gguf):
            print(f"Proof GGUF not found: {args.proof_gguf}", file=sys.stderr)
            return 1
        proof_model = load_gguf_model(args.proof_gguf, config=cfg, device=args.device)
        print(f"Loaded proof model (identity proof) from {args.proof_gguf}")

    base_model = None
    if not args.demo and args.base_gguf:
        if not os.path.exists(args.base_gguf):
            print(f"Base GGUF not found: {args.base_gguf}", file=sys.stderr)
            return 1
        base_model = load_gguf_model(args.base_gguf, config=cfg, device=args.device)
        print(f"Loaded base model (next-token predict) from {args.base_gguf}")

    cors_origins = args.cors_origins.split(",") if args.cors_origins else None
    progression = _load_progression(args.proof_gguf, args.gguf, args.checkpoint, args.config)
    if progression:
        print(f"Loaded identity-proof progression ({len(progression)} stages)")
    app = create_app(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        demo_mode=args.demo,
        demo_response=args.demo_response,
        cors_origins=cors_origins,
        api_key=args.api_key,
        identity_progression=progression,
        proof_model=proof_model,
        base_model=base_model,
    )

    import uvicorn

    print(f"Micro-Terse server running at http://{args.host}:{args.port}")
    if args.demo:
        print("Demo mode: returning placeholder responses")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
