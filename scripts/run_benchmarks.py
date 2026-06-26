#!/usr/bin/env python3
"""Terse micro — full benchmark battery (the same suites Orchid was measured with).

Runs OFF-POD on the PC after pretraining + identity tuning finish. Chains:
  1. lm-eval-harness academic suite (logprob, comparable)  — runs on the checkpoint directly
  2. Orchid benchmark_100.py (600-question generative head-to-head) — via the Terse server
  3. Orchid evaluate.py (identity / bias / reasoning)              — via the Terse server
  4. Orchid bench_speed.py (TTFT, tok/s, prefill)                 — via the Terse server

Writes a combined benchmark_results.json. Each phase is independent: a phase that fails or
whose dependency is missing is recorded and skipped, never aborts the rest.

Usage:
  python scripts/run_benchmarks.py --checkpoint checkpoints/last.pt --orchid-root ../orchid
  python scripts/run_benchmarks.py --checkpoint <ckpt> --gguf terse-micro-orpo.gguf
  python scripts/run_benchmarks.py --checkpoint <ckpt> --skip-generative   # academic only
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
TERSE_ROOT = SCRIPT_DIR.parent


def _wait_for_server(url: str, timeout: float = 120.0) -> bool:
    """Poll /v1/models until the server answers or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def _run(cmd: list[str], cwd: Path | None = None) -> dict:
    """Run a subprocess, capture stdout/stderr/returncode."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=14400)
        sys.stdout.write(p.stdout[-4000:])
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-2000:])
        return {"cmd": [str(c) for c in cmd], "returncode": p.returncode,
                "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-2000:]}
    except Exception as e:  # noqa: BLE001 — record any failure, keep going
        return {"cmd": [str(c) for c in cmd], "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained checkpoint (.pt) for lm-eval + serving")
    ap.add_argument("--config", default="configs/micro-trained.yaml", help="real ~423M config")
    ap.add_argument("--gguf", default=None, help="serve this GGUF instead of the checkpoint")
    ap.add_argument("--orchid-root", default="../orchid", help="path to the Orchid repo")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--suite", choices=["fast", "full"], default="full")
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument("--skip-academic", action="store_true")
    ap.add_argument("--skip-generative", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="benchmark_100 pilot (5/category)")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    orchid = Path(args.orchid_root).resolve()
    results: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ── 1. Academic (lm-eval-harness, logprob) — comparable numbers ──────────
    if not args.skip_academic:
        print("\n===== [1/4] lm-eval academic suite =====")
        results["academic"] = _run(
            [sys.executable, "scripts/eval.py", "--ckpt", args.checkpoint,
             "--config", args.config, "--suite", args.suite],
            cwd=TERSE_ROOT)

    # ── 2-4. Generative suites need the Terse server running ─────────────────
    if not args.skip_generative:
        serve_cmd = [sys.executable, "scripts/serve.py", "--config", args.config, "--port", str(args.port)]
        serve_cmd += (["--gguf", args.gguf] if args.gguf else ["--checkpoint", args.checkpoint])
        print(f"\n===== booting Terse server: {' '.join(serve_cmd)} =====")
        server = subprocess.Popen(serve_cmd, cwd=TERSE_ROOT)
        try:
            if not _wait_for_server(url):
                results["generative_error"] = "server did not become ready"
                print("!! server never became ready — skipping generative suites", flush=True)
            else:
                b100 = orchid / "scripts" / "benchmark_100.py"
                evalpy = orchid / "scripts" / "evaluate.py"
                speed = orchid / "scripts" / "bench_speed.py"

                print("\n===== [2/4] Orchid 600-question intelligence suite =====")
                cmd = [sys.executable, str(b100), "--orchid-url", url]
                if args.pilot:
                    cmd.append("--pilot")
                results["intelligence_600q"] = _run(cmd, cwd=orchid) if b100.exists() else {"skip": "benchmark_100.py not found"}

                print("\n===== [3/4] Identity / bias =====")
                results["identity_bias"] = _run(
                    [sys.executable, str(evalpy), "--url", url], cwd=orchid
                ) if evalpy.exists() else {"skip": "evaluate.py not found"}

                print("\n===== [4/4] Speed =====")
                results["speed"] = _run(
                    [sys.executable, str(speed), "--url", url, "--tokens", "200"], cwd=orchid
                ) if speed.exists() else {"skip": "bench_speed.py not found"}
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except Exception:
                server.kill()

    out = TERSE_ROOT / args.out
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n✅ wrote {out}")


if __name__ == "__main__":
    main()
