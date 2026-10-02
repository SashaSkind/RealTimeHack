"""One-command check of the Builders Challenge cloud stack: Cosmos3-Reason (text + image),
Cosmos Embed1, Canary-1B and W&B serverless inference. Prints status and latency, never secrets.

Credentials, in order of precedence:
  1. environment variables (already exported on the VAST VM from /config/<team>.config)
  2. KEY=VALUE lines in ./.env.event (laptop; gitignored)

  python bench/cloud_check.py                 # all checks
  python bench/cloud_check.py --wav clip.wav  # also transcribe a WAV with Canary
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GPU_HOST = "166.19.38.112"  # from the starter repo's .cursor/skills/gpu/README.md
DEFAULTS = {
    "COSMOS3_REASON_URL": f"http://{GPU_HOST}:8001",
    "COSMOS_EMBED1_URL": f"http://{GPU_HOST}:8003",
    "CANARY_1B_URL": f"http://{GPU_HOST}:8004",
}
WANDB_BASE = "https://api.inference.wandb.ai/v1"


def load_env() -> dict:
    env = dict(DEFAULTS)
    f = ROOT / ".env.event"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k in list(env) + ["GPU_BEARER_TOKEN", "WANDB_API_KEY", "WANDB_TEAM", "WANDB_PROJECT"]:
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def row(name, ok, ms=None, note=""):
    mark = "OK  " if ok else ("SKIP" if ok is None else "FAIL")
    t = f"{ms:7.0f} ms" if ms is not None else " " * 10
    print(f"[{mark}] {name:28s} {t}  {note}")


def timed(fn):
    t = time.perf_counter()
    try:
        return fn(), (time.perf_counter() - t) * 1000, None
    except Exception as e:
        return None, (time.perf_counter() - t) * 1000, f"{type(e).__name__}: {e}"[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="WAV file to transcribe with Canary-1B")
    ap.add_argument("--image", default=str(ROOT / "data/photos/cup_01.jpg"))
    ap.add_argument("--wandb-model", default=None, help="model id (default: first listed)")
    args = ap.parse_args()
    env = load_env()
    tok = env.get("GPU_BEARER_TOKEN")
    auth = {"Authorization": f"Bearer {tok}"} if tok else {}
    s = requests.Session()

    # ---- Cosmos3-Reason --------------------------------------------------------------
    base = env["COSMOS3_REASON_URL"].rstrip("/")
    if not tok:
        row("Cosmos3-Reason", None, note="GPU_BEARER_TOKEN not set (VM: /config/<team>.config)")
    else:
        r, ms, err = timed(lambda: s.get(f"{base}/v1/models", headers=auth, timeout=10))
        if err or r.status_code != 200:
            row("Cosmos3-Reason /v1/models", False, ms, err or f"HTTP {r.status_code} {r.text[:100]}")
        else:
            model = r.json()["data"][0]["id"]
            row("Cosmos3-Reason /v1/models", True, ms, model)
            body = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
                    "max_tokens": 16, "temperature": 0}
            r, ms, err = timed(lambda: s.post(f"{base}/v1/chat/completions", headers=auth, json=body, timeout=60))
            ok = not err and r.status_code == 200
            row("Cosmos3-Reason text", ok, ms,
                err or (r.json()["choices"][0]["message"]["content"][:60] if ok else f"HTTP {r.status_code}"))
            # image, through the same client the apps use (strict JSON, box)
            import cv2
            from handguide.slow_loop import CosmosClient
            rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
            for mode in ("schema", "prompt"):
                res = CosmosClient(f"{base}/v1", model=model, api_key=tok, json_mode=mode, timeout_s=60).ask(
                    rgb, "pick up the cup")
                row(f"Cosmos3-Reason image ({mode})", res.ok, res.latency_s * 1000,
                    (f"target={res.target!r} box={'yes' if res.box else 'no'} | {res.instruction[:50]}"
                     if res.ok else res.error[:120]))

    # ---- Cosmos Embed1 -----------------------------------------------------------------
    eb = env["COSMOS_EMBED1_URL"].rstrip("/")
    if not tok:
        row("Cosmos Embed1", None, note="no token")
    else:
        def emb():
            m = s.get(f"{eb}/v1/models", headers=auth, timeout=10).json()["data"][0]["id"]
            r = s.post(f"{eb}/v1/embeddings", headers=auth, timeout=30, json={
                "input": "a blender on a kitchen counter", "model": m, "request_type": "query",
                "encoding_format": "float"})
            r.raise_for_status()
            return len(r.json()["data"][0]["embedding"])
        dim, ms, err = timed(emb)
        row("Cosmos Embed1 text", not err and dim == 256, ms, err or f"dim={dim} (expect 256)")

    # ---- Canary-1B ---------------------------------------------------------------------
    cb = env["CANARY_1B_URL"].rstrip("/")
    if not tok:
        row("Canary-1B", None, note="no token")
    else:
        r, ms, err = timed(lambda: s.get(f"{cb}/v1/health/ready", headers=auth, timeout=10))
        row("Canary-1B health", not err and r.status_code == 200, ms, err or r.text[:60])
        if args.wav:
            def asr():
                with open(args.wav, "rb") as fh:
                    r = s.post(f"{cb}/v1/audio/transcriptions", headers=auth, timeout=60,
                               files={"file": fh}, data={"model": "nvidia/canary-1b"})
                r.raise_for_status()
                return r.json().get("text", r.text)[:80]
            txt, ms, err = timed(asr)
            row("Canary-1B transcribe", not err, ms, err or repr(txt))

    # ---- W&B serverless inference ------------------------------------------------------
    key = env.get("WANDB_API_KEY")
    if not key:
        row("W&B inference", None, note="WANDB_API_KEY not set (VM: already in the environment)")
    else:
        hdr = {"Authorization": f"Bearer {key}"}
        if env.get("WANDB_TEAM") and env.get("WANDB_PROJECT"):
            hdr["OpenAI-Project"] = f"{env['WANDB_TEAM']}/{env['WANDB_PROJECT']}"  # usage tracking
        r, ms, err = timed(lambda: s.get(f"{WANDB_BASE}/models", headers=hdr, timeout=15))
        if err or r.status_code != 200:
            row("W&B /v1/models", False, ms, err or f"HTTP {r.status_code} {r.text[:100]}")
        else:
            ids = [m["id"] for m in r.json().get("data", [])]
            row("W&B /v1/models", True, ms, f"{len(ids)} models")
            for i in ids:
                print(f"         - {i}")
            model = args.wandb_model or (ids[0] if ids else None)
            if model:
                body = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
                        "max_tokens": 16, "temperature": 0}
                r, ms, err = timed(lambda: s.post(f"{WANDB_BASE}/chat/completions", headers=hdr, json=body, timeout=60))
                ok = not err and r.status_code == 200
                row(f"W&B chat ({model})"[:28], ok, ms,
                    err or (r.json()["choices"][0]["message"]["content"][:60] if ok else f"HTTP {r.status_code} {r.text[:80]}"))


if __name__ == "__main__":
    main()
