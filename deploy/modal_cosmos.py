"""Cosmos Reason 2 on Modal: vLLM with an OpenAI-compatible API, same shape as Friday's CoreWeave setup.

Deploy:   modal deploy deploy/modal_cosmos.py              (prints the URL; also written to .env.cosmos)
GPU:      COSMOS_GPU=A100-80GB modal deploy deploy/modal_cosmos.py   (default H100)
Model:    COSMOS_HF_MODEL=nvidia/Cosmos-Reason2-2B modal deploy ...
Stop:     modal app stop cosmos-reason2   (containers also scale to zero after 5 idle minutes)

Secrets: the HF token (for the gated model) is read from ~/.cache/huggingface/token and the vLLM API key
from .env.cosmos (generated on first deploy). Both are sent to Modal as an inline secret at deploy time.
The endpoint URL is public, so vLLM requires that API key on every request.
"""
import os
import secrets
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env.cosmos"
MODEL = os.environ.get("COSMOS_HF_MODEL", "nvidia/Cosmos-Reason2-8B")
GPU = os.environ.get("COSMOS_GPU", "H100")
APP_NAME = "cosmos-reason2" + ("-2b" if MODEL.endswith("2B") else "")


def _local_secret() -> dict:
    """Runs only on the deploying machine; returns {} inside the container."""
    tok_file = Path.home() / ".cache" / "huggingface" / "token"
    if not tok_file.exists():
        return {}
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    key = env.get("COSMOS_API_KEY") or secrets.token_urlsafe(32)
    if "COSMOS_API_KEY" not in env:
        ENV_FILE.write_text(f"COSMOS_API_KEY={key}\nCOSMOS_MODEL=cosmos\n")
        ENV_FILE.chmod(0o600)
    return {"HF_TOKEN": tok_file.read_text().strip(), "VLLM_API_KEY": key}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
)
hf_cache = modal.Volume.from_name("cosmos-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("cosmos-vllm-cache", create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache/hf": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_dict(_local_secret())],
    scaledown_window=5 * 60,   # scale to zero after 5 idle minutes; a cold start costs ~2-4 min
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", "cosmos",
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "8192",
        "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
        "--gpu-memory-utilization", "0.90",
    ]  # vLLM reads VLLM_API_KEY from the environment and then requires it
    subprocess.Popen(cmd)
