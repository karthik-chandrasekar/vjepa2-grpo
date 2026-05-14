#!/usr/bin/env bash
# setup.sh — one-shot pod setup. Idempotent. Respects existing conventions:
#   - venv lives on container-local /root/ (ephemeral, fast)
#   - all heavy I/O on MooseFS /workspace
#   - UV_LINK_MODE=copy, HF_HUB_ENABLE_HF_TRANSFER=0
set -euxo pipefail

export UV_LINK_MODE=copy
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/workspace/.hf_cache
export TRANSFORMERS_CACHE=/workspace/.hf_cache
export TORCH_HOME=/workspace/.torch_cache

mkdir -p /workspace/.hf_cache /workspace/.torch_cache /workspace/logs \
         /workspace/data /workspace/checkpoints

# System deps
apt-get update
apt-get install -y --no-install-recommends \
    git tmux htop ffmpeg libgl1 libosmesa6-dev patchelf \
    build-essential pkg-config

# venv
if [ ! -d /root/venv ]; then
    python3 -m venv /root/venv
fi
source /root/venv/bin/activate
pip install -U pip wheel setuptools uv

# Pinned torch (CUDA 12.4)
uv pip install \
    "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu124

# Project deps
uv pip install -e .

# Flash-attn (optional; falls back to SDPA if it fails)
uv pip install "flash-attn==2.7.2" --no-build-isolation || echo "FA2 install failed; will use SDPA"

# Sim stack (optional, only needed for LIBERO eval)
uv pip install ".[sim]" || echo "Sim deps failed; LIBERO eval will not run"

echo ""
echo "=== Install complete. Run: python scripts/sanity.py ==="
