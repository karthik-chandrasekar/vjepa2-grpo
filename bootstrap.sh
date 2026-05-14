#!/usr/bin/env bash
# bootstrap.sh — one-command setup for a fresh pod with the network volume
# re-attached at /workspace.
#
# A fresh RunPod container keeps NOTHING on /root (venv, pip packages, the
# libero .pth, the libero config all live there). The network volume keeps
# the repo, the LIBERO clone, the data, and the HF cache. This script rebuilds
# everything container-local so a reclaimed pod is ~15 min from working
# instead of replaying a whole debugging session.
#
# Usage on a fresh pod:
#   cd /workspace/vjepa2_grpo && bash bootstrap.sh
#
# Idempotent — safe to re-run.
set -euo pipefail

REPO=/workspace/vjepa2_grpo
VENV=/root/venv
LIBERO_DIR=/workspace/LIBERO

echo "=== bootstrap: fresh-pod setup ==="

# --- 0. sanity: volume must be mounted with our state on it ---
for d in "$REPO" "$LIBERO_DIR" /workspace/data /workspace/.hf_cache; do
    if [ ! -d "$d" ]; then
        echo "FATAL: $d not found. Is the network volume attached at /workspace?"
        exit 1
    fi
done
echo "[ok] network volume present (repo, LIBERO, data, hf_cache)"

# --- 1. base environment (apt + venv + torch + project deps + flash-attn) ---
# setup.sh handles apt deps, venv creation, torch, `pip install -e .`, sim extras.
echo "[1/5] running setup.sh ..."
bash "$REPO/setup.sh"

source "$VENV/bin/activate"

# --- 2. packages setup.sh misses (transitive deps that aren't pinned) ---
echo "[2/5] installing packages setup.sh doesn't pin ..."
pip install -q huggingface_hub pytest

# --- 3. flash-attn (correct version string; setup.sh used a non-existent one) ---
echo "[3/5] flash-attn ..."
if python -c "import flash_attn" 2>/dev/null; then
    echo "[ok] flash-attn already present"
else
    MAX_JOBS=4 pip install flash-attn==2.7.2.post1 --no-build-isolation --no-cache-dir \
        || echo "[WARN] flash-attn build failed; SDPA will fall back to mem-efficient kernel"
fi

# --- 4. LIBERO: make `import libero.libero` work without its broken setup.py ---
#   LIBERO's setup.py produces an empty wheel (missing top-level __init__.py +
#   find_packages() misconfig). We bypass pip entirely: ensure the top-level
#   package marker exists, then put the repo root on the venv path via a .pth.
echo "[4/5] LIBERO path setup ..."
touch "$LIBERO_DIR/libero/__init__.py"
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$LIBERO_DIR" > "$SITE_PKGS/libero_local.pth"
python -c "import libero.libero" >/dev/null 2>&1 \
    && echo "[ok] import libero.libero works" \
    || echo "[WARN] import libero.libero still failing — check $LIBERO_DIR layout"

# --- 5. LIBERO config (lives on container-local /root; restore from volume) ---
echo "[5/5] LIBERO config ..."
if [ -f /workspace/.libero/config.yaml ]; then
    mkdir -p /root/.libero
    cp /workspace/.libero/config.yaml /root/.libero/config.yaml
    echo "[ok] restored /root/.libero/config.yaml from volume"
else
    echo "[WARN] /workspace/.libero/config.yaml not found — first 'import libero.libero'"
    echo "       will prompt interactively; answer with dataset path /workspace/data/libero"
    echo "       then: mkdir -p /workspace/.libero && cp /root/.libero/config.yaml /workspace/.libero/"
fi

echo ""
echo "=== bootstrap complete ==="
echo "verify:  python scripts/sanity.py"
