"""V-JEPA-2 encoder HTTP server (venv1 process).

OpenVLA-OFT needs an old transformers (4.40.1, from moojink's fork); V-JEPA-2
needs a recent transformers (5.x). They can't share a venv. This server lets
the GRPO process (in venv-oft) reach the V-JEPA-2 encoder (in venv1) over
localhost HTTP.

Architecture:
    venv-oft GRPO process  --(HTTP localhost:8765)-->  venv1 encoder server

Lifecycle is independent: start the server first in one tmux pane, launch GRPO
in another. Killing one doesn't affect the other; the GRPO client retries on
ConnectionError so a server restart doesn't kill a long run.

Usage (in venv1):
    source /root/venv/bin/activate
    python scripts/encoder_server.py --port 8765

Endpoints:
    GET  /health        -> {"status": "ready", "device": "cuda:0", "pool_hw": 8}
    POST /encode_clip   -> body: {"array_b64": "..."} where array is [T,H,W,3] uint8
                          returns {"array_b64": "..."} where array is [T_,8,8,1408] fp16
    POST /encode_single -> body: {"array_b64": "..."} where array is [H,W,3] uint8
                          returns {"array_b64": "..."} where array is [8,8,1408] fp16

Wire format:  np.save(BytesIO, arr) -> base64 -> JSON.
This carries dtype + shape automatically; client just np.load(BytesIO(b64decode(...))).
"""
from __future__ import annotations
import argparse
import base64
import io
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from flask import Flask, request, jsonify

from vjepa2_grpo.encoder import VJepa2Encoder, EMBED_DIM


# ---------------------------------------------------------------------------
# np-array <-> base64 (server side; the client has an identical pair)
# ---------------------------------------------------------------------------

def np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def np_from_b64(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app(encoder: VJepa2Encoder, device: str) -> Flask:
    app = Flask(__name__)
    # serialize requests — V-JEPA-2 forward isn't designed for concurrent CUDA streams
    # (Flask dev server is single-threaded by default; we make that explicit at run())

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ready",
            "device": str(device),
            "pool_hw": encoder.pool_hw,
            "embed_dim": EMBED_DIM,
            "model_dtype": str(encoder.dtype),
        })

    @app.post("/encode_clip")
    def encode_clip():
        try:
            payload = request.get_json(force=True)
            frames = np_from_b64(payload["array_b64"])         # [T,H,W,3] uint8
            if frames.ndim != 4 or frames.shape[-1] != 3:
                return jsonify({"error": f"bad shape {frames.shape}; need [T,H,W,3]"}), 400
            t0 = time.perf_counter()
            with torch.inference_mode():
                z = encoder.encode_clip(frames)                # [T_, 8, 8, 1408] in encoder.dtype, on CPU
            # bf16 -> fp16 for transport (numpy has no bf16)
            arr = z.float().numpy().astype(np.float16)
            dt = time.perf_counter() - t0
            return jsonify({"array_b64": np_to_b64(arr),
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "encode_ms": int(dt * 1000)})
        except Exception as e:
            import traceback
            return jsonify({"error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()}), 500

    @app.post("/encode_single")
    def encode_single():
        try:
            payload = request.get_json(force=True)
            frame = np_from_b64(payload["array_b64"])          # [H,W,3] uint8
            if frame.ndim != 3 or frame.shape[-1] != 3:
                return jsonify({"error": f"bad shape {frame.shape}; need [H,W,3]"}), 400
            t0 = time.perf_counter()
            with torch.inference_mode():
                z = encoder.encode_single_observation(frame)   # [8, 8, 1408] in encoder.dtype, on CPU
            arr = z.float().numpy().astype(np.float16)
            dt = time.perf_counter() - t0
            return jsonify({"array_b64": np_to_b64(arr),
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "encode_ms": int(dt * 1000)})
        except Exception as e:
            import traceback
            return jsonify({"error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()}), 500

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address. Default localhost (do NOT expose externally)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pool-hw", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[encoder-server] loading V-JEPA-2 on {args.device}...", flush=True)
    encoder = VJepa2Encoder(pool_hw=args.pool_hw,
                            dtype=torch.bfloat16,
                            device=args.device)
    n_params = sum(p.numel() for p in encoder.model.parameters()) / 1e9
    print(f"[encoder-server] ready. params={n_params:.2f}B  pool_hw={args.pool_hw}",
          flush=True)

    app = build_app(encoder, args.device)
    # threaded=False: serialize requests (cleanest for a single-GPU model);
    # use_reloader=False: don't double-load the model when started in __main__.
    print(f"[encoder-server] listening on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
