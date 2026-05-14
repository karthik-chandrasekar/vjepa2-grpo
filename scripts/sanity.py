"""Sanity check: verify the environment, GPU, and V-JEPA-2 load before D1.

Run this FIRST after setup.sh:
    python scripts/sanity.py

If anything fails, fix it before kicking off Phase 1.
"""
from __future__ import annotations
import sys
import traceback


def section(title):
    print(f"\n=== {title} ===")


def main():
    ok = True

    section("Python & basic deps")
    print("python:", sys.version)
    try:
        import torch
        print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            print("[FAIL] no CUDA"); ok = False
        else:
            print("device:", torch.cuda.get_device_name(0),
                  "cap:", torch.cuda.get_device_capability(0))
    except Exception:
        traceback.print_exc(); ok = False

    section("transformers + V-JEPA-2 weights")
    try:
        import transformers
        print("transformers:", transformers.__version__)
        from transformers import AutoVideoProcessor, AutoModel
        ENCODER_ID = "facebook/vjepa2-vitg-fpc64-384"
        print(f"loading {ENCODER_ID}... (will download ~4GB on first run)")
        proc = AutoVideoProcessor.from_pretrained(ENCODER_ID)
        model = AutoModel.from_pretrained(
            ENCODER_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda").eval()
        print("encoder params:", sum(p.numel() for p in model.parameters()) / 1e9, "B")

        # Tiny forward
        import numpy as np
        dummy = (np.random.rand(64, 384, 384, 3) * 255).astype(np.uint8)
        inputs = proc(videos=[dummy], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model(**inputs)
        lhs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        print("encoder out:", lhs.shape, lhs.dtype)
        # Expected: [1, 32 * 576, 1408] = [1, 18432, 1408]
        if lhs.shape[-1] != 1408:
            print(f"[WARN] unexpected embed dim {lhs.shape[-1]} (expected 1408)")
    except Exception:
        traceback.print_exc(); ok = False

    section("flash-attn / SDPA")
    try:
        try:
            import flash_attn
            print("flash_attn:", flash_attn.__version__)
        except ImportError:
            print("flash_attn: NOT installed (will fall back to SDPA)")
        # BF16 sdpa
        x = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.bfloat16)
        y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
        print("bf16 sdpa OK:", y.shape, y.dtype)
    except Exception:
        traceback.print_exc(); ok = False

    section("faiss")
    try:
        import faiss
        print("faiss:", faiss.__version__, "n_gpus:", faiss.get_num_gpus())
        import numpy as np
        idx = faiss.IndexFlatL2(128)
        idx.add(np.random.randn(1000, 128).astype("float32"))
        D, I = idx.search(np.random.randn(10, 128).astype("float32"), 5)
        print("faiss query OK:", D.shape, I.shape)
    except Exception:
        traceback.print_exc(); ok = False

    section("LIBERO (optional)")
    try:
        from libero.libero import benchmark
        bench = benchmark.get_benchmark_dict()["libero_spatial"]()
        print("libero_spatial tasks:", bench.n_tasks)
    except Exception as e:
        print("[INFO] LIBERO not importable; install via setup.sh's sim extras")
        print("       ", e)

    section("vjepa2_grpo package import")
    try:
        import vjepa2_grpo
        print("vjepa2_grpo:", vjepa2_grpo.__version__)
        from vjepa2_grpo import (
            VJepa2Encoder, BlockCausalACPredictor, ProgressCritic, FaissAnchorBuffer,
        )
        # Architecture smoke test
        p = BlockCausalACPredictor(d_model=256, n_layers=2, n_heads=4)  # tiny
        print("predictor params:", sum(x.numel() for x in p.parameters()) / 1e6, "M")
        c = ProgressCritic(d_model=128, n_layers=2, n_heads=4)
        print("critic params:", sum(x.numel() for x in c.parameters()) / 1e6, "M")
    except Exception:
        traceback.print_exc(); ok = False

    section("Summary")
    print("OK" if ok else "FAIL — fix above before proceeding")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
