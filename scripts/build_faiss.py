"""Build the FAISS anchor buffer from precomputed embeddings.

Run AFTER precompute_embeddings.py has finished. Typical invocation:

    python scripts/build_faiss.py \
        --emb-root /workspace/data/embeddings \
        --out /workspace/data/anchor_buffer \
        --nlist 4096 \
        --subsample 5

Notes:
  - The buffer stores per-frame embeddings (8x8 patches mean-pooled to one
    1408-d vector). Subsampling=5 reduces a 10M-frame corpus to 2M vectors
    (~11 GB in fp32) which fits comfortably in CPU + GPU index.
  - GPU build is much faster (~5min) but uses ~6 GB of VRAM; use --cpu if
    you need to free the GPU for parallel training.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vjepa2_grpo.faiss_anchor import FaissAnchorBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-root", required=True,
                    help="root containing precomputed .npy embeddings")
    ap.add_argument("--out", required=True, help="output dir for index.ivf + ids.npy")
    ap.add_argument("--dim", type=int, default=1408)
    ap.add_argument("--nlist", type=int, default=4096,
                    help="IVF clusters; 4096 is a good default for 2M-10M vectors")
    ap.add_argument("--subsample", type=int, default=5,
                    help="keep every Nth frame from each demo (default 5)")
    ap.add_argument("--train-size", type=int, default=200_000,
                    help="IVF training subset size")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU-only build (slower; releases GPU for training)")
    args = ap.parse_args()

    buf = FaissAnchorBuffer.build_from_embeddings(
        emb_dir=args.emb_root,
        out_dir=args.out,
        dim=args.dim,
        nlist=args.nlist,
        subsample=args.subsample,
        train_size=args.train_size,
        use_gpu_for_build=not args.cpu,
    )

    # Smoke test: query a few random vectors
    import numpy as np
    q = np.random.randn(8, args.dim).astype(np.float32)
    D, I = buf.query(q, k=4)
    print(f"\n[build_faiss] smoke query: D.shape={D.shape}  mean_d={D.mean():.3f}")
    print(f"[build_faiss] index at {args.out}/index.ivf")


if __name__ == "__main__":
    main()
