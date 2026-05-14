"""FAISS-backed reconstruction-consistency anchor buffer.

The anchor is the core methodological contribution: for every rolled-out
latent z_tilde_t, we compute L_anchor(z_tilde_t) = || z_tilde_t - NN(z_tilde_t) ||^2
where NN(.) is the L2-nearest neighbor in a frozen buffer of real-encoder
embeddings. When the predictor drifts off the manifold of real V-JEPA-2
embeddings, L_anchor grows and dominates the reward.

Implementation choices:
  - Per-frame anchor (mean-pool the 8x8 spatial patches to one vector per
    frame). The 8x8 spatial structure is preserved in the critic, but the
    anchor is per-frame to keep the buffer < 2 TB and the query < 2 ms.
  - IVF index with nlist=4096, nprobe=16 (tunable). Recall@1 on 2M-vector
    buffer at these settings is typically > 95%.
  - GPU index for build, GPU or CPU for query (GPU is faster but uses ~6 GB).
"""
from __future__ import annotations
import os
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List


class FaissAnchorBuffer:
    """Wraps a FAISS index plus the id->source mapping.

    Usage:
        buf = FaissAnchorBuffer.build_from_embeddings(
            emb_dir="/workspace/data/embeddings",
            out_dir="/workspace/data/anchor_buffer",
            dim=1408, nlist=4096, subsample=5,
        )
        # later
        buf = FaissAnchorBuffer.load("/workspace/data/anchor_buffer")
        dists, ids = buf.query(z_pool_np, k=1)
    """

    def __init__(self, index, ids: np.ndarray, dim: int = 1408,
                 nprobe: int = 16, on_gpu: bool = False):
        self.index = index
        self.ids = ids
        self.dim = dim
        self.nprobe = nprobe
        self.on_gpu = on_gpu
        try:
            self.index.nprobe = nprobe
        except Exception:
            pass

    @classmethod
    def build_from_embeddings(
        cls,
        emb_dir: str,
        out_dir: str,
        dim: int = 1408,
        nlist: int = 4096,
        subsample: int = 5,
        train_size: int = 200_000,
        use_gpu_for_build: bool = True,
    ) -> "FaissAnchorBuffer":
        """Build the FAISS index from .npy embedding files.

        Each .npy file is expected to contain [T, pool_hw, pool_hw, D] float16
        or bfloat16 (we cast to float32 for FAISS).

        Returns a constructed `FaissAnchorBuffer` and writes:
          out_dir/index.ivf
          out_dir/ids.npy
        """
        import faiss
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # First pass: collect mean-pooled per-frame embeddings + source ids
        zs, ids = [], []
        emb_root = Path(emb_dir)
        files = sorted(emb_root.rglob("*.npy"))
        print(f"[anchor] scanning {len(files)} embedding files under {emb_root}")
        for f in files:
            z = np.load(f, mmap_mode="r")  # [T, H, W, D]
            if z.ndim != 4:
                continue
            zf = z.astype(np.float32).mean(axis=(1, 2))  # [T, D]
            zf = zf[::subsample]
            zs.append(zf)
            for t_idx in range(zf.shape[0]):
                ids.append((str(f.relative_to(emb_root)), t_idx * subsample))
        Z = np.concatenate(zs, axis=0).astype(np.float32)
        print(f"[anchor] total vectors: {Z.shape[0]:,}  dim={Z.shape[1]}")
        assert Z.shape[1] == dim, f"expected dim={dim}, got {Z.shape[1]}"

        # Build IVF index
        quant = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)

        train_idx = np.random.RandomState(0).choice(
            Z.shape[0], min(train_size, Z.shape[0]), replace=False
        )
        if use_gpu_for_build and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[anchor] training IVF on GPU...")
            gpu_index.train(Z[train_idx])
            print("[anchor] adding vectors to GPU index...")
            # Add in chunks to avoid OOM
            chunk = 200_000
            for i in range(0, Z.shape[0], chunk):
                gpu_index.add(Z[i : i + chunk])
            index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            print("[anchor] training IVF on CPU...")
            index.train(Z[train_idx])
            chunk = 200_000
            for i in range(0, Z.shape[0], chunk):
                index.add(Z[i : i + chunk])

        # Persist
        faiss.write_index(index, str(out / "index.ivf"))
        np.save(out / "ids.npy", np.array(ids, dtype=object), allow_pickle=True)
        print(f"[anchor] wrote {out}/index.ivf and ids.npy")

        return cls(index=index, ids=np.array(ids, dtype=object), dim=dim, nprobe=16)

    @classmethod
    def load(cls, path: str, to_gpu: bool = True) -> "FaissAnchorBuffer":
        import faiss
        p = Path(path)
        index = faiss.read_index(str(p / "index.ivf"))
        ids = np.load(p / "ids.npy", allow_pickle=True)
        on_gpu = False
        if to_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            on_gpu = True
        return cls(index=index, ids=ids, on_gpu=on_gpu)

    def query(
        self,
        z: np.ndarray | torch.Tensor,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (squared_L2_distances [N, k], neighbor_ids [N, k])."""
        if isinstance(z, torch.Tensor):
            z = z.detach().float().cpu().numpy()
        z = np.ascontiguousarray(z.astype(np.float32))
        D, I = self.index.search(z, k)
        return D, I

    def anchor_distance(
        self,
        z_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: take [B, T, P, D] predicted latents, mean-pool spatial,
        return [B, T] squared distance to nearest real-encoder embedding.

        This is the reward-side L_anchor used in GRPO.
        """
        assert z_pred.ndim == 4, f"expected [B,T,P,D], got {z_pred.shape}"
        B, T, P, D = z_pred.shape
        z_pool = z_pred.mean(dim=2)                     # [B, T, D]
        flat = z_pool.reshape(B * T, D)
        d, _ = self.query(flat, k=1)
        d = torch.from_numpy(d).to(z_pred.device).reshape(B, T)
        return d
