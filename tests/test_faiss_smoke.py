"""FAISS anchor buffer smoke test.

Builds a tiny synthetic embedding directory, constructs the index, queries it,
and verifies the round-trip works without GPU.
"""
import json
import numpy as np
import pytest
import torch
from pathlib import Path


def test_anchor_buffer_build_and_query(tmp_path):
    pytest.importorskip("faiss")
    from vjepa2_grpo.faiss_anchor import FaissAnchorBuffer

    # Synthetic embeddings: 4 demos, each [T_=8, 8, 8, 32]
    emb_root = tmp_path / "emb"
    emb_root.mkdir()
    for i in range(4):
        d = emb_root / f"task_{i}"
        d.mkdir()
        z = np.random.randn(8, 8, 8, 32).astype(np.float16)
        np.save(d / f"demo_{i}.npy", z)
        (d / f"demo_{i}.json").write_text(json.dumps({
            "actions": [[0.0] * 7] * 16,
            "proprio": [[0.0] * 14] * 16,
            "instruction": "test", "success": 1, "task_key": f"task_{i}",
        }))

    out_dir = tmp_path / "buf"
    buf = FaissAnchorBuffer.build_from_embeddings(
        emb_dir=str(emb_root), out_dir=str(out_dir),
        dim=32, nlist=4, subsample=1, train_size=16, use_gpu_for_build=False,
    )
    # Query 10 random vectors
    q = np.random.randn(10, 32).astype(np.float32)
    D, I = buf.query(q, k=2)
    assert D.shape == (10, 2)
    assert I.shape == (10, 2)

    # Load round-trip
    buf2 = FaissAnchorBuffer.load(str(out_dir), to_gpu=False)
    D2, I2 = buf2.query(q, k=2)
    np.testing.assert_array_equal(D, D2)
    np.testing.assert_array_equal(I, I2)


def test_anchor_distance_torch(tmp_path):
    pytest.importorskip("faiss")
    from vjepa2_grpo.faiss_anchor import FaissAnchorBuffer

    # Build a tiny buffer in memory
    emb_root = tmp_path / "emb"
    emb_root.mkdir()
    d = emb_root / "t"
    d.mkdir()
    z = np.random.randn(8, 4, 4, 16).astype(np.float16)
    np.save(d / "demo.npy", z)
    (d / "demo.json").write_text(json.dumps({
        "actions": [[0.0] * 7] * 16, "proprio": [[0.0] * 14] * 16,
        "instruction": "", "success": 1, "task_key": "t",
    }))

    buf = FaissAnchorBuffer.build_from_embeddings(
        emb_dir=str(emb_root), out_dir=str(tmp_path / "buf2"),
        dim=16, nlist=2, subsample=1, train_size=4, use_gpu_for_build=False,
    )

    # anchor_distance over [B,T,P,D]
    z_pred = torch.randn(2, 3, 4, 16)
    d_out = buf.anchor_distance(z_pred)
    assert d_out.shape == (2, 3)
    assert (d_out >= 0).all()
