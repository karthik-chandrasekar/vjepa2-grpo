"""Precompute V-JEPA-2 embeddings for the demonstration corpus.

For each demo (LIBERO HDF5, RoboCasa HDF5, or Ego4D-HO MP4):
  - Read RGB frames at the encoder's native rate
  - Chunk into 64-frame clips (with stride / overlap, configurable)
  - Encode each clip through V-JEPA-2 ViT-g/384
  - Spatially pool to 8x8 patches per temporal slot
  - Write `[T_, 8, 8, 1408]` float16 to .npy, plus a sidecar .json with
    actions, proprio, language instruction, and success flag.

This script is designed to be restartable: if the .npy + .json pair exists
for a demo, skip it.

Usage:
    python scripts/precompute_embeddings.py \
        --input /workspace/data/libero \
        --output /workspace/data/embeddings/libero \
        --source libero \
        --workers 4

For Ego4D-HO:
    python scripts/precompute_embeddings.py \
        --input /workspace/data/ego4d_ho \
        --output /workspace/data/embeddings/ego4d_ho \
        --source ego4d
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm

# Make package importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vjepa2_grpo.encoder import (
    VJepa2Encoder, FRAMES_PER_CLIP, TUBELET, EMBED_DIM,
)


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

def iter_libero_demos(root: str) -> Iterator[Tuple[Path, dict]]:
    """Yield (clip_id, demo_meta) for every LIBERO demo in `root`.

    LIBERO ships demos as HDF5 with keys:
        data/demo_<i>/obs/agentview_rgb   [T, H, W, 3] uint8
        data/demo_<i>/actions             [T, 7]
        data/demo_<i>/obs/ee_states       [T, ...] etc.
        attrs                              instruction
    """
    import h5py
    root = Path(root)
    for h5_path in sorted(root.rglob("*.hdf5")):
        try:
            with h5py.File(h5_path, "r") as f:
                if "data" not in f:
                    continue
                problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                lang = problem_info.get("language_instruction", "")
                # Per-task instruction sometimes lives in env_args
                if not lang:
                    env_args = json.loads(f["data"].attrs.get("env_args", "{}"))
                    lang = env_args.get("language_instruction", "")
                for demo_key in f["data"].keys():
                    demo_grp = f["data"][demo_key]
                    rgb = np.array(demo_grp["obs/agentview_rgb"][...])  # [T,H,W,3] uint8
                    actions = np.array(demo_grp["actions"][...])
                    # Proprio: assemble eef pos + quat + gripper
                    parts = []
                    for k in ["ee_pos", "ee_ori", "gripper_states"]:
                        full = f"obs/{k}"
                        if full in demo_grp:
                            parts.append(np.array(demo_grp[full][...]))
                    proprio = np.concatenate(parts, axis=-1) if parts \
                              else np.zeros((rgb.shape[0], 14), dtype=np.float32)
                    yield (
                        h5_path.stem + "__" + demo_key,
                        {
                            "rgb": rgb,
                            "actions": actions.astype(np.float32),
                            "proprio": proprio.astype(np.float32),
                            "instruction": lang,
                            "success": 1,  # LIBERO demos are successful by construction
                            "task_key": h5_path.stem,
                        },
                    )
        except Exception as e:
            print(f"[WARN] failed to read {h5_path}: {e}")


def iter_ego4d_demos(root: str) -> Iterator[Tuple[Path, dict]]:
    """Iterate Ego4D Hand-Object clips. Expects MP4 + JSON sidecars.

    This is a stub matching the conventional Ego4D-HO format. Adapt to your
    actual layout.
    """
    import decord
    root = Path(root)
    for mp4 in sorted(root.rglob("*.mp4")):
        sidecar = mp4.with_suffix(".json")
        if not sidecar.exists():
            continue
        meta = json.loads(sidecar.read_text())
        try:
            vr = decord.VideoReader(str(mp4))
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [T,H,W,3]
        except Exception as e:
            print(f"[WARN] cannot read {mp4}: {e}")
            continue
        T = frames.shape[0]
        # Ego4D has no robot actions/proprio; synthesize zero arrays.
        yield (
            mp4.stem,
            {
                "rgb": frames,
                "actions": np.zeros((T, 7), dtype=np.float32),
                "proprio": np.zeros((T, 14), dtype=np.float32),
                "instruction": meta.get("narration", ""),
                "success": 1,
                "task_key": meta.get("task", "ego4d"),
            },
        )


def iter_robocasa_demos(root: str) -> Iterator[Tuple[Path, dict]]:
    """RoboCasa-GR1 demos. Similar to LIBERO HDF5 but different group keys.
    Verify the exact schema in your dataset."""
    import h5py
    root = Path(root)
    for h5_path in sorted(root.rglob("*.hdf5")):
        try:
            with h5py.File(h5_path, "r") as f:
                if "data" not in f:
                    continue
                for demo_key in f["data"].keys():
                    d = f["data"][demo_key]
                    rgb = np.array(d["obs/robot0_agentview_image"][...])
                    actions = np.array(d["actions"][...])
                    proprio_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
                    parts = [np.array(d[f"obs/{k}"][...]) for k in proprio_keys
                             if f"obs/{k}" in d]
                    proprio = np.concatenate(parts, axis=-1) if parts \
                              else np.zeros((rgb.shape[0], 14), dtype=np.float32)
                    lang = d.attrs.get("language_instruction", "")
                    success = int(d.attrs.get("success", 1))
                    yield (
                        h5_path.stem + "__" + demo_key,
                        {
                            "rgb": rgb, "actions": actions.astype(np.float32),
                            "proprio": proprio.astype(np.float32),
                            "instruction": str(lang), "success": success,
                            "task_key": h5_path.stem,
                        },
                    )
        except Exception as e:
            print(f"[WARN] failed to read {h5_path}: {e}")


SOURCE_FNS = {
    "libero": iter_libero_demos,
    "ego4d": iter_ego4d_demos,
    "robocasa": iter_robocasa_demos,
}


# ---------------------------------------------------------------------------
# Chunking + encoding
# ---------------------------------------------------------------------------

def encode_demo(
    encoder: VJepa2Encoder,
    rgb: np.ndarray,
    clip_stride: int = FRAMES_PER_CLIP,
) -> np.ndarray:
    """Encode a full demo as concatenated clip embeddings.

    Args:
        rgb: [T, H, W, 3] uint8
        clip_stride: stride between consecutive 64-frame clips. With
                     stride=64, clips don't overlap (fastest). With stride=32
                     they overlap by 50% (smoother temporal coverage).

    Returns:
        z: [T_, pool_hw, pool_hw, EMBED_DIM] float16 where
           T_ = num_clips * TEMPORAL_PATCHES (32 latent steps per 64-frame clip)
    """
    T = rgb.shape[0]
    if T < FRAMES_PER_CLIP:
        # Just pad and encode one clip
        chunk = encoder.encode_clip(rgb)
        return chunk.cpu().numpy().astype(np.float16)

    embeddings = []
    for start in range(0, T - FRAMES_PER_CLIP + 1, clip_stride):
        clip = rgb[start : start + FRAMES_PER_CLIP]
        z = encoder.encode_clip(clip)  # [T_, pH, pW, D] in BF16 on CPU
        embeddings.append(z.cpu().numpy().astype(np.float16))
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to source root")
    ap.add_argument("--output", required=True, help="path to write embeddings + sidecars")
    ap.add_argument("--source", required=True, choices=list(SOURCE_FNS.keys()))
    ap.add_argument("--clip-stride", type=int, default=FRAMES_PER_CLIP,
                    help="stride between 64-frame clips (default: no overlap)")
    ap.add_argument("--pool-hw", type=int, default=8)
    ap.add_argument("--max-demos", type=int, default=None,
                    help="cap demos for smoke tests")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[precompute] loading V-JEPA-2 encoder...")
    encoder = VJepa2Encoder(dtype=dtype, pool_hw=args.pool_hw,
                            attn_impl=args.attn_impl, device="cuda")
    print(f"[precompute] encoder ready; params=" + 
          f"{sum(p.numel() for p in encoder.parameters()) / 1e9:.2f}B")

    iterator = SOURCE_FNS[args.source](args.input)
    n_processed = 0
    n_skipped = 0
    t_start = time.time()
    pbar = tqdm(iterator, desc="encoding")
    for clip_id, meta in pbar:
        if args.max_demos is not None and n_processed >= args.max_demos:
            break

        task_dir = out_root / meta["task_key"]
        task_dir.mkdir(parents=True, exist_ok=True)
        emb_path = task_dir / f"{clip_id}.npy"
        meta_path = task_dir / f"{clip_id}.json"

        if emb_path.exists() and meta_path.exists():
            n_skipped += 1
            pbar.set_postfix(skipped=n_skipped, ok=n_processed)
            continue

        try:
            z = encode_demo(encoder, meta["rgb"], clip_stride=args.clip_stride)
        except Exception as e:
            print(f"[WARN] encode failed for {clip_id}: {e}")
            continue

        np.save(emb_path, z)
        meta_path.write_text(json.dumps({
            "actions": meta["actions"].tolist(),
            "proprio": meta["proprio"].tolist(),
            "instruction": meta["instruction"],
            "success": int(meta["success"]),
            "task_key": meta["task_key"],
            "rgb_T": int(meta["rgb"].shape[0]),
            "latent_T": int(z.shape[0]),
            "pool_hw": args.pool_hw,
            "embed_dim": EMBED_DIM,
        }))
        n_processed += 1
        pbar.set_postfix(ok=n_processed, skipped=n_skipped)

    dt = time.time() - t_start
    print(f"\n[precompute] done in {dt/60:.1f} min")
    print(f"  processed: {n_processed}    skipped: {n_skipped}")
    print(f"  output:    {out_root}")


if __name__ == "__main__":
    main()
