"""Datasets for predictor and critic training.

Two main loaders:

1) `PredictorDataset`: yields (z_hist, actions, proprio, lang, z_targets)
   for teacher-forcing + rollout-consistency training. Reads cached
   .npy embeddings and the corresponding LIBERO/RoboCasa demo metadata.

2) `CriticDataset`: yields (z_window, lang, progress_label, success_flag).
   Built on top of the same embedding cache plus pseudo-labels.

Both loaders assume the precompute_embeddings.py script has written:
    /workspace/data/embeddings/<source>/<task_id>/<demo_id>.npy
        shape [T_, pool_hw, pool_hw, D]
plus a sidecar metadata JSON per demo with action sequence, proprio sequence,
language instruction, and success flag.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset

from .pseudo_labels import dense_progress, windowed_pseudo_labels


# --- temporal alignment ------------------------------------------------------
#
# V-JEPA-2 has tubelet=2 (one latent step ~= 2 RGB frames). LIBERO demos record
# actions and proprio at the frame rate of the simulator. We align by taking
# the action / proprio at frame t*2 as the action that "produced" latent step t.
#
TUBELET = 2


class DemoIndex:
    """Builds an index of (file_path, demo_id, T_) over the embedding cache.

    Each demo's metadata is read on-demand to keep startup fast.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.entries: List[Dict] = []
        for emb_file in sorted(self.root.rglob("*.npy")):
            meta_file = emb_file.with_suffix(".json")
            if not meta_file.exists():
                continue
            arr = np.load(emb_file, mmap_mode="r")
            self.entries.append({
                "emb_path": emb_file,
                "meta_path": meta_file,
                "T_": arr.shape[0],
            })

    def __len__(self):
        return len(self.entries)


class PredictorDataset(Dataset):
    """Yields windows for predictor training.

    Each item:
        z_hist:    [T_hist, P_h*P_w, D]
        actions:   [T_hist + horizon, action_dim]
        proprio:   [T_hist + horizon, proprio_dim]
        lang:      [L, lang_dim]
        z_targets: [horizon, P_h*P_w, D]
    """

    def __init__(
        self,
        root: str,
        T_hist: int = 8,
        horizon: int = 4,
        lang_dim: int = 4096,
        max_lang_tokens: int = 32,
        action_dim: int = 7,
        proprio_dim: int = 8,
    ):
        self.index = DemoIndex(root)
        self.T_hist = T_hist
        self.horizon = horizon
        self.lang_dim = lang_dim
        self.max_lang_tokens = max_lang_tokens
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim

        # Build a flat index of valid (entry_idx, start_t) pairs
        self.windows: List[Tuple[int, int]] = []
        win_len = T_hist + horizon
        for i, e in enumerate(self.index.entries):
            T_ = e["T_"]
            for t0 in range(0, T_ - win_len + 1):
                self.windows.append((i, t0))

    def __len__(self):
        return len(self.windows)

    def _flatten_patches(self, z: np.ndarray) -> np.ndarray:
        # z: [T_, H, W, D] -> [T_, H*W, D]
        T_, H, W, D = z.shape
        return z.reshape(T_, H * W, D)

    def _load_lang(self, lang_str: str) -> np.ndarray:
        """Load precomputed language embedding for this instruction.

        Convention: embeddings stored at
            /workspace/data/lang_emb/<sha256>.npy
        Falls back to zero embedding if missing (so tests can run without
        the lang-emb pipeline).
        """
        import hashlib
        h = hashlib.sha256(lang_str.encode()).hexdigest()[:16]
        p = Path("/workspace/data/lang_emb") / f"{h}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape[0] > self.max_lang_tokens:
                arr = arr[: self.max_lang_tokens]
            return arr.astype(np.float32)
        return np.zeros((self.max_lang_tokens, self.lang_dim), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict:
        entry_idx, t0 = self.windows[idx]
        entry = self.index.entries[entry_idx]
        z = np.load(entry["emb_path"], mmap_mode="r")            # [T_, H, W, D]
        meta = json.loads(Path(entry["meta_path"]).read_text())

        win_len = self.T_hist + self.horizon
        z_win = self._flatten_patches(np.array(z[t0 : t0 + win_len]))   # [win, P, D]
        z_hist = z_win[: self.T_hist]
        z_targets = z_win[self.T_hist :]

        # Actions / proprio: align by tubelet
        rgb_start = t0 * TUBELET
        rgb_end = rgb_start + win_len * TUBELET
        actions = np.array(meta["actions"])[rgb_start : rgb_end : TUBELET]
        proprio = np.array(meta["proprio"])[rgb_start : rgb_end : TUBELET]

        # Pad if necessary (last demo segment may be short)
        def _pad(arr, target_len, dim):
            if arr.shape[0] < target_len:
                pad = np.zeros((target_len - arr.shape[0], dim), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            return arr.astype(np.float32)

        actions = _pad(actions, win_len, self.action_dim)
        proprio = _pad(proprio, win_len, self.proprio_dim)

        lang = self._load_lang(meta.get("instruction", ""))

        return {
            "z_hist": torch.from_numpy(z_hist.astype(np.float32)),
            "z_targets": torch.from_numpy(z_targets.astype(np.float32)),
            "actions": torch.from_numpy(actions),
            "proprio": torch.from_numpy(proprio),
            "lang": torch.from_numpy(lang),
        }


class CriticDataset(Dataset):
    """Yields windows for critic training.

    Each item:
        z_window:   [K, P, D]
        lang:       [L, lang_dim]
        progress:   scalar float in [0,1]
        success:    int {0,1}
    """

    def __init__(
        self,
        root: str,
        window_K: int = 8,
        lang_dim: int = 4096,
        max_lang_tokens: int = 32,
    ):
        self.index = DemoIndex(root)
        self.K = window_K
        self.lang_dim = lang_dim
        self.max_lang_tokens = max_lang_tokens

        self.windows: List[Tuple[int, int, float, int]] = []
        for i, e in enumerate(self.index.entries):
            T_ = e["T_"]
            meta = json.loads(Path(e["meta_path"]).read_text())
            success = int(meta.get("success", 0))
            prog = dense_progress(bool(success), T_)
            win_labels = windowed_pseudo_labels(prog, window_K)
            for t0, lab in enumerate(win_labels):
                self.windows.append((i, t0, float(lab), success))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        i, t0, label, success = self.windows[idx]
        entry = self.index.entries[i]
        z = np.load(entry["emb_path"], mmap_mode="r")
        z_win = np.array(z[t0 : t0 + self.K])                  # [K, H, W, D]
        z_win = z_win.reshape(self.K, -1, z_win.shape[-1])     # [K, P, D]

        meta = json.loads(Path(entry["meta_path"]).read_text())
        lang_str = meta.get("instruction", "")

        # Reuse PredictorDataset's lang loader trick
        import hashlib
        h = hashlib.sha256(lang_str.encode()).hexdigest()[:16]
        lang_p = Path("/workspace/data/lang_emb") / f"{h}.npy"
        if lang_p.exists():
            lang = np.load(lang_p)
            if lang.shape[0] > self.max_lang_tokens:
                lang = lang[: self.max_lang_tokens]
        else:
            lang = np.zeros((self.max_lang_tokens, self.lang_dim), dtype=np.float32)

        return {
            "z_window": torch.from_numpy(z_win.astype(np.float32)),
            "lang": torch.from_numpy(lang.astype(np.float32)),
            "progress": torch.tensor(label, dtype=torch.float32),
            "success": torch.tensor(success, dtype=torch.float32),
        }


def collate_predictor(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def collate_critic(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
