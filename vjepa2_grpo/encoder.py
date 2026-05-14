"""V-JEPA-2 frozen encoder wrapper.

Loads facebook/vjepa2-vitg-fpc64-384 from HF, freezes parameters, and exposes
patch-token embeddings after spatial pooling.

IMPORTANT — verify these assumptions on first run:
  - The HF model id `facebook/vjepa2-vitg-fpc64-384` should be loadable via
    `AutoModel`. If the class name differs in your transformers version,
    update `_load_model`. Run `scripts/sanity.py` first.
  - V-JEPA-2 ViT-g/16 at 384^2, tubelet 2: 32 temporal x 576 spatial patches
    per 64-frame clip, embed dim 1408. If the output shape differs, the
    `_reshape_patches` method below needs to match.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Union
import numpy as np

# --- Architectural constants (V-JEPA-2 ViT-g at 384x384) -----------------
ENCODER_ID = "facebook/vjepa2-vitg-fpc64-384"
EMBED_DIM = 1408           # ViT-g hidden dim
PATCH_HW_RAW = 24          # 384 / 16 = 24 spatial patches per side
PATCHES_PER_FRAME_RAW = PATCH_HW_RAW * PATCH_HW_RAW  # 576
TUBELET = 2                # temporal patch size
FRAMES_PER_CLIP = 64       # fpc64 variant
TEMPORAL_PATCHES = FRAMES_PER_CLIP // TUBELET  # 32


class VJepa2Encoder(nn.Module):
    """Frozen V-JEPA-2 video encoder with spatial patch pooling.

    The full-resolution patch grid (24x24 per temporal slot) is too large
    to cache for ~10M frames (would be ~7 TB). We average-pool spatially
    to `pool_hw x pool_hw` (default 8x8), giving ~64 patches per temporal
    slot and ~2 TB cache. This is the design point in the proposal.
    """

    def __init__(
        self,
        model_id: str = ENCODER_ID,
        dtype: torch.dtype = torch.bfloat16,
        pool_hw: int = 8,
        attn_impl: str = "sdpa",
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.processor, self.model = self._load_model(model_id, dtype, attn_impl)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.pool_hw = pool_hw
        self.dtype = dtype
        if device is not None:
            self.model.to(device)

    @staticmethod
    def _load_model(model_id: str, dtype: torch.dtype, attn_impl: str):
        from transformers import AutoVideoProcessor, AutoModel
        processor = AutoVideoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        return processor, model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def encode_clip(self, frames: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Encode a 64-frame clip.

        Args:
            frames: [T, H, W, 3] uint8 ndarray or [T, 3, H, W] float tensor.
                    T should be FRAMES_PER_CLIP (64); shorter clips are padded
                    with frame replication.

        Returns:
            [T_, pool_hw, pool_hw, D] tensor in `self.dtype`, on CPU.
            T_ = TEMPORAL_PATCHES = 32.
        """
        frames = self._ensure_64_frames(frames)
        inputs = self.processor(videos=[frames], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # last_hidden_state: [1, T_*P_raw, D]
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        z = self._reshape_and_pool(hidden)
        return z.squeeze(0).to(self.dtype).cpu()

    @torch.inference_mode()
    def encode_clip_batch(self, frames_batch: torch.Tensor) -> torch.Tensor:
        """Batched version. frames_batch: list of [T,H,W,3] or [B,T,H,W,3]."""
        if isinstance(frames_batch, (list, tuple)):
            frames_batch = [self._ensure_64_frames(f) for f in frames_batch]
            inputs = self.processor(videos=frames_batch, return_tensors="pt")
        else:
            assert frames_batch.ndim == 5
            inputs = self.processor(videos=list(frames_batch), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        z = self._reshape_and_pool(hidden)
        return z.to(self.dtype).cpu()

    @torch.inference_mode()
    def encode_single_observation(self, frame: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """For single-frame inputs at GRPO/eval time.

        V-JEPA-2 has a fixed temporal length (64 frames). Tile the observation
        across all 64 slots, encode, and return the last temporal patch slot.
        This is the simplest fix; alternatives include keeping a 64-frame
        ring buffer of recent observations.

        Returns: [pool_hw, pool_hw, D]
        """
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if frame.ndim == 3:
            frame = np.expand_dims(frame, 0)
        frames = np.repeat(frame, FRAMES_PER_CLIP, axis=0)
        z = self.encode_clip(frames)
        return z[-1]

    # --- internals -------------------------------------------------------

    def _ensure_64_frames(self, frames):
        if isinstance(frames, torch.Tensor):
            T = frames.shape[0]
            if T < FRAMES_PER_CLIP:
                pad = frames[-1:].expand(FRAMES_PER_CLIP - T, *frames.shape[1:])
                frames = torch.cat([frames, pad], dim=0)
            elif T > FRAMES_PER_CLIP:
                frames = frames[:FRAMES_PER_CLIP]
        else:
            T = frames.shape[0]
            if T < FRAMES_PER_CLIP:
                pad = np.repeat(frames[-1:], FRAMES_PER_CLIP - T, axis=0)
                frames = np.concatenate([frames, pad], axis=0)
            elif T > FRAMES_PER_CLIP:
                frames = frames[:FRAMES_PER_CLIP]
        return frames

    def _reshape_and_pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B, T_*P_raw, D] -> [B, T_, pool_hw, pool_hw, D]"""
        B, N, D = hidden.shape
        assert N == TEMPORAL_PATCHES * PATCHES_PER_FRAME_RAW, (
            f"Unexpected token count {N}; expected "
            f"{TEMPORAL_PATCHES * PATCHES_PER_FRAME_RAW}. Check encoder output."
        )
        z = hidden.reshape(B, TEMPORAL_PATCHES, PATCH_HW_RAW, PATCH_HW_RAW, D)
        # [B, T_, H, W, D] -> [B*T_, D, H, W] for pooling
        z = z.permute(0, 1, 4, 2, 3).contiguous()
        z = z.reshape(B * TEMPORAL_PATCHES, D, PATCH_HW_RAW, PATCH_HW_RAW)
        z = F.adaptive_avg_pool2d(z, (self.pool_hw, self.pool_hw))
        z = z.reshape(B, TEMPORAL_PATCHES, D, self.pool_hw, self.pool_hw)
        z = z.permute(0, 1, 3, 4, 2).contiguous()  # [B, T_, pH, pW, D]
        return z


def load_encoder(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    pool_hw: int = 8,
    attn_impl: str = "sdpa",
) -> VJepa2Encoder:
    """Convenience constructor."""
    return VJepa2Encoder(dtype=dtype, pool_hw=pool_hw, attn_impl=attn_impl, device=device)
