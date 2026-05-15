"""HTTP client for the V-JEPA-2 encoder server.

Drop-in replacement for `vjepa2_grpo.encoder.VJepa2Encoder`: same method names
(`encode_clip`, `encode_single_observation`), same return shapes, same dtypes
on output. Used by the GRPO process in venv-oft, which can't import the local
encoder because it would pull in a transformers version incompatible with OFT.

Usage in venv-oft code:
    from vjepa2_grpo.encoder_client import RemoteVJepa2Encoder
    encoder = RemoteVJepa2Encoder("http://127.0.0.1:8765")
    encoder.wait_until_ready()       # blocks until /health is 200
    z = encoder.encode_clip(frames)  # same as the local encoder

Resilience:
  - Connection errors trigger automatic retry with brief backoff (so an encoder
    restart doesn't kill a 24-hr GRPO run).
  - Each call has a configurable timeout; default 60s (plenty for V-JEPA-2 fwd).
  - Server-side errors surface as RuntimeError with the server's traceback,
    not a cryptic JSON KeyError.
"""
from __future__ import annotations
import base64
import io
import time
from typing import Union

import numpy as np
import requests


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _np_from_b64(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


class EncoderServerError(RuntimeError):
    """Raised when the server returns a 5xx with a traceback."""


class RemoteVJepa2Encoder:
    """Drop-in client for the encoder server. Mimics VJepa2Encoder's surface.

    Returns torch tensors on CPU (callers `.cuda()` as needed) to match the
    local encoder's behavior.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8765",
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 1.0,
    ):
        self.url = server_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        # cache server info from /health
        self._server_info = None

    # ----- lifecycle ---------------------------------------------------------

    def wait_until_ready(self, max_wait_s: float = 120.0) -> dict:
        """Block until the server is responsive; returns its /health info.

        Use this once at startup before any encode call. Saves ~5min of
        debugging when you launch GRPO before the server has loaded the 1B
        V-JEPA-2 weights (which takes ~10s).
        """
        t0 = time.monotonic()
        last_err = None
        while time.monotonic() - t0 < max_wait_s:
            try:
                r = requests.get(f"{self.url}/health", timeout=5.0)
                if r.status_code == 200:
                    self._server_info = r.json()
                    return self._server_info
            except requests.exceptions.ConnectionError as e:
                last_err = e
            time.sleep(1.0)
        raise RuntimeError(
            f"Encoder server at {self.url} not ready after {max_wait_s}s. "
            f"Last error: {last_err}. Did you start it in venv1?"
        )

    @property
    def pool_hw(self) -> int:
        if self._server_info is None:
            self.wait_until_ready()
        return self._server_info["pool_hw"]

    # ----- request plumbing --------------------------------------------------

    def _post(self, endpoint: str, arr: np.ndarray) -> np.ndarray:
        payload = {"array_b64": _np_to_b64(arr)}
        last_err = None
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.url}{endpoint}",
                                  json=payload,
                                  timeout=self.timeout)
                if r.status_code == 200:
                    body = r.json()
                    return _np_from_b64(body["array_b64"])
                # 4xx: client error, no retry
                if 400 <= r.status_code < 500:
                    raise EncoderServerError(
                        f"{endpoint} -> {r.status_code}: {r.text}")
                # 5xx: server error, retry
                last_err = EncoderServerError(
                    f"{endpoint} -> {r.status_code}: "
                    f"{r.json().get('error', r.text)}")
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e
            time.sleep(self.backoff * (attempt + 1))
        raise EncoderServerError(
            f"{endpoint} failed after {self.retries} retries: {last_err}")

    # ----- API mirroring VJepa2Encoder --------------------------------------

    def encode_clip(self, frames):
        """`frames`: [T,H,W,3] uint8. Returns torch.float16 [T_,8,8,1408] on CPU.

        T should be FRAMES_PER_CLIP (64); shorter clips are padded server-side.
        """
        import torch  # local import to keep startup light
        arr = self._as_numpy_uint8(frames, ndim=4)
        z_np = self._post("/encode_clip", arr)        # fp16 numpy on wire
        return torch.from_numpy(z_np)                  # fp16 tensor on CPU

    def encode_single_observation(self, frame):
        """`frame`: [H,W,3] uint8. Returns torch.float16 [8,8,1408] on CPU."""
        import torch
        arr = self._as_numpy_uint8(frame, ndim=3)
        z_np = self._post("/encode_single", arr)
        return torch.from_numpy(z_np)

    # ----- helpers -----------------------------------------------------------

    @staticmethod
    def _as_numpy_uint8(x, ndim: int) -> np.ndarray:
        """Coerce torch tensor / numpy array to a contiguous uint8 ndarray."""
        if hasattr(x, "detach"):  # torch tensor
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.dtype != np.uint8:
            # accept float [0,1] or [0,255]; round to uint8
            if x.max() <= 1.0001:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        if x.ndim != ndim:
            raise ValueError(f"expected {ndim}-D image array, got shape {x.shape}")
        return np.ascontiguousarray(x)
