"""Smoke test for the encoder IPC. Run in venv-oft after the server is up.
    source /workspace/venv-oft/bin/activate
    python scripts/smoke_encoder_ipc.py
"""
import sys, time
sys.path.insert(0, '/workspace/vjepa2_grpo')
import numpy as np
from vjepa2_grpo.encoder_client import RemoteVJepa2Encoder

enc = RemoteVJepa2Encoder("http://127.0.0.1:8765")
info = enc.wait_until_ready()
print("server:", info)

frame = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
z = enc.encode_single_observation(frame)
assert tuple(z.shape) == (8, 8, 1408), z.shape
print(f"encode_single: {tuple(z.shape)} {z.dtype}")

frames = np.random.randint(0, 256, (64, 128, 128, 3), dtype=np.uint8)
z = enc.encode_clip(frames)
assert z.shape[1:] == (8, 8, 1408), z.shape
print(f"encode_clip:   {tuple(z.shape)} {z.dtype}")

t0 = time.perf_counter()
for _ in range(10):
    enc.encode_single_observation(frame)
print(f"10x encode_single = {(time.perf_counter()-t0)*1000:.0f}ms")
