"""Gate 1 benchmark: latent-rollout throughput on H100/H200.

Measures how fast the predictor can roll out candidate action trajectories.
Real GRPO (rollout_group) processes `group_size` trajectories per call, so
this benchmark mini-batches rather than pushing all trajectories through one
giant forward pass.

Run:
    python scripts/benchmark_rollout.py
"""
from __future__ import annotations
import sys
import time
import math
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from vjepa2_grpo.predictor import BlockCausalACPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor-ckpt", type=str, default=None)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--n-states", type=int, default=100)
    ap.add_argument("--micro-batch", type=int, default=64,
                    help="trajectories per forward pass (real GRPO uses group_size=8 "
                         "per state; larger is fine for a throughput measurement)")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--action-chunk", type=int, default=8)
    ap.add_argument("--T-hist", type=int, default=8)
    ap.add_argument("--patches-per-frame", type=int, default=64)
    ap.add_argument("--n-warmup", type=int, default=2)
    ap.add_argument("--n-trials", type=int, default=5)
    args = ap.parse_args()

    B = args.group_size * args.n_states
    n_micro = math.ceil(B / args.micro_batch)

    print("=== Latent rollout benchmark ===")
    print(f"  predictor: 24L/1024w/16h, ~300M params")
    print(f"  total trajectories = group_size * n_states = {B}")
    print(f"  micro-batch = {args.micro_batch}  ->  {n_micro} forward groups")
    print(f"  horizon = {args.horizon}, action_chunk = {args.action_chunk}")
    print(f"  predictor fwd passes per trajectory = horizon * action_chunk = "
          f"{args.horizon * args.action_chunk}")

    model = BlockCausalACPredictor(
        d_model=1024, n_layers=24, n_heads=16,
        latent_dim=1408, action_dim=7, proprio_dim=14, lang_dim=4096,
        patches_per_frame=args.patches_per_frame, max_horizon=64,
        use_grad_ckpt=False,
    ).cuda().to(torch.bfloat16).eval()

    if args.predictor_ckpt:
        from vjepa2_grpo.utils import load_checkpoint
        load_checkpoint(args.predictor_ckpt, model, strict=False)

    print(f"  loaded predictor params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    mb = args.micro_batch
    z_hist = torch.randn(mb, args.T_hist, args.patches_per_frame, 1408,
                         dtype=torch.bfloat16, device="cuda")
    action_chunks = torch.randn(mb, args.action_chunk, 7,
                                dtype=torch.bfloat16, device="cuda")
    proprio_chunks = torch.zeros(mb, args.action_chunk, 14,
                                 dtype=torch.bfloat16, device="cuda")
    lang = torch.zeros(mb, 32, 4096, dtype=torch.bfloat16, device="cuda")

    def one_full_rollout():
        # roll out `mb` trajectories for the full horizon
        for _h_step in range(args.horizon):
            _ = model.rollout(z_hist, action_chunks, proprio_chunks,
                              lang, horizon=args.action_chunk)

    print(f"  warming up ({args.n_warmup} iters)...")
    with torch.inference_mode():
        for _ in range(args.n_warmup):
            one_full_rollout()
            torch.cuda.synchronize()

    print(f"  measuring ({args.n_trials} trials)...")
    times = []
    with torch.inference_mode():
        for _ in range(args.n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_micro):          # cover all B trajectories
                one_full_rollout()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    t_mean = sum(times) / len(times)
    actions_per_sec = B * args.horizon / t_mean
    print(f"\n=== Results ===")
    print(f"  time to roll out all {B} trajectories x horizon {args.horizon}: "
          f"mean {t_mean:.2f}s  min {min(times):.2f}  max {max(times):.2f}")
    print(f"  actions / sec: {actions_per_sec:.0f}")
    print(f"  V-JEPA-2-AC reference: ~50 actions/sec on 4090")
    print(f"  speedup vs reference: {actions_per_sec / 50:.1f}x")

    threshold = 200
    if actions_per_sec >= threshold:
        print(f"  >>> GATE 1 PASS: >= {threshold} actions/sec")
    else:
        print(f"  >>> GATE 1 FAIL: descope per runbook (drop horizon/group/suites)")

    print(f"  peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
