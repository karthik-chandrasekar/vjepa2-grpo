"""Unified eval driver.

Three orthogonal modes (combine as needed):

    # LIBERO 4-suite eval
    python scripts/eval_main.py \
        --policy-ckpt /workspace/checkpoints/policy/main/step_001000 \
        --suites libero_spatial libero_object libero_goal libero_long \
        --n-trials 50

    # LIBERO-Plus eval (with per-dimension breakdown)
    python scripts/eval_main.py \
        --policy-ckpt /workspace/checkpoints/policy/main/step_001000 \
        --libero-plus \
        --libero-plus-subsample 50

    # Reward-hacking diagnostics
    python scripts/eval_main.py \
        --policy-ckpt /workspace/checkpoints/policy/main/step_001000 \
        --diagnostics \
        --diag-no-anchor-ckpt /workspace/checkpoints/policy/no_anchor/step_001000

    # Everything in one go
    python scripts/eval_main.py --policy-ckpt ... --suites ... --libero-plus --diagnostics
"""
from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf


def load_policy(model_id, lora_path, dtype=torch.bfloat16):
    from vjepa2_grpo.policy import OpenVLAOFTPolicy
    policy = OpenVLAOFTPolicy(model_id=model_id, dtype=dtype, device="cuda")
    if lora_path:
        policy.load_lora(lora_path)
    return policy


def cmd_libero(args, policy):
    from vjepa2_grpo.eval_libero import evaluate_all_libero
    out = evaluate_all_libero(
        policy=policy,
        out_dir=str(Path(args.out_dir) / "libero"),
        n_trials_per_task=args.n_trials,
        suites=tuple(args.suites),
    )
    print(json.dumps(out, indent=2))


def cmd_libero_plus(args, policy):
    from vjepa2_grpo.eval_libero_plus import evaluate_libero_plus
    out = evaluate_libero_plus(
        policy=policy,
        out_path=str(Path(args.out_dir) / "libero_plus" / "summary.json"),
        n_trials_per_task=args.libero_plus_n_trials,
        subsample_per_dim=args.libero_plus_subsample,
    )
    print(json.dumps({"total": out["total"], "per_dim": out["per_dim"]}, indent=2))


def cmd_diagnostics(args, policy):
    from vjepa2_grpo.diagnostics import (
        collect_rollouts, compute_diagnostics, plot_diagnostics,
    )
    from vjepa2_grpo.encoder import VJepa2Encoder
    from vjepa2_grpo.predictor import BlockCausalACPredictor
    from vjepa2_grpo.critic import ProgressCritic
    from vjepa2_grpo.faiss_anchor import FaissAnchorBuffer
    from vjepa2_grpo.utils import load_checkpoint
    from vjepa2_grpo.eval_libero import make_libero_env

    print("[diag] loading frozen modules...")
    encoder = VJepa2Encoder(dtype=torch.bfloat16, pool_hw=8, device="cuda")
    predictor = BlockCausalACPredictor(
        d_model=1024, n_layers=24, n_heads=16,
        latent_dim=1408, action_dim=7, proprio_dim=8, lang_dim=4096,
        patches_per_frame=64, use_grad_ckpt=False,
    ).cuda().to(torch.bfloat16)
    load_checkpoint(args.diag_predictor_ckpt, predictor)
    critic = ProgressCritic(
        d_model=768, n_layers=6, n_heads=12, latent_dim=1408,
        lang_dim=4096, n_ensemble=4, window_K=8, patches_per_frame=64,
    ).cuda().to(torch.bfloat16)
    load_checkpoint(args.diag_critic_ckpt, critic)
    anchor_buf = FaissAnchorBuffer.load(args.diag_anchor_dir, to_gpu=True)

    diag_out = Path(args.out_dir) / "diagnostics"
    diag_out.mkdir(parents=True, exist_ok=True)

    # Collect rollouts for both runs at each step folder under args.policy_ckpt
    def _collect_for_run(lora_root, label):
        results_per_step = {}
        # If lora_root is a single step directory, just use it once
        step_dirs = sorted([d for d in Path(lora_root).iterdir()
                            if d.is_dir() and d.name.startswith("step_")])
        if not step_dirs:
            step_dirs = [Path(lora_root)]
        for sd in step_dirs:
            print(f"[diag] {label} @ {sd.name}: loading + collecting rollouts...")
            policy.load_lora(str(sd))
            env, _ = make_libero_env("libero_spatial", 0, render_size=384)
            data = collect_rollouts(
                policy, predictor, critic, anchor_buf, encoder, env,
                n_rollouts=args.diag_n_rollouts,
                horizon=10, action_chunk=8,
            )
            env.close()
            diag = compute_diagnostics(data)
            results_per_step[sd.name.replace("step_", "").lstrip("0") or "0"] = diag
            print(f"  corr={diag['diag1_corr']:.3f}  "
                  f"gap={diag['diag3_gap']:.3f}  "
                  f"succ_rate={diag['mean_success']:.3f}")
        return results_per_step

    main_run = _collect_for_run(args.policy_ckpt, "main")
    no_anchor_run = _collect_for_run(args.diag_no_anchor_ckpt, "no_anchor")

    plot_diagnostics(main_run, no_anchor_run, str(diag_out))
    print(f"[diag] plots + json dump -> {diag_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-ckpt", required=True,
                    help="LoRA dir for the policy under test")
    ap.add_argument("--model-id", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--out-dir", default="/workspace/eval_results")

    ap.add_argument("--suites", nargs="*", default=[],
                    help="LIBERO suites to run (any of: libero_spatial libero_object "
                         "libero_goal libero_long libero_90)")
    ap.add_argument("--n-trials", type=int, default=50)

    ap.add_argument("--libero-plus", action="store_true")
    ap.add_argument("--libero-plus-n-trials", type=int, default=1)
    ap.add_argument("--libero-plus-subsample", type=int, default=None,
                    help="cap tasks per perturbation dim (default: full eval)")

    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--diag-no-anchor-ckpt", type=str, default=None,
                    help="LoRA dir for the no-anchor ablation run; required for --diagnostics")
    ap.add_argument("--diag-predictor-ckpt", default="/workspace/checkpoints/predictor/final.pt")
    ap.add_argument("--diag-critic-ckpt", default="/workspace/checkpoints/critic/final.pt")
    ap.add_argument("--diag-anchor-dir", default="/workspace/data/anchor_buffer")
    ap.add_argument("--diag-n-rollouts", type=int, default=200)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[eval] loading policy from {args.policy_ckpt}")
    policy = load_policy(args.model_id, args.policy_ckpt)

    if args.suites:
        cmd_libero(args, policy)

    if args.libero_plus:
        cmd_libero_plus(args, policy)

    if args.diagnostics:
        if args.diag_no_anchor_ckpt is None:
            raise ValueError("--diagnostics requires --diag-no-anchor-ckpt")
        cmd_diagnostics(args, policy)

    print("[eval] done.")


if __name__ == "__main__":
    main()
