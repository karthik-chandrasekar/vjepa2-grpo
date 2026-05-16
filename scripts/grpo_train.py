"""GRPO post-training driver.

Two configurations of interest:
  --tag main         (default GRPO with anchored reward)
  --tag no_anchor    (ablation; pass --override grpo.lam_anchor_max=0.0)

    # main run
    python scripts/grpo_train.py \
        --config configs/grpo.yaml \
        --tag main

    # no-anchor ablation
    python scripts/grpo_train.py \
        --config configs/grpo.yaml \
        --tag no_anchor \
        --override grpo.lam_anchor_max=0.0 grpo.lam_anchor_warmup_steps=0

Both runs share the same predictor / critic / anchor buffer; they differ
only in the GRPO reward and the resulting policy.
"""
from __future__ import annotations
# CPU software rendering for MuJoCo offscreen — RunPod GPU containers lack
# working EGL/GLX. MUST be set before any mujoco/robosuite import.
import os as _os
_os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from omegaconf import OmegaConf

from vjepa2_grpo.policy import OpenVLAOFTPolicy
from vjepa2_grpo.predictor import BlockCausalACPredictor
from vjepa2_grpo.critic import ProgressCritic
from vjepa2_grpo.faiss_anchor import FaissAnchorBuffer
from vjepa2_grpo.encoder_client import RemoteVJepa2Encoder
from vjepa2_grpo.grpo import LatentGRPOTrainer
from vjepa2_grpo.utils import (
    set_seed, load_checkpoint, maybe_init_wandb,
)


def parse_overrides(overrides):
    """OmegaConf-style 'key=value' overrides."""
    return OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create()


def make_env_factory(cfg):
    """Returns a callable () -> vec-env. Lazy import so this script can be
    imported on machines without LIBERO installed."""
    from vjepa2_grpo.eval_libero import make_libero_env
    from libero.libero import benchmark

    def factory():
        bench = benchmark.get_benchmark_dict()[cfg.env.suite]()
        envs = []
        for i in range(cfg.env.n_envs):
            tid = i % bench.n_tasks
            env, _ = make_libero_env(cfg.env.suite, tid, cfg.env.render_size)
            envs.append(env)
        return envs

    return factory


def init_state_fn_factory(envs, encoder, instruction_lookup):
    """Returns a callable that builds the per-step `init_states` list."""
    import torch as _torch

    def fn(n_envs):
        states = []
        for env in envs[:n_envs]:
            obs = env.reset()
            instr = getattr(env, "language_instruction",
                            instruction_lookup.get(env.task_meta["name"], ""))
            # Initial latent (single-frame tiled-to-64 hack)
            # Encoder returns fp16; predictor is bf16. Cast here so all latents are bf16.
            z0 = encoder.encode_single_observation(obs["agentview_image"]).cuda().to(torch.bfloat16)
            pH, pW, D = z0.shape
            z_hist = z0.reshape(1, 1, pH * pW, D)

            # Initial proprio: pull standard LIBERO keys
            # Predictor was trained on the SAME canonical 8-dim proprio as the
            # policy: eef_pos[3] + quat2axisangle(eef_quat)[3] + gripper_qpos[2].
            # (The naive eef_pos+eef_quat+gripper concat is 9-dim and mismatches
            #  the predictor's prop_in which expects 8.)
            proprio = _torch.from_numpy(_libero_obs_to_state(obs)).float()

            # Lang emb: load from cache if exists, else zeros (placeholder)
            # cast to bf16 to match the predictor's lang_in dtype
            lang_emb = _load_lang_emb(instr, dim=4096, n_tokens=32).to(torch.bfloat16)

            # OFT policy needs an 8-dim obs["state"]; raw LIBERO splits proprio
            # across eef_pos/eef_quat/gripper_qpos. Compose it canonically.
            obs = dict(obs)  # shallow copy so we don't mutate the env's buffer
            obs["state"] = _libero_obs_to_state(obs)
            states.append({
                "obs": obs,
                "instruction": instr,
                "proprio0": proprio.cuda(),
                "lang_emb": lang_emb.cuda(),
                "z_hist": z_hist,
            })
        return states

    return fn


def _libero_obs_to_state(obs):
    """Compose the 8-dim OFT proprio vector from raw LIBERO obs keys.

    Matches openvla-oft/experiments/robot/libero/run_libero_eval.py:257 exactly:
      state = concat(eef_pos[3], quat2axisangle(eef_quat)[3], gripper_qpos[2])
    The quaternion->axis-angle conversion (4->3) is why this is 8-dim not 9.
    """
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),            # 3
        quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),  # 4->3
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),       # 2
    ]).astype(np.float32)                                               # = 8


def _load_lang_emb(text, dim=4096, n_tokens=32):
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()[:16]
    p = Path("/workspace/data/lang_emb") / f"{h}.npy"
    if p.exists():
        arr = np.load(p)
        if arr.shape[0] > n_tokens:
            arr = arr[:n_tokens]
        return torch.from_numpy(arr.astype(np.float32))
    return torch.zeros(n_tokens, dim, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="main", help="suffix for output dir")
    ap.add_argument("--override", nargs="*", default=[],
                    help="dot-list overrides, e.g. grpo.lam_anchor_max=0.0")
    ap.add_argument("--resume", default="no",
                    help="'no' (default), 'auto' (latest step_*/interrupt_* in output_dir), "
                         "or explicit path to a saved ckpt directory")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.train.output_dir = str(Path(cfg.train.output_dir).parent / args.tag)
    if args.override:
        cfg = OmegaConf.merge(cfg, parse_overrides(args.override))
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.train.output_dir) / "config.yaml")
    set_seed(cfg.train.seed)

    print(f"[grpo:{args.tag}] config:\n{OmegaConf.to_yaml(cfg)}")

    # ---- Load frozen modules ------------------------------------------------
    # V-JEPA-2 runs in venv1 as an HTTP server (transformers 5.x), accessed
    # here from venv-oft (transformers 4.40.1) via RemoteVJepa2Encoder.
    # Fail fast if the encoder server is not running.
    encoder_url = "http://127.0.0.1:8765"
    print(f"[grpo] connecting to encoder server at {encoder_url}...")
    encoder = RemoteVJepa2Encoder(encoder_url)
    try:
        info = encoder.wait_until_ready(max_wait_s=30.0)
        print(f"[grpo] encoder ready: pool_hw={info['pool_hw']} "
              f"embed_dim={info['embed_dim']}")
    except RuntimeError as e:
        raise RuntimeError(
            f"Encoder server not reachable at {encoder_url}.\n"
            f"Start it first in venv1:\n"
            f"  tmux new -s encserver -d\n"
            f"  tmux send-keys -t encserver 'source /root/venv/bin/activate && "
            f"python scripts/encoder_server.py' Enter\n"
            f"Then re-run GRPO.\n\nOriginal error: {e}"
        ) from e

    print(f"[grpo] loading predictor from {cfg.predictor_ckpt}")
    predictor = BlockCausalACPredictor(
        d_model=1024, n_layers=24, n_heads=16,
        latent_dim=1408, action_dim=cfg.policy.action_dim,
        proprio_dim=8, lang_dim=4096, patches_per_frame=64,
        use_grad_ckpt=False,
    ).cuda().to(torch.bfloat16)
    load_checkpoint(cfg.predictor_ckpt, predictor)
    predictor.eval()

    print(f"[grpo] loading critic from {cfg.critic_ckpt}")
    critic = ProgressCritic(
        d_model=768, n_layers=6, n_heads=12,
        latent_dim=1408, lang_dim=4096, n_ensemble=4,
        window_K=8, patches_per_frame=64,
    ).cuda().to(torch.bfloat16)
    load_checkpoint(cfg.critic_ckpt, critic)
    critic.eval()

    print(f"[grpo] loading anchor buffer from {cfg.anchor_index_dir}")
    anchor_buf = FaissAnchorBuffer.load(cfg.anchor_index_dir, to_gpu=True)

    # ---- Build trainable policy --------------------------------------------
    print(f"[grpo] loading policy {cfg.policy.model_id}")
    dtype = torch.bfloat16 if cfg.policy.dtype == "bfloat16" else torch.float16
    policy = OpenVLAOFTPolicy(
        model_id=cfg.policy.model_id,
        action_dim=cfg.policy.action_dim,
        action_chunk=cfg.policy.action_chunk,
        dtype=dtype, device="cuda",
        lora_r=cfg.policy.lora_r,
        lora_alpha=cfg.policy.lora_alpha,
        lora_dropout=cfg.policy.lora_dropout,
        lora_target_modules=list(cfg.policy.lora_target_modules),
        action_log_std_init=cfg.policy.action_log_std_init,
    )

    # ---- Env factory + init-state fn ---------------------------------------
    env_factory = make_env_factory(cfg)
    envs = env_factory()
    init_fn = init_state_fn_factory(envs, encoder, instruction_lookup={})

    # ---- Trainer ------------------------------------------------------------
    run = maybe_init_wandb(cfg.wandb.project,
                           f"{cfg.wandb.name}_{args.tag}",
                           OmegaConf.to_container(cfg, resolve=True),
                           mode=cfg.wandb.mode)

    trainer = LatentGRPOTrainer(
        policy=policy,
        predictor=predictor,
        critic=critic,
        anchor_buf=anchor_buf,
        encoder=encoder,
        env_factory=env_factory,
        group_size=cfg.grpo.group_size,
        horizon=cfg.grpo.horizon,
        action_chunk=cfg.grpo.action_chunk,
        n_envs=cfg.env.n_envs,
        lr=cfg.grpo.lr,
        weight_decay=cfg.grpo.weight_decay,
        kl_coef=cfg.grpo.kl_coef,
        kl_tripwire=cfg.grpo.kl_tripwire,
        lam_anchor_max=cfg.grpo.lam_anchor_max,
        lam_anchor_warmup_steps=cfg.grpo.lam_anchor_warmup_steps,
        lam_unc=cfg.grpo.lam_unc,
        clip_grad=cfg.grpo.clip_grad,
        success_rate_filter=cfg.grpo.success_rate_filter,
        dynamic_sample_max_trials=cfg.grpo.dynamic_sample_max_trials,
        output_dir=cfg.train.output_dir,
        log_every=cfg.train.log_every,
        eval_every=cfg.train.eval_every,
        save_every=cfg.train.save_every,
        wandb_run=run,
        device="cuda",
    )

    # ---- Resume handling ---------------------------------------------------
    start_step = 0
    if args.resume != "no":
        import re
        out_dir = Path(cfg.train.output_dir)
        if args.resume == "auto":
            # Find latest step_*/interrupt_*  (NOT final_*, that's a completed run)
            cands = []
            for d in list(out_dir.glob("step_*")) + list(out_dir.glob("interrupt_*")):
                if not d.is_dir() or d.name.endswith(".tmp"):
                    continue
                m = re.search(r"_(\d+)$", d.name)
                if m:
                    cands.append((int(m.group(1)), d))
            if cands:
                cands.sort(key=lambda x: x[0])
                resume_path = cands[-1][1]
                start_step = trainer.load_resume(resume_path)
                print(f"[grpo] resuming from {resume_path} at step {start_step}")
            else:
                print(f"[grpo] --resume auto but no ckpts found in {out_dir}; starting fresh")
        else:
            resume_path = Path(args.resume)
            start_step = trainer.load_resume(resume_path)
            print(f"[grpo] resuming from {resume_path} at step {start_step}")

    # ---- Interrupt handler -------------------------------------------------
    # SIGINT / SIGTERM saves an interrupt_<step>/ ckpt and exits cleanly.
    # RunPod's host-reclaim sends SIGTERM, so this gives us a recoverable
    # snapshot of LoRA + optim if the pod is preempted mid-training.
    import signal as _signal
    def _on_signal(signum, frame):
        s = trainer.step_idx
        print(f"\n[grpo] received signal {signum}; saving interrupt ckpt at step {s}...",
              flush=True)
        try:
            trainer._save(s, kind="interrupt")
        except Exception as e:
            print(f"[grpo] interrupt save FAILED: {e}", flush=True)
        sys.exit(0)
    _signal.signal(_signal.SIGINT, _on_signal)
    _signal.signal(_signal.SIGTERM, _on_signal)

    trainer.train(total_steps=cfg.train.total_steps, init_state_fn=init_fn,
                  start_step=start_step)


if __name__ == "__main__":
    main()
