"""Process critic training driver.

Run AFTER the predictor is at least partially trained (the critic only consumes
cached embeddings, not predictor outputs, so it can technically run in parallel
on a second GPU — but on one GPU we run it sequentially after the predictor).

    python scripts/train_critic.py --config configs/critic.yaml
    python scripts/train_critic.py --config configs/critic.yaml --resume auto
    python scripts/train_critic.py --config configs/critic.yaml --resume <path.pt>

Outputs (under cfg.train.ckpt_dir):
    step_<NNNNNN>.pt        periodic checkpoints (rolling-pruned)
    interrupt_<NNNNNN>.pt   written if the run is interrupted / crashes
    final.pt                final checkpoint

Hardened for multi-hour runs (~10-12 hr at 20k steps):
  - checkpoints are written atomically (.tmp + os.replace) via save_checkpoint
  - optimizer AND lr-scheduler state saved/restored (faithful resume)
  - `--resume auto` picks up the newest step_/interrupt_ checkpoint
  - rolling retention: keep last N + every M-step milestone (see config)
  - any interruption (Ctrl-C, exception, OOM, pod reclaim) tries to save
    interrupt_<step>.pt before propagating
"""
from __future__ import annotations
import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from vjepa2_grpo.critic import ProgressCritic
from vjepa2_grpo.datasets import CriticDataset, collate_critic
from vjepa2_grpo.losses import critic_loss
from vjepa2_grpo.utils import (
    set_seed, save_checkpoint, load_checkpoint, get_lr_schedule,
    maybe_init_wandb, count_params,
    find_latest_checkpoint, prune_checkpoints,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", type=str, default=None,
                    help="path to a checkpoint, or 'auto' to resume the newest "
                         "step_/interrupt_ checkpoint in cfg.train.ckpt_dir")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.train.seed)

    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # retention policy (tunable via config, sensible defaults otherwise)
    keep_last = int(cfg.train.get("ckpt_keep_last", 2))
    milestone_every = int(cfg.train.get("ckpt_milestone_every", 10000))

    # --- Dataset ---
    print("[critic] building dataset...")
    ds = CriticDataset(
        root=cfg.data.embedding_root,
        window_K=cfg.data.window_K,
        lang_dim=cfg.data.lang_dim,
        max_lang_tokens=cfg.data.max_lang_tokens,
    )
    print(f"[critic] dataset has {len(ds)} windows")
    if len(ds) == 0:
        raise RuntimeError(f"No training windows under {cfg.data.embedding_root}")

    dl = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_critic,
    )

    # --- Model ---
    model = ProgressCritic(
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        latent_dim=cfg.model.latent_dim,
        lang_dim=cfg.model.lang_dim,
        n_ensemble=cfg.model.n_ensemble,
        window_K=cfg.model.window_K,
        patches_per_frame=cfg.model.patches_per_frame,
        dropout=cfg.model.dropout,
    ).cuda()
    if cfg.train.bf16:
        model = model.to(torch.bfloat16)
    print(f"[critic] params: {count_params(model)/1e6:.1f}M")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
    )
    sched = get_lr_schedule(optim, cfg.schedule.warmup,
                            cfg.schedule.total_steps, cfg.optim.lr)

    # --- Resume ---
    # scheduler state restored from the checkpoint (the previous step-replay
    # hack `for _ in range(start_step): sched.step()` was coincidentally
    # correct here because grad_accum=1, but it's still slower and brittle —
    # the new path is one state_dict load.
    start_step = 0
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(str(ckpt_dir))
        if resume_path is None:
            print(f"[resume] --resume auto: no checkpoint in {ckpt_dir}; "
                  "starting from scratch")
        else:
            print(f"[resume] --resume auto -> {resume_path}")
    if resume_path is not None:
        start_step, _ = load_checkpoint(resume_path, model, optim, scheduler=sched)
        print(f"[resume] resumed at step {start_step}")

    run = maybe_init_wandb(cfg.wandb.project, cfg.wandb.name,
                           OmegaConf.to_container(cfg, resolve=True),
                           mode=cfg.wandb.mode)

    # --- Train ---
    model.train()
    step = start_step
    t_last = time.time()
    total_steps = cfg.schedule.total_steps

    def _checkpoint(tag_step: int, kind: str = "step"):
        p = ckpt_dir / f"{kind}_{tag_step:06d}.pt"
        save_checkpoint(str(p), model, optim, step=tag_step, scheduler=sched)
        print(f"[ckpt] -> {p}")
        if kind == "step":
            prune_checkpoints(str(ckpt_dir), keep_last=keep_last,
                              milestone_every=milestone_every)

    try:
        while step < total_steps:
            for batch in dl:
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                if cfg.train.bf16:
                    batch = {k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v
                             for k, v in batch.items()}

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.bf16):
                    out = critic_loss(
                        model, batch,
                        bootstrap_p=cfg.train.loss.bootstrap_p,
                        w_mono=cfg.train.loss.w_mono,
                        success_threshold=cfg.train.loss.success_threshold,
                    )
                optim.zero_grad()
                out["loss"].backward()
                if cfg.train.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
                optim.step()
                sched.step()

                if step % cfg.train.log_every == 0:
                    dt = time.time() - t_last
                    t_last = time.time()
                    speed = cfg.train.log_every * cfg.train.batch_size / max(dt, 1e-6)
                    print(f"[step {step}/{total_steps}]  loss={out['loss'].item():.4f}  "
                          f"prog={out['L_progress'].item():.4f}  "
                          f"mono={out['L_mono'].item():.4f}  "
                          f"phat={out['p_hat_mean'].item():.3f}  "
                          f"sigma={out['sigma_mean'].item():.3f}  "
                          f"lr={sched.get_last_lr()[0]:.2e}  {speed:.0f}/s")
                    if run is not None:
                        run.log({
                            "loss": out["loss"].item(),
                            "L_progress": out["L_progress"].item(),
                            "L_mono": out["L_mono"].item(),
                            "p_hat_mean": out["p_hat_mean"].item(),
                            "sigma_mean": out["sigma_mean"].item(),
                            "lr": sched.get_last_lr()[0],
                        }, step=step)

                if step > 0 and step % cfg.train.ckpt_every == 0:
                    _checkpoint(step, kind="step")

                step += 1
                if step >= total_steps:
                    break

    except BaseException as e:
        # Ctrl-C, exception, OOM, pod-reclaim SIGTERM: best-effort save
        ipath = ckpt_dir / f"interrupt_{step:06d}.pt"
        print(f"\n[interrupt] caught {type(e).__name__} at step {step}; "
              f"attempting to save {ipath}")
        try:
            save_checkpoint(str(ipath), model, optim, step=step, scheduler=sched)
            print(f"[interrupt] saved. resume with:  --resume {ipath}")
            print(f"[interrupt] (or simply:          --resume auto)")
        except Exception as se:
            print(f"[interrupt] FAILED to save interrupt checkpoint: {se}")
        raise

    final = ckpt_dir / "final.pt"
    save_checkpoint(str(final), model, optim, step=step, scheduler=sched)
    print(f"[critic] done. final ckpt: {final}")


if __name__ == "__main__":
    main()
