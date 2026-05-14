"""Process critic training driver.

Run AFTER the predictor is partially trained (you don't need it fully trained;
critic only consumes embeddings, not predictor outputs). Typical schedule:

    1. Start predictor (3-4 day run)
    2. After ~50k predictor steps, start critic in parallel on a second GPU
    3. Critic converges in ~12 hours (20k steps)

    python scripts/train_critic.py --config configs/critic.yaml
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
    set_seed, save_checkpoint, get_lr_schedule, maybe_init_wandb,
    count_params,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.train.seed)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    start_step = 0
    if args.resume is not None:
        from vjepa2_grpo.utils import load_checkpoint
        start_step, _ = load_checkpoint(args.resume, model, optim)
        for _ in range(start_step):
            sched.step()

    run = maybe_init_wandb(cfg.wandb.project, cfg.wandb.name,
                           OmegaConf.to_container(cfg, resolve=True),
                           mode=cfg.wandb.mode)

    model.train()
    step = start_step
    t_last = time.time()
    while step < cfg.schedule.total_steps:
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
                print(f"[step {step}]  loss={out['loss'].item():.4f}  "
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
                p = ckpt_dir / f"step_{step:06d}.pt"
                save_checkpoint(str(p), model, optim, step=step)
                print(f"[ckpt] -> {p}")

            step += 1
            if step >= cfg.schedule.total_steps:
                break

    final = ckpt_dir / "final.pt"
    save_checkpoint(str(final), model, optim, step=step)
    print(f"[critic] done. final ckpt: {final}")


if __name__ == "__main__":
    main()
