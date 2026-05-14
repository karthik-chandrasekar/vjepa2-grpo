"""Predictor training driver.

Run after scripts/precompute_embeddings.py has cached embeddings.

    python scripts/train_predictor.py --config configs/predictor.yaml

Outputs:
    <ckpt_dir>/step_<NNNNNN>.pt   periodic checkpoints
    <ckpt_dir>/final.pt           final checkpoint
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
from tqdm import tqdm

from vjepa2_grpo.predictor import BlockCausalACPredictor
from vjepa2_grpo.datasets import PredictorDataset, collate_predictor
from vjepa2_grpo.losses import predictor_loss
from vjepa2_grpo.utils import (
    set_seed, save_checkpoint, get_lr_schedule, maybe_init_wandb,
    count_params, trainable_params,
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
    print("[predictor] building dataset index...")
    ds = PredictorDataset(
        root=cfg.data.embedding_root,
        T_hist=cfg.data.T_hist,
        horizon=cfg.data.horizon,
        lang_dim=cfg.data.lang_dim,
        max_lang_tokens=cfg.data.max_lang_tokens,
        action_dim=cfg.data.action_dim,
        proprio_dim=cfg.data.proprio_dim,
    )
    print(f"[predictor] dataset has {len(ds)} windows")
    if len(ds) == 0:
        raise RuntimeError(
            f"No training windows found under {cfg.data.embedding_root}. "
            "Did precompute_embeddings.py write .npy + .json pairs?"
        )

    dl = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_predictor,
    )

    # --- Model ---
    model = BlockCausalACPredictor(
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.data.action_dim,
        proprio_dim=cfg.data.proprio_dim,
        lang_dim=cfg.data.lang_dim,
        patches_per_frame=cfg.model.patches_per_frame,
        max_horizon=cfg.model.max_horizon,
        max_lang_tokens=cfg.model.max_lang_tokens,
        dropout=cfg.model.dropout,
        ffn_mult=cfg.model.ffn_mult,
        use_grad_ckpt=cfg.model.use_grad_ckpt,
    ).cuda()

    if cfg.train.bf16:
        model = model.to(torch.bfloat16)
    print(f"[predictor] params: {count_params(model)/1e6:.1f}M  "
          f"trainable: {trainable_params(model)/1e6:.1f}M")

    # --- Optim + sched ---
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
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

    # --- Train ---
    model.train()
    step = start_step
    accum = cfg.train.grad_accum
    t_last = time.time()
    while step < cfg.schedule.total_steps:
        for batch in dl:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            if cfg.train.bf16:
                batch = {k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v
                         for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.bf16):
                out = predictor_loss(
                    model, batch,
                    w_tf=cfg.train.loss.w_tf, w_rc=cfg.train.loss.w_rc,
                )
            loss = out["loss"] / accum
            loss.backward()

            if (step + 1) % accum == 0:
                if cfg.train.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  cfg.train.clip_grad)
                optim.step()
                sched.step()
                optim.zero_grad()

            if step % cfg.train.log_every == 0:
                dt = time.time() - t_last
                t_last = time.time()
                speed = cfg.train.log_every * cfg.train.batch_size / max(dt, 1e-6)
                msg = (f"[step {step}/{cfg.schedule.total_steps}]  "
                       f"loss={out['loss'].item():.4f}  "
                       f"tf={out['L_tf'].item():.4f}  rc={out['L_rc'].item():.4f}  "
                       f"lr={sched.get_last_lr()[0]:.2e}  "
                       f"{speed:.1f} samples/s")
                print(msg)
                if run is not None:
                    run.log({
                        "loss": out["loss"].item(),
                        "L_tf": out["L_tf"].item(),
                        "L_rc": out["L_rc"].item(),
                        "lr": sched.get_last_lr()[0],
                        "samples_per_sec": speed,
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
    print(f"[predictor] done. final ckpt: {final}")


if __name__ == "__main__":
    main()
