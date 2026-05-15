"""Predictor training driver.

Run after scripts/precompute_embeddings.py has cached embeddings.

    python scripts/train_predictor.py --config configs/predictor.yaml
    python scripts/train_predictor.py --config configs/predictor.yaml --resume auto
    python scripts/train_predictor.py --config configs/predictor.yaml --resume <path.pt>

Outputs (under cfg.train.ckpt_dir):
    step_<NNNNNN>.pt        periodic checkpoints (rolling-pruned)
    interrupt_<NNNNNN>.pt   written if the run is interrupted / crashes
    final.pt                final checkpoint

Hardened for multi-day runs:
  - checkpoints are written atomically (.tmp + os.replace)
  - optimizer AND lr-scheduler state are saved/restored (faithful resume)
  - `--resume auto` picks up the newest step_/interrupt_ checkpoint
  - rolling retention: keep last N + every M-step milestone (see config)
  - any interruption (Ctrl-C, exception, OOM) tries to save interrupt_<step>.pt
    before propagating, so a crash costs minutes, not hours
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
    set_seed, save_checkpoint, load_checkpoint, get_lr_schedule,
    maybe_init_wandb, count_params, trainable_params,
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

    # --- Resume ---
    # scheduler state is restored from the checkpoint directly (LambdaLR has a
    # state_dict). The previous step-replay hack — `for _ in range(start_step):
    # sched.step()` — advanced the schedule once per *step*, but the loop calls
    # sched.step() only once per grad_accum (every `accum` steps), so it
    # fast-forwarded the LR schedule grad_accum-times too far. Fixed here.
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
    accum = cfg.train.grad_accum
    t_last = time.time()
    total_steps = cfg.schedule.total_steps

    def _checkpoint(tag_step: int, kind: str = "step"):
        """Write a checkpoint (atomic) and, for periodic ones, prune."""
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
                    msg = (f"[step {step}/{total_steps}]  "
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
                    _checkpoint(step, kind="step")

                step += 1
                if step >= total_steps:
                    break

    except BaseException as e:
        # Ctrl-C, exception, CUDA OOM, pod SIGTERM-on-reclaim: try to save
        # before propagating so the run is resumable from very close to here.
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
    print(f"[predictor] done. final ckpt: {final}")


if __name__ == "__main__":
    main()
