"""Shared utilities: logging, seeding, checkpoint helpers.

Checkpoint helpers are hardened for long (multi-day) training runs:
  - save_checkpoint  : atomic write (.tmp + os.replace) so a mid-write crash
                       cannot corrupt an existing checkpoint; also persists
                       optimizer AND lr-scheduler state.
  - load_checkpoint  : restores model + optimizer + scheduler.
  - prune_checkpoints: rolling retention (keep last N + step milestones) so a
                       120k-step run doesn't fill the volume with ~3.7GB ckpts.
  - find_latest_checkpoint: locate the newest resumable ckpt for `--resume auto`.
"""
from __future__ import annotations
import os
import re
import random
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    step: int = 0,
    extras: Dict[str, Any] = None,
    scheduler=None,
):
    """Atomically save a training checkpoint.

    Writes to `<path>.tmp` then os.replace()s onto `<path>`. os.replace is
    atomic within a filesystem, so an interrupted write leaves the previous
    checkpoint (if any) intact rather than truncating it.

    `scheduler` is appended last in the signature so existing positional
    callers — save_checkpoint(path, model, optim, step=...) — are unaffected.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extras:
        state["extras"] = extras

    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, p)  # atomic on a single filesystem


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    strict: bool = True,
    scheduler=None,
):
    """Restore model (+ optimizer + scheduler) from a checkpoint.

    `scheduler` appended last for backward-compatible positional calls.
    Returns (step, extras).
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(state["model"], strict=strict)
    print(f"[load] {path}  ->  {msg}")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    elif scheduler is not None:
        print("[load] WARNING: checkpoint has no scheduler state; "
              "lr schedule will not be exactly resumed")
    return state.get("step", 0), state.get("extras", {})


_STEP_RE = re.compile(r"(?:step|interrupt)_(\d+)\.pt$")


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return the path of the newest resumable checkpoint in `ckpt_dir`.

    Considers `step_*.pt` and `interrupt_*.pt` (the latter written when a run
    is interrupted). Ignores `final.pt` — if that exists, training finished and
    there is nothing to resume. Ignores stray `*.tmp` from interrupted writes.
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    cands = []
    for p in list(d.glob("step_*.pt")) + list(d.glob("interrupt_*.pt")):
        m = _STEP_RE.search(p.name)
        if m:
            cands.append((int(m.group(1)), p))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return str(cands[-1][1])


def prune_checkpoints(
    ckpt_dir: str,
    keep_last: int = 2,
    milestone_every: int = 10000,
):
    """Rolling-retention prune of periodic `step_*.pt` checkpoints.

    Keeps:
      - the `keep_last` most recent step checkpoints
      - every checkpoint whose step is a multiple of `milestone_every`
    Never touches `interrupt_*.pt` or `final.pt`.
    Also clears stale `*.tmp` files left by interrupted writes.

    With keep_last=2, milestone_every=10000 on a 120k-step run, peak on-disk
    is ~14 checkpoints (12 milestones + 2 rolling) ~= 52GB.
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return

    # clear stale temp files from interrupted writes
    for tmp in d.glob("*.tmp"):
        try:
            tmp.unlink()
        except OSError:
            pass

    steps = []
    for p in d.glob("step_*.pt"):
        m = re.match(r"step_(\d+)\.pt$", p.name)
        if m:
            steps.append((int(m.group(1)), p))
    if len(steps) <= keep_last:
        return
    steps.sort(key=lambda x: x[0])

    keep = set(p for _, p in steps[-keep_last:])
    if milestone_every and milestone_every > 0:
        for s, p in steps:
            if s % milestone_every == 0:
                keep.add(p)

    removed = []
    for s, p in steps:
        if p not in keep:
            try:
                p.unlink()
                removed.append(p.name)
            except OSError as e:
                print(f"[prune] could not remove {p.name}: {e}")
    if removed:
        print(f"[prune] removed {len(removed)} old checkpoint(s); "
              f"kept {len(keep)} (last {keep_last} + milestones)")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def write_json(path: str, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def read_json(path: str):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# LR schedule + wandb
# ---------------------------------------------------------------------------

def get_lr_schedule(optim, n_warmup: int, n_total: int, base_lr: float):
    """Linear warmup then cosine decay. Returns a LambdaLR (has state_dict)."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(step):
        if step < n_warmup:
            return step / max(1, n_warmup)
        progress = (step - n_warmup) / max(1, n_total - n_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optim, lr_lambda=lr_lambda)


def maybe_init_wandb(project: str, name: str, config: Dict, mode: str = "online"):
    try:
        import wandb
        run = wandb.init(project=project, name=name, config=config, mode=mode)
        return run
    except Exception as e:
        print(f"[wandb] init failed: {e}; continuing without wandb")
        return None
