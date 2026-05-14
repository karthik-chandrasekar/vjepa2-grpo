"""Shared utilities: logging, seeding, checkpoint helpers."""
from __future__ import annotations
import os
import random
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    step: int = 0,
    extras: Dict[str, Any] = None,
):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extras:
        state["extras"] = extras
    torch.save(state, p)


def load_checkpoint(path: str, model, optimizer=None, strict: bool = True):
    state = torch.load(path, map_location="cpu")
    msg = model.load_state_dict(state["model"], strict=strict)
    print(f"[load] {path}  ->  {msg}")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("step", 0), state.get("extras", {})


def write_json(path: str, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def read_json(path: str):
    return json.loads(Path(path).read_text())


def get_lr_schedule(optim, n_warmup: int, n_total: int, base_lr: float):
    """Linear warmup then cosine decay."""
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
