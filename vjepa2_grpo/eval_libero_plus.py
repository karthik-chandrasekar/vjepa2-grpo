"""LIBERO-Plus evaluation harness.

LIBERO-Plus (Fei et al. 2025, arXiv 2510.13626) extends LIBERO with 7
perturbation factors and 21 sub-dimensions over 10,030 tasks. The two columns
where current VLAs collapse hardest — and where our frozen V-JEPA-2 video
prior should help most — are:

  - Camera (viewpoint shifts): OpenVLA-OFT 56.4, pi_0-FAST 65.1
  - Robot state (initial pose perturbations): OpenVLA-OFT 31.9, UniVLA best 46.2

Goal: report per-dimension success rates with stratified subsampling.

This relies on the github.com/sylvestf/LIBERO-plus repo being cloned at
/workspace/LIBERO-plus. The repo exposes:
  - libero_plus.benchmark.get_libero_plus_dict()
  - per-dimension task lists
"""
from __future__ import annotations
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import json


PERTURBATION_DIMS = [
    "objects_layout",   # = "Layout"
    "camera",           # = "Camera"
    "robot",            # = "Robot"
    "language",         # = "Language"
    "light",            # = "Light"
    "background",       # = "Background"
    "noise",            # = "Noise"
]


def _ensure_libero_plus_on_path():
    p = Path("/workspace/LIBERO-plus")
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def evaluate_libero_plus(
    policy,
    out_path: str,
    n_trials_per_task: int = 1,
    max_steps_per_trial: int = 600,
    render_size: int = 384,
    dimensions: Optional[List[str]] = None,
    subsample_per_dim: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Run policy on LIBERO-Plus with per-dimension breakdown.

    Args:
        subsample_per_dim: if set, sample at most this many tasks per
            perturbation dimension (useful for intermediate-checkpoint evals
            during the 3-week sprint). For full eval, leave as None.
    """
    _ensure_libero_plus_on_path()
    try:
        from libero_plus.benchmark import get_libero_plus_dict
    except ImportError:
        raise RuntimeError(
            "libero_plus not importable. Make sure /workspace/LIBERO-plus "
            "is on sys.path and pip-installed (pip install -e .)"
        )

    bench_dict = get_libero_plus_dict()
    if dimensions is None:
        dimensions = PERTURBATION_DIMS

    per_dim_results: Dict[str, Dict] = {}
    for dim in dimensions:
        if dim not in bench_dict:
            print(f"[WARN] dimension {dim} not in libero_plus benchmark; skipping")
            continue
        bench = bench_dict[dim]()
        n_tasks = bench.n_tasks
        task_ids = list(range(n_tasks))
        if subsample_per_dim is not None and subsample_per_dim < n_tasks:
            rng = np.random.RandomState(42)
            task_ids = sorted(rng.choice(task_ids, subsample_per_dim, replace=False).tolist())

        per_task = []
        for task_id in tqdm(task_ids, desc=f"libero_plus/{dim}", disable=not verbose):
            task = bench.get_task(task_id)
            from libero.libero.envs import OffScreenRenderEnv
            env = OffScreenRenderEnv(
                bddl_file_name=task.bddl_file,
                camera_heights=render_size,
                camera_widths=render_size,
            )
            successes = 0
            for trial in range(n_trials_per_task):
                obs = env.reset()
                success = False
                for _ in range(max_steps_per_trial):
                    action = policy.act(obs["agentview_image"], task.language)
                    if action.ndim == 2:
                        for a in action:
                            obs, _, done, info = env.step(
                                a.cpu().numpy() if hasattr(a, "cpu") else a
                            )
                            if info.get("success", False):
                                success = True
                                break
                            if done:
                                break
                    else:
                        obs, _, done, info = env.step(
                            action.cpu().numpy() if hasattr(action, "cpu") else action
                        )
                        if info.get("success", False):
                            success = True
                    if success or done:
                        break
                successes += int(success)
            env.close()
            per_task.append({
                "task_id": task_id, "name": task.name,
                "n_success": successes, "n_trials": n_trials_per_task,
                "rate": successes / n_trials_per_task,
            })

        rate = float(np.mean([t["rate"] for t in per_task]))
        per_dim_results[dim] = {
            "n_tasks_eval": len(task_ids),
            "n_tasks_total": n_tasks,
            "mean_success_rate": rate,
            "per_task": per_task,
        }
        print(f"  {dim}: {rate*100:.1f}%   ({len(task_ids)}/{n_tasks} tasks)")

    total = float(np.mean(
        [per_dim_results[d]["mean_success_rate"] for d in per_dim_results]
    ))
    summary = {
        "total": total,
        "per_dim": {d: per_dim_results[d]["mean_success_rate"]
                     for d in per_dim_results},
        "raw": per_dim_results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"  LIBERO-Plus TOTAL: {total*100:.1f}%")
    return summary
