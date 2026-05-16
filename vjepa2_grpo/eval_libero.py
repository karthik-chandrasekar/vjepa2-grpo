"""LIBERO evaluation harness.

Runs a trained policy against the 4 LIBERO suites:
  - libero_spatial
  - libero_object
  - libero_goal
  - libero_long

For LIBERO-90: pass suite="libero_90".

Success criterion follows the LIBERO paper: the goal predicate must hold
continuously for >= 10 timesteps. The LIBERO env's `info["success"]` already
implements this, so we just track that.

Resolution: stock LIBERO renders at 128x128. We rerender at 384x384 to match
the V-JEPA-2 encoder's training resolution. This is set via the env init.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm


def make_libero_env(suite: str, task_id: int, render_size: int = 384):
    """Build a single LIBERO env at the given resolution."""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    # Use the benchmark's resolver to get the full path; `task.bddl_file` is
    # often just a basename, which fails OffScreenRenderEnv's os.path.exists check.
    bddl_path = bench.get_task_bddl_file_path(task_id)
    env_args = dict(
        bddl_file_name=bddl_path,
        camera_heights=render_size,
        camera_widths=render_size,
    )
    env = OffScreenRenderEnv(**env_args)
    env.language_instruction = task.language
    env.task_meta = {"task_id": task_id, "name": task.name}
    return env, task


def evaluate_suite(
    policy,
    suite: str,
    n_trials_per_task: int = 50,
    max_steps_per_trial: int = 600,
    render_size: int = 384,
    out_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Returns aggregate + per-task results."""
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()[suite]()
    n_tasks = bench.n_tasks

    per_task = []
    for task_id in tqdm(range(n_tasks), desc=f"{suite}", disable=not verbose):
        env, task = make_libero_env(suite, task_id, render_size=render_size)
        successes = 0
        for trial in range(n_trials_per_task):
            obs = env.reset()
            done = False
            success = False
            for _ in range(max_steps_per_trial):
                action = policy.act(obs["agentview_image"], task.language)
                # OpenVLA-OFT predict_action returns the full 8-step chunk.
                # Standard practice: execute every action in the chunk before re-querying.
                if action.ndim == 2:
                    for a in action:
                        obs, _, done, info = env.step(a.cpu().numpy() if hasattr(a, "cpu") else a)
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
            "task_id": task_id,
            "name": task.name,
            "n_trials": n_trials_per_task,
            "n_success": successes,
            "rate": successes / n_trials_per_task,
        })

    agg = {
        "suite": suite,
        "n_tasks": n_tasks,
        "n_trials_per_task": n_trials_per_task,
        "mean_success_rate": float(np.mean([t["rate"] for t in per_task])),
        "per_task": per_task,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        import json
        Path(out_path).write_text(json.dumps(agg, indent=2))
    return agg


def evaluate_all_libero(
    policy,
    out_dir: str,
    n_trials_per_task: int = 50,
    suites: List[str] = ("libero_spatial", "libero_object", "libero_goal", "libero_long"),
):
    """Run all suites and return a combined dict + write JSONs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for s in suites:
        r = evaluate_suite(
            policy, s, n_trials_per_task=n_trials_per_task,
            out_path=str(out / f"{s}.json"),
        )
        results[s] = r
        print(f"  {s}: {r['mean_success_rate']*100:.1f}%")

    avg = float(np.mean([results[s]["mean_success_rate"] for s in suites]))
    summary = {"suites": suites, "per_suite": {s: results[s]["mean_success_rate"]
                                                for s in suites},
               "average": avg}
    import json
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  AVERAGE: {avg*100:.1f}%")
    return summary
