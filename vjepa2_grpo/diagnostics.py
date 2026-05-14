"""Reward-hacking diagnostics — the publication-defining measurements.

Four plots, all comparing the main run to the no-anchor ablation across
GRPO training steps:

  (1) corr(p_hat, true_success)
        Pearson correlation between predicted rollout-mean progress and
        true LIBERO success across held-out rollouts. Should remain high
        (>0.7) with the anchor; will drop with it disabled.

  (2) anchor_distance(t)
        Mean ||z_tilde_t - NN(z_tilde_t)||^2 as a function of rollout step.
        Direct measure of latent-drift; should be flat with the anchor,
        growing with it disabled.

  (3) gap(predicted_progress, true_success_rate)
        E[p_hat_rollout_mean] - true_success_rate. Growing positive gap
        = reward hacking. THIS IS THE HEADLINE PLOT.

  (4) ensemble_sigma vs rollout_step / perturbation_severity
        How well-calibrated the critic is under distribution shift.

All four are computed from the same data: a pool of held-out rollouts that
includes both imagined (latent-only) and real-env (pixel -> encoder ->
latent) trajectories at each checkpoint.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Callable
import json


def collect_rollouts(
    policy,
    predictor,
    critic,
    anchor_buf,
    encoder,
    env,
    n_rollouts: int = 200,
    horizon: int = 10,
    action_chunk: int = 8,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Run `n_rollouts` rollouts both imagined (latent-only) and real (in env),
    record everything needed for the four diagnostics.

    Returns a dict of arrays:
      imagined_phat:       [n_rollouts, horizon*chunk]
      imagined_sigma:      [n_rollouts, horizon*chunk]
      imagined_anchor_d:   [n_rollouts, horizon*chunk]
      real_success:        [n_rollouts] in {0,1}
      real_phat:           [n_rollouts, horizon*chunk]  (critic scored on
                            encoder embeddings of real env frames)
    """
    out = {
        "imagined_phat": [],
        "imagined_sigma": [],
        "imagined_anchor_d": [],
        "real_success": [],
        "real_phat": [],
        "real_anchor_d": [],
    }

    for r in range(n_rollouts):
        obs = env.reset()
        instr = env.language_instruction if hasattr(env, "language_instruction") \
                else env.task.language
        proprio0 = torch.zeros(14, device=device)  # placeholder; fill from obs
        # initial latent
        z0 = encoder.encode_single_observation(obs["agentview_image"]).to(device)
        pH, pW, D = z0.shape
        z_hist = z0.reshape(1, 1, pH * pW, D)

        # lang emb: placeholder zero; replace with your lang encoder
        lang_emb = torch.zeros(32, 4096, device=device)

        # --- Imagined rollout ---
        from .rollout import rollout_group
        g = rollout_group(
            policy=policy, predictor=predictor, critic=critic,
            anchor_buf=anchor_buf,
            obs_pixels=obs["agentview_image"], instruction=instr,
            proprio0=proprio0, lang_emb=lang_emb, z_hist=z_hist,
            group_size=1, horizon=horizon, action_chunk=action_chunk,
            lam_anchor=0.0, lam_unc=0.0, device=device,    # raw scores
        )
        out["imagined_phat"].append(
            g["p_hat"].squeeze(0).reshape(-1).cpu().numpy()
        )
        out["imagined_sigma"].append(
            g["sigma"].squeeze(0).reshape(-1).cpu().numpy()
        )
        out["imagined_anchor_d"].append(
            g["anchor_d"].squeeze(0).reshape(-1).cpu().numpy()
        )

        # --- Real rollout: execute the deterministic policy in the env ---
        success = 0
        real_phat_per_step = []
        real_anchor_per_step = []
        steps_left = horizon * action_chunk
        cur_obs = obs
        z_real_window = [z_hist.squeeze(0).squeeze(0)]   # list of [P, D]
        K = critic.window_K
        for t in range(steps_left):
            actions = policy.act(cur_obs["agentview_image"], instr)
            # take 1 action from the chunk
            a = actions[0] if actions.ndim > 1 else actions
            cur_obs, _, done, info = env.step(a.detach().cpu().numpy())
            if info.get("success", False):
                success = 1
                break

            # Encode the new frame, build window, score critic + anchor
            z_t = encoder.encode_single_observation(cur_obs["agentview_image"]).to(device)
            z_t_flat = z_t.reshape(-1, D)
            z_real_window.append(z_t_flat)
            window = torch.stack(z_real_window[-K:], dim=0) if len(z_real_window) >= K \
                     else torch.stack(
                         [z_real_window[0]] * (K - len(z_real_window)) + z_real_window,
                         dim=0,
                     )
            window = window.unsqueeze(0)  # [1, K, P, D]
            p, s = critic(window, lang_emb.unsqueeze(0))
            real_phat_per_step.append(p.item())
            real_anchor_per_step.append(
                anchor_buf.anchor_distance(window.unsqueeze(0).squeeze(0).unsqueeze(0))[0, -1].item()
            )

        out["real_success"].append(success)
        out["real_phat"].append(np.array(real_phat_per_step))
        out["real_anchor_d"].append(np.array(real_anchor_per_step))

    # Pad ragged arrays to the same length
    def _pad(lst, fill=np.nan):
        L = max(len(x) for x in lst)
        arr = np.full((len(lst), L), fill, dtype=np.float32)
        for i, x in enumerate(lst):
            arr[i, : len(x)] = x
        return arr

    return {
        "imagined_phat": np.array(out["imagined_phat"]),
        "imagined_sigma": np.array(out["imagined_sigma"]),
        "imagined_anchor_d": np.array(out["imagined_anchor_d"]),
        "real_success": np.array(out["real_success"], dtype=np.int32),
        "real_phat": _pad(out["real_phat"]),
        "real_anchor_d": _pad(out["real_anchor_d"]),
    }


def compute_diagnostics(rollout_data: Dict[str, np.ndarray]) -> Dict[str, float | np.ndarray]:
    """Compute the four diagnostics from a rollout_data bundle.

    Returns:
        diag1_corr:       float  Pearson corr(p_hat_mean, success)
        diag2_drift:      [horizon*chunk]  mean anchor distance per step
        diag3_gap:        float  E[p_hat_mean] - mean(success)
        diag4_sigma:      [horizon*chunk]  mean sigma per step
    """
    imag_phat = rollout_data["imagined_phat"]            # [N, T]
    success = rollout_data["real_success"]               # [N]
    sigma = rollout_data["imagined_sigma"]
    anchor = rollout_data["imagined_anchor_d"]

    phat_mean = np.nanmean(imag_phat, axis=1)            # [N]
    valid = ~np.isnan(phat_mean) & ~np.isnan(success)
    if valid.sum() > 2:
        corr = float(np.corrcoef(phat_mean[valid], success[valid])[0, 1])
    else:
        corr = float("nan")

    drift = np.nanmean(anchor, axis=0)                   # [T]
    gap = float(np.nanmean(phat_mean) - np.nanmean(success))
    sigma_per_step = np.nanmean(sigma, axis=0)           # [T]

    return {
        "diag1_corr": corr,
        "diag2_drift": drift,
        "diag3_gap": gap,
        "diag4_sigma": sigma_per_step,
        "mean_success": float(np.nanmean(success)),
        "mean_phat": float(np.nanmean(phat_mean)),
    }


def plot_diagnostics(
    main_run: Dict[str, Dict],     # {step: diag_dict}
    no_anchor_run: Dict[str, Dict],
    out_dir: str,
):
    """Generate the 4 plots from main vs no-anchor runs across training steps."""
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps_main = sorted(int(s) for s in main_run.keys())
    steps_na = sorted(int(s) for s in no_anchor_run.keys())

    # Plot 1: corr over training steps
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps_main, [main_run[str(s)]["diag1_corr"] for s in steps_main],
            "o-", label="main (anchored)", color="C0")
    ax.plot(steps_na, [no_anchor_run[str(s)]["diag1_corr"] for s in steps_na],
            "x--", label="no-anchor", color="C3")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"corr$(\hat p, $ success$)$")
    ax.set_title("Critic calibration during training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "diag1_corr.pdf")
    plt.close(fig)

    # Plot 2: drift trajectory at final step
    final_main = main_run[str(steps_main[-1])]
    final_na = no_anchor_run[str(steps_na[-1])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(final_main["diag2_drift"], label="main (anchored)", color="C0")
    ax.plot(final_na["diag2_drift"], label="no-anchor", color="C3")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"$\|\tilde z_t - NN(\tilde z_t)\|^2$")
    ax.set_title("Latent drift along rollout")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "diag2_drift.pdf")
    plt.close(fig)

    # Plot 3: headline plot — predicted-vs-true gap
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps_main, [main_run[str(s)]["diag3_gap"] for s in steps_main],
            "o-", label="main (anchored)", color="C0")
    ax.plot(steps_na, [no_anchor_run[str(s)]["diag3_gap"] for s in steps_na],
            "x--", label="no-anchor", color="C3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"$E[\hat p]$ - success rate")
    ax.set_title("Reward hacking gap (HEADLINE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "diag3_gap.pdf")
    plt.close(fig)

    # Plot 4: sigma per step at final checkpoint
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(final_main["diag4_sigma"], label="main (anchored)", color="C0")
    ax.plot(final_na["diag4_sigma"], label="no-anchor", color="C3")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"$\sigma$ (ensemble disagreement)")
    ax.set_title("Critic uncertainty along rollout")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "diag4_sigma.pdf")
    plt.close(fig)

    # Dump raw numbers as JSON for the paper
    (out / "diagnostics_dump.json").write_text(json.dumps({
        "main_run": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                          for kk, vv in v.items()}
                      for k, v in main_run.items()},
        "no_anchor_run": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                               for kk, vv in v.items()}
                           for k, v in no_anchor_run.items()},
    }, indent=2))

    print(f"[diagnostics] wrote 4 plots + dump to {out}")
