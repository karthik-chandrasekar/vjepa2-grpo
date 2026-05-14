"""Pseudo-label generation for the process critic.

Following VLA-RL (Lu et al. 2025):
  - For a successful trajectory of length T: monotone linear ramp 0 -> 1.
  - For a failed trajectory: a Beta-CDF prior over the per-frame "almost
    succeeded" time, capped at 0.5 to leave headroom for successes.

The Beta prior with alpha=beta=2 puts most of its mass in the middle of the
trajectory, reflecting the observation that most failures happen mid-rollout
(too-early grasp attempts, contact failures), not at t=0 or t=T-1.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import beta as beta_dist


def dense_progress(
    success: bool,
    T: int,
    failed_cap: float = 0.5,
    alpha: float = 2.0,
    beta: float = 2.0,
    smooth: bool = True,
) -> np.ndarray:
    """Return a length-T array of progress labels in [0, failed_cap or 1].

    Args:
        success:    True if the trajectory ended in success.
        T:          length of the trajectory.
        failed_cap: max label on a failed trajectory (default 0.5).
        alpha,beta: Beta distribution params for the failure-time prior.
        smooth:     monotone smoothing on successful trajectories
                    (mostly cosmetic; the linear ramp is already monotone).

    Returns:
        np.ndarray [T] of float32 progress labels.
    """
    if success:
        prog = np.linspace(0.0, 1.0, T, dtype=np.float32)
        if smooth:
            # Slight S-curve so the gradient near 0 and 1 is gentler
            prog = 0.5 * (1 - np.cos(np.pi * prog))
        return prog
    else:
        # Beta CDF over [0,1], scaled to [0, failed_cap]
        ts = np.linspace(0.0, 1.0, T)
        prog = beta_dist.cdf(ts, alpha, beta).astype(np.float32) * failed_cap
        return prog


def windowed_pseudo_labels(
    progress: np.ndarray,
    window_K: int,
) -> np.ndarray:
    """Slide a window of size K and return the progress at the *last* step
    of each window as the window's label.

    Args:
        progress: [T] dense progress labels
        window_K: window size for the critic

    Returns:
        [T - K + 1] labels for the windows starting at indices [0..T-K].
    """
    if len(progress) < window_K:
        return np.array([], dtype=np.float32)
    return progress[window_K - 1 :].astype(np.float32)


# ----------------------------------------------------------------------------
# Batched torch versions for in-graph use during training
# ----------------------------------------------------------------------------

def dense_progress_torch(
    success: torch.Tensor,           # [B] bool/float
    T: int,
    failed_cap: float = 0.5,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> torch.Tensor:
    """Vectorized torch version. Returns [B, T] progress labels."""
    device = success.device
    B = success.shape[0]
    ts = torch.linspace(0.0, 1.0, T, device=device)

    # Successful: smoothed linear ramp
    prog_succ = 0.5 * (1 - torch.cos(torch.pi * ts))  # [T]

    # Failed: Beta CDF (use the regularized incomplete beta = I_x(a,b))
    # torch.special.betainc exists in recent torch
    if hasattr(torch.special, "betainc"):
        a = torch.full_like(ts, alpha)
        b = torch.full_like(ts, beta)
        prog_fail = torch.special.betainc(a, b, ts) * failed_cap
    else:
        # Fallback to scipy (slower; only used if torch version lacks betainc)
        prog_fail_np = beta_dist.cdf(ts.cpu().numpy(), alpha, beta) * failed_cap
        prog_fail = torch.from_numpy(prog_fail_np).to(device).float()

    s = success.float().view(B, 1)
    return s * prog_succ.view(1, T) + (1.0 - s) * prog_fail.view(1, T)
