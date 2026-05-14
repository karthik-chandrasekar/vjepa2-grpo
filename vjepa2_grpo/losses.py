"""Loss functions for the predictor and critic.

`predictor_loss`:
    Combines teacher-forcing (T=1 conditioned on ground-truth history) with
    a 2-step rollout-consistency term (V-JEPA-2-AC Eq. 3). Stop-grad on the
    1-step prediction when used as history for the 2-step prediction, matching
    JEPA-style stop-gradient.

`critic_loss`:
    BCE on the dense progress label, with:
      - monotonicity penalty on successful trajectories only
      - bootstrap-mask BCE per ensemble head
      - optional anchor-aware term (off by default during pretraining; the
        anchor is enforced at GRPO time inside the reward, not the critic loss)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def predictor_loss(
    model,
    batch: Dict[str, torch.Tensor],
    w_tf: float = 0.5,
    w_rc: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    batch:
        z_hist:    [B, T_hist, P, D_lat]
        z_targets: [B, horizon, P, D_lat]    (horizon must be >= 2 for RC term)
        actions:   [B, T_hist + horizon, A]
        proprio:   [B, T_hist + horizon, Q]
        lang:      [B, L, D_lang]
    """
    z_hist = batch["z_hist"]
    z_targets = batch["z_targets"]
    actions = batch["actions"]
    proprio = batch["proprio"]
    lang = batch["lang"]

    B, T_hist, P, D = z_hist.shape
    horizon = z_targets.shape[1]
    assert horizon >= 2, "predictor_loss expects horizon >= 2 for RC term"

    # ---- teacher-forcing: predict z_{T_hist} ... z_{T_hist+horizon-1}
    z_pred_tf = model(z_hist, actions, proprio, lang, horizon=horizon)
    L_tf = F.smooth_l1_loss(z_pred_tf, z_targets)

    # ---- 2-step rollout consistency:
    # Feed the model its own 1-step prediction (stop-grad) as the latest
    # history, then predict the next step. Compare to z_targets[:, 1].
    z_pred_step1 = z_pred_tf[:, 0:1].detach()                       # [B,1,P,D]
    # Slide z_hist by 1 (drop oldest, append predicted)
    z_hist_shifted = torch.cat([z_hist[:, 1:], z_pred_step1], dim=1)
    # Need actions/proprio shifted by 1 as well
    actions_shifted = actions[:, 1:]
    proprio_shifted = proprio[:, 1:]
    # Predict horizon-1 future steps from the shifted window
    z_pred_rc = model(z_hist_shifted, actions_shifted, proprio_shifted,
                      lang, horizon=horizon - 1)
    L_rc = F.smooth_l1_loss(z_pred_rc, z_targets[:, 1:])

    loss = w_tf * L_tf + w_rc * L_rc
    return {
        "loss": loss,
        "L_tf": L_tf.detach(),
        "L_rc": L_rc.detach(),
    }


def critic_loss(
    critic,
    batch: Dict[str, torch.Tensor],
    bootstrap_p: float = 0.75,
    w_mono: float = 0.05,
    success_threshold: float = 0.95,
) -> Dict[str, torch.Tensor]:
    """
    batch:
        z_window: [B, K, P, D]
        lang:     [B, L, D_lang]
        progress: [B]   in [0,1]
        success:  [B]   in {0,1}
    """
    z_win = batch["z_window"]
    lang = batch["lang"]
    progress = batch["progress"]
    success = batch["success"]

    B = z_win.shape[0]
    device = z_win.device

    # Bootstrap mask: each head sees a different ~bootstrap_p fraction of batch
    keep_mask = (torch.rand(B, critic.n_ensemble, device=device) < bootstrap_p).float()
    # ensure no empty head
    for h in range(critic.n_ensemble):
        if keep_mask[:, h].sum() == 0:
            keep_mask[0, h] = 1.0

    logits, mask = critic.forward_with_bootstrap_mask(z_win, lang, keep_mask)
    # logits: [B, n_ensemble]

    target = progress.unsqueeze(-1).expand_as(logits)               # [B, E]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    L_progress = (bce * mask).sum() / mask.sum().clamp(min=1.0)

    # Monotonicity penalty on successful trajectories only.
    # Approximation here: within a batch, look at adjacent samples with the
    # same source demo and apply the penalty. Since we don't track demo-ids
    # in the current batch, we apply it on a per-batch "successful sample"
    # cohort: sort by progress and penalize non-monotone p_hat among them.
    p_hat = torch.sigmoid(logits.mean(dim=-1))                      # [B]
    L_mono = torch.tensor(0.0, device=device)
    succ_mask = (success > success_threshold)
    if succ_mask.sum() >= 2:
        prog_succ = progress[succ_mask]
        ph_succ = p_hat[succ_mask]
        order = torch.argsort(prog_succ)
        ph_sorted = ph_succ[order]
        # Penalize violations of monotonicity in the sorted order
        L_mono = (ph_sorted[:-1] - ph_sorted[1:]).clamp(min=0).mean()

    loss = L_progress + w_mono * L_mono

    return {
        "loss": loss,
        "L_progress": L_progress.detach(),
        "L_mono": L_mono.detach(),
        "p_hat_mean": p_hat.mean().detach(),
        "sigma_mean": logits.std(dim=-1).mean().detach(),
    }
