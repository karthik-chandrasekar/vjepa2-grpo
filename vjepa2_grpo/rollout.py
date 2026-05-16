"""Latent rollout machinery used inside the GRPO loop.

Two main functions:

  - `gather_initial_states(env, n)`: from a vectorized LIBERO env, get the
    initial latent z_0 (via the frozen V-JEPA-2 encoder), proprio, and
    language instruction. Used to seed group rollouts.

  - `rollout_group(...)`: for one (z_0, lang, prop_0) tuple, sample G
    action chunks from the policy, roll forward in latent space via the
    predictor, and return everything the critic + anchor need.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Dict, List, Tuple
from einops import rearrange


@torch.no_grad()
def gather_initial_state(
    env_obs: Dict[str, np.ndarray],
    encoder,
    proprio_keys: List[str] = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
) -> Dict[str, torch.Tensor]:
    """Encode the current LIBERO observation into a single-step latent + proprio.

    Returns:
        z0:       [1, 1, P, D_lat] latent for the current frame
        proprio:  [proprio_dim] current proprio
        pixels:   [H, W, 3] for debugging / pixel-baseline ablation
    """
    # LIBERO obs typically has "agentview_image" or "robot0_eye_in_hand_image"
    img_key = "agentview_image"
    pixels = env_obs[img_key]  # [H, W, 3] uint8 (probably 128x128 in stock LIBERO)
    z_single = encoder.encode_single_observation(pixels)   # [pH, pW, D]
    pH, pW, D = z_single.shape
    z0 = z_single.reshape(1, 1, pH * pW, D)

    proprio_parts = [np.atleast_1d(env_obs[k]) for k in proprio_keys if k in env_obs]
    proprio = np.concatenate(proprio_parts).astype(np.float32) if proprio_parts \
              else np.zeros(8, dtype=np.float32)

    return {
        "z0": z0,
        "proprio": torch.from_numpy(proprio),
        "pixels": pixels,
    }


@torch.no_grad()
def rollout_group(
    policy,
    predictor,
    critic,
    anchor_buf,
    obs,                # dict with agentview_image, robot0_eye_in_hand_image, state
    instruction: str,
    proprio0: torch.Tensor,
    lang_emb: torch.Tensor,
    z_hist: torch.Tensor,
    group_size: int = 8,
    horizon: int = 10,
    action_chunk: int = 8,
    lam_anchor: float = 1.0,
    lam_unc: float = 0.1,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Roll out G action sequences in latent space, score every step.

    Returns:
        actions:    [G, horizon, action_chunk, action_dim]
        log_probs:  [G, horizon]  per-chunk log-prob under sampling policy
        z_rollout:  [G, horizon, action_chunk, P, D_lat]
        p_hat:      [G, horizon, action_chunk]
        sigma:      [G, horizon, action_chunk]
        anchor_d:   [G, horizon, action_chunk]
        reward:     [G, horizon, action_chunk]
        reward_sum: [G]  total reward per trajectory
    """
    B, T_hist, P, D = z_hist.shape
    assert B == 1, "rollout_group expects a single initial state; broadcast inside"

    # Broadcast to group
    z = z_hist.expand(group_size, -1, -1, -1).contiguous().to(device)
    lang_g = lang_emb.unsqueeze(0).expand(group_size, -1, -1).contiguous().to(device)
    proprio0_g = proprio0.unsqueeze(0).expand(group_size, -1).to(device)
    # Initial proprio: replicate proprio0 across all horizon steps as a starting point.
    # The policy will produce action chunks that determine subsequent proprio via the predictor.
    # Since the predictor is action-conditioned (not proprio-predicting), we treat proprio
    # as known/observable. In practice, you'd either run a proprio decoder OR feed the
    # ground-truth env proprio in a hybrid loop. Here we keep proprio0 fixed for simplicity.
    proprio_seq = proprio0_g.unsqueeze(1).expand(-1, horizon * action_chunk, -1)

    all_actions, all_logp = [], []
    all_z, all_phat, all_sigma, all_anchor, all_reward = [], [], [], [], []

    z_cur = z   # [G, T_hist, P, D]

    for h in range(horizon):
        # Policy proposes 1 action chunk per group member.
        # NOTE: this calls the policy `group_size` times. For efficiency on
        # the same observation, batch the policy. Here we call once and
        # sample group_size noisy versions (matching SimpleVLA-RL recipe).
        actions, logp = policy.sample(
            obs, instruction, n_samples=group_size,
        )
        # actions: [G, action_chunk, action_dim]
        all_actions.append(actions)
        all_logp.append(logp)

        # Predictor rolls action_chunk steps forward in latent space.
        # rollout_cached = KV-cached growing-context-within-chunk fast path.
        # (predictor.rollout is the sliding-window reference; rollout_cached's
        #  semantics match predictor._rollout_growing_naive — see the parity test.)
        # Inputs need [G, action_chunk, action_dim] and [G, action_chunk, proprio_dim]
        # Coerce all predictor inputs to its weight dtype. The encoder returns fp16,
        # the policy returns fp32, the predictor is bf16. Cast once here so the
        # rest of the pipeline doesn't have to track dtype.
        pdt = next(predictor.parameters()).dtype
        prop_chunk = proprio0_g.unsqueeze(1).expand(-1, action_chunk, -1).contiguous().to(pdt)
        z_next = predictor.rollout_cached(
            z_hist=z_cur.to(pdt),
            action_chunks=actions.to(device).to(pdt),
            proprio_chunks=prop_chunk,
            lang=lang_g.to(pdt),
            horizon=action_chunk,
        )                                                  # [G, action_chunk, P, D]
        all_z.append(z_next)

        # Critic over a sliding K-window of the predicted latents
        K = critic.window_K
        if z_next.shape[1] >= K:
            # take last K from rolling window over z_cur||z_next
            z_aug = torch.cat([z_cur, z_next], dim=1)
            phat_steps, sigma_steps = [], []
            for s in range(action_chunk):
                # window ending at this rollout step
                end_idx = z_cur.shape[1] + s + 1
                start_idx = max(0, end_idx - K)
                window = z_aug[:, start_idx:end_idx]
                if window.shape[1] < K:
                    # pad front with first frame
                    pad_len = K - window.shape[1]
                    pad = window[:, :1].expand(-1, pad_len, -1, -1)
                    window = torch.cat([pad, window], dim=1)
                p, s_unc = critic(window, lang_g)
                phat_steps.append(p)
                sigma_steps.append(s_unc)
            phat = torch.stack(phat_steps, dim=1)          # [G, action_chunk]
            sigma = torch.stack(sigma_steps, dim=1)
        else:
            phat = torch.zeros(group_size, action_chunk, device=device)
            sigma = torch.zeros(group_size, action_chunk, device=device)

        # Anchor distance per step
        anchor_d = anchor_buf.anchor_distance(z_next)       # [G, action_chunk]

        # Reward
        reward = phat - lam_anchor * anchor_d - lam_unc * sigma

        all_phat.append(phat)
        all_sigma.append(sigma)
        all_anchor.append(anchor_d)
        all_reward.append(reward)

        # Slide z_cur: drop oldest action_chunk steps, append predicted ones,
        # keeping T_hist constant. For T_hist=1 and action_chunk=8, keep only last
        # latent. For longer T_hist, slide naturally.
        if T_hist >= action_chunk:
            z_cur = torch.cat([z_cur[:, action_chunk:], z_next], dim=1)
        else:
            z_cur = z_next[:, -T_hist:]

    actions_t = torch.stack(all_actions, dim=1)            # [G, horizon, chunk, A]
    logp_t = torch.stack(all_logp, dim=1)                  # [G, horizon]
    z_t = torch.stack(all_z, dim=1)                        # [G, horizon, chunk, P, D]
    phat_t = torch.stack(all_phat, dim=1)                  # [G, horizon, chunk]
    sigma_t = torch.stack(all_sigma, dim=1)
    anchor_t = torch.stack(all_anchor, dim=1)
    reward_t = torch.stack(all_reward, dim=1)
    reward_sum = reward_t.sum(dim=(1, 2))                  # [G]
    # Per-component trajectory sums [G], so grpo.py can recombine with the
    # anchor term normalized to unit cross-rollout variance (Plan B).
    phat_sum   = phat_t.sum(dim=(1, 2))                    # [G]
    anchor_sum = anchor_t.sum(dim=(1, 2))                  # [G]
    sigma_sum  = sigma_t.sum(dim=(1, 2))                   # [G]

    return {
        "actions": actions_t,
        "log_probs_sampling": logp_t,
        "z_rollout": z_t,
        "p_hat": phat_t,
        "sigma": sigma_t,
        "anchor_d": anchor_t,
        "reward": reward_t,
        "reward_sum": reward_sum,
        "phat_sum": phat_sum,
        "anchor_sum": anchor_sum,
        "sigma_sum": sigma_sum,
        "obs": obs,
        "instruction": instruction,
    }
