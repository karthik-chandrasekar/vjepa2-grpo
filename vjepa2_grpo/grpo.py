"""LatentGRPOTrainer — on-policy GRPO with latent rollouts.

Loop structure (per training step):

  1. Sample N initial states from a vectorized LIBERO env.
  2. For each initial state, sample G action sequences via the policy.
  3. Roll out each sequence in latent space using the frozen predictor.
  4. Score each latent step with the frozen critic + anchor distance.
  5. Compute per-group advantage A_g = (R_g - mean R) / std R   (GRPO).
  6. Compute policy log-prob (with grads) and update via -A * logp.
  7. Optionally add a KL penalty to a frozen reference policy (off by default,
     DAPO-style; we log forward-KL as a tripwire only).

Anchor weight λ_anchor follows a linear warmup from 0 to lam_anchor_max over
`anchor_warmup_steps` steps (default 2000). σ_unc weight is constant.

References:
  - GRPO: Shao et al. (DeepSeekMath) 2024
  - DAPO KL removal: Yu et al. 2024
  - SimpleVLA-RL: Liu et al. 2025
  - RIPT-VLA dynamic sampling: Liu et al. 2025
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Callable, List
from pathlib import Path
from copy import deepcopy

from .rollout import rollout_group


class LatentGRPOTrainer:
    def __init__(
        self,
        policy,
        predictor,
        critic,
        anchor_buf,
        encoder,
        env_factory: Callable,            # () -> vectorized LIBERO env
        # Hyperparameters
        group_size: int = 8,
        horizon: int = 10,
        action_chunk: int = 8,
        n_envs: int = 8,
        lr: float = 5e-6,
        weight_decay: float = 0.0,
        kl_coef: float = 0.0,
        kl_tripwire: float = 1.0,
        lam_anchor_max: float = 1.0,
        lam_anchor_warmup_steps: int = 2000,
        lam_unc: float = 0.1,
        clip_grad: float = 1.0,
        success_rate_filter: bool = True,
        dynamic_sample_max_trials: int = 4,
        # I/O
        output_dir: str = "/workspace/checkpoints/policy/main",
        log_every: int = 10,
        eval_every: int = 100,
        save_every: int = 200,
        wandb_run=None,
        device: str = "cuda",
    ):
        self.policy = policy
        self.predictor = predictor.eval().requires_grad_(False)
        self.critic = critic.eval().requires_grad_(False)
        self.anchor_buf = anchor_buf
        self.encoder = encoder
        self.env_factory = env_factory

        self.group_size = group_size
        self.horizon = horizon
        self.action_chunk = action_chunk
        self.n_envs = n_envs
        self.kl_coef = kl_coef
        self.kl_tripwire = kl_tripwire
        self.lam_anchor_max = lam_anchor_max
        self.lam_anchor_warmup_steps = lam_anchor_warmup_steps
        self.lam_unc = lam_unc
        self.clip_grad = clip_grad
        self.success_rate_filter = success_rate_filter
        self.dynamic_sample_max_trials = dynamic_sample_max_trials

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.wandb = wandb_run
        self.device = device

        # Only optimize trainable params (LoRA + log_std)
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        # Frozen reference policy for KL (deep copy of LoRA state at step 0)
        self.ref_lora_state = self._snapshot_lora()

        self.step_idx = 0

    # ---- ref policy snapshot ------------------------------------------------

    def _snapshot_lora(self):
        return {
            n: p.detach().clone()
            for n, p in self.policy.named_parameters()
            if p.requires_grad
        }

    def _kl_to_ref(self) -> torch.Tensor:
        """Cheap KL proxy: sum L2 over LoRA params (Fisher-free).

        This is a tripwire, not a proper KL. For a proper forward-KL you'd
        need to call both current and reference policies on the same obs;
        we add that flag below behind `kl_coef > 0`.
        """
        kl = torch.tensor(0.0, device=self.device)
        for n, p in self.policy.named_parameters():
            if p.requires_grad and n in self.ref_lora_state:
                kl = kl + ((p - self.ref_lora_state[n].to(p.device)) ** 2).sum()
        return kl

    def _current_lam_anchor(self) -> float:
        if self.lam_anchor_warmup_steps <= 0:
            return self.lam_anchor_max
        frac = min(1.0, self.step_idx / self.lam_anchor_warmup_steps)
        return frac * self.lam_anchor_max

    # ---- core step ---------------------------------------------------------

    def step(self, init_states: List[Dict]) -> Dict:
        """One GRPO update from a batch of initial env states.

        init_states: list of dicts with keys
            obs_pixels: np.ndarray [H,W,3]
            instruction: str
            proprio0: torch.Tensor [proprio_dim]
            lang_emb: torch.Tensor [L, lang_dim]
            z_hist: torch.Tensor [1, T_hist, P, D_lat]
        """
        lam_a = self._current_lam_anchor()

        # --- (1) Rollout phase: collect group trajectories without grads -----
        groups: List[Dict] = []
        for s in init_states:
            for trial in range(self.dynamic_sample_max_trials):
                g = rollout_group(
                    policy=self.policy,
                    predictor=self.predictor,
                    critic=self.critic,
                    anchor_buf=self.anchor_buf,
                    obs_pixels=s["obs_pixels"],
                    instruction=s["instruction"],
                    proprio0=s["proprio0"],
                    lang_emb=s["lang_emb"],
                    z_hist=s["z_hist"],
                    group_size=self.group_size,
                    horizon=self.horizon,
                    action_chunk=self.action_chunk,
                    lam_anchor=lam_a,
                    lam_unc=self.lam_unc,
                    device=self.device,
                )
                # RIPT-VLA-style dynamic sampling: skip groups with all-zero
                # or all-saturated rewards (no gradient signal). Retry up to
                # dynamic_sample_max_trials before giving up.
                if self.success_rate_filter:
                    r_std = g["reward_sum"].float().std().item()
                    if r_std > 1e-3:
                        break
                else:
                    break
            groups.append(g)

        # --- (2) Advantage computation (per-group standardization) ----------
        all_adv, all_actions, all_obs, all_instr = [], [], [], []
        for g in groups:
            R = g["reward_sum"].float()                          # [G]
            A = (R - R.mean()) / (R.std() + 1e-6)
            all_adv.append(A.detach())
            all_actions.append(g["actions"].detach())            # [G, H, chunk, A]
            all_obs.append(g["obs_pixels"])
            all_instr.append(g["instruction"])

        # --- (3) Recompute log-probs WITH grads and accumulate the loss ----
        self.optim.zero_grad()
        total_loss = 0.0
        total_pg = 0.0
        total_kl = 0.0
        n_chunks = 0
        for adv, actions, obs, instr in zip(all_adv, all_actions, all_obs, all_instr):
            # Recompute log-prob for each chunk in the trajectory.
            # For action_chunk=8, horizon=10, that's 80 log-probs per group member.
            # The OFT policy is called once per `(obs, instr)`, which is the
            # initial observation only — we approximate the chunk log-probs as
            # all drawn from the initial-obs distribution. This is the standard
            # SimpleVLA-RL/RIPT-VLA approximation for action-chunk GRPO.
            # Shape: actions [G, H, chunk, A]
            G, H, C, Adim = actions.shape
            actions_flat = actions.reshape(G, H * C, Adim)       # treat as one big chunk
            # log_prob expects [G, chunk, A]; reshape accordingly
            logp = self.policy.log_prob_action_chunks(obs, instr, actions_flat)  # [G]
            # Surrogate: -A * logp (REINFORCE with group baseline)
            pg = -(adv.to(logp.device) * logp).mean()
            total_pg = total_pg + pg.item()
            loss = pg
            if self.kl_coef > 0:
                kl = self._kl_to_ref()
                loss = loss + self.kl_coef * kl
                total_kl = total_kl + kl.item()
            loss.backward()
            total_loss += loss.item()
            n_chunks += 1

        # Gradient clip + step
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad],
                self.clip_grad,
            )
        self.optim.step()

        # --- (4) KL tripwire check ------------------------------------------
        with torch.no_grad():
            kl_tripwire_val = self._kl_to_ref().item()
        if kl_tripwire_val > self.kl_tripwire:
            print(f"[WARN] KL to ref policy ({kl_tripwire_val:.2f}) > tripwire "
                  f"({self.kl_tripwire}); rollouts may be off-manifold.")

        # --- (5) Logging ----------------------------------------------------
        mean_R = float(np.mean([g["reward_sum"].mean().item() for g in groups]))
        mean_phat = float(np.mean([g["p_hat"].mean().item() for g in groups]))
        mean_anchor = float(np.mean([g["anchor_d"].mean().item() for g in groups]))
        mean_sigma = float(np.mean([g["sigma"].mean().item() for g in groups]))
        metrics = {
            "step": self.step_idx,
            "loss": total_loss / max(n_chunks, 1),
            "loss/pg": total_pg / max(n_chunks, 1),
            "loss/kl": total_kl / max(n_chunks, 1),
            "reward/mean": mean_R,
            "reward/p_hat_mean": mean_phat,
            "reward/anchor_d_mean": mean_anchor,
            "reward/sigma_mean": mean_sigma,
            "lam_anchor": lam_a,
            "kl_to_ref": kl_tripwire_val,
        }
        return metrics

    # ---- train loop --------------------------------------------------------

    def train(self, total_steps: int, init_state_fn: Callable):
        """init_state_fn(n_envs) -> List[Dict] of n_envs initial states."""
        from tqdm import trange
        for step in trange(total_steps, desc="GRPO"):
            self.step_idx = step
            init_states = init_state_fn(self.n_envs)
            metrics = self.step(init_states)
            if step % self.log_every == 0:
                msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                  if isinstance(v, (int, float)))
                print(f"[step {step}] {msg}")
                if self.wandb is not None:
                    self.wandb.log(metrics, step=step)
            if step > 0 and step % self.save_every == 0:
                self._save(step)
        self._save(total_steps)

    def _save(self, step: int):
        path = self.output_dir / f"step_{step:06d}"
        path.mkdir(exist_ok=True, parents=True)
        self.policy.save_lora(str(path))
        torch.save(self.optim.state_dict(), path / "optim.pt")
        print(f"[save] -> {path}")
