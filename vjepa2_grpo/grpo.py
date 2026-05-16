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

    def _calibrated_R(self, g: Dict, lam_a: float):
        """Plan B: recombine reward from components with the anchor term
        normalized to UNIT cross-rollout variance within the group, so lam_a
        is exactly the anchor:task spread ratio (independent of the raw L2
        magnitude of anchor distances). Returns R [G]."""
        phat_s   = g["phat_sum"].float()
        anchor_s = g["anchor_sum"].float()
        sigma_s  = g["sigma_sum"].float()
        anchor_norm = (anchor_s - anchor_s.mean()) / (anchor_s.std() + 1e-6)
        return phat_s - lam_a * anchor_norm - self.lam_unc * sigma_s

    def step(self, init_states: List[Dict]) -> Dict:
        """One GRPO update from a batch of initial env states.

        init_states: list of dicts with keys
            obs: dict with agentview_image, robot0_eye_in_hand_image, state
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
                    obs=s["obs"],
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
                    # Plan B: gate on the std of the CALIBRATED reward (what
                    # actually drives the gradient), not the raw fused
                    # reward_sum (dominated by un-normalized anchor L2).
                    r_std = self._calibrated_R(g, lam_a).std().item()
                    if r_std > 1e-3:
                        break
                else:
                    break
            groups.append(g)

        # --- (2) Advantage computation (per-group standardization) ----------
        all_adv, all_actions, all_obs, all_instr = [], [], [], []
        for g in groups:
            # Plan B: calibrated reward (anchor normalized to unit group std)
            R = self._calibrated_R(g, lam_a)                     # [G]
            A = (R - R.mean()) / (R.std() + 1e-6)
            all_adv.append(A.detach())
            all_actions.append(g["actions"].detach())            # [G, H, chunk, A]
            all_obs.append(g["obs"])
            all_instr.append(g["instruction"])

        # --- (3) Recompute log-probs WITH grads and accumulate the loss ----
        self.optim.zero_grad()
        total_loss = 0.0
        total_pg = 0.0
        total_kl = 0.0
        n_chunks = 0
        for adv, actions, obs, instr in zip(all_adv, all_actions, all_obs, all_instr):
            # Recompute log-prob for each chunk in the trajectory.
            # actions: [G, H, chunk, A].  The OFT policy conditions only on the
            # initial obs, so all H chunks share the same policy distribution.
            # policy.recompute_log_prob handles the [G, H, C, A] shape directly
            # (computes the mean ONCE, evaluates Gaussian log-prob of every
            # action against it, sums across H, C, A).  Standard SimpleVLA-RL /
            # RIPT-VLA approximation for action-chunk GRPO.
            logp = self.policy.recompute_log_prob(obs, instr, actions)  # [G]
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
            "diag/phat_sum_std": float(np.mean(
                [g["phat_sum"].float().std().item() for g in groups])),
            "diag/anchor_sum_std_raw": float(np.mean(
                [g["anchor_sum"].float().std().item() for g in groups])),
            "diag/r_std_mean": float(np.mean(
                [self._calibrated_R(g, lam_a).std().item() for g in groups])),
        }
        return metrics

    # ---- train loop --------------------------------------------------------

    def train(self, total_steps: int, init_state_fn: Callable, start_step: int = 0):
        """init_state_fn(n_envs) -> List[Dict] of n_envs initial states.

        start_step: resume offset. The loop runs from start_step to total_steps.
        """
        from tqdm import trange
        for step in trange(start_step, total_steps, desc="GRPO", initial=start_step,
                           total=total_steps):
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
                self._save(step, kind="step")
        self._save(total_steps, kind="final")
        print(f"[grpo] done. final ckpt: {self.output_dir / f'final_{total_steps:06d}'}")

    def _save(self, step: int, kind: str = "step"):
        """Atomic save of policy LoRA + optim state.

        kind: 'step' for periodic checkpoints, 'interrupt' for SIGINT-triggered,
              'final' for the end-of-training snapshot.
        Writes to <kind>_NNNNNN.tmp/, then os.replace to <kind>_NNNNNN/.
        os.replace on a directory is atomic within a filesystem, so a SIGKILL
        mid-save leaves the previous ckpt intact rather than half-written.
        """
        import os, shutil
        final_path = self.output_dir / f"{kind}_{step:06d}"
        tmp_path = self.output_dir / f"{kind}_{step:06d}.tmp"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True)

        # LoRA adapter (peft save_pretrained writes adapter_model.safetensors
        # and adapter_config.json into the dir) + policy_extras.pt (log_std)
        self.policy.save_lora(str(tmp_path))

        # Training state: optimizer + step
        torch.save({
            "optimizer": self.optim.state_dict(),
            "step": step,
            "step_idx": self.step_idx,
        }, tmp_path / "train_state.pt")

        # Atomic directory swap
        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(str(tmp_path), str(final_path))
        print(f"[save] {kind}_{step:06d} -> {final_path}")

        # Rolling prune: keep last 3 step_* (final/interrupt always kept)
        if kind == "step":
            self._prune_step_dirs(keep_last=3, milestone_every=200)

    def _prune_step_dirs(self, keep_last: int = 3, milestone_every: int = 200):
        """Rolling retention for step_NNNNNN/ checkpoints.

        Keeps the last `keep_last` and every milestone (every milestone_every
        steps). Never touches interrupt_* or final_*."""
        import re, shutil
        steps = []
        for d in self.output_dir.glob("step_*"):
            if not d.is_dir() or d.name.endswith(".tmp"):
                continue
            m = re.match(r"step_(\d+)$", d.name)
            if m:
                steps.append((int(m.group(1)), d))
        # also nuke any .tmp dirs left behind by a previous crashed save
        for tmp in self.output_dir.glob("*.tmp"):
            if tmp.is_dir():
                shutil.rmtree(tmp, ignore_errors=True)

        if len(steps) <= keep_last:
            return
        steps.sort(key=lambda x: x[0])
        keep_ids = set(s for s, _ in steps[-keep_last:])
        if milestone_every > 0:
            for s, _ in steps:
                if s % milestone_every == 0:
                    keep_ids.add(s)
        removed = 0
        for s, d in steps:
            if s not in keep_ids:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        if removed:
            print(f"[prune] removed {removed} old step dirs; kept {len(keep_ids)}")

    def load_resume(self, ckpt_dir: Path):
        """Restore LoRA + optim from a saved checkpoint directory.

        Returns the saved step number (0 if no ckpt found).
        """
        import re
        train_state_file = ckpt_dir / "train_state.pt"
        if not train_state_file.exists():
            raise FileNotFoundError(f"no train_state.pt in {ckpt_dir}")
        # Load policy LoRA
        self.policy.load_lora(str(ckpt_dir))
        # Load optim + step
        state = torch.load(train_state_file, map_location="cpu", weights_only=False)
        self.optim.load_state_dict(state["optimizer"])
        self.step_idx = state.get("step_idx", state.get("step", 0))
        print(f"[resume] loaded from {ckpt_dir} at step {self.step_idx}")
        return self.step_idx
