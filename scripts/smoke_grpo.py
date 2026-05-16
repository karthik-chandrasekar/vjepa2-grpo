"""Phase 5 smoke test: one full GRPO step end-to-end.

Run AFTER starting the encoder server in venv1:

    # in tmux pane A (venv1):
    source /root/venv/bin/activate
    python scripts/encoder_server.py

    # in tmux pane B (venv-oft):
    source /workspace/venv-oft/bin/activate
    python scripts/smoke_grpo.py

What this validates (each step is a potential failure point we want to catch
BEFORE a 24-hour real run):
  1. All four components load in venv-oft: encoder client, policy, predictor,
     critic, FAISS anchor buffer.
  2. The encoder server is reachable and returns sensible shapes for the
     init_state encoding path.
  3. trainer.step() runs without exception (rollout + recompute + backward
     + optimizer step + KL tripwire).
  4. Loss is finite, LoRA params receive nonzero gradients.
  5. _save() writes an atomic checkpoint directory.
  6. load_resume() restores from that checkpoint without complaint.

This bypasses LIBERO entirely — we synthesize a fake init_state with the
right shapes (random images + random proprio + zero lang_emb). The point is
to exercise the *pipeline*, not the env. If this passes, the only remaining
risk in the real launch is LIBERO env setup, which is itself well-tested
upstream.

If any step fails, do NOT launch the real GRPO run; paste the trace.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/vjepa2_grpo")

import numpy as np
import torch
from omegaconf import OmegaConf

from vjepa2_grpo.policy import OpenVLAOFTPolicy
from vjepa2_grpo.predictor import BlockCausalACPredictor
from vjepa2_grpo.critic import ProgressCritic
from vjepa2_grpo.faiss_anchor import FaissAnchorBuffer
from vjepa2_grpo.encoder_client import RemoteVJepa2Encoder
from vjepa2_grpo.grpo import LatentGRPOTrainer
from vjepa2_grpo.utils import load_checkpoint, set_seed


CONFIG_PATH = "/workspace/vjepa2_grpo/configs/grpo.yaml"
ENCODER_URL = "http://127.0.0.1:8765"
SMOKE_OUTPUT_DIR = Path("/tmp/smoke_grpo_output")  # ephemeral; we wipe it


def main():
    print("=" * 72)
    print("smoke_grpo: full one-step GRPO pipeline test")
    print("=" * 72)
    set_seed(0)

    # ---------------- 1. Config ----------------
    cfg = OmegaConf.load(CONFIG_PATH)
    # Reduce to single-env, small group for fast smoke
    cfg.env.n_envs = 1
    cfg.grpo.group_size = 2
    cfg.grpo.horizon = 3
    cfg.grpo.dynamic_sample_max_trials = 1   # don't retry on no-reward-variance
    cfg.grpo.success_rate_filter = False
    cfg.train.output_dir = str(SMOKE_OUTPUT_DIR)
    if SMOKE_OUTPUT_DIR.exists():
        import shutil; shutil.rmtree(SMOKE_OUTPUT_DIR)
    SMOKE_OUTPUT_DIR.mkdir(parents=True)

    print(f"[1/8] config loaded: n_envs={cfg.env.n_envs} G={cfg.grpo.group_size} "
          f"H={cfg.grpo.horizon} C={cfg.grpo.action_chunk}")

    # ---------------- 2. Encoder client ----------------
    print(f"[2/8] connecting to encoder at {ENCODER_URL}...")
    encoder = RemoteVJepa2Encoder(ENCODER_URL)
    info = encoder.wait_until_ready(max_wait_s=30.0)
    print(f"      encoder ready: pool_hw={info['pool_hw']} embed_dim={info['embed_dim']}")

    # ---------------- 3. Frozen modules ----------------
    print(f"[3/8] loading predictor from {cfg.predictor_ckpt}")
    predictor = BlockCausalACPredictor(
        d_model=1024, n_layers=24, n_heads=16,
        latent_dim=1408, action_dim=cfg.policy.action_dim,
        proprio_dim=8, lang_dim=4096, patches_per_frame=64,
        use_grad_ckpt=False,
    ).cuda().to(torch.bfloat16)
    load_checkpoint(cfg.predictor_ckpt, predictor)
    predictor.eval()

    print(f"      loading critic from {cfg.critic_ckpt}")
    critic = ProgressCritic(
        d_model=768, n_layers=6, n_heads=12,
        latent_dim=1408, lang_dim=4096, n_ensemble=4,
        window_K=8, patches_per_frame=64,
    ).cuda().to(torch.bfloat16)
    load_checkpoint(cfg.critic_ckpt, critic)
    critic.eval()

    print(f"      loading anchor buffer from {cfg.anchor_index_dir}")
    anchor_buf = FaissAnchorBuffer.load(cfg.anchor_index_dir, to_gpu=True)

    # ---------------- 4. Policy ----------------
    print(f"[4/8] loading policy {cfg.policy.model_id} (~30s)...")
    policy = OpenVLAOFTPolicy(
        model_id=cfg.policy.model_id,
        action_dim=cfg.policy.action_dim,
        action_chunk=cfg.policy.action_chunk,
        dtype=torch.bfloat16, device="cuda",
        lora_r=cfg.policy.lora_r,
        lora_alpha=cfg.policy.lora_alpha,
        lora_dropout=cfg.policy.lora_dropout,
        lora_target_modules=list(cfg.policy.lora_target_modules),
        action_log_std_init=cfg.policy.action_log_std_init,
    )

    # ---------------- 5. Trainer ----------------
    print("[5/8] building trainer")
    trainer = LatentGRPOTrainer(
        policy=policy, predictor=predictor, critic=critic,
        anchor_buf=anchor_buf, encoder=encoder,
        env_factory=lambda: [],           # we won't call env_factory in this smoke
        group_size=cfg.grpo.group_size,
        horizon=cfg.grpo.horizon,
        action_chunk=cfg.grpo.action_chunk,
        n_envs=cfg.env.n_envs,
        lr=cfg.grpo.lr,
        weight_decay=cfg.grpo.weight_decay,
        kl_coef=cfg.grpo.kl_coef,
        kl_tripwire=cfg.grpo.kl_tripwire,
        lam_anchor_max=cfg.grpo.lam_anchor_max,
        lam_anchor_warmup_steps=cfg.grpo.lam_anchor_warmup_steps,
        lam_unc=cfg.grpo.lam_unc,
        clip_grad=cfg.grpo.clip_grad,
        success_rate_filter=cfg.grpo.success_rate_filter,
        dynamic_sample_max_trials=cfg.grpo.dynamic_sample_max_trials,
        output_dir=cfg.train.output_dir,
        log_every=1, eval_every=1000, save_every=1000,
        wandb_run=None, device="cuda",
    )

    # ---------------- 6. Synthetic init_state ----------------
    # Skip LIBERO sim: build one init_state with random images + zero proprio
    # + zero lang_emb. Same shapes as init_state_fn would produce.
    print("[6/8] synthesizing one init_state (random images, zero proprio/lang_emb)")
    H_img, W_img = 256, 256                          # LIBERO render size
    obs = {
        "agentview_image":         np.random.randint(0, 256, (H_img, W_img, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(0, 256, (H_img, W_img, 3), dtype=np.uint8),
        "state":                   np.zeros((8,), dtype=np.float32),
    }
    instr = "put the moka pot on the stove"

    # Initial latent via encoder server (this is the IPC hot path)
    # Encoder returns fp16; predictor is bf16. Cast here so all latents are bf16.
    z0 = encoder.encode_single_observation(obs["agentview_image"]).cuda().to(torch.bfloat16)
    pH, pW, D_lat = z0.shape
    z_hist = z0.reshape(1, 1, pH * pW, D_lat)        # [1, T_hist=1, P=64, D]
    proprio0 = torch.zeros(8, dtype=torch.float32, device="cuda")
    lang_emb = torch.zeros(32, 4096, dtype=torch.bfloat16, device="cuda")
    init_states = [{
        "obs": obs, "instruction": instr,
        "proprio0": proprio0, "lang_emb": lang_emb,
        "z_hist": z_hist,
    }]

    # ---------------- 7. One trainer.step() ----------------
    print("[7/8] running trainer.step() — full rollout + recompute + backward + step")
    trainer.step_idx = 0
    metrics = trainer.step(init_states)
    print(f"      metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"        {k:24s} = {v:.6f}")
        else:
            print(f"        {k:24s} = {v}")

    # Validate finiteness + grad coverage
    assert np.isfinite(metrics["loss"]), f"loss is not finite: {metrics['loss']}"
    assert np.isfinite(metrics["reward/mean"]), "reward not finite"

    lora_with_grad = sum(
        1 for n, p in policy.named_parameters()
        if p.requires_grad and "lora" in n.lower()
        and p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert lora_with_grad >= 128, (
        f"only {lora_with_grad}/256 LoRA params got grad in real step (expected >= 128)"
    )
    print(f"      ✓ loss is finite: {metrics['loss']:.4f}")
    print(f"      ✓ LoRA grad coverage: {lora_with_grad}/256")

    # ---------------- 8. Checkpoint save + load roundtrip ----------------
    print("[8/8] checkpoint save + load_resume roundtrip")
    trainer.step_idx = 42  # mock the trainer state so save reflects step 42
    trainer._save(step=42, kind="step")
    ckpt_dir = Path(cfg.train.output_dir) / "step_000042"
    assert ckpt_dir.is_dir(), f"ckpt dir not created: {ckpt_dir}"
    assert (ckpt_dir / "train_state.pt").is_file(), "train_state.pt missing"
    assert (ckpt_dir / "policy_extras.pt").is_file(), "policy_extras.pt missing"
    # peft adapter files
    adapter_files = list(ckpt_dir.glob("adapter*"))
    assert adapter_files, f"no adapter_* files in {ckpt_dir}: {list(ckpt_dir.iterdir())}"
    print(f"      ✓ saved: {[p.name for p in sorted(ckpt_dir.iterdir())]}")

    restored = trainer.load_resume(ckpt_dir)
    assert restored == 42, f"resume step wrong: {restored} (expected 42)"
    print(f"      ✓ load_resume returned step={restored}")

    # cleanup
    import shutil; shutil.rmtree(SMOKE_OUTPUT_DIR)

    print()
    print("=" * 72)
    print("SMOKE PASSED — GRPO pipeline is ready for the real launch.")
    print("=" * 72)


if __name__ == "__main__":
    main()
