"""Smoke test for OpenVLAOFTPolicy. Run in venv-oft after the encoder server is up.

    source /workspace/venv-oft/bin/activate
    python scripts/smoke_policy.py

Validates:
  1. Policy loads (backbone + action_head + proprio_projector + LoRA).
  2. act() returns a deterministic action chunk of correct shape.
  3. sample() returns G stochastic samples + finite log_probs.
  4. recompute_log_prob() returns log_probs that MATCH sample()'s log_probs
     when called on the same (obs, instr, actions) — they should be ~identical
     (within fp tolerance) at the moment of sampling.
  5. backward() on recompute_log_prob's output propagates gradients to LoRA
     params and to log_std. This is the test that catches the grad-flow bug
     class — without it, GRPO would silently fail to train.

If anything here fails, do NOT launch GRPO; the policy wrapper is broken.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspace/vjepa2_grpo")

import numpy as np
import torch

from vjepa2_grpo.policy import OpenVLAOFTPolicy

print("=" * 70)
print("smoke_policy: loading OpenVLA-OFT policy (this takes ~30s)")
print("=" * 70)

policy = OpenVLAOFTPolicy(
    model_id="moojink/openvla-7b-oft-finetuned-libero-spatial",
    action_dim=7,
    action_chunk=8,
    proprio_dim=8,
    num_images_in_input=2,
    dtype=torch.bfloat16,
    device="cuda:0",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)

# Build a fake observation matching the LIBERO format.
# Real obs would come from the env; for the smoke test, random uint8 images
# + zero proprio works (predict_action will produce some valid action chunk).
H, W = 256, 256   # LIBERO renders at 256x256 by default; processor resizes
obs = {
    "agentview_image": np.random.randint(0, 256, (H, W, 3), dtype=np.uint8),
    "robot0_eye_in_hand_image": np.random.randint(0, 256, (H, W, 3), dtype=np.uint8),
    "state": np.zeros((8,), dtype=np.float32),
}
instr = "pick up the alphabet soup"

print("\n[1/5] act() — deterministic action chunk")
with torch.no_grad():
    a = policy.act(obs, instr)
print(f"  action shape: {tuple(a.shape)}  dtype: {a.dtype}  "
      f"min/max: {a.min().item():.3f} / {a.max().item():.3f}")
assert a.shape == (8, 7), f"bad shape {a.shape}"
assert torch.isfinite(a).all(), "non-finite action"

print("\n[2/5] sample(n_samples=4) — stochastic samples + log_probs")
with torch.no_grad():
    actions, lp = policy.sample(obs, instr, n_samples=4)
print(f"  actions: {tuple(actions.shape)}  log_prob: {tuple(lp.shape)} "
      f"values: {lp.tolist()}")
assert actions.shape == (4, 8, 7)
assert lp.shape == (4,)
assert torch.isfinite(lp).all(), "non-finite log_prob"

print("\n[3/5] recompute_log_prob — should ~match sample's log_prob")
# Use the SAME actions; policy hasn't changed between sample and recompute,
# so the two log_probs must be (approximately) equal. Tolerance allows for
# bf16 noise in the mean recompute.
lp_recomputed = policy.recompute_log_prob(obs, instr, actions)
diff = (lp.detach() - lp_recomputed.detach()).abs().max().item()
print(f"  recomputed: {lp_recomputed.detach().tolist()}")
print(f"  max abs diff vs sample: {diff:.4e}")
assert diff < 2e-1, (
    f"sample and recompute_log_prob disagree by {diff:.3e}. "
    "Either grad context is leaking or predict_action is nondeterministic. "
    "Drift >0.2 suggests a real bug; <0.2 is bf16 precision noise between "
    "upstream predict_action (numpy unnorm) and our torch unnorm."
)

print("\n[4/5] backward() flows gradients to LoRA params and log_std")
# Sum the recomputed log_probs and backward. Verify gradient ARRIVES on
# a few key trainable params. This is the critical test.
loss = lp_recomputed.sum()
loss.backward()

# Count LoRA params that received gradients
lora_with_grad = 0
lora_total = 0
log_std_grad_norm = policy.log_std.grad.norm().item() if policy.log_std.grad is not None else None
for name, p in policy.named_parameters():
    if not p.requires_grad:
        continue
    if "lora" in name.lower():
        lora_total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            lora_with_grad += 1

print(f"  log_std.grad norm: {log_std_grad_norm}")
print(f"  LoRA params with nonzero grad: {lora_with_grad} / {lora_total}")
assert log_std_grad_norm is not None and log_std_grad_norm > 0, (
    "log_std got no gradient — Gaussian log-prob backprop is broken"
)
assert lora_with_grad > 0, (
    "NO LoRA param received gradient. The recompute path is not differentiable "
    "through the Llama backbone. predict_action may be wrapping in no_grad "
    "internally — see policy.py docstring for the fallback path."
)
assert lora_with_grad >= lora_total // 2, (
    f"Only {lora_with_grad}/{lora_total} LoRA params got gradients. "
    "Expected lora_B (~50%%) to receive grad on step 0; lora_A activates after step 1."
)

print("\n[5/5] Smoke test complete — policy is ready for GRPO")
print("=" * 70)
print(f"  Sample log_prob:     {[round(x,2) for x in lp.tolist()]}")
print(f"  Recompute log_prob:  {[round(x,2) for x in lp_recomputed.detach().tolist()]}")
print(f"  log_std grad norm:   {log_std_grad_norm:.4e}")
print(f"  LoRA grad coverage:  {lora_with_grad}/{lora_total} = "
      f"{100*lora_with_grad/lora_total:.0f}%")
print("=" * 70)
