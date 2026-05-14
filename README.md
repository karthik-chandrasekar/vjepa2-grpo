# vjepa2-grpo

On-policy GRPO for VLA policies inside the latent space of a frozen V-JEPA-2 ViT-g/384 encoder, with a reconstruction-anchored process critic. CoRL 2026 submission.

**Scientific claim (pre-registered):** stable GRPO is achievable inside a frozen self-supervised video encoder's latent space, and a reconstruction-consistency anchor to the frozen encoder's manifold demonstrably prevents the reward hacking that emerges without it.

## Layout

```
vjepa2_grpo/
├── encoder.py          # V-JEPA-2 frozen encoder wrapper (HF)
├── masks.py            # Block-causal attention masks
├── predictor.py        # 24L/1024w action-conditioned latent predictor
├── critic.py           # 6L/768w process critic with 4-head ensemble
├── faiss_anchor.py     # FAISS-backed nearest-neighbor anchor buffer
├── policy.py           # OpenVLA-OFT wrapper + LoRA + action-chunk sampling
├── pseudo_labels.py    # VLA-RL-style dense progress labels
├── datasets.py         # Embedding + demo dataloaders
├── losses.py           # Predictor and critic loss functions
├── rollout.py          # Latent rollout machinery
├── grpo.py             # LatentGRPOTrainer
├── diagnostics.py      # Reward-hacking diagnostics (4 plots)
├── eval_libero.py      # LIBERO-90 / 4-suite eval
├── eval_libero_plus.py # LIBERO-Plus eval with per-perturbation breakdown
└── utils.py            # Logging, seeding, ckpt helpers

scripts/
├── sanity.py
├── precompute_embeddings.py
├── build_faiss.py
├── train_predictor.py
├── train_critic.py
├── grpo_train.py
├── benchmark_rollout.py
└── eval_main.py

configs/
├── predictor.yaml
├── critic.yaml
└── grpo.yaml
```

## Install

```bash
bash setup.sh   # creates /root/venv, installs torch+cu124, FA2, transformers, faiss, LIBERO
source /root/venv/bin/activate
python scripts/sanity.py
```

## Run order (v1, LIBERO-only)

```bash
# Phase 1: cache embeddings
python scripts/precompute_embeddings.py --input /workspace/data/libero --output /workspace/data/embeddings/libero
python scripts/build_faiss.py --emb-root /workspace/data/embeddings --out /workspace/data/anchor_buffer

# Phase 2: train predictor
python scripts/train_predictor.py --config configs/predictor.yaml

# Phase 3: train critic (after predictor partially trained)
python scripts/train_critic.py --config configs/critic.yaml

# Phase 4: GRPO (main + no-anchor)
python scripts/grpo_train.py --config configs/grpo.yaml --tag main
python scripts/grpo_train.py --config configs/grpo.yaml --tag no_anchor --override lam_anchor_max=0.0

# Phase 5: eval + diagnostics
python scripts/eval_main.py --policy-ckpt /workspace/checkpoints/policy/main --suites libero_spatial libero_object libero_goal libero_long
python scripts/eval_main.py --policy-ckpt /workspace/checkpoints/policy/main --libero-plus
python scripts/eval_main.py --policy-ckpt /workspace/checkpoints/policy/main --diagnostics
```

See `RUNBOOK.md` (separate document) for the day-by-day plan and decision gates.

## Verify-before-D0

These are runtime assumptions you should confirm before kicking off:

1. `transformers >= 4.50` with V-JEPA-2 support — run `scripts/sanity.py`.
2. The HF V-JEPA-2 output shape matches what `encoder.py` expects (ViT-g/16 at 384², tubelet 2 → 32 temporal × 576 spatial patches, 1408 dim). If not, edit `encoder.py::VJepa2Encoder.encode_clip`.
3. OpenVLA-OFT LoRA target modules — print `model.named_modules()` and update `configs/grpo.yaml::policy.lora_target_modules`.
4. LIBERO success criterion + 384² rendering — render a sample task at 384² and confirm physics + success predicate behave.
