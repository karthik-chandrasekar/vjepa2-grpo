"""Parity test for the KV-cached rollout.

rollout_cached() must produce the same latents as _rollout_growing_naive()
(the non-cached implementation of the *same* growing-context semantics).

WHY fp32 + MATH backend:
  The cached path's incremental steps run mask-free (flash kernel); the naive
  path runs masked (memory-efficient kernel). Different kernels => different
  floating-point reduction order => bf16 results would differ at ~1e-2, too
  loose to catch a real bug. In fp32 under the MATH backend both paths use the
  same reference kernel, so any real discrepancy (off-by-one in the cache,
  wrong PE offset, wrong mask) shows up as a large diff, not noise.

A SEPARATE bf16 smoke check confirms the path runs in the production dtype and
gives finite, sane-magnitude outputs.

Run on the pod:
    python -m pytest tests/test_rollout_cached_parity.py -v -s
"""
import pytest
import torch

from vjepa2_grpo.predictor import BlockCausalACPredictor


def _tiny_model(dtype, seed=0):
    torch.manual_seed(seed)
    m = BlockCausalACPredictor(
        d_model=128, n_layers=4, n_heads=4,
        latent_dim=48, action_dim=7, proprio_dim=14, lang_dim=64,
        patches_per_frame=16, max_horizon=16, max_lang_tokens=16,
        use_grad_ckpt=False,
    )
    return m.to(dtype).eval()


def _inputs(B, T_hist, horizon, P, D_lat, L, dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, T_hist, P, D_lat, generator=g, dtype=torch.float32)
    a = torch.randn(B, horizon, 7, generator=g, dtype=torch.float32)
    p = torch.randn(B, horizon, 14, generator=g, dtype=torch.float32)
    l = torch.randn(B, L, 64, generator=g, dtype=torch.float32)
    return (z.to(dtype), a.to(dtype), p.to(dtype), l.to(dtype))


@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_rollout_cached_matches_growing_naive_fp32(horizon):
    """Tight fp32 parity under the MATH backend — the real correctness gate."""
    from torch.nn.attention import sdpa_kernel, SDPBackend

    B, T_hist, P, D_lat, L = 2, 8, 16, 48, 5
    m = _tiny_model(torch.float32)
    z, a, p, l = _inputs(B, T_hist, horizon, P, D_lat, L, torch.float32)

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        r_naive = m._rollout_growing_naive(z, a, p, l, horizon)
        r_cached = m.rollout_cached(z, a, p, l, horizon)

    assert r_naive.shape == r_cached.shape == (B, horizon, P, D_lat)
    max_abs = (r_naive - r_cached).abs().max().item()
    rel = max_abs / (r_naive.abs().max().item() + 1e-8)
    print(f"\n[horizon={horizon}] max_abs_diff={max_abs:.2e}  rel={rel:.2e}")
    # MATH backend, fp32, same math in different order: expect ~1e-5 or better.
    # 1e-3 is a generous ceiling; a real bug blows past it (or NaNs).
    assert max_abs < 1e-3, (
        f"rollout_cached diverged from growing-naive reference: "
        f"max_abs_diff={max_abs:.3e} (horizon={horizon}). KV-cache is WRONG."
    )


def test_prefill_embedding_matches_assemble_sequence():
    """The cached path's prefill embedding must equal _assemble_sequence's
    output (lang + history portion). Catches PE-offset / mod-embedding bugs."""
    B, T_hist, P, D_lat, L = 2, 8, 16, 48, 5
    m = _tiny_model(torch.float32)
    z, a, p, l = _inputs(B, T_hist, 1, P, D_lat, L, torch.float32)

    # what _assemble_sequence produces for [lang + T_hist history steps]
    a_hist = torch.zeros(B, T_hist, 7)
    p_hist = torch.zeros(B, T_hist, 14)
    with torch.inference_mode():
        seq_ref, L_ref, T_ref, P_ref = m._assemble_sequence(z, a_hist, p_hist, l)
        # what the cached path builds
        lang_tok = m._embed_lang_tokens(l)
        hist_tok = m._embed_step_tokens(z, a_hist, p_hist, global_start=0, L=L)
        seq_cached = torch.cat([lang_tok, hist_tok], dim=1)

    assert seq_ref.shape == seq_cached.shape
    max_abs = (seq_ref - seq_cached).abs().max().item()
    print(f"\n[prefill embed] max_abs_diff={max_abs:.2e}")
    assert max_abs < 1e-5, (
        f"cached-path prefill embedding != _assemble_sequence: {max_abs:.3e}. "
        f"PE offset or modality embedding is wrong."
    )


def test_rollout_cached_bf16_runs_and_is_finite():
    """Production-dtype smoke check: bf16 path runs, shapes right, finite,
    sane magnitude. (Not a tight parity check — see the fp32 test for that.)"""
    if not torch.cuda.is_available():
        pytest.skip("bf16 SDPA path needs CUDA")

    B, T_hist, P, D_lat, L, horizon = 2, 8, 16, 48, 5, 8
    m = _tiny_model(torch.bfloat16).cuda()
    z, a, p, l = _inputs(B, T_hist, horizon, P, D_lat, L, torch.bfloat16)
    z, a, p, l = z.cuda(), a.cuda(), p.cuda(), l.cuda()

    with torch.inference_mode():
        r_cached = m.rollout_cached(z, a, p, l, horizon)
        r_naive = m._rollout_growing_naive(z, a, p, l, horizon)

    assert r_cached.shape == (B, horizon, P, D_lat)
    assert torch.isfinite(r_cached).all(), "rollout_cached produced non-finite values"
    # loose bf16 cross-kernel agreement — just a sanity bound, not the gate
    max_abs = (r_cached.float() - r_naive.float()).abs().max().item()
    print(f"\n[bf16 smoke] max_abs_diff(cached,naive)={max_abs:.2e}")
    assert max_abs < 0.5, (
        f"bf16 cached vs naive diverged by {max_abs:.3e} — far beyond "
        f"cross-kernel fp noise; likely a real bug."
    )
