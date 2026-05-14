"""Predictor smoke test: tiny config, forward + backward, shape checks."""
import pytest
import torch

from vjepa2_grpo.predictor import BlockCausalACPredictor


def _build_tiny():
    return BlockCausalACPredictor(
        d_model=64,
        n_layers=2,
        n_heads=4,
        latent_dim=32,
        action_dim=7,
        proprio_dim=14,
        lang_dim=128,
        patches_per_frame=16,
        max_horizon=8,
        max_lang_tokens=8,
        use_grad_ckpt=False,
    )


def test_forward_shape():
    model = _build_tiny()
    B, T_hist, horizon = 2, 4, 3
    P, D = 16, 32
    z_hist = torch.randn(B, T_hist, P, D)
    actions = torch.randn(B, T_hist + horizon, 7)
    proprio = torch.randn(B, T_hist + horizon, 14)
    lang = torch.randn(B, 4, 128)

    out = model(z_hist, actions, proprio, lang, horizon=horizon)
    assert out.shape == (B, horizon, P, D)


def test_backward_runs():
    model = _build_tiny()
    B, T_hist, horizon = 2, 4, 2
    P, D = 16, 32
    z_hist = torch.randn(B, T_hist, P, D, requires_grad=False)
    actions = torch.randn(B, T_hist + horizon, 7)
    proprio = torch.randn(B, T_hist + horizon, 14)
    lang = torch.randn(B, 4, 128)
    target = torch.randn(B, horizon, P, D)

    out = model(z_hist, actions, proprio, lang, horizon=horizon)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    # At least one parameter must have a non-zero grad
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters() if p.requires_grad
    )
    assert has_grad, "no gradients flowing through predictor"


def test_rollout_shape():
    model = _build_tiny()
    model.eval()
    B = 2
    P, D = 16, 32
    z_hist = torch.randn(B, 4, P, D)
    action_chunks = torch.randn(B, 5, 7)
    proprio_chunks = torch.randn(B, 5, 14)
    lang = torch.randn(B, 4, 128)

    out = model.rollout(z_hist, action_chunks, proprio_chunks, lang, horizon=5)
    assert out.shape == (B, 5, P, D)
