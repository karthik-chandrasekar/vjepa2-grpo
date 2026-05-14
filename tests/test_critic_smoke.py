"""Critic smoke test: shapes, ensemble disagreement, loss runs."""
import torch

from vjepa2_grpo.critic import ProgressCritic
from vjepa2_grpo.losses import critic_loss


def _build_tiny():
    return ProgressCritic(
        d_model=64,
        n_layers=2,
        n_heads=4,
        latent_dim=32,
        lang_dim=128,
        n_ensemble=4,
        window_K=4,
        patches_per_frame=8,
    )


def test_forward_shapes():
    model = _build_tiny()
    B, K, P, D = 3, 4, 8, 32
    z_win = torch.randn(B, K, P, D)
    lang = torch.randn(B, 5, 128)

    p_hat, sigma = model(z_win, lang)
    assert p_hat.shape == (B,)
    assert sigma.shape == (B,)
    assert (p_hat >= 0).all() and (p_hat <= 1).all()


def test_per_head_returns():
    model = _build_tiny()
    B = 3
    z_win = torch.randn(B, 4, 8, 32)
    lang = torch.randn(B, 5, 128)
    p_hat, sigma, p_per_head = model(z_win, lang, return_per_head=True)
    assert p_per_head.shape == (B, 4)


def test_loss_runs():
    model = _build_tiny()
    B = 4
    batch = {
        "z_window": torch.randn(B, 4, 8, 32),
        "lang": torch.randn(B, 5, 128),
        "progress": torch.rand(B),
        "success": (torch.rand(B) > 0.5).float(),
    }
    out = critic_loss(model, batch)
    assert "loss" in out
    assert torch.isfinite(out["loss"]).item()
    out["loss"].backward()
