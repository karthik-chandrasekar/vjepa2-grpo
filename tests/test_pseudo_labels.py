"""Unit tests for pseudo_labels."""
import numpy as np
import pytest
import torch

from vjepa2_grpo.pseudo_labels import (
    dense_progress, windowed_pseudo_labels, dense_progress_torch,
)


def test_success_monotonic():
    p = dense_progress(success=True, T=50)
    assert p.shape == (50,)
    # Monotone non-decreasing
    assert (np.diff(p) >= -1e-6).all()
    # Endpoints
    assert p[0] == pytest.approx(0.0, abs=1e-6)
    assert p[-1] == pytest.approx(1.0, abs=1e-6)


def test_failure_capped():
    p = dense_progress(success=False, T=50, failed_cap=0.5)
    assert p.max() <= 0.5 + 1e-6
    assert p.min() >= 0.0


def test_failure_monotonic():
    p = dense_progress(success=False, T=50)
    # Beta CDF is monotone
    assert (np.diff(p) >= -1e-6).all()


def test_windowed_labels_length():
    p = np.linspace(0, 1, 20)
    w = windowed_pseudo_labels(p, window_K=8)
    assert w.shape == (20 - 8 + 1,)
    # Label = last entry of window
    assert w[0] == pytest.approx(p[7])
    assert w[-1] == pytest.approx(p[-1])


def test_windowed_labels_too_short():
    p = np.array([0.0, 0.5])
    w = windowed_pseudo_labels(p, window_K=8)
    assert w.shape == (0,)


def test_torch_version_matches_numpy():
    if not hasattr(torch.special, "betainc"):
        pytest.skip("torch.special.betainc unavailable in this torch")
    s = torch.tensor([0.0, 1.0])
    p_torch = dense_progress_torch(s, T=10)
    p_np_succ = dense_progress(True, 10)
    p_np_fail = dense_progress(False, 10)
    np.testing.assert_allclose(p_torch[1].numpy(), p_np_succ, atol=1e-5)
    np.testing.assert_allclose(p_torch[0].numpy(), p_np_fail, atol=1e-3)
