"""Unit tests for masks.block_causal_mask."""
import pytest
import torch

from vjepa2_grpo.masks import block_causal_mask, step_token_indices


def test_mask_shape():
    L, T, P = 4, 3, 5
    tps = P + 2
    m = block_causal_mask(L=L, T=T, P=P, device="cpu", dtype=torch.float32)
    assert m.shape == (L + T * tps, L + T * tps)


def test_lang_visible_to_everyone():
    L, T, P = 4, 3, 5
    m = block_causal_mask(L=L, T=T, P=P, device="cpu", dtype=torch.float32)
    # Every row should be able to attend to all language columns
    assert (m[:, :L] == 0.0).all(), "Language tokens should be visible everywhere"


def test_step_causality():
    L, T, P = 2, 4, 3
    tps = P + 2
    m = block_causal_mask(L=L, T=T, P=P, device="cpu", dtype=torch.float32)
    # Step 0 tokens should NOT see step 1 tokens
    s0 = slice(L, L + tps)
    s1 = slice(L + tps, L + 2 * tps)
    assert torch.isinf(m[s0, s1]).all(), "Step 0 must not see step 1"

    # Step 2 tokens SHOULD see step 0 tokens
    s2 = slice(L + 2 * tps, L + 3 * tps)
    assert (m[s2, s0] == 0.0).all(), "Step 2 must see step 0"


def test_intra_step_bidirectional():
    L, T, P = 2, 2, 3
    tps = P + 2
    m = block_causal_mask(L=L, T=T, P=P, device="cpu", dtype=torch.float32)
    # Within a step, all positions should see all positions
    s = slice(L, L + tps)
    assert (m[s, s] == 0.0).all()


def test_step_token_indices():
    L, T, P = 2, 3, 4
    lat, act, prop = step_token_indices(L, T, P)
    assert len(lat) == T
    assert len(act) == T
    assert len(prop) == T
    assert len(lat[0]) == P
    # First step's latent indices should start right after language
    assert lat[0][0] == L
    # Action token = base + P
    assert act[0] == L + P
    # Proprio token = base + P + 1
    assert prop[0] == L + P + 1
    # Step 2 should be offset by 2 * (P+2)
    assert lat[2][0] == L + 2 * (P + 2)
