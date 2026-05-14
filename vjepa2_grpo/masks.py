"""Block-causal attention masks for the action-conditioned predictor.

Sequence layout (following V-JEPA-2-AC Appendix B):

    [LANG_tokens] [STEP_1_tokens] [STEP_2_tokens] ... [STEP_T_tokens]

where each STEP_t_tokens block contains:

    [z_t (P latent patches)] [a_t (1 action token)] [p_t (1 proprio token)]

The mask is block-causal:
  - LANG tokens: visible to everything (bidirectional within lang block; readable
    by all step tokens).
  - Within STEP_t: all tokens can attend to each other (intra-step bidirectional).
  - Across steps: STEP_t tokens can attend to STEP_s for all s <= t.

This is implemented as an additive float mask of shape [S, S] with values 0
(visible) or -inf (masked).
"""
from __future__ import annotations
import torch
from functools import lru_cache


@lru_cache(maxsize=32)
def block_causal_mask(
    L: int,            # number of language tokens
    T: int,            # number of timesteps (history + horizon)
    P: int,            # latent patches per timestep
    tokens_per_step: int = None,   # P + 1 (action) + 1 (proprio) by default
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct the block-causal attention mask.

    Returns:
        [S, S] additive mask where S = L + T * tokens_per_step.
        Value 0 means visible, -inf means masked.
    """
    if tokens_per_step is None:
        tokens_per_step = P + 2  # P latents + 1 action + 1 proprio
    S = L + T * tokens_per_step
    mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)

    # Language block: bidirectional within, visible to everyone
    mask[:L, :L] = 0.0
    mask[L:, :L] = 0.0  # all step tokens can read language

    # Step blocks: causal across steps, bidirectional within step
    for t in range(T):
        t_start = L + t * tokens_per_step
        t_end = t_start + tokens_per_step
        # within step t: bidirectional
        mask[t_start:t_end, t_start:t_end] = 0.0
        # to all previous steps (s < t): visible
        if t > 0:
            mask[t_start:t_end, L:t_start] = 0.0

    return mask


def step_token_indices(L: int, T: int, P: int, tokens_per_step: int = None):
    """Return (latent_idx, action_idx, proprio_idx) lists for each timestep.

    Useful for extracting predictions or applying per-modality losses.
    """
    if tokens_per_step is None:
        tokens_per_step = P + 2
    lat_idx, act_idx, prop_idx = [], [], []
    for t in range(T):
        base = L + t * tokens_per_step
        lat_idx.append(list(range(base, base + P)))
        act_idx.append(base + P)
        prop_idx.append(base + P + 1)
    return lat_idx, act_idx, prop_idx
