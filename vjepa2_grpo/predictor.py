"""Block-causal action-conditioned latent predictor.

Port of V-JEPA-2-AC's predictor architecture, extended with language and
proprioception conditioning. Operates on pooled V-JEPA-2 patch embeddings.

Architecture (V-JEPA-2-AC paper Sec 4.1, Appendix B):
  - 24 transformer layers, hidden = 1024, 16 heads
  - block-causal attention over (lang, [z_t, a_t, p_t]) sequence
  - 3D-RoPE for latent patches in the original; we use additive sinusoidal PE
    as a stand-in (swap for 3D-RoPE for exact parity — note in the paper)
  - Predicts the next-step latent z_{t+1} given history up to step t

THREE EXECUTION PATHS (all share the same weights):

  forward()              — teacher-forced, full block-causal attention over the
                           whole window. Training + the parity reference.
                           UNTOUCHED by the KV-cache work below.

  rollout()              — sliding-window autoregressive rollout. Each step
                           re-runs forward() over a fixed T_hist window.
                           Kept as a correctness reference.

  rollout_cached()       — growing-context-within-chunk autoregressive rollout
                           with a per-call KV-cache. Each step does two cheap
                           partial passes (predict + cache-commit) instead of a
                           full-window forward. This is the GRPO fast path.
                           NB: growing-context != sliding-window, so this does
                           NOT match rollout() — it matches _rollout_growing_naive(),
                           which is the same growing-context semantics without
                           the cache. The parity test asserts that equality.

IMPLEMENTATION NOTE (fused attention):
  The custom block calls F.scaled_dot_product_attention directly with a
  *boolean* block-causal mask (fused memory-efficient SDPA kernel in bf16).
  The cached path's incremental steps are mask-free -> flash kernel.
"""
from __future__ import annotations
import math
from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List

from .masks import step_token_indices


# ---------------------------------------------------------------------------
# Boolean block-causal mask (SDPA convention: True = attend, False = masked)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _block_causal_bool_mask(
    L: int,
    T: int,
    tokens_per_step: int,
    device: str,
) -> torch.Tensor:
    """Boolean block-causal mask for SDPA.

    Sequence layout: [LANG (L tokens)] [STEP_0] [STEP_1] ... [STEP_{T-1}]
    where each STEP block has `tokens_per_step` tokens (P latents + action + proprio).

    Rule (identical to masks.block_causal_mask, just boolean):
      - lang tokens are visible to everyone (and bidirectional within lang)
      - within a step: fully bidirectional
      - across steps: step t sees steps 0..t

    Returns: [S, S] bool tensor, S = L + T * tokens_per_step.
             True at [i, j] means query i may attend to key j.
    """
    S = L + T * tokens_per_step
    m = torch.zeros(S, S, dtype=torch.bool, device=device)
    # Everyone (lang rows included) attends to all lang columns.
    m[:, :L] = True
    for t in range(T):
        ts = L + t * tokens_per_step
        te = ts + tokens_per_step
        m[ts:te, ts:te] = True          # intra-step bidirectional
        if t > 0:
            m[ts:te, L:ts] = True       # attend to all previous steps
    return m


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Additive sinusoidal PE. Cheap stand-in for 3D-RoPE during development.
    Replace with proper 3D-RoPE (spatial x temporal) for the final paper run.

    `offset` lets the cached rollout path embed a single step at its correct
    absolute sequence position without materializing the whole sequence.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        end = offset + x.size(1)
        assert end <= self.max_len, (
            f"PE overflow: need position {end} but max_len={self.max_len}. "
            f"Increase max_horizon / max_lang_tokens."
        )
        return x + self.pe[:, offset:end].to(x.dtype)


# ---------------------------------------------------------------------------
# Fused attention block
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention via F.scaled_dot_product_attention.

    Separate q/k/v projections so the cached path can concatenate past K/V.

    Two entry points:
      forward()        — the training / full-forward path. UNCHANGED.
      forward_cached() — the KV-cache path used only by rollout_cached().
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _shape(self, t: torch.Tensor) -> torch.Tensor:
        # [B, S, d] -> [B, n_heads, S, head_dim]
        B, S, _ = t.shape
        return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Training / full-forward path. UNCHANGED from the fused rewrite."""
        B, S, d = x.shape
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        # attn_mask: [S, S] bool, broadcasts over [B, H, S, S]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, S, d)
        return self.out_proj(out)

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        append_to_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """KV-cache path. Inference only (no dropout).

        Args:
            x:               [B, S_new, d] new tokens to process
            attn_mask:       [S_new, S_total] bool, or None (None -> flash kernel)
            past_kv:         (k, v) each [B, H, S_past, hd], or None
            append_to_cache: if True, return the full (k, v) as the new cache entry

        Returns:
            (out, new_kv) where out is [B, S_new, d] and new_kv is
            (k_full, v_full) if append_to_cache else None.
        """
        B, S_new, d = x.shape
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).reshape(B, S_new, d)
        out = self.out_proj(out)
        new_kv = (k, v) if append_to_cache else None
        return out, new_kv


class PredictorBlock(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Training / full-forward path. UNCHANGED."""
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        append_to_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """KV-cache path."""
        attn_out, new_kv = self.attn.forward_cached(
            self.norm1(x), attn_mask, past_kv, append_to_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class BlockCausalACPredictor(nn.Module):
    """Action-conditioned latent predictor with block-causal attention.

    Inputs:
        z_hist:   [B, T_hist, P, D_lat]  history of latent patches
        actions:  [B, T_hist + horizon, D_act]
        proprio:  [B, T_hist + horizon, D_prop]
        lang:     [B, L, D_lang]   (e.g. SigLIP or Llama text features)

    Output:
        z_pred:   [B, horizon, P, D_lat]  predicted latents for steps
                  T_hist .. T_hist + horizon - 1
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        latent_dim: int = 1408,
        action_dim: int = 7,
        proprio_dim: int = 8,
        lang_dim: int = 4096,
        patches_per_frame: int = 64,
        max_horizon: int = 32,
        max_lang_tokens: int = 64,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        use_grad_ckpt: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.P = patches_per_frame
        self.max_horizon = max_horizon
        self.max_lang_tokens = max_lang_tokens
        self.use_grad_ckpt = use_grad_ckpt

        # Modality embeddings
        self.lat_in = nn.Linear(latent_dim, d_model)
        self.act_in = nn.Linear(action_dim, d_model)
        self.prop_in = nn.Linear(proprio_dim, d_model)
        self.lang_in = nn.Linear(lang_dim, d_model)

        # Learned modality type embeddings (0=lang 1=lat 2=act 3=prop)
        self.mod_emb = nn.Embedding(4, d_model)

        # Positional encoding. max_len must cover lang + (T_hist+horizon) steps;
        # be generous so the cached rollout (which grows context) never overflows.
        self.pe = SinusoidalPositionalEncoding(
            d_model,
            max_len=2 * max_lang_tokens + 2 * max_horizon * (patches_per_frame + 2),
        )

        # Transformer trunk (pre-norm, fused SDPA blocks)
        self.blocks = nn.ModuleList([
            PredictorBlock(d_model, n_heads, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output head
        self.out_norm = nn.LayerNorm(d_model)
        self.lat_out = nn.Linear(d_model, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ----- embedding helpers -------------------------------------------------

    def _assemble_sequence(
        self,
        z: torch.Tensor,        # [B, T, P, D_lat]
        a: torch.Tensor,        # [B, T, D_act]
        p: torch.Tensor,        # [B, T, D_prop]
        lang: torch.Tensor,     # [B, L, D_lang]
    ) -> Tuple[torch.Tensor, int, int, int]:
        """Full-sequence embedding used by forward(). UNCHANGED."""
        B, T, P, _ = z.shape
        L = lang.size(1)

        z_tok = self.lat_in(z)                              # [B,T,P,d]
        a_tok = self.act_in(a).unsqueeze(2)                 # [B,T,1,d]
        p_tok = self.prop_in(p).unsqueeze(2)                # [B,T,1,d]
        step_tok = torch.cat([z_tok, a_tok, p_tok], dim=2)  # [B,T,P+2,d]
        step_tok = rearrange(step_tok, "b t k d -> b (t k) d")

        mod_ids_step = torch.tensor(
            [1] * P + [2, 3], device=z.device, dtype=torch.long
        )
        mod_ids_step = mod_ids_step.repeat(T)
        step_tok = step_tok + self.mod_emb(mod_ids_step).unsqueeze(0)

        lang_tok = self.lang_in(lang)
        lang_tok = lang_tok + self.mod_emb(
            torch.zeros(L, dtype=torch.long, device=z.device)
        ).unsqueeze(0)

        seq = torch.cat([lang_tok, step_tok], dim=1)
        seq = self.pe(seq)
        return seq, L, T, P

    def _embed_lang_tokens(self, lang: torch.Tensor) -> torch.Tensor:
        """Embed language tokens at absolute positions 0..L-1.

        Constructed to be byte-for-byte equivalent to the lang portion of
        _assemble_sequence (verified by the rollout parity test)."""
        B, L, _ = lang.shape
        lang_tok = self.lang_in(lang)
        lang_tok = lang_tok + self.mod_emb(
            torch.zeros(L, dtype=torch.long, device=lang.device)
        ).unsqueeze(0)
        return self.pe(lang_tok, offset=0)

    def _embed_step_tokens(
        self,
        z: torch.Tensor,        # [B, T, P, D_lat]
        a: torch.Tensor,        # [B, T, D_act]
        p: torch.Tensor,        # [B, T, D_prop]
        global_start: int,      # index of the first step (0 == first history step)
        L: int,                 # number of language tokens (for PE offset)
    ) -> torch.Tensor:
        """Embed T consecutive step blocks at their correct absolute positions.

        Step `global_start + t` occupies sequence positions
        [L + (global_start+t)*(P+2), L + (global_start+t+1)*(P+2)).

        Constructed to be byte-for-byte equivalent to the step portion of
        _assemble_sequence when global_start=0 (verified by the parity test)."""
        B, T, P, _ = z.shape
        z_tok = self.lat_in(z)
        a_tok = self.act_in(a).unsqueeze(2)
        p_tok = self.prop_in(p).unsqueeze(2)
        step_tok = torch.cat([z_tok, a_tok, p_tok], dim=2)  # [B,T,P+2,d]
        step_tok = rearrange(step_tok, "b t k d -> b (t k) d")

        mod_ids = torch.tensor(
            [1] * P + [2, 3], device=z.device, dtype=torch.long
        ).repeat(T)
        step_tok = step_tok + self.mod_emb(mod_ids).unsqueeze(0)

        pe_start = L + global_start * (P + 2)
        return self.pe(step_tok, offset=pe_start)

    # ----- forward (training / reference) ------------------------------------

    def forward(
        self,
        z_hist: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor,
        lang: torch.Tensor,
        horizon: int = 1,
    ) -> torch.Tensor:
        """One-shot prediction over `horizon` future steps.

        Teacher-forcing training: pass full ground-truth z; the last `horizon`
        output slots are the predictions (see losses.predictor_loss).

        Rollout: pass z with future slots zero-filled; the predictor's output
        at those slots is the prediction. Actions/proprio are required for all
        T_hist + horizon steps (the predictor is action-conditioned).
        """
        B, T_hist, P, _ = z_hist.shape
        T_total = T_hist + horizon
        assert actions.shape[1] == T_total, (
            f"actions must have length T_hist+horizon={T_total}, got {actions.shape[1]}"
        )
        assert proprio.shape[1] == T_total

        if horizon > 0:
            future_z = torch.zeros(
                B, horizon, P, self.latent_dim,
                device=z_hist.device, dtype=z_hist.dtype,
            )
            z = torch.cat([z_hist, future_z], dim=1)
        else:
            z = z_hist

        seq, L, T, P = self._assemble_sequence(z, actions, proprio, lang)
        mask = _block_causal_bool_mask(L, T, P + 2, str(seq.device))

        h = seq
        for blk in self.blocks:
            if self.use_grad_ckpt and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, mask, use_reentrant=False,
                )
            else:
                h = blk(h, mask)
        h = self.norm(h)

        lat_idx, _, _ = step_token_indices(L, T, P)
        future_idx = []
        for t in range(T_hist, T_total):
            future_idx.extend(lat_idx[t])
        future_idx = torch.tensor(future_idx, device=h.device)
        h_future = h.index_select(1, future_idx)              # [B, horizon*P, d]
        h_future = self.out_norm(h_future)
        z_pred = self.lat_out(h_future)                       # [B, horizon*P, D_lat]
        z_pred = rearrange(z_pred, "b (h p) d -> b h p d", h=horizon, p=P)
        return z_pred

    # ----- rollout: sliding-window reference ---------------------------------

    @torch.inference_mode()
    def rollout(
        self,
        z_hist: torch.Tensor,
        action_chunks: torch.Tensor,
        proprio_chunks: torch.Tensor,
        lang: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Sliding-window autoregressive rollout. Correctness reference.

        Each step re-runs forward() over a fixed-length T_hist window. This is
        NOT the GRPO fast path — use rollout_cached(). Kept because it is the
        simplest obviously-correct implementation.
        """
        B, T_hist, P, D = z_hist.shape
        z = z_hist
        a_hist = torch.zeros(B, T_hist, action_chunks.size(-1),
                             device=z.device, dtype=action_chunks.dtype)
        p_hist = torch.zeros(B, T_hist, proprio_chunks.size(-1),
                             device=z.device, dtype=proprio_chunks.dtype)
        preds = []
        for h in range(horizon):
            a_full = torch.cat([a_hist, action_chunks[:, h : h + 1]], dim=1)
            p_full = torch.cat([p_hist, proprio_chunks[:, h : h + 1]], dim=1)
            z_next = self.forward(z, a_full, p_full, lang, horizon=1)  # [B,1,P,D]
            preds.append(z_next)
            z = torch.cat([z[:, 1:], z_next], dim=1)
            a_hist = torch.cat([a_hist[:, 1:], action_chunks[:, h : h + 1]], dim=1)
            p_hist = torch.cat([p_hist[:, 1:], proprio_chunks[:, h : h + 1]], dim=1)
        return torch.cat(preds, dim=1)

    # ----- rollout: growing-context naive (parity reference for cached) ------

    @torch.inference_mode()
    def _rollout_growing_naive(
        self,
        z_hist: torch.Tensor,
        action_chunks: torch.Tensor,
        proprio_chunks: torch.Tensor,
        lang: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Growing-context autoregressive rollout, no cache.

        Identical semantics to rollout_cached() but implemented by re-running
        forward() over the *growing* context each step. This is the parity
        reference: rollout_cached() must match this within numerical tolerance.
        """
        B, T_hist, P, D = z_hist.shape
        z = z_hist
        a_hist = torch.zeros(B, T_hist, action_chunks.size(-1),
                             device=z.device, dtype=action_chunks.dtype)
        p_hist = torch.zeros(B, T_hist, proprio_chunks.size(-1),
                             device=z.device, dtype=proprio_chunks.dtype)
        preds = []
        for h in range(horizon):
            a_full = torch.cat([a_hist, action_chunks[:, h : h + 1]], dim=1)
            p_full = torch.cat([p_hist, proprio_chunks[:, h : h + 1]], dim=1)
            z_next = self.forward(z, a_full, p_full, lang, horizon=1)  # [B,1,P,D]
            preds.append(z_next)
            # GROW the context (do not slide).
            z = torch.cat([z, z_next], dim=1)
            a_hist = torch.cat([a_hist, action_chunks[:, h : h + 1]], dim=1)
            p_hist = torch.cat([p_hist, proprio_chunks[:, h : h + 1]], dim=1)
        return torch.cat(preds, dim=1)

    # ----- rollout: KV-cached fast path --------------------------------------

    @torch.inference_mode()
    def rollout_cached(
        self,
        z_hist: torch.Tensor,
        action_chunks: torch.Tensor,
        proprio_chunks: torch.Tensor,
        lang: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Growing-context autoregressive rollout with a per-call KV-cache.

        This is the GRPO fast path. Semantics match _rollout_growing_naive()
        (the parity reference), NOT rollout() (sliding-window).

        Per-call structure:
          1. Prefill: embed [lang + T_hist history steps], run all layers with
             the block-causal mask, populate the KV-cache.
          2. For each of `horizon` steps:
             a. Predict pass — embed the step's [P zero-latents, action, proprio]
                placeholder, run all layers attending to the cache (no mask ->
                flash kernel). The latent outputs are the prediction.
             b. Commit pass — embed the step's [P predicted-latents, action,
                proprio], run all layers, append the resulting K/V to the cache.

        The cache lives only for this call (one action_chunk).
        """
        B, T_hist, P, D = z_hist.shape
        L = lang.size(1)
        device = z_hist.device
        n_layers = len(self.blocks)

        # --- 1. Prefill: lang + history steps ---
        a_hist = torch.zeros(B, T_hist, action_chunks.size(-1),
                             device=device, dtype=action_chunks.dtype)
        p_hist = torch.zeros(B, T_hist, proprio_chunks.size(-1),
                             device=device, dtype=proprio_chunks.dtype)
        lang_tok = self._embed_lang_tokens(lang)                          # [B, L, d]
        hist_tok = self._embed_step_tokens(z_hist, a_hist, p_hist,
                                           global_start=0, L=L)           # [B, T_hist*(P+2), d]
        seq = torch.cat([lang_tok, hist_tok], dim=1)
        prefill_mask = _block_causal_bool_mask(L, T_hist, P + 2, str(device))

        cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        h = seq
        for blk in self.blocks:
            h, kv = blk.forward_cached(h, prefill_mask, past_kv=None, append_to_cache=True)
            cache.append(kv)
        # prefill output `h` is discarded — we only needed the cache

        # --- 2. Autoregressive steps ---
        preds = []
        for i in range(horizon):
            global_step = T_hist + i
            a_i = action_chunks[:, i : i + 1]          # [B, 1, D_act]
            p_i = proprio_chunks[:, i : i + 1]         # [B, 1, D_prop]

            # 2a. Predict pass: zero-latent placeholder for this step.
            #     Attends to cache (lang + steps 0..global_step-1) + own tokens
            #     = everything <= this step -> no mask needed -> flash kernel.
            z_ph = torch.zeros(B, 1, P, self.latent_dim, device=device, dtype=z_hist.dtype)
            step_tok = self._embed_step_tokens(z_ph, a_i, p_i,
                                               global_start=global_step, L=L)  # [B, P+2, d]
            hA = step_tok
            for layer_idx, blk in enumerate(self.blocks):
                hA, _ = blk.forward_cached(hA, attn_mask=None,
                                           past_kv=cache[layer_idx],
                                           append_to_cache=False)
            hA = self.norm(hA)
            # latent tokens are the first P of the (P+2)-token step block
            h_lat = self.out_norm(hA[:, :P])                     # [B, P, d]
            z_pred = self.lat_out(h_lat).unsqueeze(1)            # [B, 1, P, D_lat]
            preds.append(z_pred)

            # 2b. Commit pass: re-embed the step with the *predicted* latents,
            #     run all layers, append the resulting K/V to the cache so the
            #     next step sees this step's real representation.
            step_tok_real = self._embed_step_tokens(z_pred, a_i, p_i,
                                                    global_start=global_step, L=L)
            hB = step_tok_real
            for layer_idx, blk in enumerate(self.blocks):
                hB, kv = blk.forward_cached(hB, attn_mask=None,
                                            past_kv=cache[layer_idx],
                                            append_to_cache=True)
                cache[layer_idx] = kv
            # commit-pass output `hB` is discarded

        return torch.cat(preds, dim=1)                           # [B, horizon, P, D_lat]
