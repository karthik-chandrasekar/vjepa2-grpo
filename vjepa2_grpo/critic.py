"""Process critic with reconstruction-consistency awareness.

A 6-layer / 768-width transformer that takes a window of K=8 predicted
latent timesteps plus the language instruction and outputs:
  - p_hat in [0,1]: progress score
  - sigma:          ensemble disagreement (4 heads)

The reconstruction-anchor term L_anchor is NOT computed inside this module;
it lives in `faiss_anchor.py` and is combined with `p_hat` at GRPO time in
`grpo.py`. This separation keeps the critic differentiable end-to-end during
its supervised pretraining and lets the anchor be a non-differentiable
inference-time penalty.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


class ProgressCritic(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        latent_dim: int = 1408,
        lang_dim: int = 4096,
        n_ensemble: int = 4,
        window_K: int = 8,
        patches_per_frame: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_ensemble = n_ensemble
        self.window_K = window_K
        self.P = patches_per_frame

        self.lat_in = nn.Linear(latent_dim, d_model)
        self.lang_in = nn.Linear(lang_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable per-timestep + per-modality positional embedding
        # max positions = 1 (cls) + max_lang + K * P
        max_lang = 64
        self.pe = nn.Parameter(
            torch.zeros(1, 1 + max_lang + window_K * patches_per_frame, d_model)
        )
        nn.init.trunc_normal_(self.pe, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Ensemble heads (shared trunk, separate small MLPs)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
            for _ in range(n_ensemble)
        ])

    def _embed(self, z_win: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        # z_win: [B, K, P, D_lat]; lang: [B, L, D_lang]
        B, K, P, _ = z_win.shape
        z = self.lat_in(z_win)                         # [B,K,P,d]
        z = rearrange(z, "b k p d -> b (k p) d")
        l = self.lang_in(lang)                         # [B,L,d]
        cls = self.cls_token.expand(B, -1, -1)         # [B,1,d]
        seq = torch.cat([cls, l, z], dim=1)
        seq = seq + self.pe[:, : seq.size(1)]
        return seq

    def forward(
        self,
        z_window: torch.Tensor,
        lang: torch.Tensor,
        return_per_head: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (p_hat, sigma).

        p_hat:  [B] progress in [0,1] (ensemble mean of sigmoid logits)
        sigma:  [B] ensemble std over the logits
        """
        seq = self._embed(z_window, lang)
        h = self.trunk(seq)
        h = self.norm(h)
        cls = h[:, 0]                                    # [B, d]
        logits = torch.stack([head(cls).squeeze(-1) for head in self.heads], dim=-1)
        # logits: [B, n_ensemble]
        p_per_head = torch.sigmoid(logits)
        p_hat = p_per_head.mean(dim=-1)
        sigma = logits.std(dim=-1)
        if return_per_head:
            return p_hat, sigma, p_per_head
        return p_hat, sigma

    def forward_with_bootstrap_mask(
        self,
        z_window: torch.Tensor,
        lang: torch.Tensor,
        head_keep_mask: torch.Tensor,   # [B, n_ensemble] in {0,1}
    ) -> torch.Tensor:
        """For ensemble training: each head sees a different bootstrap subset
        of the batch. Returns per-head logits [B, n_ensemble]."""
        seq = self._embed(z_window, lang)
        h = self.trunk(seq)
        h = self.norm(h)
        cls = h[:, 0]
        logits = torch.stack([head(cls).squeeze(-1) for head in self.heads], dim=-1)
        return logits, head_keep_mask
