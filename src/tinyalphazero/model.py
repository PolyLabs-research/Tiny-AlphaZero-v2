"""ChessNet: transformer trunk + bilinear policy + WDL value + aux legality (spec §4)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from tinyalphazero.config import ModelConfig
from tinyalphazero.encoding import VOCAB_SIZE

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class ModelOutput:
    policy_ft: torch.Tensor       # (B, 64, 64) raw logits, unmasked
    promo: torch.Tensor           # (B, 64, 64, 4) logits
    value_wdl: torch.Tensor       # (B, 3) logits — order [loss, draw, win] from side to move
    legality: torch.Tensor | None  # (B, 64, 64) logits; None when cfg.aux_legality=False


# ---------------------------------------------------------------------------
# Standalone helpers (public contract for later chunks)
# ---------------------------------------------------------------------------


def masked_policy(
    policy_ft: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply legal mask and return softmax probabilities.

    policy_ft:  (B, 64, 64) raw logits
    legal_mask: (B, 64, 64) bool, True = legal
    returns:    (B, 4096) probabilities; 0 where illegal
    """
    B = policy_ft.size(0)
    logits = policy_ft.reshape(B, 4096)
    mask = legal_mask.reshape(B, 4096).bool()
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def value_scalar(value_wdl: torch.Tensor) -> torch.Tensor:
    """P(win) − P(loss) from WDL logits; output in (−1, 1).

    value_wdl: (B, 3) logits, order [loss, draw, win]
    returns:   (B,)
    """
    probs = torch.softmax(value_wdl, dim=-1)
    return probs[:, 2] - probs[:, 0]


# ---------------------------------------------------------------------------
# ChessNet
# ---------------------------------------------------------------------------


class ChessNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        dp = cfg.d_policy

        # ---- Token embeddings ----
        self.token_emb = nn.Embedding(VOCAB_SIZE, d)

        # ---- Factored rank+file positional encoding for squares 0-63 ----
        # Plus an 8-slot flag embedding for state tokens 64-71.
        # Precomputed into a (72, d_model) table each forward via index lookup
        # (one call per batch, not per token — avoids v1's per-call rebuild).
        self.rank_emb = nn.Embedding(8, d)
        self.file_emb = nn.Embedding(8, d)
        self.flag_emb = nn.Embedding(8, d)

        # Static index buffers for pos table construction
        self.register_buffer("_sq_ranks", torch.arange(64) // 8)  # (64,) int
        self.register_buffer("_sq_files", torch.arange(64) % 8)   # (64,) int
        self.register_buffer("_flag_ids", torch.arange(8))        # (8,)  int

        # ---- Transformer trunk (pre-norm, GELU, no causal mask) ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            norm=nn.LayerNorm(d),
        )

        # ---- Policy head: bilinear from→to (§4.3) ----
        # score(i,j) = (W_from h_i) · (W_to h_j) / sqrt(d_p)
        # Cost: 2 * d * d_p ≈ 65K params (vs 2.2M in v1)
        self.W_from = nn.Linear(d, dp, bias=False)
        self.W_to = nn.Linear(d, dp, bias=False)

        # ---- Promotion sub-head (§4.3) ----
        # Linear(2*d, 4) on [h_from; h_to], factored as two Linear(d, 4) summed:
        # promo(i,j) = promo_from(h_i) + promo_to(h_j)
        # Equivalent to the joint linear but avoids materialising (B,64,64,2d).
        self.promo_from = nn.Linear(d, 4, bias=True)
        self.promo_to = nn.Linear(d, 4, bias=False)

        # ---- Value head: attention pool + MLP (§4.4) ----
        # Single learned query attends over all 72 token features.
        self.value_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.value_query, std=0.02)
        self.value_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 3),
        )

        # ---- Optional legality auxiliary head ----
        # Same bilinear structure, separate projections (§4, Phase 1 only).
        if cfg.aux_legality:
            self.W_from_aux = nn.Linear(d, dp, bias=False)
            self.W_to_aux = nn.Linear(d, dp, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> ModelOutput:
        """
        tokens: (B, 72) LongTensor
        returns: ModelOutput with all heads computed
        """
        x = self._trunk(tokens)          # (B, 72, d)
        h_sq = x[:, :64, :]             # (B, 64, d) — square features only

        policy_ft = self._policy(h_sq)  # (B, 64, 64)
        promo = self._promo_dense(h_sq) # (B, 64, 64, 4)
        value_wdl = self._value(x)      # (B, 3)

        legality: torch.Tensor | None = None
        if hasattr(self, "W_from_aux"):
            # Attached (not detached) on purpose: per spec §4.4 the aux head exists
            # to push a dense mobility signal into the trunk. Detaching it here made
            # it a linear probe and starved the trunk of its main gradient.
            legality = self._legality(h_sq)  # (B, 64, 64)

        return ModelOutput(
            policy_ft=policy_ft,
            promo=promo,
            value_wdl=value_wdl,
            legality=legality,
        )

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk and return (h_all, h_sq) without computing heads.

        Useful for training code that needs to call promo_logits on specific pairs.
        h_all: (B, 72, d_model), h_sq: (B, 64, d_model)
        """
        x = self._trunk(tokens)
        return x, x[:, :64, :]

    def promo_logits(
        self,
        h_sq: torch.Tensor,
        from_sq: torch.Tensor,
        to_sq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute promotion logits for specific (from, to) pairs.

        Memory-efficient path for training: only computes the requested pairs.
        h_sq:    (B, 64, d_model)
        from_sq: (B,) long
        to_sq:   (B,) long
        returns: (B, 4)
        """
        idx = torch.arange(h_sq.size(0), device=h_sq.device)
        h_from = h_sq[idx, from_sq]  # (B, d)
        h_to = h_sq[idx, to_sq]      # (B, d)
        return self.promo_from(h_from) + self.promo_to(h_to)

    def param_count(self) -> dict[str, int]:
        """Return total and policy-head parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        policy = sum(
            p.numel()
            for p in [
                *self.W_from.parameters(),
                *self.W_to.parameters(),
                *self.promo_from.parameters(),
                *self.promo_to.parameters(),
            ]
        )
        return {"total": total, "policy_head": policy}

    # ------------------------------------------------------------------
    # Private sub-computations
    # ------------------------------------------------------------------

    def _pos_table(self) -> torch.Tensor:
        """Build (72, d_model) positional table from current embedding parameters."""
        sq_pos = self.rank_emb(self._sq_ranks) + self.file_emb(self._sq_files)  # (64, d)
        flag_pos = self.flag_emb(self._flag_ids)                                  # (8,  d)
        return torch.cat([sq_pos, flag_pos], dim=0)                               # (72, d)

    def _trunk(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embed + positional table + transformer. Returns (B, 72, d)."""
        pos = self._pos_table()                          # (72, d)
        x = self.token_emb(tokens) + pos.unsqueeze(0)   # (B, 72, d)
        return self.transformer(x)                        # (B, 72, d)

    def _policy(self, h_sq: torch.Tensor) -> torch.Tensor:
        """Bilinear from→to policy logits. Returns (B, 64, 64)."""
        u = self.W_from(h_sq)  # (B, 64, d_p)
        v = self.W_to(h_sq)    # (B, 64, d_p)
        return torch.bmm(u, v.transpose(1, 2)) / math.sqrt(self.cfg.d_policy)

    def _promo_dense(self, h_sq: torch.Tensor) -> torch.Tensor:
        """Dense promotion logits over all (from, to) pairs. Returns (B, 64, 64, 4).

        Uses factored projection to avoid materialising (B, 64, 64, 2*d_model).
        Broadcasting: (B,64,1,4) + (B,1,64,4) → (B,64,64,4).
        """
        return self.promo_from(h_sq).unsqueeze(2) + self.promo_to(h_sq).unsqueeze(1)

    def _value(self, x: torch.Tensor) -> torch.Tensor:
        """Attention-pool all 72 tokens via a learned query, then MLP. Returns (B, 3)."""
        scores = (x @ self.value_query) / math.sqrt(self.cfg.d_model)  # (B, 72)
        weights = torch.softmax(scores, dim=-1)                          # (B, 72)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)                 # (B, d)
        return self.value_mlp(pooled)                                    # (B, 3)

    def _legality(self, h_sq: torch.Tensor) -> torch.Tensor:
        """Auxiliary legality head (Phase 1 only). Returns (B, 64, 64)."""
        u = self.W_from_aux(h_sq)
        v = self.W_to_aux(h_sq)
        return torch.bmm(u, v.transpose(1, 2)) / math.sqrt(self.cfg.d_policy)
