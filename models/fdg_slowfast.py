from __future__ import annotations

import math

import torch
import torch.nn as nn

from .blocks import FeatureBottleneck
from .fdg import FDG, row_normalize_adjacency
from .fdg_temporal import TemporalHistoryEncoder
from .gnn_head import GNNHead


def _logit_init(prob: float) -> torch.Tensor:
    clipped = min(max(float(prob), 1e-4), 1.0 - 1e-4)
    return torch.tensor(math.log(clipped / (1.0 - clipped)), dtype=torch.float32)


def _masked_matrix_mse(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    diff = (A - B).square()
    if mask is None:
        return diff.mean()
    valid = mask.to(diff.dtype)
    pair_mask = valid.unsqueeze(-1) * valid.unsqueeze(-2)
    denom = pair_mask.sum().clamp_min(1.0)
    return (diff * pair_mask).sum() / denom


def _masked_tensor_mse(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    diff = (A - B).square()
    if mask is None:
        return diff.mean()
    valid = mask.to(diff.dtype).unsqueeze(-1)
    denom = valid.sum().clamp_min(1.0) * diff.shape[-1]
    return (diff * valid).sum() / denom


class SlowFastTemporalFDGRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int,
        d_hidden: int,
        tau: float = 1.0,
        dropout: float = 0.1,
        b_init: str = "identity_perturbed",
        bottleneck_dim: int | list[int] = 64,
        conv_channels: int = 16,
        temporal_kernel_size: int = 3,
        graph_slow_init: float = 0.8,
        value_fast_init: float = 0.8,
        fast_graph_mix_init: float = 0.1,
        graph_smooth_weight: float = 0.0,
        assignment_smooth_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.temporal_encoder = TemporalHistoryEncoder(
            d_out=d_in,
            conv_channels=conv_channels,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.graph_fusion_norm = nn.LayerNorm(d_in * 2)
        self.graph_fusion = nn.Sequential(
            nn.Linear(d_in * 2, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_in),
        )
        self.value_fusion_norm = nn.LayerNorm(d_in * 2)
        self.value_fusion = nn.Sequential(
            nn.Linear(d_in * 2, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_in),
        )
        self.graph_slow_logit = nn.Parameter(_logit_init(graph_slow_init))
        self.value_fast_logit = nn.Parameter(_logit_init(value_fast_init))
        self.fast_graph_mix_logit = nn.Parameter(_logit_init(fast_graph_mix_init))
        self.dropout = nn.Dropout(dropout)
        self.fdg = FDG(d_in=d_in, rank=rank, tau=tau, b_init=b_init)
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)
        self.graph_smooth_weight = float(graph_smooth_weight)
        self.assignment_smooth_weight = float(assignment_smooth_weight)
        self._last_reg_loss: torch.Tensor | None = None
        self._last_reg_terms: dict[str, float] = {}

    def _build_states(self, X_fast: torch.Tensor, X_slow: torch.Tensor, mask: torch.Tensor | None):
        fusion_input = torch.cat([X_fast, X_slow], dim=-1)
        graph_delta = self.dropout(self.graph_fusion(self.graph_fusion_norm(fusion_input)))
        value_delta = self.dropout(self.value_fusion(self.value_fusion_norm(fusion_input)))

        graph_slow = torch.sigmoid(self.graph_slow_logit)
        value_fast = torch.sigmoid(self.value_fast_logit)

        graph_state = graph_slow * X_slow + (1.0 - graph_slow) * X_fast + graph_delta
        value_state = value_fast * X_fast + (1.0 - value_fast) * X_slow + value_delta

        if mask is not None:
            valid = mask.unsqueeze(-1).to(graph_state.dtype)
            graph_state = graph_state * valid
            value_state = value_state * valid
        return graph_state, value_state

    def _build_regularizer(
        self,
        A_slow: torch.Tensor,
        A_fast: torch.Tensor,
        S_slow: torch.Tensor,
        S_fast: torch.Tensor,
        R_slow: torch.Tensor,
        R_fast: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        reference = A_slow.sum() * 0.0
        total = reference
        terms: dict[str, float] = {}

        if self.graph_smooth_weight > 0.0:
            graph_term = _masked_matrix_mse(A_slow, A_fast, mask=mask)
            total = total + self.graph_smooth_weight * graph_term
            terms["graph_smooth"] = float(graph_term.detach().cpu())

        if self.assignment_smooth_weight > 0.0:
            assignment_term = 0.5 * (
                _masked_tensor_mse(S_slow, S_fast, mask=mask) + _masked_tensor_mse(R_slow, R_fast, mask=mask)
            )
            total = total + self.assignment_smooth_weight * assignment_term
            terms["assignment_smooth"] = float(assignment_term.detach().cpu())

        self._last_reg_terms = terms
        return total

    def regularization_loss(self) -> torch.Tensor:
        if self._last_reg_loss is None:
            parameter = next(self.parameters())
            return parameter.sum() * 0.0
        return self._last_reg_loss

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_fast = self.encoder(X, mask=mask)
        X_slow = self.temporal_encoder(history=history, mask=mask, reference=X_fast)
        graph_state, value_state = self._build_states(X_fast=X_fast, X_slow=X_slow, mask=mask)

        A_slow, S_slow, R_slow = self.fdg(graph_state, mask=mask)
        A_fast, S_fast, R_fast = self.fdg(value_state, mask=mask)
        fast_graph_mix = torch.sigmoid(self.fast_graph_mix_logit)
        A = row_normalize_adjacency((1.0 - fast_graph_mix) * A_slow + fast_graph_mix * A_fast, mask=mask)
        self._last_reg_loss = self._build_regularizer(
            A_slow=A_slow,
            A_fast=A_fast,
            S_slow=S_slow,
            S_fast=S_fast,
            R_slow=R_slow,
            R_fast=R_fast,
            mask=mask,
        )

        y_hat = self.head(A, value_state, mask=mask)
        if return_graph:
            return y_hat, {
                "A": A,
                "A_slow": A_slow,
                "A_fast": A_fast,
                "X_fast": X_fast,
                "X_slow": X_slow,
                "graph_state": graph_state,
                "value_state": value_state,
                "reg_terms": dict(self._last_reg_terms),
            }
        return y_hat
