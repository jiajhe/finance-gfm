from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import FeatureBottleneck
from .fdg import FDG
from .gnn_head import GNNHead


class TemporalHistoryEncoder(nn.Module):
    def __init__(
        self,
        d_out: int,
        conv_channels: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = max(0, int(kernel_size) // 2)
        self.temporal = nn.Sequential(
            nn.Conv1d(1, int(conv_channels), kernel_size=int(kernel_size), padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(int(conv_channels), int(conv_channels), kernel_size=int(kernel_size), padding=padding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(int(conv_channels), d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        history: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history is None:
            if reference is None:
                raise ValueError("TemporalHistoryEncoder requires either `history` or `reference`.")
            out = reference.new_zeros(reference.shape)
            if mask is not None:
                out = out * mask.unsqueeze(-1).to(out.dtype)
            return out

        if history.dim() != 3:
            raise ValueError("`history` must have shape [B, N, L].")

        batch_size, num_nodes, seq_len = history.shape
        temporal_input = history.reshape(batch_size * num_nodes, 1, seq_len)
        hidden = self.temporal(temporal_input).squeeze(-1)
        out = self.dropout(self.proj(hidden)).reshape(batch_size, num_nodes, -1)
        out = self.norm(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class TemporalFDGRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int,
        d_hidden: int,
        tau: float = 1.0,
        dropout: float = 0.1,
        b_init: str = "identity_perturbed",
        bottleneck_dim: int = 64,
        conv_channels: int = 16,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.temporal_encoder = TemporalHistoryEncoder(
            d_out=d_in,
            conv_channels=conv_channels,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.fusion_norm = nn.LayerNorm(d_in * 2)
        self.fusion = nn.Sequential(
            nn.Linear(d_in * 2, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_in),
        )
        self.fdg = FDG(d_in=d_in, rank=rank, tau=tau, b_init=b_init)
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_snapshot = self.encoder(X, mask=mask)
        X_history = self.temporal_encoder(history=history, mask=mask, reference=X_snapshot)
        fused_input = torch.cat([X_snapshot, X_history], dim=-1)
        X_model = X_snapshot + self.dropout(self.fusion(self.fusion_norm(fused_input)))
        if mask is not None:
            X_model = X_model * mask.unsqueeze(-1).to(X_model.dtype)

        A, S, R = self.fdg(X_model, mask=mask)
        y_hat = self.head(A, X_model, mask=mask)

        if return_graph:
            return y_hat, {
                "A": A,
                "S": S,
                "R": R,
                "X_snapshot": X_snapshot,
                "X_history": X_history,
                "X_model": X_model,
            }
        return y_hat
