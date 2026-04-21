from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..blocks import FeatureBottleneck, GraphResidualBlock, ResidualMLPBlock
from ..fdg import FDG, row_normalize_adjacency
from ..temporal_graph import EntropyStockGraph, RollingCorrelationGraph, topk_sparsify_adjacency


def _logit(prob: float) -> torch.Tensor:
    clipped = min(max(float(prob), 1e-4), 1.0 - 1e-4)
    return torch.tensor(math.log(clipped / (1.0 - clipped)), dtype=torch.float32)


class MLPBaseline(nn.Module):
    def __init__(self, d_in, d_hidden, dropout, bottleneck_dim=64, residual_layers=2):
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.input_proj = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(dim=d_hidden, dropout=dropout) for _ in range(int(residual_layers))
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, X, mask, history=None):
        X = self.encoder(X, mask=mask)
        hidden = self.input_proj(X)
        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, mask=mask)
        y_hat = self.head(hidden).squeeze(-1)
        return torch.where(mask, y_hat, torch.zeros_like(y_hat))


class GraphResidualMLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        dropout,
        rank=16,
        tau=1.0,
        b_init="identity_perturbed",
        bottleneck_dim=64,
        residual_layers=2,
        graph_layers=1,
        graph_kind="fdg_roll",
        roll_topk=20,
        entropy_topk=20,
        entropy_bins=8,
        final_topk=20,
        graph_mix_init=0.7,
    ):
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.input_proj = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(dim=d_hidden, dropout=dropout) for _ in range(int(residual_layers))
        )
        self.graph_kind = str(graph_kind)
        self.final_topk = int(final_topk)
        self.fdg = FDG(d_in=d_in, rank=int(rank), tau=float(tau), b_init=str(b_init))
        self.roll_graph = RollingCorrelationGraph(topk=int(roll_topk))
        self.entropy_graph = EntropyStockGraph(topk=int(entropy_topk), num_bins=int(entropy_bins))
        self.graph_mix_logit = nn.Parameter(_logit(graph_mix_init))
        self.graph_blocks = nn.ModuleList(
            GraphResidualBlock(dim=d_hidden, dropout=dropout) for _ in range(int(graph_layers))
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def _build_graph(self, X_model, history, mask):
        A_fdg, S, R = self.fdg(X_model, mask=mask)
        mix = torch.sigmoid(self.graph_mix_logit)

        if self.graph_kind == "fdg":
            A = A_fdg
            A_aux = A_fdg
        elif self.graph_kind == "roll":
            A_aux = self.roll_graph(history=history, mask=mask)
            A = A_aux
        elif self.graph_kind == "entropy":
            A_aux = self.entropy_graph(history=history, mask=mask)
            A = A_aux
        elif self.graph_kind == "fdg_entropy":
            A_aux = self.entropy_graph(history=history, mask=mask)
            A = row_normalize_adjacency(mix * A_fdg + (1.0 - mix) * A_aux, mask=mask)
        else:
            A_aux = self.roll_graph(history=history, mask=mask)
            A = row_normalize_adjacency(mix * A_fdg + (1.0 - mix) * A_aux, mask=mask)

        if self.final_topk > 0:
            A = topk_sparsify_adjacency(A, topk=self.final_topk, mask=mask)
        return A, A_fdg, A_aux, S, R, mix

    def forward(self, X, mask, history=None, return_graph: bool = False):
        X_model = self.encoder(X, mask=mask)
        hidden = self.input_proj(X_model)
        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, mask=mask)

        A, A_fdg, A_aux, S, R, mix = self._build_graph(X_model=X_model, history=history, mask=mask)
        for block in self.graph_blocks:
            hidden = block(hidden, A=A, mask=mask)

        y_hat = self.head(hidden).squeeze(-1)
        y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        if return_graph:
            return y_hat, {
                "A": A,
                "A_fdg": A_fdg,
                "A_aux": A_aux,
                "S": S,
                "R": R,
                "mix": mix,
            }
        return y_hat


class TemporalGraphResidualMLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        dropout,
        rank=16,
        tau=1.0,
        b_init="identity_perturbed",
        bottleneck_dim=64,
        residual_layers=2,
        graph_layers=1,
        graph_kind="fdg_roll",
        roll_topk=20,
        entropy_topk=20,
        entropy_bins=8,
        final_topk=20,
        graph_mix_init=0.7,
        history_window=20,
        temporal_layers=1,
        temporal_heads=4,
    ):
        super().__init__()
        if int(d_hidden) % max(1, int(temporal_heads)) != 0:
            raise ValueError("d_hidden must be divisible by temporal_heads.")

        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.feature_proj = nn.Linear(d_in, d_hidden)
        self.temporal_proj = nn.Linear(1, d_hidden)
        self.temporal_pos = nn.Parameter(torch.zeros(1, int(history_window), d_hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=int(temporal_heads),
            dim_feedforward=max(d_hidden, 2 * int(d_hidden)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(temporal_layers))
        self.market_gate = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
        )
        self.post_fuse_norm = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(dim=d_hidden, dropout=dropout) for _ in range(int(residual_layers))
        )
        self.graph_kind = str(graph_kind)
        self.final_topk = int(final_topk)
        self.fdg = FDG(d_in=d_hidden, rank=int(rank), tau=float(tau), b_init=str(b_init))
        self.roll_graph = RollingCorrelationGraph(topk=int(roll_topk))
        self.entropy_graph = EntropyStockGraph(topk=int(entropy_topk), num_bins=int(entropy_bins))
        self.graph_mix_logit = nn.Parameter(_logit(graph_mix_init))
        self.graph_blocks = nn.ModuleList(
            GraphResidualBlock(dim=d_hidden, dropout=dropout) for _ in range(int(graph_layers))
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        mask_f = mask.to(hidden.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (hidden * mask_f).sum(dim=1) / denom

    def _encode_temporal(self, history: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if history is None:
            raise ValueError("TemporalGraphResidualMLP requires `history` with shape [B, N, L].")
        batch_size, n_nodes, seq_len = history.shape
        seq = history.reshape(batch_size * n_nodes, seq_len, 1)
        seq = self.temporal_proj(seq)
        pos = self.temporal_pos[:, :seq_len, :]
        seq = self.temporal_encoder(seq + pos)
        temporal_hidden = seq[:, -1, :].reshape(batch_size, n_nodes, -1)
        if mask is not None:
            temporal_hidden = temporal_hidden * mask.unsqueeze(-1).to(temporal_hidden.dtype)
        return temporal_hidden

    def _build_graph(self, hidden, history, mask):
        A_fdg, S, R = self.fdg(hidden, mask=mask)
        mix = torch.sigmoid(self.graph_mix_logit)

        if self.graph_kind == "fdg":
            A = A_fdg
            A_aux = A_fdg
        elif self.graph_kind == "roll":
            A_aux = self.roll_graph(history=history, mask=mask)
            A = A_aux
        elif self.graph_kind == "entropy":
            A_aux = self.entropy_graph(history=history, mask=mask)
            A = A_aux
        elif self.graph_kind == "fdg_entropy":
            A_aux = self.entropy_graph(history=history, mask=mask)
            A = row_normalize_adjacency(mix * A_fdg + (1.0 - mix) * A_aux, mask=mask)
        else:
            A_aux = self.roll_graph(history=history, mask=mask)
            A = row_normalize_adjacency(mix * A_fdg + (1.0 - mix) * A_aux, mask=mask)

        if self.final_topk > 0:
            A = topk_sparsify_adjacency(A, topk=self.final_topk, mask=mask)
        return A, A_fdg, A_aux, S, R, mix

    def forward(self, X, mask, history=None, return_graph: bool = False):
        X_model = self.encoder(X, mask=mask)
        feature_hidden = self.feature_proj(X_model)
        temporal_hidden = self._encode_temporal(history=history, mask=mask)

        market_context = self._masked_mean(temporal_hidden, mask=mask)
        gate = torch.sigmoid(self.market_gate(market_context)).unsqueeze(1)
        hidden = feature_hidden + gate * temporal_hidden
        hidden = self.post_fuse_norm(hidden)
        hidden = self.dropout(hidden)
        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)

        for block in self.blocks:
            hidden = block(hidden, mask=mask)

        A, A_fdg, A_aux, S, R, mix = self._build_graph(hidden=hidden, history=history, mask=mask)
        for block in self.graph_blocks:
            hidden = block(hidden, A=A, mask=mask)

        y_hat = self.head(hidden).squeeze(-1)
        y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        if return_graph:
            return y_hat, {
                "A": A,
                "A_fdg": A_fdg,
                "A_aux": A_aux,
                "S": S,
                "R": R,
                "mix": mix,
                "gate": gate,
            }
        return y_hat
