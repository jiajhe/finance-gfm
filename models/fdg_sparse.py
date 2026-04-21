from __future__ import annotations

import math

import torch
import torch.nn as nn

from .blocks import FeatureBottleneck
from .fdg import FDG, row_normalize_adjacency
from .gnn_head import GNNHead


def _zero_diagonal(A: torch.Tensor) -> torch.Tensor:
    n_nodes = A.shape[-1]
    diag_mask = torch.eye(n_nodes, device=A.device, dtype=torch.bool).unsqueeze(0)
    return A.masked_fill(diag_mask, 0.0)


def topk_sparsify_adjacency(
    A: torch.Tensor,
    topk: int | None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = _zero_diagonal(A)
    if topk is None or int(topk) <= 0:
        return row_normalize_adjacency(scores, mask=mask)

    n_nodes = scores.shape[-1]
    k = max(1, min(int(topk), max(1, n_nodes - 1)))
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(-2), float("-inf"))

    values, indices = torch.topk(scores, k=k, dim=-1)
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    sparse = torch.zeros_like(scores).scatter(-1, indices, values)
    return row_normalize_adjacency(sparse, mask=mask)


def edge_dropout_adjacency(
    A: torch.Tensor,
    dropout: float,
    training: bool,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    adjacency = _zero_diagonal(A)
    if not training or float(dropout) <= 0.0:
        return row_normalize_adjacency(adjacency, mask=mask)

    keep_prob = 1.0 - float(dropout)
    edge_keep = torch.rand_like(adjacency).le(keep_prob)
    edge_keep = edge_keep.to(adjacency.dtype)
    adjacency = adjacency * edge_keep / max(keep_prob, 1e-6)
    return row_normalize_adjacency(adjacency, mask=mask)


def _correlation_similarity(
    history: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    centered = history - history.mean(dim=-1, keepdim=True)
    denom = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / denom
    if mask is not None:
        normalized = normalized * mask.unsqueeze(-1).to(normalized.dtype)
    sim = torch.matmul(normalized, normalized.transpose(-1, -2)) / max(1, history.shape[-1])
    sim = torch.relu(sim)
    return _zero_diagonal(sim)


def _logit_init(prob: float) -> torch.Tensor:
    clipped = min(max(float(prob), 1e-4), 1.0 - 1e-4)
    return torch.tensor(math.log(clipped / (1.0 - clipped)), dtype=torch.float32)


class SparseRollingCorrelationGraph(nn.Module):
    def __init__(self, topk: int = 20, edge_dropout: float = 0.0) -> None:
        super().__init__()
        self.topk = int(topk)
        self.edge_dropout = float(edge_dropout)

    def forward(self, history: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
        if history is None:
            raise ValueError("SparseRollingCorrelationGraph requires `history` with shape [B, N, L].")
        A = _correlation_similarity(history=history, mask=mask)
        A = topk_sparsify_adjacency(A, topk=self.topk, mask=mask)
        return edge_dropout_adjacency(A, dropout=self.edge_dropout, training=self.training, mask=mask)


class SparseFDGBranch(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int,
        tau: float = 1.0,
        b_init: str = "identity_perturbed",
        topk: int | None = None,
        edge_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fdg = FDG(d_in=d_in, rank=rank, tau=tau, b_init=b_init)
        self.topk = None if topk is None else int(topk)
        self.edge_dropout = float(edge_dropout)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        A, S, R = self.fdg(X, mask=mask)
        A = topk_sparsify_adjacency(A, topk=self.topk, mask=mask)
        A = edge_dropout_adjacency(A, dropout=self.edge_dropout, training=self.training, mask=mask)
        return A, S, R


class SparseRollingFDGRegressor(nn.Module):
    """
    Standalone sparse FDG regressor with a learned FDG branch and a rolling
    correlation branch. Both branches are explicitly sparsified, then mixed by
    a learnable scalar gate before message passing.
    """

    def __init__(
        self,
        d_in: int,
        rank: int,
        d_hidden: int,
        tau: float = 1.0,
        dropout: float = 0.1,
        b_init: str = "identity_perturbed",
        bottleneck_dim: int = 64,
        fdg_topk: int | None = 20,
        roll_topk: int = 20,
        final_topk: int | None = None,
        edge_dropout: float = 0.0,
        mix_init: float = 0.8,
    ) -> None:
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.fdg_branch = SparseFDGBranch(
            d_in=d_in,
            rank=rank,
            tau=tau,
            b_init=b_init,
            topk=fdg_topk,
            edge_dropout=edge_dropout,
        )
        self.roll_branch = SparseRollingCorrelationGraph(topk=roll_topk, edge_dropout=edge_dropout)
        self.mix_logit = nn.Parameter(_logit_init(mix_init))
        self.final_topk = None if final_topk is None else int(final_topk)
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_model = self.encoder(X, mask=mask)
        A_fdg, S, R = self.fdg_branch(X_model, mask=mask)
        A_roll = self.roll_branch(history=history, mask=mask)

        mix = torch.sigmoid(self.mix_logit)
        A = row_normalize_adjacency(mix * A_fdg + (1.0 - mix) * A_roll, mask=mask)
        if self.final_topk is not None and self.final_topk > 0:
            A = topk_sparsify_adjacency(A, topk=self.final_topk, mask=mask)

        y_hat = self.head(A, X_model, mask=mask)
        if return_graph:
            return y_hat, {
                "A": A,
                "A_fdg": A_fdg,
                "A_roll": A_roll,
                "S": S,
                "R": R,
                "mix": mix,
            }
        return y_hat


__all__ = [
    "SparseFDGBranch",
    "SparseRollingCorrelationGraph",
    "SparseRollingFDGRegressor",
    "edge_dropout_adjacency",
    "topk_sparsify_adjacency",
]
