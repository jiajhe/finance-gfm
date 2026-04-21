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
    if topk is None or int(topk) <= 0:
        return row_normalize_adjacency(_zero_diagonal(A), mask=mask)

    n_nodes = A.shape[-1]
    k = max(1, min(int(topk), max(1, n_nodes - 1)))
    scores = _zero_diagonal(A)
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(-2), float("-inf"))

    values, indices = torch.topk(scores, k=k, dim=-1)
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    sparse = torch.zeros_like(A).scatter(-1, indices, values)
    return row_normalize_adjacency(sparse, mask=mask)


def _correlation_similarity(history: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    centered = history - history.mean(dim=-1, keepdim=True)
    denom = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / denom
    if mask is not None:
        normalized = normalized * mask.unsqueeze(-1).to(normalized.dtype)
    sim = torch.matmul(normalized, normalized.transpose(-1, -2)) / max(1, history.shape[-1])
    sim = torch.relu(sim)
    return _zero_diagonal(sim)


def _normalized_histogram_entropy(history: torch.Tensor, num_bins: int) -> torch.Tensor:
    flat = history.reshape(-1, history.shape[-1])
    row_min = flat.min(dim=-1, keepdim=True).values
    row_max = flat.max(dim=-1, keepdim=True).values
    scale = (row_max - row_min).clamp_min(1e-6)
    normalized = ((flat - row_min) / scale).clamp(0.0, 1.0 - 1e-6)

    bin_idx = torch.clamp((normalized * int(num_bins)).long(), min=0, max=int(num_bins) - 1)
    probs = torch.zeros(flat.shape[0], int(num_bins), device=flat.device, dtype=flat.dtype)
    probs.scatter_add_(1, bin_idx, torch.ones_like(flat))
    probs = probs / max(1, flat.shape[-1])
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
    if int(num_bins) > 1:
        entropy = entropy / math.log(int(num_bins))
    return entropy.reshape(history.shape[0], history.shape[1])


def _logit_init(prob: float) -> torch.Tensor:
    clipped = min(max(float(prob), 1e-4), 1.0 - 1e-4)
    return torch.tensor(math.log(clipped / (1.0 - clipped)), dtype=torch.float32)


class RollingCorrelationGraph(nn.Module):
    def __init__(self, topk: int = 20) -> None:
        super().__init__()
        self.topk = int(topk)

    def forward(self, history: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
        if history is None:
            raise ValueError("RollingCorrelationGraph requires `history` with shape [B, N, L].")
        sim = _correlation_similarity(history=history, mask=mask)
        return topk_sparsify_adjacency(sim, topk=self.topk, mask=mask)


class EntropyStockGraph(nn.Module):
    def __init__(self, topk: int = 20, num_bins: int = 8) -> None:
        super().__init__()
        self.topk = int(topk)
        self.num_bins = int(num_bins)

    def forward(self, history: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
        if history is None:
            raise ValueError("EntropyStockGraph requires `history` with shape [B, N, L].")

        entropy = _normalized_histogram_entropy(history=history, num_bins=self.num_bins)
        energy = history.square().mean(dim=-1)
        local_similarity = _correlation_similarity(history=history, mask=mask)

        receiver_entropy = entropy.unsqueeze(-1)
        sender_entropy = entropy.unsqueeze(-2)
        entropy_gap = torch.relu(receiver_entropy - sender_entropy)

        receiver_energy = energy.unsqueeze(-1)
        sender_energy = energy.unsqueeze(-2)
        energy_ratio = sender_energy / (sender_energy + receiver_energy + 1e-6)

        # Low-entropy, high-energy senders are encouraged to influence
        # noisier receivers, then pruned into explicit neighbors.
        adjacency = local_similarity * (0.05 + entropy_gap) * energy_ratio
        adjacency = _zero_diagonal(adjacency)
        return topk_sparsify_adjacency(adjacency, topk=self.topk, mask=mask)


class FDGAuxGraphRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int,
        d_hidden: int,
        aux_graph: nn.Module,
        tau: float = 1.0,
        dropout: float = 0.1,
        b_init: str = "identity_perturbed",
        bottleneck_dim: int = 64,
        mix_init: float = 0.8,
        final_topk: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.fdg = FDG(d_in=d_in, rank=rank, tau=tau, b_init=b_init)
        self.aux_graph = aux_graph
        self.mix_logit = nn.Parameter(_logit_init(mix_init))
        self.final_topk = None if final_topk is None else int(final_topk)
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_model = self.encoder(X, mask=mask)
        A_fdg, S, R = self.fdg(X_model, mask=mask)
        A_aux = self.aux_graph(history=history, mask=mask)
        mix = torch.sigmoid(self.mix_logit)
        A = row_normalize_adjacency(mix * A_fdg + (1.0 - mix) * A_aux, mask=mask)
        if self.final_topk is not None and self.final_topk > 0:
            A = topk_sparsify_adjacency(A, topk=self.final_topk, mask=mask)
        y_hat = self.head(A, X_model, mask=mask)
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


class EntropyGraphRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        graph: nn.Module,
        dropout: float = 0.1,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.graph = graph
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_model = self.encoder(X, mask=mask)
        A = self.graph(history=history, mask=mask)
        y_hat = self.head(A, X_model, mask=mask)
        if return_graph:
            return y_hat, {"A": A}
        return y_hat
