from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import FeatureBottleneck
from .fdg import FDG
from .gnn_head import GNNHead
from .temporal_graph import topk_sparsify_adjacency


def _masked_row_entropy(A: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    probs = A.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    if mask is None:
        return entropy.mean()
    valid = mask.to(entropy.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (entropy * valid).sum() / denom


def _assignment_entropy(assignments: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    probs = assignments.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    if mask is None:
        return entropy.mean()
    valid = mask.to(entropy.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (entropy * valid).sum() / denom


class RegularizedFDGRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int,
        d_hidden: int,
        tau: float = 1.0,
        dropout: float = 0.1,
        b_init: str = "identity_perturbed",
        bottleneck_dim: int = 64,
        adjacency_topk: int | None = None,
        core_reg_weight: float = 0.0,
        graph_entropy_weight: float = 0.0,
        assignment_entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.fdg = FDG(d_in=d_in, rank=rank, tau=tau, b_init=b_init)
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)
        self.adjacency_topk = None if adjacency_topk is None else int(adjacency_topk)
        self.core_reg_weight = float(core_reg_weight)
        self.graph_entropy_weight = float(graph_entropy_weight)
        self.assignment_entropy_weight = float(assignment_entropy_weight)
        self._last_reg_loss: torch.Tensor | None = None
        self._last_reg_terms: dict[str, float] = {}

    def _build_regularizer(
        self,
        A: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        reference = A.sum() * 0.0
        total = reference
        terms: dict[str, float] = {}

        if self.core_reg_weight > 0.0:
            core_term = self.fdg.B.square().mean()
            total = total + self.core_reg_weight * core_term
            terms["core"] = float(core_term.detach().cpu())

        if self.graph_entropy_weight > 0.0:
            graph_entropy = _masked_row_entropy(A=A, mask=mask)
            total = total + self.graph_entropy_weight * graph_entropy
            terms["graph_entropy"] = float(graph_entropy.detach().cpu())

        if self.assignment_entropy_weight > 0.0:
            assignment_entropy = 0.5 * (
                _assignment_entropy(assignments=S, mask=mask) + _assignment_entropy(assignments=R, mask=mask)
            )
            total = total + self.assignment_entropy_weight * assignment_entropy
            terms["assignment_entropy"] = float(assignment_entropy.detach().cpu())

        self._last_reg_terms = terms
        return total

    def regularization_loss(self) -> torch.Tensor:
        if self._last_reg_loss is None:
            parameter = next(self.parameters())
            return parameter.sum() * 0.0
        return self._last_reg_loss

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_model = self.encoder(X, mask=mask)
        A, S, R = self.fdg(X_model, mask=mask)
        if self.adjacency_topk is not None and self.adjacency_topk > 0:
            A = topk_sparsify_adjacency(A, topk=self.adjacency_topk, mask=mask)

        self._last_reg_loss = self._build_regularizer(A=A, S=S, R=R, mask=mask)
        y_hat = self.head(A, X_model, mask=mask)
        if return_graph:
            return y_hat, {
                "A": A,
                "S": S,
                "R": R,
                "reg_terms": dict(self._last_reg_terms),
            }
        return y_hat
