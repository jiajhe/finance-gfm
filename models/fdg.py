from __future__ import annotations

import math

import torch
import torch.nn as nn


def initialize_core_matrix(matrix: torch.Tensor, b_init: str = "identity_perturbed") -> None:
    if b_init == "identity_perturbed":
        with torch.no_grad():
            if matrix.shape[0] == matrix.shape[1]:
                matrix.copy_(torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype))
            else:
                nn.init.xavier_uniform_(matrix)
            matrix.add_(0.01 * torch.randn_like(matrix))
    elif b_init == "random":
        nn.init.normal_(matrix, mean=0.0, std=1.0 / math.sqrt(max(1, matrix.shape[-1])))
    else:
        raise ValueError(f"Unsupported b_init: {b_init}")


def row_normalize_adjacency(A: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        valid = mask.to(A.dtype)
        A = A * valid.unsqueeze(-1) * valid.unsqueeze(-2)

    row_sums = A.sum(dim=-1, keepdim=True)
    non_zero_rows = row_sums.abs() >= 1e-8
    safe_row_sums = torch.where(non_zero_rows, row_sums, torch.ones_like(row_sums))
    A = torch.where(non_zero_rows, A / safe_row_sums, torch.zeros_like(A))

    if mask is not None:
        A = A * mask.to(A.dtype).unsqueeze(-1)
    return A


class FDG(nn.Module):
    """
    Factorized Directed Graph.

    A = softmax(X W_s / tau) @ B @ softmax(X W_r / tau).T

    HIST's hidden-concept branch is recovered by setting W_s = W_r and
    B = I_r (symmetric bipartite). FDG generalizes to the directed asymmetric
    case with a learnable core influence matrix B.

    The returned adjacency is row-normalized after masking so each valid row
    sums to 1 whenever its pre-normalization mass is non-zero.
    """

    def __init__(
        self,
        d_in: int,
        rank: int,
        tau: float = 1.0,
        b_init: str = "identity_perturbed",
        core_mode: str = "asymmetric",
        share_sr_weights: bool = False,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.tau = float(tau)
        self.core_mode = str(core_mode).lower()
        self.share_sr_weights = bool(share_sr_weights)
        if self.core_mode not in {"asymmetric", "symmetric"}:
            raise ValueError(f"Unsupported core_mode: {core_mode}")
        self.W_s = nn.Linear(d_in, rank, bias=False)
        if self.share_sr_weights:
            self.W_r = self.W_s
        else:
            self.W_r = nn.Linear(d_in, rank, bias=False)
        self.B = nn.Parameter(torch.empty(rank, rank))
        self.reset_parameters(b_init=b_init)

    def reset_parameters(self, b_init: str = "identity_perturbed") -> None:
        nn.init.xavier_uniform_(self.W_s.weight)
        if not self.share_sr_weights:
            nn.init.xavier_uniform_(self.W_r.weight)
        initialize_core_matrix(self.B, b_init=b_init)

    def core_matrix(self) -> torch.Tensor:
        if self.core_mode == "symmetric":
            return 0.5 * (self.B + self.B.transpose(-1, -2))
        return self.B

    def forward(self, X, mask=None):
        """
        Args:
          X: [B, N, d]
          mask: [B, N] bool

        Returns:
          A: [B, N, N]
          S: [B, N, r]
          R: [B, N, r]
        """

        logits_s = self.W_s(X) / self.tau
        logits_r = self.W_r(X) / self.tau

        if mask is not None:
            expanded_mask = mask.unsqueeze(-1)
            logits_s = logits_s.masked_fill(~expanded_mask, -1e9)
            logits_r = logits_r.masked_fill(~expanded_mask, -1e9)
        else:
            expanded_mask = None

        S = torch.softmax(logits_s, dim=-1)
        R = torch.softmax(logits_r, dim=-1)

        if expanded_mask is not None:
            S = S * expanded_mask.to(S.dtype)
            R = R * expanded_mask.to(R.dtype)

        A = S @ self.core_matrix() @ R.transpose(-1, -2)

        A = row_normalize_adjacency(A, mask=mask)
        return A, S, R
