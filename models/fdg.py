from __future__ import annotations

import math

import torch
import torch.nn as nn


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
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.tau = float(tau)
        self.W_s = nn.Linear(d_in, rank, bias=False)
        self.W_r = nn.Linear(d_in, rank, bias=False)
        self.B = nn.Parameter(torch.empty(rank, rank))
        self.reset_parameters(b_init=b_init)

    def reset_parameters(self, b_init: str = "identity_perturbed") -> None:
        nn.init.xavier_uniform_(self.W_s.weight)
        nn.init.xavier_uniform_(self.W_r.weight)
        if b_init == "identity_perturbed":
            with torch.no_grad():
                self.B.copy_(torch.eye(self.rank))
                self.B.add_(0.01 * torch.randn_like(self.B))
        elif b_init == "random":
            nn.init.normal_(self.B, mean=0.0, std=1.0 / math.sqrt(self.rank))
        else:
            raise ValueError(f"Unsupported b_init: {b_init}")

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

        A = S @ self.B @ R.transpose(-1, -2)

        if mask is not None:
            valid = mask.to(A.dtype)
            A = A * valid.unsqueeze(-1) * valid.unsqueeze(-2)

        row_sums = A.sum(dim=-1, keepdim=True)
        non_zero_rows = row_sums.abs() >= 1e-8
        safe_row_sums = torch.where(non_zero_rows, row_sums, torch.ones_like(row_sums))
        A = torch.where(non_zero_rows, A / safe_row_sums, torch.zeros_like(A))

        if mask is not None:
            A = A * mask.to(A.dtype).unsqueeze(-1)

        return A, S, R
