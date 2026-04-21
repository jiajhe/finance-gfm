from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class FeatureBottleneck(nn.Module):
    def __init__(self, d_in: int, bottleneck_dim: int | Sequence[int], dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        if bottleneck_dim is None:
            hidden_dims = []
        elif isinstance(bottleneck_dim, Sequence) and not isinstance(bottleneck_dim, (str, bytes)):
            hidden_dims = [max(4, int(dim)) for dim in bottleneck_dim]
        else:
            hidden_dims = [max(4, int(bottleneck_dim))]
        dims = [d_in] + hidden_dims + [d_in]
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(self.layers) == 1 and self.layers[0].in_features == self.layers[0].out_features:
            out = X
            if mask is not None:
                out = out * mask.unsqueeze(-1).to(out.dtype)
            return out
        hidden = self.norm(X)
        for layer in self.layers[:-1]:
            hidden = self.dropout(self.act(layer(hidden)))
        out = X + self.dropout(self.layers[-1](hidden))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, expansion: int = 2) -> None:
        super().__init__()
        hidden_dim = max(dim, int(dim * expansion))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.fc1(self.norm(X))
        hidden = self.dropout(self.act(hidden))
        out = X + self.dropout(self.fc2(hidden))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class GraphResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, X: torch.Tensor, A: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.norm(X)
        hidden = torch.matmul(A, self.value(hidden))
        hidden = self.dropout(self.act(hidden))
        update = self.dropout(self.out_proj(hidden))
        out = X + torch.sigmoid(self.mix_logit) * update
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out
