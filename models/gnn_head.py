from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import GraphResidualBlock


class GNNHead(nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        d_out=1,
        dropout=0.1,
        graph_gate_init=None,
        use_graph_branch=True,
        use_skip_branch=True,
        skip_hidden_dim=None,
        graph_layers: int = 1,
    ):
        super().__init__()
        if not use_graph_branch and not use_skip_branch:
            raise ValueError("GNNHead requires at least one active branch.")
        self.graph_layers = max(1, int(graph_layers))
        self.W_v = nn.Linear(d_in, d_hidden)
        skip_hidden_dim = d_hidden if skip_hidden_dim is None else int(skip_hidden_dim)
        if skip_hidden_dim <= 0:
            raise ValueError("skip_hidden_dim must be positive.")
        self.skip_hidden_dim = skip_hidden_dim
        if skip_hidden_dim == d_hidden:
            self.W_skip = nn.Linear(d_in, d_hidden)
        else:
            self.W_skip = nn.Sequential(
                nn.Linear(d_in, skip_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(skip_hidden_dim, d_hidden),
            )
        self.norm = nn.LayerNorm(d_hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.use_graph_branch = bool(use_graph_branch)
        self.use_skip_branch = bool(use_skip_branch)
        self.graph_gate_logit = None
        if graph_gate_init is not None:
            self.graph_gate_logit = nn.Parameter(torch.tensor(float(graph_gate_init)))
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
        self.graph_blocks = nn.ModuleList(
            GraphResidualBlock(dim=d_hidden, dropout=dropout) for _ in range(self.graph_layers - 1)
        )

    def graph_gate(self):
        if self.graph_gate_logit is None:
            return None
        return torch.sigmoid(self.graph_gate_logit)

    def forward(self, A, X, mask=None):
        skip = self.W_skip(X) if self.use_skip_branch else None
        graph_message = None
        if self.use_graph_branch:
            values = self.W_v(X)
            graph_message = torch.matmul(A, values)
        gate = self.graph_gate()
        if gate is not None and graph_message is not None:
            graph_message = gate * graph_message

        if graph_message is None:
            graph_message = torch.zeros_like(skip)
        if skip is None:
            skip = torch.zeros_like(graph_message)

        H = graph_message + skip
        for block in self.graph_blocks:
            H = block(H, A, mask=mask)
        Z = self.norm(self.act(H))
        Z = self.dropout(Z)
        y_hat = self.mlp(Z).squeeze(-1)
        if mask is not None:
            y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        return y_hat
