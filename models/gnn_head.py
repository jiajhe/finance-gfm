from __future__ import annotations

import torch
import torch.nn as nn


class GNNHead(nn.Module):
    def __init__(self, d_in, d_hidden, d_out=1, dropout=0.1):
        super().__init__()
        self.W_v = nn.Linear(d_in, d_hidden)
        self.W_skip = nn.Linear(d_in, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, A, X, mask=None):
        values = self.W_v(X)
        skip = self.W_skip(X)
        H = torch.matmul(A, values) + skip
        Z = self.norm(self.act(H))
        Z = self.dropout(Z)
        y_hat = self.mlp(Z).squeeze(-1)
        if mask is not None:
            y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        return y_hat
