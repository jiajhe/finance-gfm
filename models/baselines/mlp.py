from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, d_in, d_hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, X, mask):
        y_hat = self.net(X).squeeze(-1)
        return torch.where(mask, y_hat, torch.zeros_like(y_hat))
