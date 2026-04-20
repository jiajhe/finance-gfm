from __future__ import annotations

import torch.nn as nn

from .baselines.mlp import MLPBaseline
from .fdg import FDG
from .gnn_head import GNNHead


class FDGRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int,
        d_hidden: int,
        tau: float = 1.0,
        dropout: float = 0.1,
        b_init: str = "identity_perturbed",
    ) -> None:
        super().__init__()
        self.fdg = FDG(d_in=d_in, rank=rank, tau=tau, b_init=b_init)
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)

    def forward(self, X, mask=None, return_graph: bool = False):
        A, S, R = self.fdg(X, mask=mask)
        y_hat = self.head(A, X, mask=mask)
        if return_graph:
            return y_hat, A, S, R
        return y_hat


def build_model(model_cfg: dict, d_in: int) -> nn.Module:
    name = model_cfg["name"].lower()
    if name == "fdg":
        return FDGRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
        )
    if name == "mlp":
        return MLPBaseline(
            d_in=d_in,
            d_hidden=int(model_cfg["d_hidden"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise NotImplementedError(f"Phase 1 does not support model `{model_cfg['name']}` yet.")
