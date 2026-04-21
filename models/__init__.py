from __future__ import annotations

import torch
import torch.nn as nn

from .baselines.mlp import GraphResidualMLP, MLPBaseline, TemporalGraphResidualMLP
from .blocks import FeatureBottleneck
from .fdg import FDG, initialize_core_matrix, row_normalize_adjacency
from .fdg_regularized import RegularizedFDGRegressor
from .fdg_slowfast import SlowFastTemporalFDGRegressor
from .fdg_sparse import SparseRollingFDGRegressor
from .fdg_temporal import TemporalFDGRegressor
from .gnn_head import GNNHead
from .prior import ClusterPrior, build_cluster_centroids
from .temporal_graph import (
    EntropyGraphRegressor,
    EntropyStockGraph,
    FDGAuxGraphRegressor,
    RollingCorrelationGraph,
)


def _default_bottleneck_dim(d_in: int, model_cfg: dict) -> int:
    return int(model_cfg.get("bottleneck_dim", min(64, max(16, d_in // 2))))


def _bottleneck_spec(d_in: int, model_cfg: dict):
    if "bottleneck_layers" in model_cfg:
        return list(model_cfg["bottleneck_layers"])
    return _default_bottleneck_dim(d_in=d_in, model_cfg=model_cfg)


class FDGRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int,
        d_hidden: int,
        tau: float = 1.0,
        dropout: float = 0.1,
        b_init: str = "identity_perturbed",
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.fdg = FDG(d_in=d_in, rank=rank, tau=tau, b_init=b_init)
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_model = self.encoder(X, mask=mask)
        A, S, R = self.fdg(X_model, mask=mask)
        y_hat = self.head(A, X_model, mask=mask)
        if return_graph:
            return y_hat, A, S, R
        return y_hat


class FDGPriorRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int,
        d_hidden: int,
        prior_centroids,
        prior_mean,
        prior_scale,
        prior_fusion: str,
        tau: float = 1.0,
        dropout: float = 0.1,
        b_init: str = "identity_perturbed",
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.fdg = FDG(d_in=d_in, rank=rank, tau=tau, b_init=b_init)
        self.prior = ClusterPrior(
            centroids=prior_centroids,
            mean=prior_mean,
            scale=prior_scale,
        )
        self.prior_fusion = str(prior_fusion)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.head = GNNHead(d_in=d_in, d_hidden=d_hidden, d_out=1, dropout=dropout)

        if self.prior_fusion == "gate":
            self.prior_core = nn.Parameter(torch.empty(self.prior.num_groups, self.prior.num_groups))
            initialize_core_matrix(self.prior_core, b_init=b_init)
        elif self.prior_fusion == "shared":
            self.U_s = nn.Parameter(torch.empty(self.prior.num_groups, rank))
            self.U_r = nn.Parameter(torch.empty(self.prior.num_groups, rank))
            self.reset_prior_parameters()
        else:
            raise ValueError(f"Unsupported prior fusion mode: {self.prior_fusion}")

    def reset_prior_parameters(self) -> None:
        if self.prior.num_groups == self.fdg.rank:
            with torch.no_grad():
                base = torch.eye(self.prior.num_groups)
                self.U_s.copy_(base)
                self.U_r.copy_(base)
                self.U_s.add_(0.01 * torch.randn_like(self.U_s))
                self.U_r.add_(0.01 * torch.randn_like(self.U_r))
        else:
            nn.init.xavier_uniform_(self.U_s)
            nn.init.xavier_uniform_(self.U_r)

    def _build_prior_adjacency(self, X_raw, mask=None):
        C = self.prior.memberships(X_raw, mask=mask)
        if self.prior_fusion == "gate":
            A_prior = C @ self.prior_core @ C.transpose(-1, -2)
        else:
            S_prior = C @ self.U_s
            R_prior = C @ self.U_r
            A_prior = S_prior @ self.fdg.B @ R_prior.transpose(-1, -2)
        A_prior = row_normalize_adjacency(A_prior, mask=mask)
        return A_prior, C

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_model = self.encoder(X, mask=mask)
        A_fdg, S, R = self.fdg(X_model, mask=mask)
        A_prior, C = self._build_prior_adjacency(X_raw=X, mask=mask)
        mix = torch.sigmoid(self.mix_logit)
        A = row_normalize_adjacency(mix * A_fdg + (1.0 - mix) * A_prior, mask=mask)
        y_hat = self.head(A, X_model, mask=mask)
        if return_graph:
            return y_hat, {
                "A": A,
                "A_fdg": A_fdg,
                "A_prior": A_prior,
                "S": S,
                "R": R,
                "C": C,
                "mix": mix,
            }
        return y_hat


def build_model(model_cfg: dict, d_in: int, train_dataset=None) -> nn.Module:
    name = model_cfg["name"].lower()
    bottleneck_dim = _bottleneck_spec(d_in=d_in, model_cfg=model_cfg)

    if name == "fdg":
        return FDGRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            bottleneck_dim=bottleneck_dim,
        )

    if name == "fdg_reg":
        return RegularizedFDGRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            bottleneck_dim=bottleneck_dim,
            adjacency_topk=model_cfg.get("adjacency_topk"),
            core_reg_weight=float(model_cfg.get("core_reg_weight", 0.0)),
            graph_entropy_weight=float(model_cfg.get("graph_entropy_weight", 0.0)),
            assignment_entropy_weight=float(model_cfg.get("assignment_entropy_weight", 0.0)),
        )

    if name == "fdg_temporal":
        return TemporalFDGRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            bottleneck_dim=bottleneck_dim,
            conv_channels=int(model_cfg.get("conv_channels", 16)),
            temporal_kernel_size=int(model_cfg.get("temporal_kernel_size", 3)),
        )

    if name == "fdg_sparse":
        return SparseRollingFDGRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            bottleneck_dim=bottleneck_dim,
            fdg_topk=model_cfg.get("fdg_topk", model_cfg.get("final_topk", 20)),
            roll_topk=int(model_cfg.get("roll_topk", 20)),
            final_topk=model_cfg.get("final_topk"),
            edge_dropout=float(model_cfg.get("edge_dropout", 0.0)),
            mix_init=float(model_cfg.get("roll_mix_init", 0.8)),
        )

    if name == "fdg_slowfast":
        return SlowFastTemporalFDGRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            bottleneck_dim=bottleneck_dim,
            conv_channels=int(model_cfg.get("conv_channels", 16)),
            temporal_kernel_size=int(model_cfg.get("temporal_kernel_size", 3)),
            graph_slow_init=float(model_cfg.get("graph_slow_init", 0.8)),
            value_fast_init=float(model_cfg.get("value_fast_init", 0.8)),
            fast_graph_mix_init=float(model_cfg.get("fast_graph_mix_init", 0.1)),
            graph_smooth_weight=float(model_cfg.get("graph_smooth_weight", 0.0)),
            assignment_smooth_weight=float(model_cfg.get("assignment_smooth_weight", 0.0)),
        )

    if name in {"fdg_prior_gate", "fdg_prior_shared"}:
        if train_dataset is None:
            raise ValueError("Prior-fused FDG requires the training dataset to build cluster centroids.")
        prior_groups = int(model_cfg.get("prior_groups", model_cfg["rank"]))
        prior_seed = int(model_cfg.get("prior_seed", 2026))
        prior_payload = build_cluster_centroids(
            train_dataset=train_dataset,
            num_groups=prior_groups,
            seed=prior_seed,
        )
        return FDGPriorRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            prior_centroids=prior_payload["centroids"],
            prior_mean=prior_payload["mean"],
            prior_scale=prior_payload["scale"],
            prior_fusion="gate" if name == "fdg_prior_gate" else "shared",
            bottleneck_dim=bottleneck_dim,
        )

    if name == "fdg_roll":
        return FDGAuxGraphRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            bottleneck_dim=bottleneck_dim,
            mix_init=float(model_cfg.get("roll_mix_init", 0.8)),
            final_topk=int(model_cfg.get("final_topk", model_cfg.get("roll_topk", 20))),
            aux_graph=RollingCorrelationGraph(topk=int(model_cfg.get("roll_topk", 20))),
        )

    if name == "entropy_gnn":
        return EntropyGraphRegressor(
            d_in=d_in,
            d_hidden=int(model_cfg["d_hidden"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            bottleneck_dim=bottleneck_dim,
            graph=EntropyStockGraph(
                topk=int(model_cfg.get("entropy_topk", 20)),
                num_bins=int(model_cfg.get("entropy_bins", 8)),
            ),
        )

    if name == "mlp":
        return MLPBaseline(
            d_in=d_in,
            d_hidden=int(model_cfg["d_hidden"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            bottleneck_dim=bottleneck_dim,
            residual_layers=int(model_cfg.get("residual_layers", 2)),
        )

    if name == "mlp_graph_plugin":
        return GraphResidualMLP(
            d_in=d_in,
            d_hidden=int(model_cfg["d_hidden"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            bottleneck_dim=bottleneck_dim,
            residual_layers=int(model_cfg.get("residual_layers", 2)),
            graph_layers=int(model_cfg.get("graph_layers", 1)),
            rank=int(model_cfg.get("rank", 16)),
            tau=float(model_cfg.get("tau", 1.0)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            graph_kind=str(model_cfg.get("graph_kind", "fdg_roll")),
            roll_topk=int(model_cfg.get("roll_topk", 20)),
            entropy_topk=int(model_cfg.get("entropy_topk", 20)),
            entropy_bins=int(model_cfg.get("entropy_bins", 8)),
            final_topk=int(model_cfg.get("final_topk", model_cfg.get("roll_topk", 20))),
            graph_mix_init=float(model_cfg.get("graph_mix_init", 0.7)),
        )

    if name == "temporal_graph_plugin":
        return TemporalGraphResidualMLP(
            d_in=d_in,
            d_hidden=int(model_cfg["d_hidden"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            bottleneck_dim=bottleneck_dim,
            residual_layers=int(model_cfg.get("residual_layers", 2)),
            graph_layers=int(model_cfg.get("graph_layers", 1)),
            rank=int(model_cfg.get("rank", 16)),
            tau=float(model_cfg.get("tau", 1.0)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            graph_kind=str(model_cfg.get("graph_kind", "fdg_roll")),
            roll_topk=int(model_cfg.get("roll_topk", 20)),
            entropy_topk=int(model_cfg.get("entropy_topk", 20)),
            entropy_bins=int(model_cfg.get("entropy_bins", 8)),
            final_topk=int(model_cfg.get("final_topk", model_cfg.get("roll_topk", 20))),
            graph_mix_init=float(model_cfg.get("graph_mix_init", 0.7)),
            history_window=int(model_cfg.get("history_window", 20)),
            temporal_layers=int(model_cfg.get("temporal_layers", 1)),
            temporal_heads=int(model_cfg.get("temporal_heads", 4)),
        )

    raise NotImplementedError(f"Unsupported model `{model_cfg['name']}`.")
