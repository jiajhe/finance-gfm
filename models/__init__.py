from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .baselines.mlp import GraphResidualMLP, MLPBaseline, TemporalGraphResidualMLP
from .blocks import FeatureBottleneck
from .fdg import FDG, initialize_core_matrix, row_normalize_adjacency
from .fdg_regularized import RegularizedFDGRegressor
from .fdg_slowfast import SlowFastTemporalFDGRegressor
from .fdg_sparse import SparseRollingFDGRegressor
from .fdg_temporal import TemporalFDGRegressor, TemporalHistoryEncoder
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


def _resolve_feature_indices(
    feature_names: Sequence[str] | None,
    *,
    include: Sequence[str] | str | None = None,
    exclude: Sequence[str] | str | None = None,
) -> list[int] | None:
    if feature_names is None:
        return None

    names = [str(name) for name in feature_names]
    include_set = None
    if include is not None:
        if isinstance(include, str):
            include_set = {include}
        else:
            include_set = {str(name) for name in include}
        missing = include_set.difference(names)
        if missing:
            raise ValueError(f"Unknown graph feature(s): {sorted(missing)}")

    if isinstance(exclude, str):
        exclude_set = {exclude}
    elif exclude is None:
        exclude_set = set()
    else:
        exclude_set = {str(name) for name in exclude}

    indices = []
    for idx, name in enumerate(names):
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        indices.append(idx)

    if not indices:
        raise ValueError("Graph feature selection removed every feature.")
    if len(indices) == len(names):
        return None
    return indices


def _resolve_feature_penalty(
    feature_names: Sequence[str] | None,
    *,
    selected_indices: Sequence[int] | None = None,
    penalty_map: dict[str, float] | None = None,
) -> list[float] | None:
    if feature_names is None or not penalty_map:
        return None

    names = [str(name) for name in feature_names]
    base = [0.0 for _ in names]
    for name, value in penalty_map.items():
        if name not in names:
            raise ValueError(f"Unknown graph feature penalty key: {name}")
        base[names.index(name)] = float(value)

    if selected_indices is None:
        return base if any(weight > 0.0 for weight in base) else None

    selected = [base[int(idx)] for idx in selected_indices]
    return selected if any(weight > 0.0 for weight in selected) else None


def _masked_rank_transform(X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)

    transformed = torch.zeros_like(X)
    for batch_idx in range(X.shape[0]):
        valid_idx = torch.nonzero(mask[batch_idx], as_tuple=False).squeeze(-1)
        n_valid = int(valid_idx.numel())
        if n_valid <= 0:
            continue
        day_values = X[batch_idx, valid_idx]
        order = torch.argsort(day_values, dim=0)
        ranks = torch.empty_like(day_values)
        base = torch.arange(1, n_valid + 1, device=X.device, dtype=X.dtype).unsqueeze(-1)
        ranks.scatter_(0, order, base.expand_as(day_values))
        transformed[batch_idx, valid_idx] = 2.0 * (ranks / float(n_valid)) - 1.0
    return transformed * mask.unsqueeze(-1).to(transformed.dtype)


def _masked_robust_zscore(
    X: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    clip: float = 5.0,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)

    transformed = torch.zeros_like(X)
    for batch_idx in range(X.shape[0]):
        valid_idx = torch.nonzero(mask[batch_idx], as_tuple=False).squeeze(-1)
        if int(valid_idx.numel()) <= 0:
            continue
        day_values = X[batch_idx, valid_idx]
        median = day_values.median(dim=0).values
        mad = (day_values - median).abs().median(dim=0).values
        scale = (1.4826 * mad).clamp_min(1e-6)
        normalized = ((day_values - median) / scale).clamp(-clip, clip)
        transformed[batch_idx, valid_idx] = normalized
    return transformed * mask.unsqueeze(-1).to(transformed.dtype)


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
        graph_feature_indices: Sequence[int] | None = None,
        graph_bottleneck_dim: int | Sequence[int] | None = None,
        graph_input_transform: str = "none",
        graph_gate_init: float | None = None,
        graph_feature_penalty: Sequence[float] | None = None,
        graph_feature_penalty_weight: float = 0.0,
        graph_mode: str = "learned",
        core_mode: str = "asymmetric",
        share_sr_weights: bool = False,
        random_graph_seed: int = 2026,
        use_graph_branch: bool = True,
        use_skip_branch: bool = True,
        skip_hidden_dim: int | None = None,
        graph_layers: int = 1,
        graph_input_source: str = "snapshot",
        graph_history_channels: int = 16,
        graph_history_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.value_encoder = FeatureBottleneck(d_in=d_in, bottleneck_dim=bottleneck_dim, dropout=dropout)
        self.graph_feature_indices = None if graph_feature_indices is None else tuple(int(idx) for idx in graph_feature_indices)
        self.graph_input_transform = str(graph_input_transform).lower()
        self.graph_input_source = str(graph_input_source).lower()
        graph_input_dim = d_in if self.graph_feature_indices is None else len(self.graph_feature_indices)
        graph_bottleneck_dim = bottleneck_dim if graph_bottleneck_dim is None else graph_bottleneck_dim
        share_graph_encoder = (
            self.graph_input_source in {"snapshot", "value", "value_encoded"}
            and self.graph_feature_indices is None
            and self.graph_input_transform in {"none", "identity"}
            and graph_bottleneck_dim == bottleneck_dim
        )
        self.graph_encoder = (
            self.value_encoder
            if share_graph_encoder
            else FeatureBottleneck(d_in=graph_input_dim, bottleneck_dim=graph_bottleneck_dim, dropout=dropout)
        )
        self.graph_history_encoder = None
        if self.graph_input_source in {"history", "temporal", "temporal_embedding"}:
            self.graph_history_encoder = TemporalHistoryEncoder(
                d_out=graph_input_dim,
                conv_channels=int(graph_history_channels),
                kernel_size=int(graph_history_kernel_size),
                dropout=dropout,
            )
        elif self.graph_input_source not in {"snapshot", "value", "value_encoded"}:
            raise ValueError(f"Unsupported graph_input_source: {graph_input_source}")
        self.graph_mode = str(graph_mode).lower()
        self.random_graph_seed = int(random_graph_seed)
        self.fdg = FDG(
            d_in=graph_input_dim,
            rank=rank,
            tau=tau,
            b_init=b_init,
            core_mode=core_mode,
            share_sr_weights=share_sr_weights,
        )
        self.head = GNNHead(
            d_in=d_in,
            d_hidden=d_hidden,
            d_out=1,
            dropout=dropout,
            graph_gate_init=graph_gate_init,
            use_graph_branch=use_graph_branch,
            use_skip_branch=use_skip_branch,
            skip_hidden_dim=skip_hidden_dim,
            graph_layers=graph_layers,
        )
        self.graph_feature_penalty_weight = float(graph_feature_penalty_weight)
        if graph_feature_penalty is None:
            self.register_buffer("graph_feature_penalty", torch.empty(0), persistent=False)
        else:
            penalty = torch.as_tensor(graph_feature_penalty, dtype=torch.float32)
            if penalty.numel() != graph_input_dim:
                raise ValueError(
                    f"graph_feature_penalty must have length {graph_input_dim}, got {penalty.numel()}"
                )
            self.register_buffer("graph_feature_penalty", penalty, persistent=False)

    def _select_graph_input(self, X: torch.Tensor) -> torch.Tensor:
        if self.graph_feature_indices is None:
            return X
        return X[..., list(self.graph_feature_indices)]

    def _transform_graph_input(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.graph_input_transform in {"none", "identity"}:
            return X
        if self.graph_input_transform == "rank":
            return _masked_rank_transform(X, mask=mask)
        if self.graph_input_transform in {"robust_zscore", "robust"}:
            return _masked_robust_zscore(X, mask=mask)
        raise ValueError(f"Unsupported graph_input_transform: {self.graph_input_transform}")

    def _identity_adjacency(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_nodes = X.shape[:2]
        eye = torch.eye(num_nodes, device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return row_normalize_adjacency(eye, mask=mask)

    def _random_adjacency(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_nodes = X.shape[:2]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_graph_seed)
        adjacency = torch.rand((batch_size, num_nodes, num_nodes), generator=generator, dtype=torch.float32)
        adjacency = adjacency.to(device=X.device, dtype=X.dtype)
        return row_normalize_adjacency(adjacency, mask=mask)

    def _build_adjacency(self, X_graph: torch.Tensor, mask: torch.Tensor | None = None):
        if self.graph_mode == "learned":
            return self.fdg(X_graph, mask=mask)
        if self.graph_mode == "identity":
            return self._identity_adjacency(X_graph, mask=mask), None, None
        if self.graph_mode == "random":
            return self._random_adjacency(X_graph, mask=mask), None, None
        raise ValueError(f"Unsupported graph_mode: {self.graph_mode}")

    def forward(self, X, mask=None, history=None, return_graph: bool = False):
        X_value = self.value_encoder(X, mask=mask)
        X_graph_input = None
        if self.graph_input_source in {"history", "temporal", "temporal_embedding"}:
            if self.graph_history_encoder is None:
                raise RuntimeError("graph_history_encoder is not initialized.")
            reference = self._select_graph_input(X)
            X_graph_raw = self.graph_history_encoder(history=history, mask=mask, reference=reference)
            X_graph_input = self._transform_graph_input(X_graph_raw, mask=mask)
            X_graph = self.graph_encoder(X_graph_input, mask=mask)
        elif self.graph_encoder is self.value_encoder and self.graph_feature_indices is None:
            X_graph = X_value
        else:
            X_graph_raw = self._select_graph_input(X)
            X_graph_input = self._transform_graph_input(X_graph_raw, mask=mask)
            X_graph = self.graph_encoder(X_graph_input, mask=mask)
        A, S, R = self._build_adjacency(X_graph, mask=mask)
        y_hat = self.head(A, X_value, mask=mask)
        if return_graph:
            return y_hat, {
                "A": A,
                "S": S,
                "R": R,
                "X_graph_input": X_graph_input,
                "X_graph": X_graph,
                "X_value": X_value,
                "graph_input_source": self.graph_input_source,
                "graph_gate": self.head.graph_gate(),
                "graph_mode": self.graph_mode,
                "core_mode": self.fdg.core_mode,
                "share_sr_weights": self.fdg.share_sr_weights,
                "B_core": self.fdg.core_matrix(),
                "skip_hidden_dim": self.head.skip_hidden_dim,
                "graph_layers": self.head.graph_layers,
                "use_graph_branch": self.head.use_graph_branch,
                "use_skip_branch": self.head.use_skip_branch,
            }
        return y_hat

    def regularization_loss(self):
        if self.graph_feature_penalty_weight <= 0.0 or self.graph_feature_penalty.numel() == 0:
            return self.fdg.B.sum() * 0.0

        penalty = self.graph_feature_penalty.to(self.fdg.W_s.weight.dtype)
        weight_sq = self.fdg.W_s.weight.square() + self.fdg.W_r.weight.square()
        return self.graph_feature_penalty_weight * (weight_sq * penalty.unsqueeze(0)).sum()


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
        if train_dataset is None and (
            model_cfg.get("graph_feature_include") is not None or model_cfg.get("graph_feature_exclude") is not None
        ):
            raise ValueError("Graph feature selection requires train_dataset.feature_names.")
        feature_names = None if train_dataset is None else getattr(train_dataset, "feature_names", None)
        graph_feature_indices = _resolve_feature_indices(
            feature_names,
            include=model_cfg.get("graph_feature_include"),
            exclude=model_cfg.get("graph_feature_exclude"),
        )
        graph_feature_penalty = _resolve_feature_penalty(
            feature_names,
            selected_indices=graph_feature_indices,
            penalty_map=model_cfg.get("graph_feature_penalty"),
        )
        return FDGRegressor(
            d_in=d_in,
            rank=int(model_cfg["rank"]),
            d_hidden=int(model_cfg["d_hidden"]),
            tau=float(model_cfg.get("tau", 1.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            b_init=str(model_cfg.get("b_init", "identity_perturbed")),
            bottleneck_dim=bottleneck_dim,
            graph_feature_indices=graph_feature_indices,
            graph_bottleneck_dim=model_cfg.get("graph_bottleneck_dim"),
            graph_input_transform=str(model_cfg.get("graph_input_transform", "none")),
            graph_gate_init=model_cfg.get("graph_gate_init"),
            graph_feature_penalty=graph_feature_penalty,
            graph_feature_penalty_weight=float(model_cfg.get("graph_feature_penalty_weight", 0.0)),
            graph_mode=str(model_cfg.get("graph_mode", "learned")),
            core_mode=str(model_cfg.get("core_mode", "asymmetric")),
            share_sr_weights=bool(model_cfg.get("share_sr_weights", False)),
            random_graph_seed=int(model_cfg.get("random_graph_seed", 2026)),
            use_graph_branch=bool(model_cfg.get("use_graph_branch", True)),
            use_skip_branch=bool(model_cfg.get("use_skip_branch", True)),
            skip_hidden_dim=model_cfg.get("skip_hidden_dim"),
            graph_layers=int(model_cfg.get("graph_layers", 1)),
            graph_input_source=str(model_cfg.get("graph_input_source", "snapshot")),
            graph_history_channels=int(model_cfg.get("graph_history_channels", 16)),
            graph_history_kernel_size=int(model_cfg.get("graph_history_kernel_size", 3)),
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
