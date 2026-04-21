from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _prior_cache_path(train_dataset, num_groups: int, seed: int) -> Path:
    cache_dir = Path(train_dataset.cache_dir).expanduser()
    return cache_dir / (
        f"prior_cluster_{train_dataset.market}_{train_dataset.start_time}_{train_dataset.end_time}_"
        f"{train_dataset.handler}_k{num_groups}_s{seed}.pt"
    )


def build_cluster_centroids(train_dataset, num_groups: int, seed: int = 2026) -> dict:
    cache_path = _prior_cache_path(train_dataset, num_groups=num_groups, seed=seed)
    if cache_path.exists():
        return _safe_torch_load(cache_path)

    feature_sums: dict[str, np.ndarray] = {}
    feature_counts: dict[str, int] = {}
    for day in train_dataset.days:
        X_day = day["X"].numpy()
        for row, instrument in zip(X_day, day["instruments"]):
            if instrument not in feature_sums:
                feature_sums[instrument] = row.astype(np.float64, copy=True)
                feature_counts[instrument] = 1
            else:
                feature_sums[instrument] += row
                feature_counts[instrument] += 1

    if not feature_sums:
        raise RuntimeError("Unable to build cluster prior: no instrument fingerprints found.")

    ordered_instruments = sorted(feature_sums)
    fingerprints = np.stack(
        [feature_sums[inst] / feature_counts[inst] for inst in ordered_instruments],
        axis=0,
    )

    scaler = StandardScaler()
    fingerprints_std = scaler.fit_transform(fingerprints)
    scale = np.where(scaler.scale_ < 1e-6, 1.0, scaler.scale_).astype(np.float32)
    n_clusters = max(1, min(int(num_groups), fingerprints_std.shape[0]))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    kmeans.fit(fingerprints_std)

    payload = {
        "centroids": kmeans.cluster_centers_.astype(np.float32),
        "mean": scaler.mean_.astype(np.float32),
        "scale": scale,
        "num_groups": int(n_clusters),
    }
    torch.save(payload, cache_path)
    return payload


class ClusterPrior(nn.Module):
    def __init__(self, centroids, mean, scale) -> None:
        super().__init__()
        centroids_tensor = torch.as_tensor(centroids, dtype=torch.float32)
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        self.register_buffer("centroids", centroids_tensor)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("scale", scale_tensor)

    @property
    def num_groups(self) -> int:
        return int(self.centroids.shape[0])

    def memberships(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        X_std = (X - self.mean) / self.scale
        centers = self.centroids.unsqueeze(0).expand(X.shape[0], -1, -1)
        distances = torch.cdist(X_std, centers)
        nearest = distances.argmin(dim=-1)
        C = F.one_hot(nearest, num_classes=self.num_groups).to(X.dtype)
        if mask is not None:
            C = C * mask.unsqueeze(-1).to(C.dtype)
        return C
