from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import qlib
import torch
from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from qlib.data import D
from torch.utils.data import Dataset
from tqdm import tqdm

_QLIB_PROVIDER_URI: str | None = None


def _init_qlib(provider_uri: str | None = None) -> None:
    global _QLIB_PROVIDER_URI
    resolved_uri = str(Path(provider_uri or "~/.qlib/qlib_data/cn_data").expanduser())
    if _QLIB_PROVIDER_URI == resolved_uri:
        return
    qlib.init(provider_uri=resolved_uri)
    _QLIB_PROVIDER_URI = resolved_uri


def _feature_spec(handler: str) -> Tuple[List[str], List[str]]:
    if handler == "Alpha158":
        exprs, names = Alpha158DL.get_feature_config()
    elif handler == "Alpha360":
        exprs, names = Alpha360DL.get_feature_config()
    else:
        raise ValueError(f"Unsupported handler: {handler}")
    return list(exprs), list(names)


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class QlibCrossSectionalDataset(Dataset):
    """
    Yields (X_t, y_t, mask_t, date_t, instruments_t) per trading day.

    X_t: [N_t, d] float tensor, Alpha158/Alpha360 features, z-scored
      cross-sectionally per day.
    y_t: [N_t] float tensor, next-day return label.
    mask_t: [N_t] bool, always True for valid rows before padding.
    date_t: pandas.Timestamp.
    instruments_t: list[str] for portfolio turnover accounting.

    Feature definitions are taken from Qlib's official Alpha158/Alpha360
    loaders, while preprocessing is performed in this dataset to keep a
    cacheable per-day tensor format.
    """

    def __init__(
        self,
        market: str,
        start_time: str,
        end_time: str,
        handler: str = "Alpha158",
        label: str = "Ref($close, -2) / Ref($close, -1) - 1",
        cache_dir: str = "~/.qlib_cache",
        provider_uri: str | None = None,
        chunk_size_days: int = 252,
        feature_limit: int | None = None,
    ) -> None:
        super().__init__()
        self.market = market
        self.start_time = start_time
        self.end_time = end_time
        self.handler = handler
        self.label = label
        self.provider_uri = provider_uri or "~/.qlib/qlib_data/cn_data"
        self.chunk_size_days = int(chunk_size_days)
        self.feature_limit = None if feature_limit is None else int(feature_limit)
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"{market}_{start_time}_{end_time}_{handler}.pt"

        if self.cache_path.exists():
            payload = _safe_torch_load(self.cache_path)
        else:
            payload = self._preprocess()
            torch.save(payload, self.cache_path)

        self.feature_names = list(payload["feature_names"])
        self.days = payload["days"]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, index: int):
        item = self.days[index]
        return item["X"], item["y"], item["mask"], item["date"], item["instruments"]

    def _preprocess(self) -> dict:
        _init_qlib(self.provider_uri)
        feature_exprs, feature_names = _feature_spec(self.handler)
        if self.feature_limit is not None:
            feature_exprs = feature_exprs[: self.feature_limit]
            feature_names = feature_names[: self.feature_limit]
        instruments = D.instruments(self.market)
        calendar = list(D.calendar(start_time=self.start_time, end_time=self.end_time))
        if not calendar:
            raise ValueError(
                f"No trading days found for market={self.market}, "
                f"start={self.start_time}, end={self.end_time}"
            )

        days = []
        label_name = "LABEL0"
        chunk_iterator = range(0, len(calendar), self.chunk_size_days)
        desc = f"preprocess {self.market} {self.start_time}->{self.end_time}"
        for start_idx in tqdm(chunk_iterator, desc=desc, leave=False):
            end_idx = min(start_idx + self.chunk_size_days, len(calendar))
            chunk_start = pd.Timestamp(calendar[start_idx]).strftime("%Y-%m-%d")
            chunk_end = pd.Timestamp(calendar[end_idx - 1]).strftime("%Y-%m-%d")
            chunk_df = D.features(
                instruments,
                feature_exprs + [self.label],
                start_time=chunk_start,
                end_time=chunk_end,
                freq="day",
            )
            if chunk_df.empty:
                continue

            chunk_df.columns = feature_names + [label_name]
            chunk_df = chunk_df.swaplevel("instrument", "datetime").sort_index()
            for date, day_df in chunk_df.groupby(level="datetime", sort=True):
                day_df = day_df.droplevel("datetime")
                processed = self._process_day(
                    date=pd.Timestamp(date),
                    features=day_df.loc[:, feature_names],
                    labels=day_df.loc[:, label_name],
                )
                if processed is not None:
                    days.append(processed)

        if not days:
            raise RuntimeError(
                f"Dataset preprocessing produced no valid days for "
                f"{self.market} {self.start_time}..{self.end_time}"
            )

        return {
            "feature_names": feature_names,
            "days": days,
        }

    def _process_day(
        self,
        date: pd.Timestamp,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> dict | None:
        valid_rows = labels.notna()
        if int(valid_rows.sum()) < 30:
            return None

        features = features.loc[valid_rows]
        labels = labels.loc[valid_rows].astype(np.float32)
        if features.empty:
            return None

        feature_values = features.to_numpy(dtype=np.float32, copy=True)
        keep_cols = ~np.isnan(feature_values).all(axis=0)
        if not np.any(keep_cols):
            return None

        normalized = np.zeros_like(feature_values, dtype=np.float32)
        subset = feature_values[:, keep_cols]
        means = np.nanmean(subset, axis=0)
        stds = np.nanstd(subset, axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)
        subset = (subset - means) / stds
        subset = np.nan_to_num(subset, nan=0.0, posinf=0.0, neginf=0.0)
        normalized[:, keep_cols] = subset

        instruments = [str(inst) for inst in features.index.tolist()]
        y = labels.to_numpy(dtype=np.float32, copy=False)
        return {
            "X": torch.from_numpy(normalized),
            "y": torch.from_numpy(y),
            "mask": torch.ones(len(y), dtype=torch.bool),
            "date": date,
            "instruments": instruments,
        }


def pad_collate(batch):
    """
    Pad variable-N batches to max N.

    Returns:
      X: [B, N_max, d]
      y: [B, N_max]
      mask: [B, N_max]
      dates: list[pd.Timestamp]
      instruments: list[list[str]]
    """

    batch_size = len(batch)
    d_in = batch[0][0].shape[-1]
    max_n = max(item[0].shape[0] for item in batch)

    X = torch.zeros(batch_size, max_n, d_in, dtype=batch[0][0].dtype)
    y = torch.zeros(batch_size, max_n, dtype=batch[0][1].dtype)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    dates = []
    instruments = []

    for idx, (x_i, y_i, mask_i, date_i, instruments_i) in enumerate(batch):
        n_i = x_i.shape[0]
        X[idx, :n_i] = x_i
        y[idx, :n_i] = y_i
        mask[idx, :n_i] = mask_i
        dates.append(date_i)
        instruments.append(instruments_i)

    return X, y, mask, dates, instruments
