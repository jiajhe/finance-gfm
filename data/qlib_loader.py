from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
import qlib
import torch
from qlib.contrib.data.handler import Alpha158 as Alpha158Handler
from qlib.contrib.data.handler import Alpha360 as Alpha360Handler
from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.data import Dataset
from tqdm import tqdm

_QLIB_INIT_STATE: tuple[str, str] | None = None


def _init_qlib(provider_uri: str | None = None, region: str = "cn") -> None:
    global _QLIB_INIT_STATE
    resolved_uri = str(Path(provider_uri or "~/.qlib/qlib_data/cn_data").expanduser())
    init_state = (resolved_uri, str(region))
    if _QLIB_INIT_STATE == init_state:
        return
    qlib.init(provider_uri=resolved_uri, region=region)
    _QLIB_INIT_STATE = init_state


def _feature_spec(handler: str) -> Tuple[List[str], List[str]]:
    if handler == "Alpha158":
        exprs, names = Alpha158DL.get_feature_config()
    elif handler == "Alpha360":
        exprs, names = Alpha360DL.get_feature_config()
    else:
        raise ValueError(f"Unsupported handler: {handler}")
    return list(exprs), list(names)


def _history_spec(history_window: int) -> Tuple[List[str], List[str]]:
    if int(history_window) <= 0:
        return [], []

    exprs: List[str] = []
    names: List[str] = []
    for lag in range(int(history_window)):
        if lag == 0:
            expr = "$close / Ref($close, 1) - 1"
        else:
            expr = f"Ref($close, {lag}) / Ref($close, {lag + 1}) - 1"
        exprs.append(expr)
        names.append(f"HIST_RET_{lag:02d}")
    return exprs, names


def _load_history_frame(
    market: str,
    start_time: str,
    end_time: str,
    history_window: int,
) -> Tuple[pd.DataFrame | None, List[str]]:
    history_exprs, history_names = _history_spec(history_window)
    if not history_names:
        return None, []

    instruments = D.instruments(market)
    history_df = D.features(
        instruments,
        history_exprs,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    if history_df.empty:
        return None, history_names

    history_df.columns = history_names
    history_df = history_df.swaplevel("instrument", "datetime").sort_index()
    return history_df, history_names


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cache_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()[:12]


def _single_series(frame_or_series: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(frame_or_series, pd.Series):
        return frame_or_series
    if frame_or_series.shape[1] != 1:
        raise ValueError(f"Expected exactly one label column, got {frame_or_series.shape[1]}")
    return frame_or_series.iloc[:, 0]


def _resolve_feature_selection(
    feature_names: Sequence[str],
    selection: Sequence[str] | str | None,
) -> list[str]:
    feature_names = [str(name) for name in feature_names]
    if selection is None:
        return []
    if isinstance(selection, str):
        if selection.lower() in {"__all__", "all", "*"}:
            return feature_names
        selection = [selection]

    selection_set = {str(name) for name in selection}
    return [name for name in feature_names if name in selection_set]


def _signed_pct_rank(series: pd.Series) -> pd.Series:
    rank = series.rank(method="average", pct=True)
    return (2.0 * rank - 1.0).astype(np.float32)


def _apply_feature_transforms(
    features: pd.DataFrame,
    *,
    rank_features: Sequence[str] | str | None = None,
    drop_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    transformed = features.copy()
    rank_cols = _resolve_feature_selection(transformed.columns, rank_features)
    if rank_cols:
        transformed.loc[:, rank_cols] = transformed.loc[:, rank_cols].apply(_signed_pct_rank, axis=0)

    drop_cols = _resolve_feature_selection(transformed.columns, drop_features)
    if drop_cols:
        transformed = transformed.drop(columns=drop_cols)
    return transformed


def _make_day_payload(
    *,
    date: pd.Timestamp,
    features: pd.DataFrame,
    labels: pd.Series,
    raw_labels: pd.Series | None = None,
    history: pd.DataFrame | None = None,
    normalize_features: bool,
) -> dict | None:
    valid_rows = labels.notna()
    if raw_labels is not None:
        valid_rows = valid_rows & raw_labels.notna()
    if int(valid_rows.sum()) < 30:
        return None

    features = features.loc[valid_rows]
    labels = labels.loc[valid_rows].astype(np.float32)
    raw_labels = labels if raw_labels is None else raw_labels.loc[valid_rows].astype(np.float32)
    if history is not None:
        history = history.loc[valid_rows]
    if features.empty:
        return None

    feature_values = features.to_numpy(dtype=np.float32, copy=True)
    if normalize_features:
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
        feature_values = normalized
    else:
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)

    history_values = None
    if history is not None:
        history_values = history.to_numpy(dtype=np.float32, copy=True)
        history_values = np.nan_to_num(history_values, nan=0.0, posinf=0.0, neginf=0.0)
        history_values = np.clip(history_values, -0.3, 0.3)

    instruments = [str(inst) for inst in features.index.tolist()]
    y = labels.to_numpy(dtype=np.float32, copy=False)
    raw_y = raw_labels.to_numpy(dtype=np.float32, copy=False)
    payload = {
        "X": torch.from_numpy(feature_values),
        "y": torch.from_numpy(y),
        "raw_y": torch.from_numpy(raw_y),
        "mask": torch.ones(len(y), dtype=torch.bool),
        "date": date,
        "instruments": instruments,
    }
    if history_values is not None:
        payload["history"] = torch.from_numpy(history_values)
    return payload


class InMemoryCrossSectionalDataset(Dataset):
    def __init__(self, days: list[dict], feature_names: Sequence[str]) -> None:
        super().__init__()
        self.days = list(days)
        self.feature_names = list(feature_names)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, index: int):
        item = self.days[index]
        return (
            item["X"],
            item["y"],
            item.get("raw_y", item["y"]),
            item["mask"],
            item["date"],
            item["instruments"],
            item.get("history"),
        )


class QlibCrossSectionalDataset(InMemoryCrossSectionalDataset):
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
        history_window: int = 0,
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
        self.history_window = int(history_window)
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / (
            f"{market}_{start_time}_{end_time}_{handler}_hist{self.history_window}.pt"
        )

        if self.cache_path.exists():
            payload = _safe_torch_load(self.cache_path)
        else:
            payload = self._preprocess()
            torch.save(payload, self.cache_path)

        super().__init__(days=payload["days"], feature_names=payload["feature_names"])

    def _preprocess(self) -> dict:
        _init_qlib(self.provider_uri)
        feature_exprs, feature_names = _feature_spec(self.handler)
        history_exprs, history_names = _history_spec(self.history_window)
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
                feature_exprs + history_exprs + [self.label],
                start_time=chunk_start,
                end_time=chunk_end,
                freq="day",
            )
            if chunk_df.empty:
                continue

            chunk_df.columns = feature_names + history_names + [label_name]
            chunk_df = chunk_df.swaplevel("instrument", "datetime").sort_index()
            for date, day_df in chunk_df.groupby(level="datetime", sort=True):
                day_df = day_df.droplevel("datetime")
                processed = _make_day_payload(
                    date=pd.Timestamp(date),
                    features=day_df.loc[:, feature_names],
                    history=None if not history_names else day_df.loc[:, history_names],
                    labels=_single_series(day_df.loc[:, [label_name]]),
                    normalize_features=True,
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


class QlibHandlerCrossSectionalDataset(InMemoryCrossSectionalDataset):
    def __init__(
        self,
        *,
        market: str,
        splits: dict,
        split_name: str,
        handler: str = "Alpha158",
        label: str = "Ref($close, -2) / Ref($close, -1) - 1",
        cache_dir: str = "~/.qlib_cache",
        provider_uri: str | None = None,
        region: str = "cn",
        fit_start_time: str | None = None,
        fit_end_time: str | None = None,
        infer_processors: list | None = None,
        learn_processors: list | None = None,
        process_type: str = "independent",
        history_window: int = 0,
        feature_rank_transform: dict | None = None,
        feature_drop_list: Sequence[str] | None = None,
    ) -> None:
        self.market = market
        self.splits = splits
        self.split_name = split_name
        self.handler = handler
        self.label = label
        self.provider_uri = provider_uri or "~/.qlib/qlib_data/cn_data"
        self.region = region
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.infer_processors = infer_processors or []
        self.learn_processors = learn_processors or []
        self.process_type = process_type
        self.history_window = int(history_window)
        self.feature_rank_transform = feature_rank_transform or {}
        self.feature_drop_list = list(feature_drop_list or [])

        cache_dir_path = Path(cache_dir).expanduser()
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_meta = {
            "market": market,
            "handler": handler,
            "split_name": split_name,
            "splits": splits,
            "fit_start_time": fit_start_time,
            "fit_end_time": fit_end_time,
            "infer_processors": self.infer_processors,
            "learn_processors": self.learn_processors,
            "process_type": process_type,
            "history_window": self.history_window,
            "label": label,
            "feature_rank_transform": self.feature_rank_transform,
            "feature_drop_list": self.feature_drop_list,
        }
        fingerprint = _cache_fingerprint(cache_meta)
        self.cache_path = cache_dir_path / f"{market}_{handler}_{split_name}_{fingerprint}.pt"

        if self.cache_path.exists():
            payload = _safe_torch_load(self.cache_path)
        else:
            payload = self._preprocess()
            torch.save(payload, self.cache_path)

        super().__init__(days=payload["days"], feature_names=payload["feature_names"])

    def _handler_class(self):
        if self.handler == "Alpha158":
            return Alpha158Handler
        if self.handler == "Alpha360":
            return Alpha360Handler
        raise ValueError(f"Unsupported handler for official pipeline: {self.handler}")

    def _preprocess(self) -> dict:
        _init_qlib(self.provider_uri, region=self.region)
        handler_cls = self._handler_class()
        overall_start = min(period[0] for period in self.splits.values())
        overall_end = max(period[1] for period in self.splits.values())

        handler_kwargs = {
            "instruments": self.market,
            "start_time": overall_start,
            "end_time": overall_end,
            "fit_start_time": self.fit_start_time,
            "fit_end_time": self.fit_end_time,
            "infer_processors": self.infer_processors,
            "learn_processors": self.learn_processors,
            "label": ([self.label], ["LABEL0"]),
        }
        if self.handler == "Alpha158":
            handler_kwargs["process_type"] = self.process_type

        handler = handler_cls(**handler_kwargs)
        dataset = DatasetH(handler=handler, segments=self.splits)
        learn_df = dataset.prepare(self.split_name, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        raw_label_df = dataset.prepare(self.split_name, col_set=["label"], data_key=DataHandlerLP.DK_R)
        history_df, _ = _load_history_frame(
            market=self.market,
            start_time=overall_start,
            end_time=overall_end,
            history_window=self.history_window,
        )

        feature_df = learn_df["feature"]
        rank_selection = None
        if self.feature_rank_transform:
            rank_selection = self.feature_rank_transform.get("features")
        feature_df = _apply_feature_transforms(
            feature_df,
            rank_features=rank_selection,
            drop_features=self.feature_drop_list,
        )
        label_series = _single_series(learn_df["label"])
        raw_label_series = _single_series(raw_label_df["label"]).reindex(learn_df.index)
        feature_names = [str(col) for col in feature_df.columns]

        days = []
        for date, feature_day in feature_df.groupby(level="datetime", sort=True):
            day_index = feature_day.index
            label_day = label_series.loc[day_index]
            raw_label_day = raw_label_series.loc[day_index]
            history_day = None
            if history_df is not None:
                try:
                    history_day = history_df.xs(date, level="datetime").reindex(
                        feature_day.droplevel("datetime").index
                    )
                except KeyError:
                    history_day = None
            processed = _make_day_payload(
                date=pd.Timestamp(date),
                features=feature_day.droplevel("datetime"),
                labels=label_day.droplevel("datetime"),
                raw_labels=raw_label_day.droplevel("datetime"),
                history=history_day,
                normalize_features=False,
            )
            if processed is not None:
                days.append(processed)

        if not days:
            raise RuntimeError(
                f"Official pipeline produced no valid days for split={self.split_name}, market={self.market}"
            )

        return {
            "feature_names": feature_names,
            "days": days,
        }


def _normalize_splits(splits: dict) -> dict:
    normalized = {}
    for split_name, period in splits.items():
        normalized[split_name] = [str(period[0]), str(period[1])]
    return normalized


def build_qlib_handler_bundle(cfg: dict, splits: dict | None = None) -> dict:
    splits = _normalize_splits(splits or cfg["splits"])
    common_kwargs = {
        "market": cfg["market"],
        "handler": cfg.get("handler", "Alpha158"),
        "label": cfg.get("label", "Ref($close, -2) / Ref($close, -1) - 1"),
        "cache_dir": cfg.get("cache_dir", "~/.qlib_cache"),
        "provider_uri": cfg.get("provider_uri"),
        "region": cfg.get("region", "cn"),
        "fit_start_time": cfg.get("fit_start_time"),
        "fit_end_time": cfg.get("fit_end_time"),
        "infer_processors": cfg.get("infer_processors", []),
        "learn_processors": cfg.get("learn_processors", []),
        "process_type": cfg.get("process_type", "independent"),
        "history_window": int(cfg.get("history_window", 0)),
        "feature_rank_transform": cfg.get("feature_rank_transform"),
        "feature_drop_list": cfg.get("feature_drop_list", []),
    }
    cache_meta = {
        **common_kwargs,
        "splits": splits,
    }
    cache_dir_path = Path(common_kwargs["cache_dir"]).expanduser()
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    bundle_cache_path = cache_dir_path / (
        f"{cfg['market']}_{common_kwargs['handler']}_handler_bundle_{_cache_fingerprint(cache_meta)}.pt"
    )

    if bundle_cache_path.exists():
        bundle = _safe_torch_load(bundle_cache_path)
    else:
        _init_qlib(common_kwargs["provider_uri"], region=common_kwargs["region"])
        if common_kwargs["handler"] == "Alpha158":
            handler_cls = Alpha158Handler
        elif common_kwargs["handler"] == "Alpha360":
            handler_cls = Alpha360Handler
        else:
            raise ValueError(f"Unsupported handler for official pipeline: {common_kwargs['handler']}")

        overall_start = min(period[0] for period in splits.values())
        overall_end = max(period[1] for period in splits.values())
        handler_kwargs = {
            "instruments": cfg["market"],
            "start_time": overall_start,
            "end_time": overall_end,
            "fit_start_time": common_kwargs["fit_start_time"],
            "fit_end_time": common_kwargs["fit_end_time"],
            "infer_processors": common_kwargs["infer_processors"],
            "learn_processors": common_kwargs["learn_processors"],
            "label": ([common_kwargs["label"]], ["LABEL0"]),
        }
        if common_kwargs["handler"] == "Alpha158":
            handler_kwargs["process_type"] = common_kwargs["process_type"]

        handler = handler_cls(**handler_kwargs)
        dataset = DatasetH(handler=handler, segments=splits)
        history_df, _ = _load_history_frame(
            market=cfg["market"],
            start_time=overall_start,
            end_time=overall_end,
            history_window=int(cfg.get("history_window", 0)),
        )

        bundle = {"feature_names": None, "splits": {}}
        for split_name in splits.keys():
            learn_df = dataset.prepare(split_name, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            raw_label_df = dataset.prepare(split_name, col_set=["label"], data_key=DataHandlerLP.DK_R)

            feature_df = learn_df["feature"]
            rank_selection = None
            if common_kwargs["feature_rank_transform"]:
                rank_selection = common_kwargs["feature_rank_transform"].get("features")
            feature_df = _apply_feature_transforms(
                feature_df,
                rank_features=rank_selection,
                drop_features=common_kwargs["feature_drop_list"],
            )
            label_series = _single_series(learn_df["label"])
            raw_label_series = _single_series(raw_label_df["label"]).reindex(learn_df.index)
            if bundle["feature_names"] is None:
                bundle["feature_names"] = [str(col) for col in feature_df.columns]

            days = []
            for date, feature_day in feature_df.groupby(level="datetime", sort=True):
                day_index = feature_day.index
                history_day = None
                if history_df is not None:
                    try:
                        history_day = history_df.xs(date, level="datetime").reindex(
                            feature_day.droplevel("datetime").index
                        )
                    except KeyError:
                        history_day = None
                processed = _make_day_payload(
                    date=pd.Timestamp(date),
                    features=feature_day.droplevel("datetime"),
                    labels=label_series.loc[day_index].droplevel("datetime"),
                    raw_labels=raw_label_series.loc[day_index].droplevel("datetime"),
                    history=history_day,
                    normalize_features=False,
                )
                if processed is not None:
                    days.append(processed)

            if not days:
                raise RuntimeError(
                    f"Official pipeline produced no valid days for split={split_name}, market={cfg['market']}"
                )
            bundle["splits"][split_name] = days

        torch.save(bundle, bundle_cache_path)

    return bundle


def build_qlib_handler_datasets(cfg: dict):
    bundle = build_qlib_handler_bundle(cfg)
    feature_names = bundle["feature_names"]
    train_ds = InMemoryCrossSectionalDataset(bundle["splits"]["train"], feature_names)
    valid_ds = InMemoryCrossSectionalDataset(bundle["splits"]["valid"], feature_names)
    test_ds = InMemoryCrossSectionalDataset(bundle["splits"]["test"], feature_names)
    return train_ds, valid_ds, test_ds


def pad_collate(batch):
    """
    Pad variable-N batches to max N.

    Returns:
      X: [B, N_max, d]
      y: [B, N_max]
      raw_y: [B, N_max]
      mask: [B, N_max]
      dates: list[pd.Timestamp]
      instruments: list[list[str]]
      history: [B, N_max, L] or None
    """

    batch_size = len(batch)
    d_in = batch[0][0].shape[-1]
    max_n = max(item[0].shape[0] for item in batch)

    X = torch.zeros(batch_size, max_n, d_in, dtype=batch[0][0].dtype)
    y = torch.zeros(batch_size, max_n, dtype=batch[0][1].dtype)
    raw_y = torch.zeros(batch_size, max_n, dtype=batch[0][2].dtype)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    dates = []
    instruments = []
    has_history = batch[0][6] is not None
    history = None
    if has_history:
        history_dim = batch[0][6].shape[-1]
        history = torch.zeros(batch_size, max_n, history_dim, dtype=batch[0][6].dtype)

    for idx, (x_i, y_i, raw_y_i, mask_i, date_i, instruments_i, history_i) in enumerate(batch):
        n_i = x_i.shape[0]
        X[idx, :n_i] = x_i
        y[idx, :n_i] = y_i
        raw_y[idx, :n_i] = raw_y_i
        mask[idx, :n_i] = mask_i
        dates.append(date_i)
        instruments.append(instruments_i)
        if history is not None and history_i is not None:
            history[idx, :n_i] = history_i

    return X, y, raw_y, mask, dates, instruments, history
