from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-12:
        return np.nan
    return float((x * y).sum() / denom)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return 0.0
    return float(np.nanmean(arr))


def _nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def ic(preds, labels, masks) -> dict:
    """
    preds, labels, masks: list of numpy arrays (one per day), variable length.
    """

    ic_values = []
    rankic_values = []

    for pred, label, mask in zip(preds, labels, masks):
        pred = np.asarray(pred)
        label = np.asarray(label)
        mask = np.asarray(mask, dtype=bool)
        pred = pred[mask]
        label = label[mask]
        if pred.size < 2:
            continue

        ic_values.append(_safe_corr(pred, label))
        rankic_values.append(_safe_corr(rankdata(pred), rankdata(label)))

    ic_mean = _nanmean(ic_values)
    ic_std = _nanstd(ic_values)
    rankic_mean = _nanmean(rankic_values)
    rankic_std = _nanstd(rankic_values)
    return {
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "ICIR": float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0,
        "RankIC_mean": rankic_mean,
        "RankICIR": float(rankic_mean / rankic_std) if rankic_std > 1e-12 else 0.0,
    }
