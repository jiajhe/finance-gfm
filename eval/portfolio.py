from __future__ import annotations

from typing import Sequence

import numpy as np


def _max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / np.maximum(running_max, 1e-12) - 1.0
    return float(np.min(drawdowns))


def _turnover(prev_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    keys = set(prev_weights) | set(new_weights)
    total = 0.0
    for key in keys:
        total += abs(new_weights.get(key, 0.0) - prev_weights.get(key, 0.0))
    return 0.5 * total


def topk_portfolio(preds, labels, masks, dates, k=50, instrument_lists: Sequence[Sequence[str]] | None = None):
    """
    Long-only top-K equal-weight portfolio, rebalanced daily.

    Turnover is reported as 0.5 * sum(|w_t - w_{t-1}|).
    """

    daily_returns = []
    turnovers = []
    prev_weights: dict[str, float] = {}

    for day_idx, (pred, label, mask, _date) in enumerate(zip(preds, labels, masks, dates)):
        pred = np.asarray(pred)
        label = np.asarray(label)
        mask = np.asarray(mask, dtype=bool)

        pred = pred[mask]
        label = label[mask]
        if pred.size == 0:
            continue

        if instrument_lists is None:
            instruments = [str(i) for i in range(pred.size)]
        else:
            day_instruments = np.asarray(list(instrument_lists[day_idx]), dtype=object)
            instruments = day_instruments[mask].tolist() if day_instruments.shape[0] == mask.shape[0] else list(instrument_lists[day_idx])

        k_eff = min(int(k), pred.size)
        order = np.argsort(pred)[::-1][:k_eff]
        selected_returns = label[order]
        daily_returns.append(float(np.mean(selected_returns)))

        weight = 1.0 / k_eff
        new_weights = {str(instruments[idx]): weight for idx in order}
        turnovers.append(_turnover(prev_weights, new_weights) if prev_weights else 0.0)
        prev_weights = new_weights

    if not daily_returns:
        return {
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
        }

    returns = np.asarray(daily_returns, dtype=np.float64)
    equity = np.cumprod(1.0 + returns)
    annual_return = float(equity[-1] ** (252.0 / len(returns)) - 1.0)
    annual_vol = float(np.std(returns, ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252.0)) if len(returns) > 1 and np.std(returns, ddof=1) > 1e-12 else 0.0
    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "turnover": float(np.mean(turnovers)) if turnovers else 0.0,
    }
