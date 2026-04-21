from __future__ import annotations

import torch


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(x.dtype)
    denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.where(mask, x, torch.zeros_like(x)).sum(dim=-1, keepdim=True) / denom


def trim_extreme_mask(y: torch.Tensor, mask: torch.Tensor, drop_pct: float) -> torch.Tensor:
    drop_pct = float(drop_pct)
    if drop_pct <= 0.0:
        return mask

    tail_pct = drop_pct / 2.0
    trimmed_rows = []
    for target_row, mask_row in zip(y, mask):
        valid_idx = torch.nonzero(mask_row, as_tuple=False).squeeze(-1)
        n_valid = int(valid_idx.numel())
        if n_valid < 10:
            trimmed_rows.append(mask_row)
            continue

        drop_per_tail = int(n_valid * tail_pct)
        if drop_per_tail <= 0 or (n_valid - 2 * drop_per_tail) < 10:
            trimmed_rows.append(mask_row)
            continue

        valid_targets = target_row[valid_idx]
        order = torch.argsort(valid_targets)
        keep = torch.ones(n_valid, dtype=torch.bool, device=mask_row.device)
        keep[order[:drop_per_tail]] = False
        keep[order[-drop_per_tail:]] = False

        trimmed = torch.zeros_like(mask_row)
        trimmed[valid_idx[keep]] = True
        trimmed_rows.append(trimmed)

    return torch.stack(trimmed_rows, dim=0)


def ic_loss(y_hat, y, mask):
    """
    Cross-sectional IC loss, averaged over batch.

    For each sample in batch:
      1. Select valid entries via mask
      2. Demean both y_hat and y
      3. Compute Pearson correlation

    Return: -mean(IC).
    """

    valid_counts = mask.sum(dim=-1)
    valid_days = valid_counts >= 10
    if not torch.any(valid_days):
        return _zero_loss(y_hat)

    y_hat_mean = _masked_mean(y_hat, mask)
    y_mean = _masked_mean(y, mask)
    y_hat_centered = torch.where(mask, y_hat - y_hat_mean, torch.zeros_like(y_hat))
    y_centered = torch.where(mask, y - y_mean, torch.zeros_like(y))

    numerator = (y_hat_centered * y_centered).sum(dim=-1)
    denom = torch.sqrt(
        y_hat_centered.square().sum(dim=-1) * y_centered.square().sum(dim=-1)
    ).clamp_min(1e-8)
    correlations = numerator / denom
    return -correlations[valid_days].mean()


def wpcc_loss(y_hat, y, mask):
    """
    Weighted Pearson loss with weights proportional to
    |rank(y) - N / 2| inside each cross-section.
    """

    losses = []
    for pred_row, target_row, mask_row in zip(y_hat, y, mask):
        valid_idx = torch.nonzero(mask_row, as_tuple=False).squeeze(-1)
        n_valid = int(valid_idx.numel())
        if n_valid < 10:
            continue

        pred = pred_row[valid_idx]
        target = target_row[valid_idx]
        rank = torch.argsort(torch.argsort(target))
        rank = rank.to(pred.dtype)
        center = (n_valid - 1) / 2.0
        weights = (rank - center).abs()
        weights = weights / weights.sum().clamp_min(1e-8)

        pred_mean = (weights * pred).sum()
        target_mean = (weights * target).sum()
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        numerator = (weights * pred_centered * target_centered).sum()
        denom = torch.sqrt(
            (weights * pred_centered.square()).sum() * (weights * target_centered.square()).sum()
        ).clamp_min(1e-8)
        losses.append(-(numerator / denom))

    if not losses:
        return _zero_loss(y_hat)
    return torch.stack(losses).mean()


def mse_loss(y_hat, y, mask):
    mask_f = mask.to(y_hat.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    squared_error = (y_hat - y).square() * mask_f
    return squared_error.sum() / denom


def build_loss(
    loss_name: str,
    drop_extreme_pct: float = 0.0,
    wpcc_weight: float = 0.0,
    ic_weight: float = 0.0,
):
    loss_name = str(loss_name).lower()
    drop_extreme_pct = float(drop_extreme_pct)
    wpcc_weight = float(wpcc_weight)
    ic_weight = float(ic_weight)

    def loss_fn(y_hat, y, mask):
        effective_mask = trim_extreme_mask(y=y, mask=mask, drop_pct=drop_extreme_pct)
        if loss_name == "ic":
            return (1.0 - wpcc_weight) * ic_loss(y_hat, y, effective_mask) + wpcc_weight * wpcc_loss(
                y_hat, y, effective_mask
            )
        if loss_name == "wpcc":
            return wpcc_loss(y_hat, y, effective_mask)
        if loss_name == "ic_wpcc":
            mix = wpcc_weight if wpcc_weight > 0.0 else 0.2
            return (1.0 - mix) * ic_loss(y_hat, y, effective_mask) + mix * wpcc_loss(
                y_hat, y, effective_mask
            )
        if loss_name == "mse":
            return mse_loss(y_hat, y, effective_mask)
        if loss_name == "mse_ic":
            mix = ic_weight if ic_weight > 0.0 else 0.1
            return mse_loss(y_hat, y, effective_mask) + mix * ic_loss(y_hat, y, effective_mask)
        if loss_name == "mse_ic_wpcc":
            ic_mix = ic_weight if ic_weight > 0.0 else 0.1
            wpcc_mix = wpcc_weight if wpcc_weight > 0.0 else 0.05
            return (
                mse_loss(y_hat, y, effective_mask)
                + ic_mix * ic_loss(y_hat, y, effective_mask)
                + wpcc_mix * wpcc_loss(y_hat, y, effective_mask)
            )
        raise ValueError(f"Unsupported loss: {loss_name}")

    return loss_fn
