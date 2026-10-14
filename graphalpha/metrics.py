"""Cross-sectional metrics: z-scoring, daily IC, Rank IC."""
from __future__ import annotations

from typing import Dict

import torch

from .ops import rank_rows

NAN = float("nan")


def zscore_rows(a: torch.Tensor, min_valid: int = 20) -> torch.Tensor:
    """Per-day cross-sectional z-score. Rows with too few finite values or no variance -> NaN."""
    finite = torch.isfinite(a)
    n = finite.sum(1, keepdim=True)
    a0 = torch.where(finite, a, torch.zeros_like(a))
    mean = a0.sum(1, keepdim=True) / n.clamp(min=1)
    d = torch.where(finite, a0 - mean, torch.zeros_like(a0))
    std = (d.pow(2).sum(1, keepdim=True) / n.clamp(min=1)).sqrt()
    z = d / std.clamp_min(1e-12)
    bad = (n < min_valid) | (std < 1e-10)
    z = torch.where(finite & ~bad, z, torch.full_like(z, NAN))
    return z


def daily_ic(a: torch.Tensor, y: torch.Tensor, min_obs: int = 20) -> torch.Tensor:
    """Pearson correlation between ``a`` and ``y`` for every day; ``[T]`` with NaN for invalid days."""
    m = torch.isfinite(a) & torch.isfinite(y)
    n = m.sum(1)
    nf = n.clamp(min=1).to(a.dtype)
    a0 = torch.where(m, a, torch.zeros_like(a))
    y0 = torch.where(m, y, torch.zeros_like(y))
    da = torch.where(m, a0 - (a0.sum(1) / nf).unsqueeze(1), torch.zeros_like(a0))
    dy = torch.where(m, y0 - (y0.sum(1) / nf).unsqueeze(1), torch.zeros_like(y0))
    cov = (da * dy).sum(1)
    va = da.pow(2).sum(1)
    vy = dy.pow(2).sum(1)
    ic = cov / (va * vy).sqrt().clamp_min(1e-12)
    bad = (n < min_obs) | (va < 1e-10) | (vy < 1e-10)
    return torch.where(bad, torch.full_like(ic, NAN), ic)


def daily_rank_ic(a: torch.Tensor, y: torch.Tensor, min_obs: int = 20) -> torch.Tensor:
    m = torch.isfinite(a) & torch.isfinite(y)
    a = torch.where(m, a, torch.full_like(a, NAN))
    y = torch.where(m, y, torch.full_like(y, NAN))
    return daily_ic(rank_rows(a), rank_rows(y), min_obs)


def nanmean(x: torch.Tensor) -> float:
    f = torch.isfinite(x)
    if not bool(f.any()):
        return NAN
    return float(x[f].mean())


def summarize(ic: torch.Tensor) -> Dict[str, float]:
    f = torch.isfinite(ic)
    if not bool(f.any()):
        return {"ic": NAN, "icir": NAN, "ic_std": NAN, "pos_frac": NAN, "n_days": 0}
    v = ic[f]
    std = float(v.std(unbiased=False))
    mean = float(v.mean())
    return {
        "ic": mean,
        "ic_std": std,
        "icir": mean / std if std > 1e-12 else NAN,
        "pos_frac": float((v > 0).float().mean()),
        "n_days": int(f.sum()),
    }
