"""Synthetic market with built-in cross-stock structure (for tests and offline smoke runs).

Planted structure so that graph operators *can* matter:
  * sector factors      -> stocks in a sector co-move
  * leader -> follower  -> a follower's return today loads on its leader's idiosyncratic return yesterday
  * sector mean reversion -> a stock that outran its sector over 5 days drifts back
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


def make_synthetic(
    path: Optional[str] = None,
    n_stocks: int = 100,
    start: str = "2011-01-03",
    end: str = "2024-12-31",
    n_sectors: int = 10,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T, N = len(dates), n_stocks

    sector = rng.permutation(np.arange(N) % n_sectors)
    beta = rng.uniform(0.6, 1.4, N)
    leaders = rng.choice(N, size=max(N // 5, 2), replace=False)
    leader_of = rng.choice(leaders, size=N)
    leader_of[leaders] = leaders                       # leaders follow themselves (no effect)
    is_follower = ~np.isin(np.arange(N), leaders)

    mkt = rng.normal(0.0003, 0.008, T)
    sec = rng.normal(0.0, 0.006, (T, n_sectors))
    idio = rng.normal(0.0, 0.012, (T, N))
    ret = np.zeros((T, N))
    rel = np.zeros((T, N))                              # return relative to sector mean
    for t in range(T):
        base = beta * mkt[t] + sec[t, sector] + idio[t]
        if t >= 1:
            base = base + np.where(is_follower, 0.30 * idio[t - 1, leader_of], 0.0)
        if t >= 6:
            drift = rel[t - 5:t].sum(axis=0)
            base = base - 0.04 * drift
        ret[t] = base
        sec_mean = np.array([ret[t, sector == s].mean() if (sector == s).any() else 0.0 for s in range(n_sectors)])
        rel[t] = ret[t] - sec_mean[sector]

    close = 10.0 * np.cumprod(1.0 + ret, axis=0)
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * (1.0 + rng.normal(0, 0.002, (T, N)))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, (T, N))))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, (T, N))))
    vol = np.exp(rng.normal(13.0, 0.4, (1, N))) * (1.0 + 25.0 * np.abs(ret)) * np.exp(rng.normal(0, 0.2, (T, N)))
    vwap = (high + low + close) / 3.0

    # staggered "IPO" dates -> leading NaNs
    late = rng.choice(N, size=max(N // 10, 1), replace=False)
    for i in late:
        k = int(rng.integers(200, 800))
        for arr in (open_, high, low, close, vol, vwap):
            arr[:k, i] = np.nan

    out = dict(
        dates=dates.strftime("%Y-%m-%d").values.astype("U10"),
        codes=np.array([f"SYN{i:03d}" for i in range(N)]),
        sector_ids=sector.astype(np.int64),
        sector_names=np.array([f"S{s}" for s in range(n_sectors)]),
        open=open_.astype(np.float32), high=high.astype(np.float32), low=low.astype(np.float32),
        close=close.astype(np.float32), volume=vol.astype(np.float32), vwap=vwap.astype(np.float32),
    )
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **out)
    return out
