"""Simple long-only top-K backtest of an ensemble signal.

* Signal at the close of day ``d`` is used to pick the K highest-scoring stocks.
* Holdings are equal-weighted and refreshed every ``rebalance`` days.
* Day-``t`` portfolio return = mean close-to-close return of the holdings over ``t``.
* Trading cost = ``cost`` x (sum of absolute weight changes) charged on rebalance days.
* Benchmark = equal-weight average of all stocks with a valid return that day.

This is an *evaluation* rule only; nothing here is learned.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

TRADING_DAYS = 244


def topk_backtest(signal: torch.Tensor, ret: torch.Tensor, lo: int, hi: int,
                  k: int = 30, rebalance: int = 20, cost: float = 0.001) -> Dict:
    sig = signal.detach().cpu().numpy()
    r = ret.detach().cpu().numpy()
    T, N = r.shape
    w = np.zeros(N)
    port, bench, turnover = [], [], []
    for t in range(lo, min(hi, T)):
        c = 0.0
        if (t - lo) % rebalance == 0 and t - 1 >= 0:
            s = sig[t - 1] if t - 1 < sig.shape[0] else None      # decision uses information through t-1
            if s is not None:
                ok = np.isfinite(s) & (np.abs(s) > 0)
                if ok.sum() >= k:
                    pick = np.argsort(np.where(ok, s, -np.inf))[-k:]
                    new_w = np.zeros(N)
                    new_w[pick] = 1.0 / k
                    turn = np.abs(new_w - w).sum()
                    c = cost * turn
                    turnover.append(turn)
                    w = new_w
        day = np.nan_to_num(r[t], nan=0.0)
        held = w > 0
        port.append(float(day[held].mean()) - c if held.any() else 0.0)
        valid = np.isfinite(r[t])
        bench.append(float(r[t][valid].mean()) if valid.any() else 0.0)

    port, bench = np.array(port), np.array(bench)
    return {"port": port, "bench": bench, **_stats(port, bench), "avg_turnover": float(np.mean(turnover)) if turnover else 0.0}


def _stats(p: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    def ann(x):
        return float((1 + x).prod() ** (TRADING_DAYS / max(len(x), 1)) - 1) if len(x) else float("nan")

    def sharpe(x):
        sd = x.std()
        return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else float("nan")

    def mdd(x):
        cum = np.cumprod(1 + x)
        return float((cum / np.maximum.accumulate(cum) - 1).min()) if len(x) else float("nan")

    return {
        "annual_return": ann(p), "sharpe": sharpe(p), "max_drawdown": mdd(p),
        "bench_annual_return": ann(b), "bench_sharpe": sharpe(b),
        "excess_annual_return": ann(p) - ann(b),
    }
