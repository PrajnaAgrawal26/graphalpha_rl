"""Stock-relationship graphs.

Every graph is stored as a *top-k neighbour list per stock per refresh chunk*:

    idx   [S, N, k]  long   neighbour indices
    mask  [S, N, k]  bool   which of the k slots are real neighbours
    day2chunk [T]    long   which chunk is active on each day

``corr`` and ``lead`` are dynamic graphs rebuilt every ``step`` days from a
rolling window of returns.  The chunk that serves day ``t`` is built from
returns up to ``t-1`` only, so graph-based alphas cannot peek ahead.

  * ``sector`` : static; same-industry peers, ranked by train-period correlation
  * ``corr``   : dynamic; most positively correlated stocks (symmetric co-movement)
  * ``lead``   : dynamic; stocks whose *yesterday* return best predicts my *today* return
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional

import numpy as np
import torch

NEG_INF = float("-inf")


class NeighborGraph:
    def __init__(self, name: str, idx: torch.Tensor, mask: torch.Tensor, day2chunk: torch.Tensor):
        self.name = name
        self.idx = idx
        self.mask = mask
        self.day2chunk = day2chunk

    @property
    def k(self) -> int:
        return self.idx.shape[-1]

    def to(self, device) -> "NeighborGraph":
        return NeighborGraph(
            self.name, self.idx.to(device), self.mask.to(device), self.day2chunk.to(device)
        )

    def gather(self, x: torch.Tensor):
        """Neighbour values for each (day, stock): returns ``(vals [T,N,k], valid [T,N,k])``."""
        T = x.shape[0]
        chunk = self.day2chunk[:T]
        idx = self.idx[chunk]                       # [T, N, k]
        msk = self.mask[chunk]
        rows = torch.arange(T, device=x.device).view(T, 1, 1)
        vals = x[rows, idx]
        return vals, msk & torch.isfinite(vals)

    def neighbor_counts(self) -> torch.Tensor:
        return self.mask.sum(-1)


# --------------------------------------------------------------------------
def _standardize(R: torch.Tensor) -> torch.Tensor:
    """Column-standardise a ``[W, N]`` return window; NaNs become 0."""
    valid = torch.isfinite(R)
    R0 = torch.where(valid, R, torch.zeros_like(R))
    n = valid.sum(0).clamp(min=1).to(R.dtype)
    mean = R0.sum(0) / n
    D = torch.where(valid, R0 - mean, torch.zeros_like(R0))
    std = (D.pow(2).sum(0) / n).sqrt().clamp_min(1e-8)
    return D / std


def _topk(score: torch.Tensor, k: int, min_score: float):
    vals, ids = score.topk(k, dim=1)
    return ids, torch.isfinite(vals) & (vals > min_score)


def build_dynamic_graph(
    ret: torch.Tensor, mode: str, window: int, step: int, k: int, min_corr: float = 0.0
) -> NeighborGraph:
    """Rolling correlation (``mode='corr'``) or lagged-correlation (``mode='lead'``) graph."""
    assert mode in ("corr", "lead")
    T, N = ret.shape
    starts = list(range(window, T, step))
    S = len(starts) + 1                              # chunk 0 = "no graph yet"
    idx = torch.zeros(S, N, k, dtype=torch.long)
    mask = torch.zeros(S, N, k, dtype=torch.bool)
    day2chunk = torch.zeros(T, dtype=torch.long)

    for s, d in enumerate(starts, start=1):
        R = ret[d - window:d]                        # returns of days d-W .. d-1  (causal)
        ok = torch.isfinite(R).sum(0) >= int(0.8 * window)
        Z = _standardize(R)
        if mode == "corr":
            C = (Z.T @ Z) / window                   # [i, j]
        else:
            C = ((Z[:-1].T @ Z[1:]) / (window - 1)).T  # [follower i, leader j]
        C = C.clone()
        C.fill_diagonal_(NEG_INF)
        C[:, ~ok] = NEG_INF
        C[~ok, :] = NEG_INF
        ids, m = _topk(C, k, min_corr)
        idx[s], mask[s] = ids, m
        day2chunk[d:d + step] = s
    return NeighborGraph(mode, idx, mask, day2chunk)


def build_sector_graph(
    ret_train: torch.Tensor, sector_ids: np.ndarray, k: int, min_obs: int = 120
) -> NeighborGraph:
    T, N = ret_train.shape
    Z = _standardize(ret_train)
    C = (Z.T @ Z) / max(T, 1)
    sec = torch.as_tensor(sector_ids, dtype=torch.long)
    same = sec.view(-1, 1) == sec.view(1, -1)
    ok = torch.isfinite(ret_train).sum(0) >= min_obs
    C = torch.where(same, C, torch.full_like(C, NEG_INF)).clone()
    C.fill_diagonal_(NEG_INF)
    C[:, ~ok] = NEG_INF
    C[~ok, :] = NEG_INF
    ids, m = _topk(C, k, -1.0)
    return NeighborGraph("sector", ids.unsqueeze(0), m.unsqueeze(0), torch.zeros(0, dtype=torch.long))


def build_graphs(
    ret: torch.Tensor,
    sector_ids: np.ndarray,
    names: List[str],
    train_hi: int,
    window: int,
    step: int,
    k: int,
    cache_dir: Optional[str] = None,
    cache_key: str = "",
) -> Dict[str, NeighborGraph]:
    """Build (or load from cache) all requested graphs on CPU."""
    T = ret.shape[0]
    sig = hashlib.md5(f"{cache_key}|{T}|{ret.shape[1]}|{window}|{step}|{k}|{train_hi}".encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"graphs_{sig}.pt") if cache_dir else None

    cached: Dict[str, NeighborGraph] = {}
    if path and os.path.exists(path):
        blob = torch.load(path)
        for n, d in blob.items():
            cached[n] = NeighborGraph(n, d["idx"], d["mask"], d["day2chunk"])

    changed = False
    for name in names:
        if name in cached:
            continue
        if name == "sector":
            g = build_sector_graph(ret[:train_hi], sector_ids, k)
        elif name in ("corr", "lead"):
            g = build_dynamic_graph(ret, name, window, step, k)
        else:
            raise ValueError(f"unknown graph '{name}'")
        if name == "sector":                         # static: same chunk every day
            g.day2chunk = torch.zeros(T, dtype=torch.long)
        cached[name] = g
        changed = True

    if path and changed:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {n: {"idx": g.idx, "mask": g.mask, "day2chunk": g.day2chunk} for n, g in cached.items()},
            path,
        )
    return {n: cached[n] for n in names}
