"""Vectorised operator library.

Every function maps tensors of shape ``[T, N]`` (days x stocks) to ``[T, N]``
and is strictly *causal*: the value at day ``t`` depends only on inputs at days
``<= t``. Missing data is carried as NaN and propagates.

Four families:
  * unary / binary element-wise
  * rolling time-series (window ``w`` over the day axis)
  * pair-rolling (Cov / Corr of two series)
  * graph operators (aggregate over each stock's neighbours on a day-varying graph)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import torch

Tensor = torch.Tensor
Operand = Union[Tensor, float]

NAN = float("nan")
EPS = 1e-8


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _windows(x: Tensor, w: int) -> Tensor:
    """View ``[T, N] -> [T, N, w]`` where slice ``t`` holds ``x[t-w+1 .. t]``.

    The first ``w-1`` days lack a full window and are NaN-padded.
    """
    pad = x.new_full((w - 1, x.shape[1]), NAN)
    return torch.cat([pad, x], dim=0).unfold(0, w, 1)


def _shift(x: Tensor, k: int) -> Tensor:
    """``out[t] = x[t-k]`` (NaN for the first ``k`` rows)."""
    if k <= 0:
        return x
    T = x.shape[0]
    out = x.new_full(x.shape, NAN)
    if k < T:
        out[k:] = x[: T - k]
    return out


def rank_rows(x: Tensor) -> Tensor:
    """Cross-sectional rank in [0, 1] per day; NaN stays NaN."""
    finite = torch.isfinite(x)
    filled = torch.where(finite, x, torch.full_like(x, float("inf")))
    ranks = filled.argsort(dim=1).argsort(dim=1).to(x.dtype)
    n = finite.sum(dim=1, keepdim=True).clamp(min=2).to(x.dtype)
    out = ranks / (n - 1.0)
    return torch.where(finite, out, torch.full_like(x, NAN))


# --------------------------------------------------------------------------
# unary
# --------------------------------------------------------------------------
def op_abs(x: Tensor) -> Tensor:
    return x.abs()


def op_log(x: Tensor) -> Tensor:
    return torch.log(x.abs().clamp_min(1e-6))


def op_csrank(x: Tensor) -> Tensor:
    return rank_rows(x)


# --------------------------------------------------------------------------
# binary (one operand may be a python float constant)
# --------------------------------------------------------------------------
def _as_pair(a: Operand, b: Operand):
    if not torch.is_tensor(a):
        a = torch.as_tensor(a, dtype=b.dtype, device=b.device)
    if not torch.is_tensor(b):
        b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    return a, b


def op_add(a: Operand, b: Operand) -> Tensor:
    a, b = _as_pair(a, b)
    return a + b


def op_sub(a: Operand, b: Operand) -> Tensor:
    a, b = _as_pair(a, b)
    return a - b


def op_mul(a: Operand, b: Operand) -> Tensor:
    a, b = _as_pair(a, b)
    return a * b


def op_div(a: Operand, b: Operand) -> Tensor:
    a, b = _as_pair(a, b)
    safe = torch.where(b.abs() < EPS, torch.full_like(b, NAN), b)
    return a / safe


def op_greater(a: Operand, b: Operand) -> Tensor:
    a, b = _as_pair(a, b)
    return torch.maximum(a, b)


def op_less(a: Operand, b: Operand) -> Tensor:
    a, b = _as_pair(a, b)
    return torch.minimum(a, b)


# --------------------------------------------------------------------------
# rolling
# --------------------------------------------------------------------------
def op_ref(x: Tensor, w: int) -> Tensor:
    return _shift(x, w)


def op_mean(x: Tensor, w: int) -> Tensor:
    return _windows(x, w).mean(dim=-1)


def op_sum(x: Tensor, w: int) -> Tensor:
    return _windows(x, w).sum(dim=-1)


def op_std(x: Tensor, w: int) -> Tensor:
    if w < 2:
        return torch.full_like(x, NAN)
    return _windows(x, w).std(dim=-1, unbiased=False)


def op_max(x: Tensor, w: int) -> Tensor:
    return _windows(x, w).amax(dim=-1)


def op_min(x: Tensor, w: int) -> Tensor:
    return _windows(x, w).amin(dim=-1)


def op_delta(x: Tensor, w: int) -> Tensor:
    return x - _shift(x, w)


def op_wma(x: Tensor, w: int) -> Tensor:
    wts = torch.arange(1, w + 1, dtype=x.dtype, device=x.device)
    wts = wts / wts.sum()
    return (_windows(x, w) * wts).sum(dim=-1)


def op_ema(x: Tensor, w: int) -> Tensor:
    alpha = 2.0 / (w + 1.0)
    # newest observation gets the largest weight
    wts = (1.0 - alpha) ** torch.arange(w - 1, -1, -1, dtype=x.dtype, device=x.device)
    wts = wts / wts.sum()
    return (_windows(x, w) * wts).sum(dim=-1)


# --------------------------------------------------------------------------
# pair rolling
# --------------------------------------------------------------------------
def op_cov(x: Tensor, y: Tensor, w: int) -> Tensor:
    if w < 2:
        return torch.full_like(x, NAN)
    wx, wy = _windows(x, w), _windows(y, w)
    return ((wx - wx.mean(-1, keepdim=True)) * (wy - wy.mean(-1, keepdim=True))).mean(-1)


def op_corr(x: Tensor, y: Tensor, w: int) -> Tensor:
    if w < 2:
        return torch.full_like(x, NAN)
    wx, wy = _windows(x, w), _windows(y, w)
    dx = wx - wx.mean(-1, keepdim=True)
    dy = wy - wy.mean(-1, keepdim=True)
    cov = (dx * dy).mean(-1)
    den = dx.pow(2).mean(-1).sqrt() * dy.pow(2).mean(-1).sqrt()
    return torch.where(den < EPS, torch.full_like(cov, NAN), cov / den)


# --------------------------------------------------------------------------
# graph operators.  ``g`` is a graphs.NeighborGraph exposing ``gather(x)``
# which returns (values [T,N,k], valid [T,N,k]).
# --------------------------------------------------------------------------
def _nbr_stats(x: Tensor, g):
    xg, valid = g.gather(x)
    cnt = valid.sum(-1)
    xz = torch.where(valid, xg, torch.zeros_like(xg))
    return xg, valid, cnt, xz


def op_nbr_mean(x: Tensor, g) -> Tensor:
    _, _, cnt, xz = _nbr_stats(x, g)
    mean = xz.sum(-1) / cnt.clamp(min=1)
    return torch.where(cnt > 0, mean, torch.full_like(mean, NAN))


def op_nbr_rel(x: Tensor, g) -> Tensor:
    return x - op_nbr_mean(x, g)


def op_nbr_std(x: Tensor, g) -> Tensor:
    _, valid, cnt, xz = _nbr_stats(x, g)
    c = cnt.clamp(min=1)
    mean = xz.sum(-1) / c
    var = (torch.where(valid, xz - mean.unsqueeze(-1), torch.zeros_like(xz)) ** 2).sum(-1) / c
    out = var.sqrt()
    return torch.where(cnt >= 2, out, torch.full_like(out, NAN))


def op_nbr_max(x: Tensor, g) -> Tensor:
    xg, valid, cnt, _ = _nbr_stats(x, g)
    out = torch.where(valid, xg, torch.full_like(xg, float("-inf"))).amax(-1)
    return torch.where(cnt > 0, out, torch.full_like(out, NAN))


def op_nbr_rank(x: Tensor, g) -> Tensor:
    """Fraction of neighbours whose value is below the stock's own value."""
    xg, valid, cnt, _ = _nbr_stats(x, g)
    below = (valid & (xg < x.unsqueeze(-1))).sum(-1).to(x.dtype)
    out = below / cnt.clamp(min=1)
    ok = (cnt > 0) & torch.isfinite(x)
    return torch.where(ok, out, torch.full_like(out, NAN))


def op_diffuse(x: Tensor, g) -> Tensor:
    """Two-hop neighbourhood mean (information diffusion)."""
    return op_nbr_mean(op_nbr_mean(x, g), g)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str          # unary | binary | rolling | pair | gop
    fn: Callable


def _specs() -> List[OpSpec]:
    return [
        OpSpec("Abs", "unary", op_abs),
        OpSpec("Log", "unary", op_log),
        OpSpec("CsRank", "unary", op_csrank),
        OpSpec("Add", "binary", op_add),
        OpSpec("Sub", "binary", op_sub),
        OpSpec("Mul", "binary", op_mul),
        OpSpec("Div", "binary", op_div),
        OpSpec("Greater", "binary", op_greater),
        OpSpec("Less", "binary", op_less),
        OpSpec("Ref", "rolling", op_ref),
        OpSpec("Mean", "rolling", op_mean),
        OpSpec("Sum", "rolling", op_sum),
        OpSpec("Std", "rolling", op_std),
        OpSpec("Max", "rolling", op_max),
        OpSpec("Min", "rolling", op_min),
        OpSpec("Delta", "rolling", op_delta),
        OpSpec("WMA", "rolling", op_wma),
        OpSpec("EMA", "rolling", op_ema),
        OpSpec("Cov", "pair", op_cov),
        OpSpec("Corr", "pair", op_corr),
        OpSpec("NbrMean", "gop", op_nbr_mean),
        OpSpec("NbrRel", "gop", op_nbr_rel),
        OpSpec("NbrStd", "gop", op_nbr_std),
        OpSpec("NbrMax", "gop", op_nbr_max),
        OpSpec("NbrRank", "gop", op_nbr_rank),
        OpSpec("Diffuse", "gop", op_diffuse),
    ]


REGISTRY: Dict[str, OpSpec] = {s.name: s for s in _specs()}

FEATURES: List[str] = ["open", "high", "low", "close", "volume", "vwap"]
WINDOWS: List[int] = [1, 5, 10, 20, 40]
CONSTANTS: List[float] = [-5.0, -2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0, 5.0]
