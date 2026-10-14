"""Plots for the report."""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_log(run_dir: str) -> Dict[str, np.ndarray]:
    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def _smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    if len(y) < k:
        return y
    pad = np.concatenate([np.full(k - 1, y[0]), y])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def plot_training_curves(runs: Dict[str, List[str]], out: str) -> None:
    """``runs``: label -> list of run dirs (seeds). Mean +- std across seeds."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    panels = [("mean_return", "Mean episode reward"), ("train_obj", "Pool IC (train)"), ("valid_ic", "Pool IC (valid)")]
    for label, dirs in runs.items():
        logs = [_read_log(d) for d in dirs]
        logs = [l for l in logs if l]
        if not logs:
            continue
        n = min(len(l["timesteps"]) for l in logs)
        x = logs[0]["timesteps"][:n]
        for a, (key, title) in zip(ax, panels):
            ys = np.stack([_smooth(l[key][:n]) for l in logs])
            m, s = np.nanmean(ys, 0), np.nanstd(ys, 0)
            a.plot(x, m, label=label, lw=1.6)
            a.fill_between(x, m - s, m + s, alpha=0.15)
            a.set_title(title, fontsize=10)
            a.set_xlabel("environment steps")
    ax[0].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_ablation(results: Dict[str, Dict[str, List[float]]], out: str) -> None:
    """``results``: label -> {metric: [value per seed]} for metrics test_ic, test_rank_ic."""
    labels = list(results)
    metrics = [("valid_ic", "Valid IC"), ("test_ic", "Test IC"), ("test_rank_ic", "Test RankIC")]
    fig, ax = plt.subplots(1, len(metrics), figsize=(12, 3.4))
    for a, (key, title) in zip(ax, metrics):
        means = [np.nanmean(results[l][key]) for l in labels]
        stds = [np.nanstd(results[l][key]) for l in labels]
        a.bar(range(len(labels)), means, yerr=stds, capsize=3, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(labels))))
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        a.set_title(title, fontsize=10)
        a.axhline(0, color="k", lw=0.6)
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_cumulative(curves: Dict[str, np.ndarray], bench: np.ndarray, out: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for label, p in curves.items():
        ax.plot(np.cumprod(1 + p), label=label, lw=1.5)
    ax.plot(np.cumprod(1 + bench), "k--", label="Equal-weight universe", lw=1.2)
    ax.set_xlabel("test trading day")
    ax.set_ylabel("cumulative return (x)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_pool_corr(matrix: np.ndarray, labels: Sequence[str], out: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title or "Pairwise mean daily correlation of pool alphas", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
