"""Evaluate trained runs on validation/test, aggregate seeds, and make the report figures.

    python evaluate.py runs/graph_all_s0 runs/graph_all_s1 runs/nograph_s0 ... --out reports

Runs are grouped by their config ``name``; metrics are averaged over seeds.
The pool snapshot with the best *validation* IC is used; the test set is touched only here.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch

from graphalpha.backtest import topk_backtest
from graphalpha.config import Config
from graphalpha.metrics import daily_ic, nanmean
from graphalpha.plotting import (plot_ablation, plot_cumulative, plot_pool_corr,
                                 plot_training_curves)
from graphalpha.pool import AlphaPool
from graphalpha.runner import build_market
from graphalpha.vocab import Vocab


def evaluate_run(run_dir: str, market_cache: Dict) -> Dict:
    cfg = Config.from_json(os.path.join(run_dir, "config.json"))
    key = (cfg.data_path, tuple(cfg.graphs), cfg.graph_k, cfg.graph_window, cfg.graph_step,
           tuple(cfg.train), tuple(cfg.valid), tuple(cfg.test), cfg.horizon)
    if key not in market_cache:
        market_cache[key] = build_market(cfg)
    market = market_cache[key]
    vocab = Vocab(cfg.graphs)
    pool = AlphaPool.load(market, cfg, vocab, os.path.join(run_dir, "pool_best_valid.json"))

    res = {"name": cfg.name, "seed": cfg.seed, "pool_size": pool.size, "train_obj": pool.obj}
    for split in ("train", "valid", "test"):
        m = pool.evaluate_split(split)
        res[f"{split}_ic"], res[f"{split}_rank_ic"], res[f"{split}_icir"] = m["ic"], m["rank_ic"], m["icir"]

    lo, hi = market.ic_slice("test")
    sig = pool.signal("test")
    bt = topk_backtest(sig, market.ret, lo, hi, k=30, rebalance=cfg.horizon)
    for k, v in bt.items():
        if k not in ("port", "bench"):
            res[f"bt_{k}"] = v

    # diversity: mean |pairwise corr| of the pool alphas on test
    sigs = [s[lo:hi] for s in pool.member_signals("test")]
    n = len(sigs)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = nanmean(daily_ic(sigs[i], sigs[j]))
            corr[i, j] = corr[j, i] = 0.0 if c != c else c
    res["pool_mean_abs_corr"] = float(np.abs(corr[np.triu_indices(n, 1)]).mean()) if n > 1 else float("nan")
    res["uses_graph"] = int(sum(m.node.uses_graph() for m in pool.members))
    res["formulas"] = [{"formula": str(m.node), "weight": float(w), "train_ic": m.ic}
                       for m, w in zip(pool.members, pool.weights())]

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({k: v for k, v in res.items()}, f, indent=2)
    res["_port"], res["_bench"], res["_corr"] = bt["port"], bt["bench"], corr
    return res


def _ms(vals: List[float]) -> str:
    v = np.array(vals, dtype=float)
    return f"{np.nanmean(v):+.4f} ± {np.nanstd(v):.4f}" if len(v) > 1 else f"{v[0]:+.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    cache: Dict = {}
    results = [evaluate_run(r, cache) for r in args.runs]
    groups: "OrderedDict[str, List[Dict]]" = OrderedDict()
    for r in results:
        groups.setdefault(r["name"], []).append(r)
    dirs: "OrderedDict[str, List[str]]" = OrderedDict()
    for r, d in zip(results, args.runs):
        dirs.setdefault(r["name"], []).append(d)

    cols = [("valid_ic", "Valid IC"), ("test_ic", "Test IC"), ("test_rank_ic", "Test RankIC"),
            ("test_icir", "Test ICIR"), ("bt_sharpe", "Sharpe"), ("bt_annual_return", "AnnRet"),
            ("bt_max_drawdown", "MaxDD"), ("pool_mean_abs_corr", "|corr|"), ("uses_graph", "#graph")]
    lines = ["| Method | seeds | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---|" + "|".join("---" for _ in cols) + "|"]
    for name, rs in groups.items():
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(_ms([r[c[0]] for r in rs]) for c in cols) + " |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "results_table.md"), "w") as f:
        f.write(table + "\n")
    print(table)

    flines = []
    for name, rs in groups.items():
        best = max(rs, key=lambda r: r["valid_ic"] if r["valid_ic"] == r["valid_ic"] else -9)
        flines.append(f"### {name} (seed {best['seed']}, valid IC {best['valid_ic']:+.4f}, test IC {best['test_ic']:+.4f})")
        for m in sorted(best["formulas"], key=lambda m: -abs(m["weight"])):
            flines.append(f"- `{m['formula']}`  (w={m['weight']:+.3f}, train IC={m['train_ic']:+.4f})")
        flines.append("")
    with open(os.path.join(args.out, "formulas.md"), "w") as f:
        f.write("\n".join(flines))

    fig = os.path.join(args.out, "figures")
    plot_training_curves(dirs, os.path.join(fig, "training_curves.png"))
    plot_ablation({n: {k: [r[k] for r in rs] for k in ("valid_ic", "test_ic", "test_rank_ic")} for n, rs in groups.items()},
                  os.path.join(fig, "ablation.png"))
    curves, bench = {}, None
    for name, rs in groups.items():
        best = max(rs, key=lambda r: r["valid_ic"] if r["valid_ic"] == r["valid_ic"] else -9)
        curves[name], bench = best["_port"], best["_bench"]
        n = best["_corr"].shape[0]
        plot_pool_corr(best["_corr"], [f"a{i+1}" for i in range(n)],
                       os.path.join(fig, f"pool_corr_{name}.png"), title=f"{name}: pool alpha correlation (test)")
    if bench is not None:
        plot_cumulative(curves, bench, os.path.join(fig, "cumulative_return.png"))
    print(f"\nSaved table, formulas and figures under {args.out}/")


if __name__ == "__main__":
    main()
