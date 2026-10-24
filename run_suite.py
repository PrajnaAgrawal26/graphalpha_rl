"""Run a full experiment suite (methods x seeds) on ONE dataset, then evaluate.

    python run_suite.py --data data/synthetic_relational.npz --tag rel --seeds 0 1 2 --steps 100000
    python run_suite.py --data data/csi300.npz               --tag real --seeds 0 1 2 --steps 300000

Runs land in ``runs/<tag>_<method>_s<seed>`` and the report in ``reports/<tag>/``.
Finished runs (those with a ``model.zip``/``pool_best_valid.json``) are skipped, so the
command is safe to re-run after an interruption. Data source is irrelevant to this script.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

DEFAULT_METHODS = ["graph_all", "nograph", "sector_only", "corr_only", "lead_only", "random_graph"]


def run_dir(tag: str, method: str, seed: int, out_dir: str) -> str:
    return os.path.join(out_dir, f"{tag}_{method}_s{seed}")


def is_done(d: str) -> bool:
    return os.path.exists(os.path.join(d, "pool_best_valid.json")) and os.path.exists(os.path.join(d, "log.csv"))


def train_cmd(method: str, seed: int, args) -> List[str]:
    overrides = [f"data_path={args.data}", f"name={args.tag}_{method}",
                 f"out_dir={args.out_dir}", f"total_timesteps={args.steps}"] + list(args.set)
    return [sys.executable, "train.py", "--config", f"configs/{method}.yaml",
            "--seed", str(seed), "--set", *overrides]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to a market .npz")
    ap.add_argument("--tag", required=True, help="short experiment tag, e.g. rel | real | mixed")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--report_dir", default=None, help="default: reports/<tag>")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra config overrides")
    ap.add_argument("--jobs", type=int, default=1, help="parallel training processes")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per job")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"data file not found: {args.data}")

    finished: List[str] = []
    todo = []
    for method in args.methods:
        if not os.path.exists(f"configs/{method}.yaml"):
            sys.exit(f"unknown method '{method}' (no configs/{method}.yaml)")
        for seed in args.seeds:
            d = run_dir(args.tag, method, seed, args.out_dir)
            finished.append(d)
            if is_done(d):
                print(f"[suite] skip (done): {d}", flush=True)
                continue
            todo.append((method, seed, d))

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(args.threads)

    def work(item):
        method, seed, d = item
        cmd = train_cmd(method, seed, args)
        print(f"[suite] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            return 0
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, f"{args.tag}_{method}_s{seed}.stdout.log"), "w") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f"[suite] {'ok  ' if rc == 0 else 'FAIL'} {d}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        codes = list(ex.map(work, todo))
    if any(codes):
        print(f"[suite] {sum(1 for c in codes if c)} run(s) failed; evaluating the finished ones", flush=True)
        finished = [d for d in finished if is_done(d)]

    if args.no_eval or args.dry_run:
        return
    report = args.report_dir or os.path.join("reports", args.tag)
    cmd = [sys.executable, "evaluate.py", *finished, "--out", report]
    print(f"[suite] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
