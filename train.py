"""Train an agent.

    python train.py --config configs/graph_all.yaml --seed 0
    python train.py --config configs/graph_all.yaml --set total_timesteps=50000 pool_capacity=8
"""
from __future__ import annotations

import argparse

from graphalpha.config import Config
from graphalpha.runner import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config, args.set)
    if args.seed is not None:
        cfg.seed = args.seed
    run_dir = train(cfg)
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
