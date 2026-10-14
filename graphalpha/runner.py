"""Shared run setup + training loops (PPO agent and random-policy baseline)."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import Config
from .env import AlphaEnv
from .market import Market
from .pool import AlphaPool
from .vocab import Vocab

LOG_FIELDS = [
    "timesteps", "episodes", "train_obj", "valid_ic", "valid_rank_ic", "pool_size",
    "mean_return", "frac_valid", "frac_graph", "seconds",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_market(cfg: Config) -> Market:
    cache = os.path.join(os.path.dirname(os.path.abspath(cfg.data_path)), "cache")
    return Market(
        cfg.data_path, cfg.horizon,
        {"train": cfg.train, "valid": cfg.valid, "test": cfg.test},
        cfg.graphs, cfg.graph_k, cfg.graph_window, cfg.graph_step,
        device=cfg.compute_device, cache_dir=cache,
    )


def build_stack(cfg: Config, market: Optional[Market] = None) -> Tuple[Market, Vocab, AlphaPool, AlphaEnv]:
    market = market or build_market(cfg)
    vocab = Vocab(cfg.graphs)
    pool = AlphaPool(market, cfg, vocab)
    env = AlphaEnv(market, pool, vocab, cfg)
    return market, vocab, pool, env


class RunLogger:
    """Writes ``log.csv`` and keeps the pool snapshot with the best validation IC."""

    def __init__(self, run_dir: str, pool: AlphaPool, env: AlphaEnv):
        self.dir, self.pool, self.env = run_dir, pool, env
        self.t0 = time.time()
        self.returns = deque(maxlen=300)
        self.status = deque(maxlen=300)
        self.used_graph = deque(maxlen=300)
        self.best_valid = -1e9
        os.makedirs(run_dir, exist_ok=True)
        self._f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=LOG_FIELDS)
        self._w.writeheader()

    def episode_end(self, ret: float, info: Dict) -> None:
        self.returns.append(ret)
        self.status.append(info.get("status") != "invalid")
        self.used_graph.append("G_" in info.get("formula", ""))

    def log(self, timesteps: int) -> Dict:
        v = self.pool.evaluate_split("valid") if self.pool.size else {}
        row = {
            "timesteps": timesteps,
            "episodes": self.env.episodes,
            "train_obj": self.pool.obj,
            "valid_ic": v.get("ic", float("nan")),
            "valid_rank_ic": v.get("rank_ic", float("nan")),
            "pool_size": self.pool.size,
            "mean_return": float(np.mean(self.returns)) if self.returns else float("nan"),
            "frac_valid": float(np.mean(self.status)) if self.status else float("nan"),
            "frac_graph": float(np.mean(self.used_graph)) if self.used_graph else float("nan"),
            "seconds": round(time.time() - self.t0, 1),
        }
        self._w.writerow(row)
        self._f.flush()
        self.pool.save(os.path.join(self.dir, "pool_latest.json"))
        vic = row["valid_ic"]
        if vic == vic and vic > self.best_valid:
            self.best_valid = vic
            self.pool.save(os.path.join(self.dir, "pool_best_valid.json"))
        return row

    def close(self) -> None:
        self._f.close()


def run_random(cfg: Config, run_dir: str, market: Optional[Market] = None, log_every: int = 1024) -> None:
    """Baseline: uniform-random *valid* actions in the same environment, same step budget."""
    _, _, pool, env = build_stack(cfg, market)
    rng = np.random.default_rng(cfg.seed)
    logger = RunLogger(run_dir, pool, env)
    steps, ep_ret, next_log = 0, 0.0, log_every
    env.reset()
    while steps < cfg.total_timesteps:
        legal = np.flatnonzero(env.action_masks())
        _, r, done, _, info = env.step(int(rng.choice(legal)))
        steps += 1
        ep_ret += r
        if done:
            logger.episode_end(ep_ret, info)
            ep_ret = 0.0
            env.reset()
        if steps >= next_log:
            row = logger.log(steps)
            print(f"[random] step {steps} obj={row['train_obj']:.4f} valid_ic={row['valid_ic']:.4f}", flush=True)
            next_log += log_every
    logger.log(steps)
    logger.close()


def run_ppo(cfg: Config, run_dir: str, market: Optional[Market] = None) -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .policy import policy_kwargs

    market, vocab, pool, env = build_stack(cfg, market)
    logger = RunLogger(run_dir, pool, env)

    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._ret = 0.0

        def _on_step(self) -> bool:
            self._ret += float(self.locals["rewards"][0])
            if bool(self.locals["dones"][0]):
                info = self.locals["infos"][0]
                logger.episode_end(self._ret, info)
                self._ret = 0.0
            return True

        def _on_rollout_end(self) -> None:
            row = logger.log(self.num_timesteps)
            print(
                f"[ppo] step {self.num_timesteps} ep={row['episodes']} obj={row['train_obj']:.4f} "
                f"valid_ic={row['valid_ic']:.4f} pool={row['pool_size']} "
                f"valid%={row['frac_valid']:.2f} graph%={row['frac_graph']:.2f} t={row['seconds']}s",
                flush=True,
            )

    venv = DummyVecEnv([lambda: env])
    model = MaskablePPO(
        "MultiInputPolicy", venv,
        learning_rate=cfg.learning_rate, n_steps=cfg.n_steps, batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef, clip_range=cfg.clip_range,
        policy_kwargs=policy_kwargs(cfg, len(vocab)),
        seed=cfg.seed, device=cfg.policy_device, verbose=0,
    )
    model.learn(total_timesteps=cfg.total_timesteps, callback=Callback())
    logger.log(model.num_timesteps)
    logger.close()
    model.save(os.path.join(run_dir, "model.zip"))


def train(cfg: Config, market: Optional[Market] = None) -> str:
    seed_everything(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))
    if cfg.agent == "random":
        run_random(cfg, run_dir, market)
    else:
        run_ppo(cfg, run_dir, market)
    return run_dir
