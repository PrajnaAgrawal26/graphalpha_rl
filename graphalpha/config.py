"""Experiment configuration (single dataclass, YAML-loadable, CLI-overridable)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Config:
    # ---- bookkeeping -------------------------------------------------
    name: str = "graph_all"
    seed: int = 0
    out_dir: str = "runs"
    agent: str = "ppo"                  # "ppo" | "random"

    # ---- data --------------------------------------------------------
    data_path: str = "data/csi300.npz"
    horizon: int = 20                   # forward-return horizon (days)
    train: List[str] = field(default_factory=lambda: ["2012-01-01", "2019-12-31"])
    valid: List[str] = field(default_factory=lambda: ["2020-01-01", "2021-12-31"])
    test: List[str] = field(default_factory=lambda: ["2022-01-01", "2024-06-30"])

    # ---- graphs ------------------------------------------------------
    graphs: List[str] = field(default_factory=lambda: ["sector", "corr", "lead"])
    graph_k: int = 8                    # neighbours kept per stock
    graph_window: int = 60              # rolling window for dynamic graphs
    graph_step: int = 5                 # graph refresh period (days)

    # ---- formula language / environment ------------------------------
    max_len: int = 15
    curriculum: List[List[Any]] = field(
        default_factory=lambda: [[0, 9, True], [2000, 12, True], [6000, 15, True]]
    )  # rows: [episode_threshold, max_len, graph_ops_allowed]

    # ---- alpha pool / reward -----------------------------------------
    pool_capacity: int = 10
    weighting: str = "ic"               # "ic" | "equal"
    dup_threshold: float = 0.99
    reward_scale: float = 10.0
    beta_div: float = 0.3               # diversity bonus weight
    gamma_len: float = 0.05             # complexity penalty weight
    invalid_penalty: float = 0.5
    dup_penalty: float = 0.2
    dense_reward: bool = False          # rejected candidates still get graded reward (fixes frozen-pool zero signal)
    robust_folds: int = 0               # >=2: reward uses mean(fold IC) - lambda*std(fold IC) over train folds
    robust_lambda: float = 1.0          # penalty on IC instability across train sub-periods
    obj_metric: str = "ic"              # "ic" (Pearson) | "rank_ic" (Spearman Rank IC)
    fast_ast_dedup: bool = True         # O(1) canonical AST hash deduplication before GPU eval

    # ---- policy / PPO ------------------------------------------------
    policy: str = "gat"                 # "gat" | "lstm"
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    total_timesteps: int = 300_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 128
    n_epochs: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    ent_coef: float = 0.02
    clip_range: float = 0.2

    # ---- devices -----------------------------------------------------
    compute_device: str = "auto"        # formula evaluation (GPU if available)
    policy_device: str = "cpu"

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.data_path == "data/csi300.npz" and not os.path.exists(cfg.data_path) and os.path.exists("data/synthetic.npz"):
            cfg.data_path = "data/synthetic.npz"
        return cfg

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[List[str]] = None) -> "Config":
        with open(path, "r") as f:
            cfg = cls.from_dict(yaml.safe_load(f) or {})
        if overrides:
            cfg.apply_overrides(overrides)
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def apply_overrides(self, items: List[str]) -> None:
        """Apply ``key=value`` strings (value parsed as YAML)."""
        valid = {f.name for f in fields(self)}
        for item in items:
            key, _, raw = item.partition("=")
            key = key.strip()
            if key not in valid:
                raise KeyError(f"Unknown config key: {key}")
            setattr(self, key, yaml.safe_load(raw))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def use_graph_ops(self) -> bool:
        return len(self.graphs) > 0

    @property
    def run_name(self) -> str:
        return f"{self.name}_s{self.seed}"
