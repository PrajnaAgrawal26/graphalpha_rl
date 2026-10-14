"""Gymnasium environment: the agent writes one formula token at a time.

MDP
    state   s_t = (token ids so far, AST edges, top-of-stack roots, context)
    action  a_t = next token (discrete, ~50 choices; invalid ones are masked)
    reward  0 until the formula ends; then pool-based reward (see pool.py)
    episode ends on <SEP> or when the length budget is used up
"""
from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .grammar import TOP_K, FormulaBuilder
from .pool import AlphaPool
from .vocab import Vocab

CTX_DIM = 4


class AlphaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, market, pool: AlphaPool, vocab: Vocab, cfg):
        super().__init__()
        self.mkt, self.pool, self.vocab, self.cfg = market, pool, vocab, cfg
        self.L = cfg.max_len
        self.builder = FormulaBuilder(vocab, max_len=self.L)
        self.episodes = 0
        self.last_info: Dict = {}

        V = len(vocab)
        self.action_space = spaces.Discrete(V)
        self.observation_space = spaces.Dict({
            "tokens": spaces.Box(0.0, float(V), (self.L,), np.float32),
            "adj": spaces.Box(0.0, 1.0, (self.L, self.L), np.float32),
            "top": spaces.Box(-1.0, float(self.L), (TOP_K,), np.float32),
            "ctx": spaces.Box(-10.0, 10.0, (CTX_DIM,), np.float32),
        })
        self._stage_len, self._stage_graph = self.L, True

    # ------------------------------------------------------------------
    def _stage(self):
        """Curriculum: (max formula length, graph operators allowed) for the current episode."""
        if not self.cfg.curriculum:
            return self.L, self.cfg.use_graph_ops
        length, graph = self.L, self.cfg.use_graph_ops
        for threshold, max_len, allow in self.cfg.curriculum:
            if self.episodes >= threshold:
                length, graph = min(int(max_len), self.L), bool(allow) and self.cfg.use_graph_ops
        return length, graph

    def _obs(self) -> Dict[str, np.ndarray]:
        b = self.builder
        tokens = np.zeros(self.L, dtype=np.float32)
        tokens[: len(b.tokens)] = np.asarray(b.tokens, dtype=np.float32) + 1.0
        top = np.asarray(b.top_positions(), dtype=np.float32)
        ctx = np.array([
            (self._stage_len - len(b.tokens)) / self.L,
            1.0 if self._stage_graph else 0.0,
            self.pool.size / max(self.pool.capacity, 1),
            float(np.clip(self.pool.obj * 10.0, -10.0, 10.0)),
        ], dtype=np.float32)
        return {"tokens": tokens, "adj": b.adj.copy(), "top": top, "ctx": ctx}

    def action_masks(self) -> np.ndarray:
        return self.builder.valid_mask()

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._stage_len, self._stage_graph = self._stage()
        self.builder.reset(self._stage_len, self._stage_graph)
        self.episodes += 1
        return self._obs(), {}

    def _finish(self, invalid: bool = False):
        b = self.builder
        if invalid or not b.is_complete():
            from .pool import AddResult
            res = AddResult(-self.cfg.invalid_penalty, "invalid")
            formula = "<incomplete>"
        else:
            node = b.node()
            res = self.pool.try_add(node, len(b), self._stage_len)
            if res.status == "accepted":
                self.pool.set_last_keys(b.keys())
            formula = str(node)
        info = {"formula": formula, "status": res.status, "ic": res.ic,
                "max_mutual": res.max_mutual, "n_tokens": len(b), "parts": res.parts}
        self.last_info = info
        return float(res.reward), info

    def step(self, action: int):
        tok = self.vocab[int(action)]
        mask = self.builder.valid_mask()
        if not mask[int(action)]:                      # should never happen with masking
            r, info = self._finish(invalid=True)
            return self._obs(), r, True, False, info
        if tok.kind == "sep":
            r, info = self._finish()
            return self._obs(), r, True, False, info
        self.builder.push(int(action))
        if len(self.builder) >= self._stage_len:       # budget exhausted: formula is complete by construction
            r, info = self._finish()
            return self._obs(), r, True, False, info
        return self._obs(), 0.0, False, False, {}
