"""Alpha pool: the part of the environment that turns a finished formula into a reward.

The pool keeps at most ``capacity`` alphas. Each member is z-scored cross-sectionally
every day and combined with *fixed, parameter-free weights* (``w_i = IC_i``: signed, so a
negatively-predictive alpha is automatically flipped). No regression is fitted — the only
thing that learns anywhere in this project is the RL policy.

Adding a candidate:
  1. invalid / constant / mostly-NaN                  -> penalty
  2. near-duplicate of a member (|mutual IC| > thr)   -> penalty
  3. otherwise tentatively add; if over capacity drop the member whose removal hurts the
     ensemble IC least (possibly the newcomer itself).
Reward = scale * (Delta ensemble IC + beta * novelty bonus) - gamma * length penalty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .expr import Node, evaluate
from .grammar import build_from_keys
from .metrics import daily_ic, daily_rank_ic, nanmean, summarize, zscore_rows
from .vocab import Vocab


@dataclass
class Member:
    node: Node
    keys: List[str]
    ic: float
    z: torch.Tensor                      # filled z-score on the train slice [T_tr, N]


@dataclass
class AddResult:
    reward: float
    status: str                          # invalid | duplicate | rejected | accepted
    ic: float = float("nan")
    max_mutual: float = float("nan")
    delta_obj: float = 0.0
    parts: Dict[str, float] = field(default_factory=dict)


class AlphaPool:
    def __init__(self, market, cfg, vocab: Vocab):
        self.mkt = market
        self.cfg = cfg
        self.vocab = vocab
        self.capacity = cfg.pool_capacity
        self.lo, self.hi = market.ic_slice("train")
        self.y = market.target[self.lo:self.hi]
        self.members: List[Member] = []
        self.obj = 0.0                   # ensemble IC on train slice
        self.eval_count = 0

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.members)

    def _weights(self, ics: List[float]) -> torch.Tensor:
        w = torch.tensor(ics, dtype=torch.float32, device=self.mkt.device)
        return torch.sign(w) if self.cfg.weighting == "equal" else w

    def weights(self) -> np.ndarray:
        if not self.members:
            return np.zeros(0)
        return self._weights([m.ic for m in self.members]).cpu().numpy()

    def _objective(self, ensemble: torch.Tensor) -> float:
        v = nanmean(daily_ic(ensemble, self.y))
        return 0.0 if v != v else v

    # ------------------------------------------------------------------
    def _candidate(self, node: Node):
        try:
            a = evaluate(node, self.mkt, self.hi)
        except Exception:
            return None
        if not torch.is_tensor(a) or tuple(a.shape) != (self.hi, self.mkt.N):
            return None
        z = zscore_rows(a[self.lo:self.hi])
        if float(torch.isfinite(z).float().mean()) < 0.3:
            return None
        ic = nanmean(daily_ic(z, self.y))
        if ic != ic:
            return None
        return z, ic

    def try_add(self, node: Node, n_tokens: int, budget: int) -> AddResult:
        cfg = self.cfg
        self.eval_count += 1
        cand = self._candidate(node)
        if cand is None:
            return AddResult(-cfg.invalid_penalty, "invalid")
        z, ic = cand
        zf = torch.nan_to_num(z, nan=0.0)

        mutual = [abs(nanmean(daily_ic(zf, m.z))) for m in self.members]
        mutual = [0.0 if v != v else v for v in mutual]
        max_mut = max(mutual) if mutual else 0.0
        if max_mut > cfg.dup_threshold:
            return AddResult(-cfg.dup_penalty, "duplicate", ic, max_mut)

        ics = [m.ic for m in self.members] + [ic]
        Z = torch.stack([m.z for m in self.members] + [zf])
        w = self._weights(ics)
        ens = torch.einsum("k,ktn->tn", w, Z)

        drop: Optional[int] = None
        if len(ics) <= self.capacity:
            obj_new = self._objective(ens)
            retained = True
        else:
            best_obj, best_j = -1e9, -1
            for j in range(len(ics)):
                o = self._objective(ens - w[j] * Z[j])
                if o > best_obj:
                    best_obj, best_j = o, j
            if best_j == len(ics) - 1:       # newcomer is the weakest link -> pool unchanged
                obj_new, retained = self.obj, False
            else:
                obj_new, retained, drop = best_obj, True, best_j

        delta = obj_new - self.obj
        novelty = cfg.beta_div * (1.0 - max_mut) * abs(ic) if retained else 0.0
        length_pen = cfg.gamma_len * n_tokens / max(budget, 1)
        reward = cfg.reward_scale * (delta + novelty) - length_pen

        if retained:
            if drop is not None:
                del self.members[drop]
            self.members.append(Member(node, [], ic, zf))   # keys filled by caller
            self.obj = obj_new
        parts = {"delta": cfg.reward_scale * delta, "novelty": cfg.reward_scale * novelty, "length": -length_pen}
        return AddResult(reward, "accepted" if retained else "rejected", ic, max_mut, delta, parts)

    def set_last_keys(self, keys: List[str]) -> None:
        if self.members:
            self.members[-1].keys = list(keys)

    # ------------------------------------------------------------------
    # evaluation on any split (weights stay frozen at their training values)
    # ------------------------------------------------------------------
    def member_signals(self, split: str) -> List[torch.Tensor]:
        lo, hi = self.mkt.ic_slice(split)
        return [
            torch.nan_to_num(zscore_rows(evaluate(m.node, self.mkt, hi)[:hi]), nan=0.0)
            for m in self.members
        ]

    def signal(self, split: str) -> torch.Tensor:
        """Ensemble signal ``[hi, N]`` for days ``[0, hi)`` of the split's scoring window."""
        sigs = self.member_signals(split)
        w = self._weights([m.ic for m in self.members])
        return torch.einsum("k,ktn->tn", w, torch.stack(sigs))

    def evaluate_split(self, split: str) -> Dict[str, float]:
        if not self.members:
            return {"ic": float("nan"), "rank_ic": float("nan"), "icir": float("nan"), "n_days": 0}
        lo, hi = self.mkt.ic_slice(split)
        sig = self.signal(split)[lo:hi]
        y = self.mkt.target[lo:hi]
        out = summarize(daily_ic(sig, y))
        out["rank_ic"] = nanmean(daily_rank_ic(sig, y))
        return out

    # ------------------------------------------------------------------
    def to_json(self) -> Dict:
        return {
            "objective_train": self.obj,
            "capacity": self.capacity,
            "weighting": self.cfg.weighting,
            "members": [
                {"formula": str(m.node), "keys": m.keys, "train_ic": m.ic, "weight": float(w)}
                for m, w in zip(self.members, self.weights())
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, market, cfg, vocab: Vocab, path: str) -> "AlphaPool":
        """Rebuild a pool from JSON; member ICs are recomputed on the train slice."""
        with open(path) as f:
            blob = json.load(f)
        pool = cls(market, cfg, vocab)
        for item in blob["members"]:
            node = build_from_keys(vocab, item["keys"])
            cand = pool._candidate(node)
            if cand is None:
                continue
            z, ic = cand
            pool.members.append(Member(node, item["keys"], ic, torch.nan_to_num(z, nan=0.0)))
        if pool.members:
            ens = torch.einsum(
                "k,ktn->tn", pool._weights([m.ic for m in pool.members]),
                torch.stack([m.z for m in pool.members]),
            )
            pool.obj = pool._objective(ens)
        return pool
