"""Stack grammar for postfix formula construction + invalid-action masking.

Formulas are emitted in *postfix* (operands first, operator last), e.g.::

    $close 5 Delta G_sector NbrRel $volume 10 Corr
        = Corr(NbrRel(Delta($close, 5), G_sector), $volume, 10)

The builder tracks a stack of typed items:
    expr   : a feature-carrying subtree (the only valid final result)
    const  : a scalar constant
    win    : a rolling window length  (consumed by a rolling / pair operator)
    graph  : a graph identifier       (consumed by a graph operator)

``valid_mask`` returns exactly the tokens that keep the formula completable
inside the length budget, so the agent can never emit a malformed formula.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .expr import Node
from .vocab import Vocab

TOP_K = 3  # how many stack roots are exposed to the policy


@dataclass
class _Item:
    type: str        # expr | const | win | graph
    node: Node
    pos: int         # position of the item's root token in the sequence


class FormulaBuilder:
    def __init__(self, vocab: Vocab, max_len: int = 15, allow_graph: bool = True):
        self.vocab = vocab
        self.cap = max_len                 # storage capacity (largest curriculum stage)
        self.reset(max_len, allow_graph)

    # ------------------------------------------------------------------
    def reset(self, max_len: Optional[int] = None, allow_graph: bool = True) -> None:
        if max_len is not None:
            self.limit = max_len
        self.allow_graph = allow_graph
        self.tokens: List[int] = []
        self.stack: List[_Item] = []
        self.adj = np.zeros((self.cap, self.cap), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.tokens)

    # ------------------------------------------------------------------
    def _types(self) -> List[str]:
        return [it.type for it in self.stack]

    def is_complete(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].type == "expr"

    def node(self) -> Node:
        if not self.is_complete():
            raise ValueError("formula is not complete")
        return self.stack[0].node

    def top_positions(self) -> List[int]:
        """Token positions of the top ``TOP_K`` stack roots (top first), ``-1`` padded."""
        pos = [it.pos for it in reversed(self.stack[-TOP_K:])]
        return pos + [-1] * (TOP_K - len(pos))

    # ------------------------------------------------------------------
    def valid_mask(self) -> np.ndarray:
        v = self.vocab
        mask = np.zeros(len(v), dtype=bool)
        types = self._types()
        m = len(types)
        top = types[-1] if m else None
        used = len(self.tokens)
        has_expr = "expr" in types

        def fits(m_after: int, expr_after: bool) -> bool:
            needed = (m_after - 1) + (0 if expr_after else 2)
            return used + 1 + needed <= self.limit

        for tok in v.tokens:
            k = tok.kind
            ok = False
            if k == "sep":
                ok = m == 1 and top == "expr"
            elif k == "feat":
                ok = (top in (None, "expr", "const")) and fits(m + 1, True)
            elif k == "const":
                ok = (top is None or top == "expr") and fits(m + 1, has_expr)
            elif k == "win":
                ok = top == "expr" and fits(m + 1, True)
            elif k == "graph":
                ok = self.allow_graph and top == "expr" and fits(m + 1, True)
            elif k == "unary":
                ok = top == "expr" and fits(m, True)
            elif k == "binary":
                ok = (
                    m >= 2
                    and types[-1] in ("expr", "const")
                    and types[-2] in ("expr", "const")
                    and not (types[-1] == "const" and types[-2] == "const")
                    and fits(m - 1, True)
                )
            elif k == "rolling":
                ok = m >= 2 and types[-1] == "win" and types[-2] == "expr" and fits(m - 1, True)
            elif k == "pair":
                ok = (
                    m >= 3 and types[-1] == "win"
                    and types[-2] == "expr" and types[-3] == "expr" and fits(m - 2, True)
                )
            elif k == "gop":
                ok = (
                    self.allow_graph and m >= 2 and types[-1] == "graph"
                    and types[-2] == "expr" and fits(m - 1, True)
                )
            mask[tok.id] = ok
        return mask

    # ------------------------------------------------------------------
    def push(self, token_id: int) -> None:
        tok = self.vocab[token_id]
        pos = len(self.tokens)
        if pos >= self.cap:
            raise ValueError("token sequence exceeds builder capacity")
        k = tok.kind
        if k == "sep":
            raise ValueError("SEP is not part of the formula; end the episode instead")
        if k == "feat":
            self.stack.append(_Item("expr", Node("feat", tok.name), pos))
        elif k == "const":
            self.stack.append(_Item("const", Node("const", value=tok.value), pos))
        elif k == "win":
            self.stack.append(_Item("win", Node("win", value=tok.value), pos))
        elif k == "graph":
            self.stack.append(_Item("graph", Node("graph", name=tok.name), pos))
        else:
            arity = {"unary": 1, "binary": 2, "rolling": 2, "gop": 2, "pair": 3}[k]
            if len(self.stack) < arity:
                raise ValueError(f"operator {tok.name} needs {arity} operands")
            kids = self.stack[-arity:]
            del self.stack[-arity:]
            for kid in kids:
                self.adj[pos, kid.pos] = 1.0
            self.stack.append(_Item("expr", Node("op", tok.name, args=tuple(c.node for c in kids)), pos))
        self.tokens.append(token_id)

    # ------------------------------------------------------------------
    def keys(self) -> List[str]:
        return [self.vocab.key(self.vocab[i]) for i in self.tokens]


def build_from_keys(vocab: Vocab, keys: Sequence[str], max_len: int = 64) -> Node:
    """Rebuild a formula tree from saved token keys (no masking, used when loading pools)."""
    b = FormulaBuilder(vocab, max_len=max_len, allow_graph=True)
    for k in keys:
        b.push(vocab.lookup(k).id)
    return b.node()
