"""Formula trees and their tensor evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch

from .ops import FEATURES, REGISTRY

_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}


@dataclass(frozen=True)
class Node:
    """A node of a formula tree.

    kind: ``feat`` | ``const`` | ``win`` | ``graph`` | ``op``
    """

    kind: str
    name: str = ""
    value: float = 0.0
    args: Tuple["Node", ...] = ()

    def __str__(self) -> str:
        if self.kind == "feat":
            return f"${self.name}"
        if self.kind == "const":
            return f"{self.value:g}"
        if self.kind == "win":
            return f"{int(self.value)}"
        if self.kind == "graph":
            return f"G_{self.name}"
        return f"{self.name}({', '.join(str(a) for a in self.args)})"

    __repr__ = __str__

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def uses_graph(self) -> bool:
        return self.kind == "graph" or any(a.uses_graph() for a in self.args)


def evaluate(node: Node, market, upto: int) -> Union[torch.Tensor, float]:
    """Evaluate ``node`` on days ``[0, upto)``; returns ``[upto, N]`` (or a float for constants)."""
    kind = node.kind
    if kind == "feat":
        return market.feat[_FEATURE_INDEX[node.name], :upto]
    if kind == "const":
        return float(node.value)
    if kind != "op":
        raise ValueError(f"cannot evaluate a bare '{kind}' node")

    spec = REGISTRY[node.name]
    a = node.args
    if spec.kind == "unary":
        return spec.fn(evaluate(a[0], market, upto))
    if spec.kind == "binary":
        return spec.fn(evaluate(a[0], market, upto), evaluate(a[1], market, upto))
    if spec.kind == "rolling":
        return spec.fn(evaluate(a[0], market, upto), int(a[1].value))
    if spec.kind == "pair":
        return spec.fn(evaluate(a[0], market, upto), evaluate(a[1], market, upto), int(a[2].value))
    if spec.kind == "gop":
        x = evaluate(a[0], market, upto)
        return spec.fn(x, market.graph(a[1].name))
    raise ValueError(f"unknown op kind {spec.kind}")
