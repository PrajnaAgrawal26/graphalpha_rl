import numpy as np
import pytest
import torch

from graphalpha import ops
from graphalpha.expr import evaluate
from graphalpha.grammar import build_from_keys
from graphalpha.graphs import build_dynamic_graph
from graphalpha.vocab import Vocab


@pytest.mark.parametrize("name,w", [("Mean", 5), ("Std", 10), ("Max", 5), ("Min", 5), ("Delta", 5),
                                    ("Ref", 5), ("WMA", 10), ("EMA", 10), ("Sum", 5)])
def test_rolling_ops_are_causal(name, w):
    torch.manual_seed(0)
    x = torch.randn(80, 12)
    fn = ops.REGISTRY[name].fn
    base = fn(x, w)
    y = x.clone()
    y[50:] = torch.randn(30, 12)                      # change the future
    pert = fn(y, w)
    assert torch.allclose(base[:50], pert[:50], equal_nan=True)


def test_pair_ops_causal_and_corr_bounded():
    torch.manual_seed(0)
    x, y = torch.randn(80, 12), torch.randn(80, 12)
    c = ops.op_corr(x, y, 10)
    fin = c[torch.isfinite(c)]
    assert fin.numel() > 0 and bool((fin.abs() <= 1 + 1e-5).all())
    z = y.clone()
    z[50:] = 0
    assert torch.allclose(c[:50], ops.op_corr(x, z, 10)[:50], equal_nan=True)


def test_rolling_mean_matches_reference():
    x = torch.arange(1, 21, dtype=torch.float32).view(20, 1).repeat(1, 3)
    out = ops.op_mean(x, 5)
    assert torch.isnan(out[3]).all()
    assert torch.allclose(out[4], torch.full((3,), 3.0))
    assert torch.allclose(out[19], torch.full((3,), 18.0))


def test_csrank_range():
    x = torch.randn(10, 30)
    r = ops.op_csrank(x)
    assert float(r.min()) == 0.0 and float(r.max()) == 1.0


def test_graph_operators_shapes_and_semantics(small_market):
    g = small_market.graph("sector")
    x = small_market.feat[3, :300]
    mean = ops.op_nbr_mean(x, g)
    assert mean.shape == x.shape
    rel = ops.op_nbr_rel(x, g)
    assert torch.allclose(rel[100:], (x - mean)[100:], equal_nan=True)
    s = ops.op_nbr_std(x, g)
    assert (s[torch.isfinite(s)] >= 0).all()
    r = ops.op_nbr_rank(x, g)
    assert ((r[torch.isfinite(r)] >= 0) & (r[torch.isfinite(r)] <= 1)).all()


def test_nbr_mean_manual_check():
    from graphalpha.graphs import NeighborGraph
    idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])             # fully connected triangle, k=2
    mask = torch.ones(1, 3, 2, dtype=torch.bool)
    g = NeighborGraph("t", idx, mask, torch.zeros(4, dtype=torch.long))
    x = torch.tensor([[1.0, 2.0, 6.0]] * 4)
    out = ops.op_nbr_mean(x, g)[0]
    assert torch.allclose(out, torch.tensor([4.0, 3.5, 1.5]))


@pytest.mark.parametrize("mode", ["corr", "lead"])
def test_dynamic_graph_has_no_lookahead(mode):
    torch.manual_seed(0)
    ret = torch.randn(300, 20) * 0.01
    g1 = build_dynamic_graph(ret, mode, window=40, step=5, k=4)
    ret2 = ret.clone()
    ret2[200:] = torch.randn(100, 20) * 0.01                    # alter the future only
    g2 = build_dynamic_graph(ret2, mode, window=40, step=5, k=4)
    # graphs serving days <= 200 only use returns of days <= 199
    for t in range(0, 201):
        a, b = g1.day2chunk[t], g2.day2chunk[t]
        assert torch.equal(g1.idx[a], g2.idx[b]) and torch.equal(g1.mask[a], g2.mask[b]), f"day {t} leaks"


def test_graph_alpha_is_causal(small_market):
    vocab = Vocab(["sector", "corr", "lead"])
    node = build_from_keys(vocab, ["feat:close", "win:5", "Delta", "graph:lead", "NbrRel"])
    full = evaluate(node, small_market, 900)
    part = evaluate(node, small_market, 700)
    a, b = full[:700], part
    both = torch.isfinite(a) & torch.isfinite(b)
    assert torch.allclose(a[both], b[both])
    assert torch.equal(torch.isfinite(a), torch.isfinite(b))
