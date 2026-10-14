import numpy as np
import torch

from graphalpha.expr import evaluate
from graphalpha.grammar import FormulaBuilder, build_from_keys
from graphalpha.vocab import Vocab


def _rollout(vocab, rng, max_len, allow_graph):
    b = FormulaBuilder(vocab, max_len=max_len, allow_graph=allow_graph)
    for _ in range(max_len + 2):
        mask = b.valid_mask()
        assert mask.any(), "no legal action: deadlock"
        a = int(rng.choice(np.flatnonzero(mask)))
        if vocab[a].kind == "sep":
            break
        b.push(a)
        if len(b) >= max_len:
            break
    return b


def test_random_rollouts_always_valid():
    vocab = Vocab(["sector", "corr", "lead"])
    rng = np.random.default_rng(0)
    for max_len in (3, 5, 7, 11, 15):
        for _ in range(400):
            b = _rollout(vocab, rng, max_len, allow_graph=True)
            assert b.is_complete(), f"incomplete formula at len {len(b)}"
            assert len(b) <= max_len


def test_no_graph_vocab_has_no_graph_tokens():
    vocab = Vocab([])
    assert "graph" not in vocab.groups and "gop" not in vocab.groups
    rng = np.random.default_rng(1)
    for _ in range(100):
        assert _rollout(vocab, rng, 15, True).is_complete()


def test_graph_masked_when_disallowed():
    vocab = Vocab(["sector", "corr"])
    b = FormulaBuilder(vocab, max_len=15, allow_graph=False)
    b.push(vocab.lookup("feat:close").id)
    m = b.valid_mask()
    assert not m[vocab.groups["graph"]].any() and not m[vocab.groups["gop"]].any()


def test_postfix_semantics_and_roundtrip():
    vocab = Vocab(["sector", "corr", "lead"])
    keys = ["feat:close", "win:5", "Delta", "graph:sector", "NbrRel", "feat:volume", "win:10", "Corr"]
    node = build_from_keys(vocab, keys)
    assert str(node) == "Corr(NbrRel(Delta($close, 5), G_sector), $volume, 10)"
    b = FormulaBuilder(vocab, max_len=20)
    for k in keys:
        b.push(vocab.lookup(k).id)
    assert b.keys() == keys and node.uses_graph()


def test_adjacency_links_operator_to_operands():
    vocab = Vocab(["corr"])
    b = FormulaBuilder(vocab, max_len=15)
    for k in ["feat:close", "win:5", "Delta"]:
        b.push(vocab.lookup(k).id)
    assert b.adj[2, 0] == 1 and b.adj[2, 1] == 1 and b.adj.sum() == 2
    assert b.top_positions()[0] == 2


def test_formula_evaluates_to_finite_values(small_market):
    vocab = Vocab(["sector", "corr", "lead"])
    keys = ["feat:close", "win:5", "Delta", "graph:corr", "NbrRel"]
    node = build_from_keys(vocab, keys)
    out = evaluate(node, small_market, 600)
    assert out.shape == (600, small_market.N)
    assert torch.isfinite(out[200:]).float().mean() > 0.5
