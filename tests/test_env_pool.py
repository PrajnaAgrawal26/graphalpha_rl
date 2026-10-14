import numpy as np
import torch

from graphalpha.env import AlphaEnv
from graphalpha.metrics import daily_ic, daily_rank_ic, zscore_rows
from graphalpha.pool import AlphaPool
from graphalpha.vocab import Vocab


def test_daily_ic_matches_numpy():
    torch.manual_seed(0)
    a, y = torch.randn(30, 60), torch.randn(30, 60)
    ic = daily_ic(a, y)
    ref = np.array([np.corrcoef(a[i].numpy(), y[i].numpy())[0, 1] for i in range(30)])
    assert np.allclose(ic.numpy(), ref, atol=1e-4)


def test_ic_handles_nan_and_perfect_signal():
    y = torch.randn(10, 50)
    a = y.clone()
    a[:, :5] = float("nan")
    ic = daily_ic(a, y)
    assert torch.allclose(ic, torch.ones(10), atol=1e-5)
    assert torch.allclose(daily_rank_ic(a, y), torch.ones(10), atol=1e-5)


def test_zscore_rows():
    z = zscore_rows(torch.randn(5, 40) * 3 + 7)
    assert torch.allclose(z.mean(1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(z.std(1, unbiased=False), torch.ones(5), atol=1e-4)
    assert torch.isnan(zscore_rows(torch.ones(3, 40))).all()      # no variance -> NaN


def _env(small_market, small_cfg):
    vocab = Vocab(small_cfg.graphs)
    pool = AlphaPool(small_market, small_cfg, vocab)
    return AlphaEnv(small_market, pool, vocab, small_cfg), pool, vocab


def test_env_episode_and_reward(small_market, small_cfg):
    env, pool, vocab = _env(small_market, small_cfg)
    rng = np.random.default_rng(0)
    seen_accept = False
    for _ in range(60):
        obs, _ = env.reset()
        done, steps = False, 0
        while not done:
            a = int(rng.choice(np.flatnonzero(env.action_masks())))
            obs, r, done, trunc, info = env.step(a)
            steps += 1
            assert steps <= small_cfg.max_len
            assert np.isfinite(r)
            assert obs["tokens"].shape == (small_cfg.max_len,)
            assert env.observation_space.contains(obs)
        assert info["status"] in ("invalid", "duplicate", "rejected", "accepted")
        seen_accept |= info["status"] == "accepted"
    assert seen_accept
    assert pool.size <= pool.capacity


def test_pool_objective_matches_reevaluation(small_market, small_cfg):
    env, pool, vocab = _env(small_market, small_cfg)
    rng = np.random.default_rng(3)
    for _ in range(80):
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    # the incrementally maintained objective equals one recomputed from scratch
    ens = torch.einsum("k,ktn->tn", pool._weights([m.ic for m in pool.members]),
                       torch.stack([m.z for m in pool.members]))
    assert abs(pool._objective(ens) - pool.obj) < 1e-5


def test_pool_save_load_roundtrip(small_market, small_cfg, tmp_path):
    env, pool, vocab = _env(small_market, small_cfg)
    rng = np.random.default_rng(5)
    for _ in range(60):
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    p = str(tmp_path / "pool.json")
    pool.save(p)
    loaded = AlphaPool.load(small_market, small_cfg, vocab, p)
    assert loaded.size == pool.size
    assert abs(loaded.obj - pool.obj) < 1e-4
    assert abs(loaded.evaluate_split("valid")["ic"] - pool.evaluate_split("valid")["ic"]) < 1e-5


def test_duplicate_is_penalised(small_market, small_cfg):
    from graphalpha.grammar import build_from_keys
    vocab = Vocab(small_cfg.graphs)
    pool = AlphaPool(small_market, small_cfg, vocab)
    node = build_from_keys(vocab, ["feat:close", "win:5", "Delta"])
    first = pool.try_add(node, 3, 15)
    again = pool.try_add(node, 3, 15)
    assert first.status == "accepted" and again.status == "duplicate"
    assert again.reward < 0
