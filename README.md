# GraphAlpha-RL

Graph-aware symbolic alpha mining with reinforcement learning.

A **MaskablePPO** agent writes stock-selection formulas one token at a time. Besides ordinary
time-series operators (`Delta`, `Mean`, `Corr`, ...), the formula language contains **graph operators**
(`NbrMean`, `NbrRel`, `NbrStd`, `NbrMax`, `NbrRank`, `Diffuse`) that aggregate a quantity over each
stock's neighbours on a **day-varying stock-relationship graph** (sector peers, rolling-correlation
peers, lead-lag leaders). The agent is rewarded for how much a finished formula improves a small,
diverse **pool** of alphas, measured by cross-sectional Information Coefficient (IC) against the
20-day forward return.

> Everything that *learns* is the RL policy. Pool weights are parameter-free (`w_i = IC_i`), graphs
> are deterministic functions of past returns, and the backtest is a fixed evaluation rule.

Prior work acknowledged: the token-by-token MDP / synergistic alpha pool idea comes from
*Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning* (Yu et al., KDD 2023).
This repository is an independent implementation; the graph operators, dynamic graphs, diversity /
complexity-shaped reward, curriculum and the AST graph-attention policy are new here.

---

## RL formulation

| | |
|---|---|
| **State** | Dict: `tokens` (`L=15` token ids, 0-padded), `adj` (`15x15` operator→operand edges of the partial formula tree), `top` (positions of the top-3 stack subtrees), `ctx` (4 numbers: remaining budget, graph-ops-enabled flag, pool fill, current pool IC). |
| **Action** | Discrete next token (≈ 50): 6 features, 10 constants, 5 windows, 3 graph ids, 3 unary + 6 binary + 9 rolling + 2 pair-rolling + 6 graph operators, `<SEP>`. A grammar stack provides an **action mask** so only completable, type-correct tokens are legal. |
| **Reward** | 0 until the formula ends, then `R = s·(ΔIC_pool + β·(1−max\|ρ\|)·\|IC\|) − γ·len/budget`; invalid formula `−0.5`, near-duplicate (`\|ρ\|>0.99`) `−0.2`. |
| **Environment** | Custom Gymnasium env (`graphalpha/env.py`): a formula builder + GPU/CPU evaluator on CSI 300 daily data + the alpha pool. |
| **Algorithm** | MaskablePPO (`sb3-contrib`) with a custom **graph-attention** feature extractor over the partial formula tree (or an LSTM baseline). |

## Repository layout

```
graphalpha/
  config.py     experiment config (dataclass <-> YAML, CLI overrides)
  data.py       Baostock CSI 300 downloader  -> data/csi300.npz
  synthetic.py  synthetic market with planted cross-stock structure (tests / smoke runs)
  market.py     tensors, forward-return target, date splits (with purging), graphs
  graphs.py     sector / rolling-correlation / lead-lag neighbour graphs (causal)
  ops.py        operator library incl. graph operators (all causal)
  expr.py       formula tree + evaluator
  vocab.py      token vocabulary (= action space)
  grammar.py    postfix stack grammar + action masking
  metrics.py    z-score, daily IC / RankIC
  pool.py       alpha pool + reward
  env.py        Gymnasium environment
  policy.py     graph-attention and LSTM feature extractors
  runner.py     training loops (PPO + random baseline), logging
  backtest.py   long-only top-K evaluation
  plotting.py   report figures
configs/        one YAML per experiment / ablation
tests/          unit tests (grammar completeness, operator causality, graph no-lookahead, metrics, env)
train.py  evaluate.py  fetch_data.py
```

## Installation

```bash
conda create -n graphalpha python=3.10 -y && conda activate graphalpha
pip install -r requirements.txt
pytest -q                         # ~1 min, runs offline on synthetic data
```

A CUDA GPU is optional (formula evaluation uses it automatically when present; the policy runs on CPU).

## Data

```bash
python fetch_data.py                       # CSI 300, 2011-2024, Baostock  -> data/csi300.npz  (needs internet)
python fetch_data.py --synthetic --out data/synthetic.npz   # offline synthetic market (smoke tests only)
```

* Universe: current CSI 300 constituents (survivorship bias — noted as a limitation).
* Features: open, high, low, close, volume, vwap (forward-adjusted). Sector = CSRC industry code (first 3 chars).
* Splits (`configs/*.yaml`): train 2012–2019, valid 2020–2021, test 2022–mid-2024. Target = 20-day forward return.
  The last 20 days of train/valid are purged so label windows never cross a split boundary.
* No lookahead: all operators are causal; the graph serving day *t* is built from returns up to *t−1*.

## Running

```bash
# smoke test (≈ 2 min, synthetic data)
python train.py --config configs/smoke.yaml

# main experiment + ablations (3 seeds each)
for cfg in graph_all nograph sector_only corr_only lead_only all_lstm all_nocurriculum all_nodiv random_graph random_nograph; do
  for s in 0 1 2; do python train.py --config configs/$cfg.yaml --seed $s; done
done

# evaluate best-validation pools on the untouched test set + figures
python evaluate.py runs/*_s* --out reports
```

Outputs: `runs/<name>_s<seed>/{config.json, log.csv, pool_best_valid.json, pool_latest.json, model.zip}`;
`reports/{results_table.md, formulas.md, figures/*.png}`.
Override any config value: `python train.py --config configs/graph_all.yaml --set total_timesteps=100000 graph_k=6`.

## Experiments

| Config | What it tests |
|---|---|
| `graph_all` | full model |
| `nograph` | same agent without graph operators (AlphaGen-style language) |
| `sector_only`, `corr_only`, `lead_only` | which graph helps |
| `all_lstm` | graph-attention policy vs. flat LSTM policy |
| `all_nocurriculum`, `all_nodiv` | curriculum and diversity-bonus ablations |
| `random_graph`, `random_nograph` | random valid-action baseline (no learning) |

## Reproducibility

Seeds are set for Python, NumPy, PyTorch and SB3. Each run saves its full config; `evaluate.py`
reloads pools from the saved token sequences, so reported formulas can be re-evaluated exactly.
