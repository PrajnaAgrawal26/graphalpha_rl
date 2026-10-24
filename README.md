# GraphAlpha-RL: Graph-Aware Symbolic Alpha Mining with Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GraphAlpha-RL** is an end-to-end Reinforcement Learning system designed to automatically discover profitable, orthogonal, and interpretable formulaic trading signals (alphas) for quantitative equity portfolios. 

Rather than treating stocks as isolated time-series in a vacuum, GraphAlpha-RL introduces **relational graph operators** that aggregate quantities across stock neighborhoods on dynamic, day-varying market graphs (rolling correlation clusters, sector peers, and lead-lag predictors). A **MaskablePPO** policy network guided by a **Tree-GAT (Graph Attention Network)** constructs mathematical syntax trees token-by-token, receiving rewards strictly based on marginal portfolio enhancement and cross-sectional Information Coefficient (IC).

---

## 1. Mathematical RL Formulation

The formula discovery problem is formalized as a finite-horizon, discrete-time Markov Decision Process (MDP) $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

### State Space ($\mathcal{S}$)
At decision step $t$, the state representation $s_t \in \mathcal{S}$ captures both the partially generated expression tree and the macroscopic state of the alpha portfolio:
1. **`tokens`** $\in \mathbb{R}^{L}$: Zero-padded sequence of emitted token IDs up to maximum length $L=15$.
2. **`adj`** $\in \{0, 1\}^{L \times L}$: Directed adjacency matrix encoding parent-child syntax tree connections. Edge $(i, j) = 1$ denotes that operator at position $i$ consumed the sub-expression rooted at position $j$.
3. **`top`** $\in \mathbb{R}^{3}$: Token indices corresponding to the top-3 subtree roots currently residing on the stack, enabling the agent to track unresolved operands.
4. **`ctx`** $\in \mathbb{R}^{8}$: 8-dimensional macroscopic context vector:
   * Remaining length budget fraction: $(L_{\text{stage}} - |tokens|) / L$
   * Graph operators active flag: $\{0, 1\}$
   * Alpha pool saturation ratio: $|\mathcal{P}| / K$
   * Current pool ensemble objective: $\text{clip}(10 \cdot \text{IC}_{\text{pool}}, -10, 10)$
   * Factor composition: proportion of pool alphas utilizing price features (`close`, `open`, `high`, `low`)
   * Factor composition: proportion of pool alphas utilizing volume/liquidity features (`volume`, `vwap`)
   * Factor composition: proportion of pool alphas utilizing relational graph operators
   * Mean pairwise correlation among existing pool members (diversity metric)

### Action Space ($\mathcal{A}$)
The action $a_t \in \mathcal{A}$ selects the discrete next token from a vocabulary of $V \approx 50$ primitive operators and terminals:
* **Feature Terminals**: `open`, `high`, `low`, `close`, `volume`, `vwap`.
* **Constants**: 10 scalar values $\{-5.0, -2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0, 5.0\}$.
* **Window Parameters**: 5 lookback lengths $\{1, 5, 10, 20, 40\}$.
* **Graph Terminals**: `G_sector`, `G_corr`, `G_lead`.
* **Operators**:
  * Unary: `Abs`, `Log`, `CsRank` (cross-sectional ranking).
  * Binary: `Add`, `Sub`, `Mul`, `Div`, `Greater`, `Less`.
  * Rolling Time-Series: `Ref`, `Mean`, `Sum`, `Std`, `Max`, `Min`, `Delta`, `WMA`, `EMA`.
  * Pair Rolling: `Cov`, `Corr`.
  * **Relational Graph Operators**: `NbrMean`, `NbrRel`, `NbrStd`, `NbrMax`, `NbrRank`, `Diffuse`.
* **Episode Terminator**: `<SEP>`.

**Grammar-Constrained Action Masking**: A deterministic postfix stack automaton enforces syntactic validity. At each step, invalid tokens (e.g. operators lacking operands, expressions exceeding maximum depth, or malformed types) are masked out with logit mask $M(s) \in \{-\infty, 0\}^{|\mathcal{A}|}$, ensuring the agent *never* emits a syntactically invalid formula.

### Reward Function ($\mathcal{R}$)
The environment awards intermediate step rewards $r_t = 0$. Upon completing a valid expression tree $f$ at terminal step $T$, the reward evaluates its marginal portfolio contribution:

$$R(f) = s \cdot \left( \Delta \text{IC}_{\text{pool}} + \beta \cdot (1 - \max_{m \in \mathcal{P}} |\rho(f, m)|) \cdot |\text{IC}(f)| \right) - \gamma \cdot \frac{\text{len}(f)}{L_{\text{budget}}}$$

* **Marginal Ensemble Gain ($\Delta \text{IC}_{\text{pool}}$)**: Change in cross-sectional ensemble Information Coefficient against 20-day forward returns when $f$ is admitted to pool $\mathcal{P}$.
* **Orthogonality / Diversity Bonus**: $\beta \cdot (1 - \max |\rho|) \cdot |\text{IC}(f)|$ penalizes collinear signals and rewards factors that add orthogonal predictive power.
* **Complexity Regularizer**: $-\gamma \cdot \frac{\text{len}(f)}{L}$ penalizes bloated formulas (Occam's razor).
* **Penalties**: Degenerate/constant formulas receive $-0.5$; near-duplicates ($|\rho| > 0.99$) receive $-0.2$.

### RL Algorithm & Architecture
* **Policy Architecture**: **Tree-GAT** (Tree Graph Attention Network). The state's AST adjacency matrix and token sequence are processed through multi-head graph self-attention layers with bidirectional message passing, followed by readout heads combining pool context and top-of-stack embeddings.
* **Optimization**: **MaskablePPO** (`sb3-contrib`), using generalized advantage estimation (GAE: $\lambda = 0.95$, $\gamma = 1.0$), clipped surrogate objective ($\epsilon = 0.2$), and entropy regularization ($c_{\text{ent}} = 0.02$).

---

## 2. Installation & Environment Setup

### Prerequisites
* Linux OS (Ubuntu 20.04+ recommended)
* Conda / Miniconda
* Python 3.10+
* NVIDIA GPU (Optional: CUDA supported automatically if available; CPU execution supported natively)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/PrajnaAgrawal26/graphalpha_rl.git
cd graphalpha_rl

# 2. Create and activate a conda environment
conda create -n graphalpha python=3.10 -y
conda activate graphalpha

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation via unit test suite
pytest tests/ -v
```

All 32 test cases should pass in under 5 seconds, validating grammar completeness, causality, and reward computation.

---

## 3. Execution Commands & Reproduction Instructions

### 1. Data Preparation
To generate the offline synthetic relational market data or download real large-cap equity market data:

```bash
python fetch_data.py --synthetic --out data/synthetic.npz

python fetch_nse.py --out data/nse100.npz
```

### 2. Fast Smoke Test (Single Agent)
Verify the complete training pipeline end-to-end (~2 minutes):
```bash
python train.py --config configs/smoke.yaml
```

### 3. Training the RL Agent
Train the Tree-GAT policy on the real-world dataset across multiple seeds:

```bash
# Train Dynamic Correlation Graph agent with multi-regime robustness (100k steps)
python train.py --config configs/corr_robust.yaml --seed 0 --set data_path=data/nse100.npz

# Or launch parallel multi-seed experiments across 3 seeds
python run_suite.py \
  --data data/nse100.npz \
  --tag nse_experiment \
  --methods corr_dense corr_robust \
  --seeds 0 1 2 \
  --steps 100000 \
  --jobs 3
```

### 4. Pretrained RL Agent Checkpoint
A trained Tree-GAT MaskablePPO agent is provided in `checkpoints/best_agent/`:

```python
from sb3_contrib import MaskablePPO

# Load the trained agent policy
model = MaskablePPO.load("checkpoints/best_agent/model.zip")
print(model.policy)
```

---
