"""Feature extractors for the PPO policy.

``GraphAttentionExtractor`` treats the partially-built formula as a *graph*: nodes are the
tokens emitted so far, edges link every operator to the operands it consumed. A few rounds of
masked multi-head attention restricted to those edges (graph attention) let information flow
between operators and operands; the readout combines a mean over nodes with the embeddings of the
current top-of-stack subtrees, which is exactly what the next-token decision depends on.

``LSTMExtractor`` reads the flat token sequence (the flat sequential baseline).
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .env import CTX_DIM
from .grammar import TOP_K


class _GraphAttentionLayer(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / self.dh ** 0.5
        att = att.masked_fill(~edge.unsqueeze(1), float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, L, d)
        x = self.n1(x + self.drop(self.out(y)))
        return self.n2(x + self.drop(self.ffn(x)))


class GraphAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, vocab_size: int, d_model: int = 64,
                 n_layers: int = 3, n_heads: int = 4):
        super().__init__(observation_space, features_dim=d_model)
        self.L = observation_space["tokens"].shape[0]
        self.d = d_model
        self.tok = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)     # 0 = padding
        self.pos = nn.Embedding(self.L, d_model)
        self.layers = nn.ModuleList(_GraphAttentionLayer(d_model, n_heads) for _ in range(n_layers))
        self.ctx = nn.Linear(CTX_DIM, d_model)
        self.head = nn.Sequential(nn.Linear((2 + TOP_K) * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))

    def forward(self, obs):
        tokens = obs["tokens"].long()
        B, L = tokens.shape
        valid = tokens > 0
        h = self.tok(tokens) + self.pos.weight[:L].unsqueeze(0)

        adj = obs["adj"] > 0.5                                       # parent -> child
        edge = adj | adj.transpose(1, 2) | torch.eye(L, dtype=torch.bool, device=h.device).unsqueeze(0)
        for layer in self.layers:
            h = layer(h, edge)

        vm = valid.unsqueeze(-1).to(h.dtype)
        mean = (h * vm).sum(1) / vm.sum(1).clamp(min=1.0)

        top = obs["top"].long()                                      # [B, TOP_K], -1 = none
        gathered = torch.gather(h, 1, top.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d))
        gathered = gathered * (top >= 0).unsqueeze(-1).to(h.dtype)
        ctx = F.gelu(self.ctx(obs["ctx"]))
        z = torch.cat([mean, ctx, gathered.flatten(1)], dim=-1)
        return self.head(z)


class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, vocab_size: int, d_model: int = 64,
                 n_layers: int = 1, **_):
        super().__init__(observation_space, features_dim=d_model)
        self.beg = vocab_size + 1
        self.tok = nn.Embedding(vocab_size + 2, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=max(1, n_layers), batch_first=True)
        self.ctx = nn.Linear(CTX_DIM, d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())

    def forward(self, obs):
        tokens = obs["tokens"].long()
        B = tokens.shape[0]
        beg = torch.full((B, 1), self.beg, dtype=torch.long, device=tokens.device)
        seq = torch.cat([beg, tokens], dim=1)
        out, _ = self.lstm(self.tok(seq))
        last = (tokens > 0).sum(1)                                   # index of last real token (BEG at 0)
        h = out[torch.arange(B, device=out.device), last]
        return self.head(torch.cat([h, F.gelu(self.ctx(obs["ctx"]))], dim=-1))


def policy_kwargs(cfg, vocab_size: int) -> dict:
    if cfg.policy == "gat":
        cls = GraphAttentionExtractor
        kw = dict(vocab_size=vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads)
    elif cfg.policy == "lstm":
        cls = LSTMExtractor
        kw = dict(vocab_size=vocab_size, d_model=cfg.d_model)
    else:
        raise ValueError(f"unknown policy '{cfg.policy}'")
    return dict(
        features_extractor_class=cls,
        features_extractor_kwargs=kw,
        net_arch=dict(pi=[cfg.d_model], vf=[cfg.d_model]),
    )
