import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphalpha.config import Config  # noqa: E402
from graphalpha.market import Market  # noqa: E402
from graphalpha.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg():
    return Config(
        name="test", data_path="", graphs=["sector", "corr", "lead"], pool_capacity=4,
        curriculum=[], max_len=15, horizon=10,
        train=["2012-01-01", "2014-12-31"], valid=["2015-01-01", "2015-12-31"], test=["2016-01-01", "2016-12-31"],
    )


@pytest.fixture(scope="session")
def small_market(tmp_path_factory, small_cfg):
    path = str(tmp_path_factory.mktemp("data") / "syn.npz")
    make_synthetic(path, n_stocks=40, start="2011-01-03", end="2016-12-30", n_sectors=5, seed=1)
    return Market(
        path, small_cfg.horizon,
        {"train": small_cfg.train, "valid": small_cfg.valid, "test": small_cfg.test},
        small_cfg.graphs, graph_k=5, graph_window=40, graph_step=5, device="cpu",
    )
