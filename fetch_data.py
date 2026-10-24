"""Fetch market data (synthetic regimes or real CSI 300). Thin wrapper over graphalpha.data.

    python fetch_data.py --synthetic --regime relational --out data/synthetic_relational.npz
    python fetch_data.py --universe builtin --out data/csi300.npz     # real prices, fixed large-cap list
    python fetch_data.py --out data/csi300.npz                        # real; live constituents, auto fallback
    python fetch_data.py --limit 10 --out data/csi_test.npz           # quick connectivity test

Real-data downloads are cached under data/cache/raw and resume if interrupted.
"""
from graphalpha.data import main

if __name__ == "__main__":
    main()
