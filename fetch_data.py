"""Fetch CSI 300 data (or a synthetic market) into data/.

    python fetch_data.py                    # Baostock CSI 300 -> data/csi300.npz
    python fetch_data.py --synthetic        # offline synthetic market for smoke tests
"""
from graphalpha.data import main

if __name__ == "__main__":
    main()
