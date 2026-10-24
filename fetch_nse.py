"""Fetch ~100 large-cap NSE stocks from Yahoo Finance -> the same .npz format as data/csi300.npz.

    pip install yfinance
    python fetch_nse.py --out data/nse100.npz
    python fetch_nse.py --limit 8 --out data/nse_test.npz      # quick connectivity test

Notes / limitations (state these in the report):
  * Universe = a fixed list of today's large caps listed before 2012 (survivorship bias).
  * Prices are split/dividend adjusted (Yahoo ``auto_adjust``).
  * Yahoo has no VWAP: ``vwap`` is approximated by the typical price (high + low + close) / 3.
  * Sectors are a hand-written map (needed for the sector graph).
Downloads are cached per stock under ``<out dir>/cache/nse`` so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from graphalpha.data import assemble

# Yahoo symbol (without ".NS") -> sector
UNIVERSE: Dict[str, str] = {
    # Banks
    "HDFCBANK": "Banks", "ICICIBANK": "Banks", "SBIN": "Banks", "KOTAKBANK": "Banks", "AXISBANK": "Banks",
    "INDUSINDBK": "Banks", "BANKBARODA": "Banks", "PNB": "Banks", "FEDERALBNK": "Banks", "CANBK": "Banks",
    # Financial services
    "BAJFINANCE": "FinServ", "BAJAJFINSV": "FinServ", "CHOLAFIN": "FinServ", "MUTHOOTFIN": "FinServ",
    "PFC": "FinServ", "RECLTD": "FinServ", "M&MFIN": "FinServ", "LICHSGFIN": "FinServ", "SRTRANSFIN": "FinServ",
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT", "MPHASIS": "IT",
    "COFORGE": "IT", "PERSISTENT": "IT",
    # Energy / utilities
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy", "GAIL": "Energy",
    "COALINDIA": "Energy", "HINDPETRO": "Energy", "PETRONET": "Energy", "OIL": "Energy",
    "NTPC": "Power", "POWERGRID": "Power", "TATAPOWER": "Power", "NHPC": "Power", "TORNTPOWER": "Power",
    # Auto
    "MARUTI": "Auto", "M&M": "Auto", "TATAMOTORS": "Auto", "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto",
    "EICHERMOT": "Auto", "ASHOKLEY": "Auto", "TVSMOTOR": "Auto", "BOSCHLTD": "Auto", "MOTHERSON": "Auto",
    "BALKRISIND": "Auto",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG", "DABUR": "FMCG",
    "GODREJCP": "FMCG", "MARICO": "FMCG", "COLPAL": "FMCG", "TATACONSUM": "FMCG", "MCDOWELL-N": "FMCG",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma", "LUPIN": "Pharma",
    "AUROPHARMA": "Pharma", "TORNTPHARM": "Pharma", "BIOCON": "Pharma", "GLENMARK": "Pharma",
    # Metals & mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals", "VEDL": "Metals", "SAIL": "Metals",
    "NMDC": "Metals", "JINDALSTEL": "Metals",
    # Cement / materials
    "ULTRACEMCO": "Materials", "GRASIM": "Materials", "SHREECEM": "Materials", "AMBUJACEM": "Materials",
    "ACC": "Materials", "UPL": "Materials", "SRF": "Materials", "PIDILITIND": "Materials",
    # Capital goods / infra
    "LT": "CapGoods", "SIEMENS": "CapGoods", "ABB": "CapGoods", "BHEL": "CapGoods", "CUMMINSIND": "CapGoods",
    "HAVELLS": "CapGoods", "VOLTAS": "CapGoods", "ADANIPORTS": "CapGoods", "CONCOR": "CapGoods",
    "DLF": "Realty",
    # Consumer discretionary
    "TITAN": "Consumer", "ASIANPAINT": "Consumer", "BERGEPAINT": "Consumer", "PAGEIND": "Consumer",
    "TRENT": "Consumer",
    # Telecom & media
    "BHARTIARTL": "Telecom", "IDEA": "Telecom", "SUNTV": "Media", "ZEEL": "Media",
}


def download_one(sym: str, start: str, end: str, cache: str, tries: int = 3) -> Optional[pd.DataFrame]:
    import yfinance as yf

    path = os.path.join(cache, f"{sym.replace('&', '_')}_{start}_{end}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    last = None
    for i in range(tries):
        try:
            h = yf.Ticker(f"{sym}.NS").history(start=start, end=end, auto_adjust=True)
            if h is not None and len(h) > 0:
                df = pd.DataFrame({
                    "open": h["Open"], "high": h["High"], "low": h["Low"],
                    "close": h["Close"], "volume": h["Volume"].astype(float),
                })
                df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
                df = df[~df.index.duplicated(keep="last")]
                df = df[df["close"] > 0]
                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0     # approximation (no VWAP on Yahoo)
                df.to_pickle(path)
                return df
            last = "empty"
        except Exception as e:  # rate limit / network
            last = str(e)
        time.sleep(2 * (i + 1))
    print(f"[nse] skip {sym}: {last}", flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/nse100.npz")
    ap.add_argument("--start", default="2011-06-01")      # a little before train start so rolling ops warm up
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--limit", type=int, default=None, help="only first N stocks (quick test)")
    ap.add_argument("--min_coverage", type=float, default=0.9,
                    help="drop stocks with fewer valid days than this fraction of the longest history")
    args = ap.parse_args()

    syms: List[str] = list(UNIVERSE)[: args.limit] if args.limit else list(UNIVERSE)
    cache = os.path.join(os.path.dirname(os.path.abspath(args.out)), "cache", "nse")
    os.makedirs(cache, exist_ok=True)
    frames: Dict[str, pd.DataFrame] = {}
    for k, s in enumerate(syms, 1):
        df = download_one(s, args.start, args.end, cache)
        if df is not None:
            frames[s] = df
        print(f"[nse] {k}/{len(syms)} {s}: {'ok ' + str(len(df)) + ' days' if df is not None else 'FAILED'}", flush=True)
        time.sleep(0.3)
    if not frames:
        raise SystemExit("no stock downloaded (check network / pip install yfinance)")

    longest = max(len(f) for f in frames.values())
    dropped = [s for s, f in frames.items() if len(f) < args.min_coverage * longest]
    for s in dropped:
        del frames[s]
    if dropped:
        print(f"[nse] dropped for short history (<{args.min_coverage:.0%}): {dropped}", flush=True)

    sectors = {s: UNIVERSE[s] for s in frames}
    counts = pd.Series(sectors).value_counts()
    print(f"[nse] {len(frames)} stocks; sector sizes:\n{counts.to_string()}", flush=True)
    assemble(frames, sectors, args.out)


if __name__ == "__main__":
    main()
