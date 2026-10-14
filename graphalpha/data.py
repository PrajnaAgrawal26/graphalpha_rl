"""Download CSI 300 daily data from Baostock and pack it into one ``.npz`` file.

Output arrays (``T`` dates x ``N`` stocks):
    dates, codes, sector_ids, sector_names, open, high, low, close, volume, vwap

Prices are forward-adjusted (Baostock ``adjustflag=2``). VWAP is ``amount/volume``
from the unadjusted series, rescaled by the adjustment ratio so it sits on the
same price scale as the adjusted OHLC. Halted days carry the last close forward
(volume 0); days before a stock's listing stay NaN.

Survivorship caveat: the universe is the *current* CSI 300 membership.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd


def _query(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def _retry(fn, tries: int = 4, wait: float = 2.0):
    last = None
    for i in range(tries):
        try:
            rs = fn()
            if rs.error_code == "0":
                return rs
            last = rs.error_msg
        except Exception as e:  # network hiccup
            last = str(e)
        time.sleep(wait * (i + 1))
    raise RuntimeError(f"Baostock query failed: {last}")


def _daily(bs, code: str, start: str, end: str, flag: str, fields: str) -> pd.DataFrame:
    rs = _retry(lambda: bs.query_history_k_data_plus(
        code, fields, start_date=start, end_date=end, frequency="d", adjustflag=flag))
    df = _query(rs)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date")


def fetch_csi300(out: str, start: str = "2011-01-01", end: str = "2024-12-31",
                 limit: Optional[int] = None) -> None:
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {lg.error_msg}")
    try:
        cons = _query(_retry(lambda: bs.query_hs300_stocks()))
        codes: List[str] = sorted(cons["code"].tolist())
        if limit:
            codes = codes[:limit]
        from tqdm import tqdm
        print(f"[data] Found {len(codes)} constituents. Downloading daily OHLCV...", flush=True)

        ind = _query(_retry(lambda: bs.query_stock_industry()))
        ind_map = {r.code: (str(r.industry)[:3] if r.industry else "UNK") for r in ind.itertuples()}

        adj_fields = "date,open,high,low,close,volume,amount"
        raw_fields = "date,close,volume,amount"
        frames = {}
        pbar = tqdm(codes, desc="[data] downloading", unit="stock")
        for code in pbar:
            a = _daily(bs, code, start, end, "2", adj_fields)
            r = _daily(bs, code, start, end, "3", raw_fields)
            if a.empty or r.empty:
                pbar.write(f"[data] skip {code}: no data")
                continue
            ratio = (a["close"] / r["close"]).replace([np.inf, -np.inf], np.nan)
            vw_raw = (r["amount"] / r["volume"]).where(r["volume"] > 0)
            a["vwap"] = (vw_raw * ratio).reindex(a.index)
            frames[code] = a
    finally:
        bs.logout()

    codes = sorted(frames)
    cal = sorted(set().union(*[set(f.index) for f in frames.values()]))
    idx = pd.DatetimeIndex(cal)

    def stack(col: str, ffill: bool) -> np.ndarray:
        df = pd.DataFrame({c: frames[c][col] for c in codes}).reindex(idx)
        if ffill:
            df = df.ffill()
        return df.to_numpy(dtype=np.float32)

    close = stack("close", True)
    out_arrays = dict(
        open=stack("open", True), high=stack("high", True), low=stack("low", True),
        close=close, volume=np.nan_to_num(stack("volume", False), nan=0.0),
    )
    vwap = stack("vwap", False)
    out_arrays["vwap"] = np.where(np.isfinite(vwap), vwap, close).astype(np.float32)
    # keep NaN where the stock was not yet listed
    pre = ~np.isfinite(stack("close", False)) & ~np.isfinite(close)
    for k in ("open", "high", "low", "vwap"):
        out_arrays[k] = np.where(pre, np.nan, out_arrays[k]).astype(np.float32)

    sectors = [ind_map.get(c, "UNK") for c in codes]
    names = sorted(set(sectors))
    sector_ids = np.array([names.index(s) for s in sectors], dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out,
        dates=idx.strftime("%Y-%m-%d").values.astype("U10"),
        codes=np.array(codes), sector_ids=sector_ids, sector_names=np.array(names),
        **out_arrays,
    )
    print(f"[data] saved {out}: T={len(idx)} N={len(codes)} sectors={len(names)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/csi300.npz")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--limit", type=int, default=None, help="only first N stocks (quick test)")
    ap.add_argument("--synthetic", action="store_true", help="write a synthetic market instead")
    ap.add_argument("--n_stocks", type=int, default=100)
    args = ap.parse_args()
    if args.synthetic:
        from .synthetic import make_synthetic
        make_synthetic(args.out, n_stocks=args.n_stocks, start=args.start, end=args.end)
        print(f"[data] synthetic market saved to {args.out}")
    else:
        fetch_csi300(args.out, args.start, args.end, args.limit)


if __name__ == "__main__":
    main()
