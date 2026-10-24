"""Download CSI 300 daily data from Baostock and pack it into one ``.npz`` file.

Output arrays (``T`` dates x ``N`` stocks):
    dates, codes, sector_ids, sector_names, open, high, low, close, volume, vwap

Prices are forward-adjusted (Baostock ``adjustflag=2``). VWAP is ``amount/volume``
from the unadjusted series, rescaled by the adjustment ratio so it sits on the
same price scale as the adjusted OHLC. Halted days carry the last close forward
(volume 0); days before a stock's listing stay NaN.

Robustness (Baostock's client has *no* socket timeout and can hang forever):
  * a global socket timeout turns a hang into a retryable error,
  * a failed query triggers a re-login before the retry,
  * every stock is cached under ``<cache>/raw`` so an interrupted run resumes where it stopped.

Survivorship caveat: the universe is the *current* constituent set (see ``universe.py``).
"""
from __future__ import annotations

import argparse
import os
import socket
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .universe import resolve_universe

ADJ_FIELDS = "date,open,high,low,close,volume,amount"
RAW_FIELDS = "date,close,volume,amount"


# ---------------------------------------------------------------------- Baostock helpers
def _query(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


class BaostockSession:
    """Login/relogin wrapper with retries; every query goes through ``call``."""

    def __init__(self, timeout: float = 20.0, tries: int = 5, wait: float = 2.0):
        import baostock as bs
        self.bs, self.tries, self.wait = bs, tries, wait
        socket.setdefaulttimeout(timeout)
        self.login()

    def login(self) -> None:
        lg = self.bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

    def relogin(self) -> None:
        try:
            self.bs.logout()
        except Exception:
            pass
        self.login()

    def call(self, fn: Callable) -> pd.DataFrame:
        last = None
        for i in range(self.tries):
            try:
                rs = fn()
                if rs.error_code == "0":
                    return _query(rs)
                last = rs.error_msg
            except Exception as e:  # network hiccup / timeout
                last = str(e)
            time.sleep(self.wait * (i + 1))
            try:
                self.relogin()
            except Exception as e:
                last = f"{last}; relogin failed: {e}"
        raise RuntimeError(f"Baostock query failed after {self.tries} tries: {last}")

    def close(self) -> None:
        try:
            self.bs.logout()
        except Exception:
            pass


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date")


# ---------------------------------------------------------------------- per-stock download
def download_stock(sess: BaostockSession, code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    def hist(flag: str, fields: str) -> pd.DataFrame:
        return _numeric(sess.call(lambda: sess.bs.query_history_k_data_plus(
            code, fields, start_date=start, end_date=end, frequency="d", adjustflag=flag)))

    a, r = hist("2", ADJ_FIELDS), hist("3", RAW_FIELDS)
    if a.empty or r.empty:
        return None
    ratio = (a["close"] / r["close"]).replace([np.inf, -np.inf], np.nan)
    vw_raw = (r["amount"] / r["volume"]).where(r["volume"] > 0)
    a["vwap"] = (vw_raw * ratio).reindex(a.index)
    return a


def download_all(sess: BaostockSession, codes: List[str], start: str, end: str, raw_dir: str) -> Dict[str, pd.DataFrame]:
    from tqdm import tqdm

    os.makedirs(raw_dir, exist_ok=True)
    frames: Dict[str, pd.DataFrame] = {}
    tag = f"{start}_{end}"
    pbar = tqdm(codes, desc="[data] downloading", unit="stock")
    for code in pbar:
        path = os.path.join(raw_dir, f"{code}_{tag}.pkl")
        if os.path.exists(path):
            df = pd.read_pickle(path)
        else:
            try:
                df = download_stock(sess, code, start, end)
            except RuntimeError as e:
                pbar.write(f"[data] skip {code}: {e}")
                continue
            if df is None:
                pbar.write(f"[data] skip {code}: no data")
                continue
            df.to_pickle(path)
        frames[code] = df
    return frames


def load_industry(sess: BaostockSession, codes: List[str], cache_path: str) -> Dict[str, str]:
    """code -> 3-char industry tag; cached; falls back to 'UNK' (with a warning) if unavailable."""
    try:
        if os.path.exists(cache_path):
            ind = pd.read_csv(cache_path)
        else:
            ind = sess.call(lambda: sess.bs.query_stock_industry())
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            ind.to_csv(cache_path, index=False)
        m = {r.code: (str(r.industry)[:3] if isinstance(r.industry, str) and r.industry else "UNK")
             for r in ind.itertuples()}
    except Exception as e:
        print(f"[data] WARNING: industry query failed ({e}); all stocks get sector 'UNK' "
              f"-> the sector graph will be uninformative", flush=True)
        m = {}
    return {c: m.get(c, "UNK") for c in codes}


# ---------------------------------------------------------------------- assembly
def assemble(frames: Dict[str, pd.DataFrame], ind_map: Dict[str, str], out: str) -> None:
    codes = sorted(frames)
    idx = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))

    def stack(col: str, ffill: bool) -> np.ndarray:
        df = pd.DataFrame({c: frames[c][col] for c in codes}).reindex(idx)
        if ffill:
            df = df.ffill()
        return df.to_numpy(dtype=np.float32)

    close = stack("close", True)
    arrays = dict(
        open=stack("open", True), high=stack("high", True), low=stack("low", True),
        close=close, volume=np.nan_to_num(stack("volume", False), nan=0.0),
    )
    vwap = stack("vwap", False)
    arrays["vwap"] = np.where(np.isfinite(vwap), vwap, close).astype(np.float32)
    pre = ~np.isfinite(stack("close", False)) & ~np.isfinite(close)   # not yet listed -> keep NaN
    for k in ("open", "high", "low", "vwap"):
        arrays[k] = np.where(pre, np.nan, arrays[k]).astype(np.float32)

    sectors = [ind_map.get(c, "UNK") for c in codes]
    names = sorted(set(sectors))
    sector_ids = np.array([names.index(s) for s in sectors], dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out, dates=idx.strftime("%Y-%m-%d").values.astype("U10"),
        codes=np.array(codes), sector_ids=sector_ids, sector_names=np.array(names), **arrays,
    )
    print(f"[data] saved {out}: T={len(idx)} N={len(codes)} sectors={len(names)}", flush=True)


def fetch_csi300(out: str, start: str = "2011-01-01", end: str = "2024-12-31",
                 limit: Optional[int] = None, universe: str = "auto",
                 codes_file: Optional[str] = None, timeout: float = 20.0) -> None:
    cache = os.path.join(os.path.dirname(os.path.abspath(out)), "cache")
    sess = BaostockSession(timeout=timeout)
    try:
        codes = resolve_universe(sess.bs, sess.call, codes_file, universe,
                                 cache_path=os.path.join(cache, "constituents.txt"))
        if limit:
            codes = codes[:limit]
        print(f"[data] {len(codes)} stocks, {start} -> {end}", flush=True)
        frames = download_all(sess, codes, start, end, os.path.join(cache, "raw"))
        if not frames:
            raise RuntimeError("no stock downloaded successfully")
        ind_map = load_industry(sess, sorted(frames), os.path.join(cache, "industry.csv"))
    finally:
        sess.close()
    assemble(frames, ind_map, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/csi300.npz")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--limit", type=int, default=None, help="only first N stocks (quick test)")
    ap.add_argument("--universe", default="auto", choices=["auto", "baostock", "builtin", "file"])
    ap.add_argument("--codes_file", default=None, help="text/CSV of Baostock codes (with --universe file)")
    ap.add_argument("--timeout", type=float, default=20.0, help="socket timeout in seconds")
    ap.add_argument("--synthetic", action="store_true", help="write a synthetic market instead")
    ap.add_argument("--regime", default="mixed", help="synthetic regime: mixed | relational")
    ap.add_argument("--n_stocks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.synthetic:
        from .synthetic import make_synthetic
        make_synthetic(args.out, n_stocks=args.n_stocks, start=args.start, end=args.end,
                       seed=args.seed, regime=args.regime)
        print(f"[data] synthetic ({args.regime}) market saved to {args.out}")
    else:
        fetch_csi300(args.out, args.start, args.end, args.limit, args.universe,
                     args.codes_file, args.timeout)


if __name__ == "__main__":
    main()
