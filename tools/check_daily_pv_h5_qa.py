import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def _pick_default_key(store: pd.HDFStore) -> str:
    keys = store.keys()
    if not keys:
        raise RuntimeError("No keys found in H5.")
    # Prefer common keys if present
    for k in ("/data", "/daily", "/day", "/features"):
        if k in keys:
            return k
    # Fallback: choose the longest key (often the main table)
    return max(keys, key=len)


def _safe_ratio_bool(s: pd.Series, mask) -> float:
    if len(s) == 0:
        return float("nan")
    return float(mask.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "QA check for exported daily_pv.h5. "
            "Checks: keys, column naming ($-prefix), required fields, date range, amount/volume NaN&0 ratios."
        )
    )
    parser.add_argument(
        "--h5",
        dest="h5_path",
        default=r"/mnt/f/dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data/daily_pv.h5",
        help="H5 file path (WSL style recommended).",
    )
    parser.add_argument(
        "--key",
        default="",
        help="HDF key to read (e.g. /data). If empty, auto-detect.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200000,
        help="Max rows to sample-read for QA (default 200000).",
    )

    args = parser.parse_args()

    h5_path = args.h5_path
    print("H5_PATH =", h5_path, "exists =", os.path.exists(h5_path))
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    store = pd.HDFStore(h5_path, mode="r")
    try:
        print("\n[H5 keys]")
        keys = store.keys()
        print(keys)

        key = args.key.strip() or _pick_default_key(store)
        print("\n[Using key]", key)

        max_rows = int(args.max_rows)
        if max_rows <= 0:
            raise ValueError("--max-rows must be positive")

        df = store.select(key, start=0, stop=max_rows)
    finally:
        store.close()

    print("\n[Head]")
    print(df.head())

    print("\n[Dtypes]")
    print(df.dtypes)

    # date range
    dt: Optional[pd.Series] = None
    if isinstance(df.index, pd.MultiIndex):
        idx_names = df.index.names
        print("\n[index names]", idx_names)
        if "datetime" in idx_names:
            dt = df.index.get_level_values("datetime")
        else:
            dt = df.index.get_level_values(-1)
    else:
        cand = [c for c in df.columns if isinstance(c, str) and ("date" in c.lower() or "time" in c.lower())]
        if cand:
            dt = pd.to_datetime(df[cand[0]], errors="coerce")

    if dt is not None and len(dt) > 0:
        print("\n[date range]", pd.to_datetime(dt).min(), "~", pd.to_datetime(dt).max())

    cols = list(df.columns)

    # $ prefix check
    dollar_cols = [c for c in cols if isinstance(c, str) and c.startswith("$")]
    print("\n[$-prefixed columns]", dollar_cols)

    # required fields check
    must = ["open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in must if c not in cols]
    print("\n[missing required columns]", missing)

    # amount stats
    if "amount" in cols:
        s = df["amount"]
        nan_ratio = _safe_ratio_bool(s, s.isna())
        zero_ratio = _safe_ratio_bool(s, s == 0)
        neg_cnt = int((s < 0).sum(skipna=True))
        non_nan = s.dropna()
        print("\n[amount stats]")
        print("  nan_ratio =", nan_ratio)
        print("  zero_ratio =", zero_ratio)
        print("  negative_cnt =", neg_cnt)
        print("  non_nan_min =", float(non_nan.min()) if len(non_nan) else None)
        print("  non_nan_max =", float(non_nan.max()) if len(non_nan) else None)

    # volume stats
    if "volume" in cols:
        s = df["volume"]
        print("\n[volume stats]")
        print("  nan_ratio =", _safe_ratio_bool(s, s.isna()))
        print("  zero_ratio =", _safe_ratio_bool(s, s == 0))

    print("\n[Done]")


if __name__ == "__main__":
    main()
