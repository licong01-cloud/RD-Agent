import argparse
import os
from typing import List, Optional

import pandas as pd


def _init_qlib(provider_uri: str, region: str = "cn"):
    import qlib

    qlib.init(provider_uri=provider_uri, region=region)


def _load_features(
    instruments,
    fields: List[str],
    start: str,
    end: str,
    freq: str = "day",
):
    from qlib.data import D

    df = D.features(instruments, fields, start_time=start, end_time=end, freq=freq)
    # Columns in qlib features are usually exactly the expressions requested
    return df.sort_index()


def _summarize_single_inst(df_inst: pd.DataFrame) -> dict:
    # Expect columns include $close and $volume (exact names)
    col_close = "$close" if "$close" in df_inst.columns else "close"
    col_vol = "$volume" if "$volume" in df_inst.columns else "volume"

    s_close = df_inst[col_close]
    s_vol = df_inst[col_vol]

    same_close = s_close.eq(s_close.shift(1))
    rule_a = (s_vol == 0) & same_close
    rule_b = s_vol.isna()

    return {
        "rows": int(len(df_inst)),
        "volume_nan": int(s_vol.isna().sum()),
        "volume_zero": int((s_vol == 0).sum()),
        "close_nan": int(s_close.isna().sum()),
        "rule_a_cnt": int(rule_a.sum()),
        "rule_b_cnt": int(rule_b.sum()),
    }


def _print_df_tail(title: str, df: pd.DataFrame, n: int = 20) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    if df.empty:
        print("<EMPTY>")
    else:
        print(df.tail(n))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check Qlib CN dataset suspension/missing behavior by scanning for:\n"
            "A) volume==0 AND close==prev_close (ffill-like / padded rows)\n"
            "B) volume is NaN (missing)\n"
            "Outputs top instruments and sample rows for both patterns."
        )
    )
    parser.add_argument(
        "--provider-uri",
        default=os.path.expanduser("~/.qlib/qlib_data/cn_data"),
        help="Qlib provider_uri path (default: ~/.qlib/qlib_data/cn_data)",
    )
    parser.add_argument(
        "--region",
        default="cn",
        help="Qlib region (default: cn)",
    )
    parser.add_argument(
        "--market",
        default="csi300",
        help="Instrument universe/market name passed to D.instruments (default: csi300). Use 'all' for full market.",
    )
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--sample-inst",
        default=None,
        help="Optional: check a single instrument first (e.g. 000001.SZ).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Top K instruments to print (default: 20)",
    )

    args = parser.parse_args()

    provider_uri = args.provider_uri
    print(f"Initializing qlib with provider_uri={provider_uri!r}, region={args.region!r}")
    _init_qlib(provider_uri=provider_uri, region=args.region)

    from qlib.data import D

    print(f"Loading instruments market={args.market!r} ...")
    instruments = D.instruments(args.market)

    fields = ["$close", "$volume"]

    if args.sample_inst:
        print(f"\n[SingleInstrumentCheck] {args.sample_inst}")
        df1 = _load_features([args.sample_inst], fields, args.start, args.end)
        # df1 is multiindex with instrument level; filter
        try:
            df1_inst = df1.xs(args.sample_inst, level="instrument")
        except Exception:
            # fallback: maybe lowercase instrument in index
            df1_inst = df1.reset_index()
            df1_inst = df1_inst[df1_inst["instrument"].astype(str).str.upper() == args.sample_inst.upper()]
            df1_inst = df1_inst.set_index("datetime").sort_index()[fields]
        summary = _summarize_single_inst(df1_inst)
        print("summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        s_close = df1_inst["$close"]
        s_vol = df1_inst["$volume"]
        same_close = s_close.eq(s_close.shift(1))
        sus_a = df1_inst[(s_vol == 0) & same_close]
        sus_b = df1_inst[s_vol.isna()]
        _print_df_tail(f"[SingleInstrumentCheck] RuleA (volume==0 & same_close) tail", sus_a)
        _print_df_tail(f"[SingleInstrumentCheck] RuleB (volume is NaN) tail", sus_b)

    print(
        f"\n[MarketScan] market={args.market!r}, start={args.start!r}, end={args.end!r}, fields={fields}"
    )
    df = _load_features(instruments, fields, args.start, args.end)

    # Ensure expected columns exist
    missing_cols = [c for c in fields if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing columns from qlib D.features result: {missing_cols}; got {list(df.columns)}")

    # Group by instrument to compute prev_close
    g = df.groupby(level="instrument", group_keys=False)
    same_close = g["$close"].apply(lambda s: s.eq(s.shift(1)))

    cand_a = df[(df["$volume"] == 0) & same_close]
    cand_b = df[df["$volume"].isna()]

    print("\nCounts:")
    print(f"  total rows: {len(df)}")
    print(f"  A(volume==0 & same_close) rows: {len(cand_a)}")
    print(f"  B(volume is NaN) rows: {len(cand_b)}")

    def _top_counts(cand: pd.DataFrame, name: str) -> pd.Series:
        if cand.empty:
            return pd.Series(dtype=int, name=name)
        return cand.reset_index().groupby("instrument").size().sort_values(ascending=False).rename(name)

    top_a = _top_counts(cand_a, "ruleA_cnt")
    top_b = _top_counts(cand_b, "ruleB_cnt")

    print(f"\nTop {args.topk} instruments by RuleA count:")
    if top_a.empty:
        print("<EMPTY>")
    else:
        print(top_a.head(args.topk))

    print(f"\nTop {args.topk} instruments by RuleB count:")
    if top_b.empty:
        print("<EMPTY>")
    else:
        print(top_b.head(args.topk))

    _print_df_tail("Sample RuleA rows (tail)", cand_a)
    _print_df_tail("Sample RuleB rows (tail)", cand_b)

    # additionally: print a few candidate instruments for manual inspection
    if not top_a.empty:
        print("\nSuggested candidate instruments (RuleA top 10):")
        print(list(top_a.head(10).index))
    if not top_b.empty:
        print("\nSuggested candidate instruments (RuleB top 10):")
        print(list(top_b.head(10).index))


if __name__ == "__main__":
    main()
