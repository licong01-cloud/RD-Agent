import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def export_sh000300_benchmark(
    provider_uri: str,
    instrument: str,
    target_path: str,
    start_time: str = "2010-01-07",
    end_time: str = "2025-12-01",
    use_return: bool = True,
) -> None:
    """Export 000300.SH from a qlib bin as a benchmark series.

    This script does **not** assume any specific internal qlib cn_data layout.
    It simply:
    - reads `$close` of the given instrument from the specified provider_uri
    - builds a daily time series (date, close[, return])
    - if target_path already exists, creates a timestamped backup next to it
    - writes the new data as CSV to target_path

    You can then inspect this CSV and decide whether to:
      - move/rename it to qlib's benchmark location, or
      - adapt it to the exact format of your existing benchmark file.
    """

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing target if any
    if target.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = target.with_suffix(target.suffix + f".bak_{ts}")
        print(f"[INFO] Target exists, creating backup: {backup}")
        shutil.copy2(target, backup)

    # Lazy import qlib only when needed
    try:
        import qlib
        from qlib.data import D
    except ImportError as e:
        print("[ERROR] qlib is not installed in this environment.")
        print(e)
        sys.exit(1)

    print(f"[INFO] Initializing qlib with provider_uri={provider_uri!r}")
    qlib.init(provider_uri=provider_uri, region="cn")

    print(f"[INFO] Loading $close for instrument={instrument!r} from {start_time} to {end_time}")
    df = D.features(
        [instrument],
        ["$close"],
        start_time=start_time,
        end_time=end_time,
    )

    if df.empty:
        print("[WARN] No data loaded for the specified instrument and time range.")
        print("       Nothing will be written.")
        return

    # df index is MultiIndex(instrument, datetime)
    df = df.reset_index()
    if "datetime" not in df.columns:
        raise RuntimeError("Unexpected data format: 'datetime' column missing after reset_index().")

    df = df.rename(columns={"datetime": "date", "$close": "close"})
    df = df.sort_values("date")

    if use_return:
        df["return"] = df["close"].pct_change().fillna(0.0)

    # For a benchmark series we typically only need date + close/return
    cols = ["date", "close"] + (["return"] if use_return else [])
    out = df[cols]

    print(f"[INFO] Writing benchmark CSV to {target}")
    out.to_csv(target, index=False)
    print("[INFO] Done. Please inspect the CSV and align it with your qlib benchmark format if necessary.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export 000300.SH from a qlib bin as a benchmark series and "
            "optionally backup an existing target file."
        )
    )
    parser.add_argument(
        "--provider-uri",
        type=str,
        default="/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209",
        help="qlib provider_uri pointing to your bin (default: current qlib_bin_20251209)",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="000300.SH",
        help="instrument code to export as benchmark (default: 000300.SH)",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        required=True,
        help=(
            "Path to the benchmark file to write. If the file exists, a timestamped "
            "backup will be created next to it before overwriting."
        ),
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default="2010-01-07",
        help="start date for exporting data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default="2025-12-01",
        help="end date for exporting data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-return",
        action="store_true",
        help="if set, do not compute daily return column; only date+close are written",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    export_sh000300_benchmark(
        provider_uri=args.provider_uri,
        instrument=args.instrument,
        target_path=args.target_path,
        start_time=args.start_time,
        end_time=args.end_time,
        use_return=not args.no_return,
    )


if __name__ == "__main__":
    main()
