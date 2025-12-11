"""Quick script to test the AIstock Qlib snapshot from WSL.

Usage (in WSL):

    conda activate rdagent-gpu
    python test_qlib_snapshot.py

This will:
- init qlib with the given provider_uri
- print some available instruments
- fetch a small slice of daily data
- (optionally) try to fetch 1-minute data
"""

from pathlib import Path
import traceback

import qlib
from qlib.data import D

# WSL can access the Windows path via /mnt/c/...
PROVIDER_URI = "/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_20251206"
REGION = "cn"

DAY_START = "2025-06-01"
DAY_END = "2025-06-06"


def main() -> None:
    print(f"Initializing qlib with provider_uri={PROVIDER_URI!r}, region={REGION!r}")
    qlib.init(
        provider_uri=PROVIDER_URI,
        region=REGION,
        auto_mount=False,  # avoid qlib modifying the path
    )

    print("\n[1] Get instruments (market='all') and test daily features:")
    try:
        insts = D.instruments(market="all")
        print("D.instruments(market='all') type:", type(insts))
        print("D.instruments(market='all') repr (truncated):", repr(insts)[:200], "...")
    except Exception as e:  # noqa: BLE001
        print("Failed to get instruments:", repr(e))
        return

    print(f"\n[2] Try loading daily features for instruments=insts, {DAY_START}~{DAY_END}, with default freq:")
    try:
        df_day = D.features(
            insts,
            fields=["$close", "$open", "$high", "$low", "$volume"],
            start_time=DAY_START,
            end_time=DAY_END,
        )
        print("Loaded daily features successfully.")
        print(df_day.head())
        print("\nDaily rows:", len(df_day))
    except Exception as e:  # noqa: BLE001
        print("Failed to load daily data:", repr(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
