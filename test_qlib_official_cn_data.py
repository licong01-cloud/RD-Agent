"""Test whether the current qlib installation can read the official cn_data.

Usage (in WSL):

    conda activate rdagent-gpu
    python test_qlib_official_cn_data.py

This will:
- init qlib with provider_uri="~/.qlib/qlib_data/cn_data"
- get instruments for market="csi300"
- try to load a small slice of daily close prices
"""

import os

import qlib
from qlib.data import D

PROVIDER_URI = os.path.expanduser("~/.qlib/qlib_data/cn_data")
REGION = "cn"
DAY_START = "2020-01-01"
DAY_END = "2020-01-10"


def main() -> None:
    print(f"Initializing qlib with provider_uri={PROVIDER_URI!r}, region={REGION!r}")
    qlib.init(
        provider_uri=PROVIDER_URI,
        region=REGION,
        auto_mount=False,
    )

    print("\n[1] Get instruments for market='csi300':")
    try:
        insts = D.instruments(market="csi300")
        print("D.instruments(market='csi300') type:", type(insts))
        print("D.instruments(market='csi300') repr (truncated):", repr(insts)[:200], "...")
    except Exception as e:  # noqa: BLE001
        print("Failed to get instruments:", repr(e))
        return

    print(f"\n[2] Try loading daily features for instruments=insts, {DAY_START}~{DAY_END}, freq='day':")
    try:
        df_day = D.features(
            insts,
            fields=["$close"],
            start_time=DAY_START,
            end_time=DAY_END,
            freq="day",
        )
        print("Loaded daily features successfully.")
        print(df_day.head())
        print("\nDaily rows:", len(df_day))
    except Exception as e:  # noqa: BLE001
        print("Failed to load daily data:", repr(e))


if __name__ == "__main__":
    main()
