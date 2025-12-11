import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Pre-compute stock daily basic factor table from daily_basic.h5.

    - Input  H5:  <snapshot_root>/daily_basic.h5
    - Output H5:  C:/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors/result.h5

    Index: MultiIndex(datetime, instrument)
    Columns (examples):
        value_pe_inv          1 / db_pe_ttm（估值越低越大）
        value_pb_inv          1 / db_pb
        size_log_mv           log(db_circ_mv)
        liquidity_turnover    db_turnover_rate
        liquidity_vol_ratio   db_volume_ratio
    """

    snapshot_root = os.environ.get(
        "AIstock_SNAPSHOT_ROOT",
        r"C:/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_20251209",
    )

    # 在 Windows 下直接使用 C:/... 路径；在 WSL/Linux 下，将 C:/ 前缀转换为 /mnt/c/
    if os.name != "nt" and (snapshot_root.startswith("C:/") or snapshot_root.startswith("c:/")):
        snapshot_root_path = Path("/mnt/c") / Path(snapshot_root[3:])
    else:
        snapshot_root_path = Path(snapshot_root)

    daily_basic_path = snapshot_root_path / "daily_basic.h5"
    if not daily_basic_path.exists():
        raise FileNotFoundError(f"daily_basic.h5 not found at {daily_basic_path}")

    print(f"[INFO] Loading daily_basic from {daily_basic_path}")
    db = pd.read_hdf(daily_basic_path, key="data")

    df = pd.DataFrame(index=db.index)

    # 估值相关
    if "db_pe_ttm" in db.columns:
        df["value_pe_inv"] = 1.0 / db["db_pe_ttm"].replace(0, np.nan)
    elif "db_pe" in db.columns:
        df["value_pe_inv"] = 1.0 / db["db_pe"].replace(0, np.nan)

    if "db_pb" in db.columns:
        df["value_pb_inv"] = 1.0 / db["db_pb"].replace(0, np.nan)

    # 市值（规模）相关
    mv_col = None
    for c in ["db_circ_mv", "db_total_mv"]:
        if c in db.columns:
            mv_col = c
            break
    if mv_col is not None:
        df["size_log_mv"] = np.log(db[mv_col].where(db[mv_col] > 0)).replace(-np.inf, np.nan)

    # 流动性相关
    if "db_turnover_rate" in db.columns:
        df["liquidity_turnover"] = db["db_turnover_rate"]

    if "db_volume_ratio" in db.columns:
        df["liquidity_vol_ratio"] = db["db_volume_ratio"]

    # 输出根目录：在 Windows 下直接写入 C:/，在 WSL/Linux 下映射到 /mnt/c/
    output_root_str = r"C:/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors"
    if os.name != "nt" and (output_root_str.startswith("C:/") or output_root_str.startswith("c:/")):
        output_root = Path("/mnt/c") / Path(output_root_str[3:])
    else:
        output_root = Path(output_root_str)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "result.h5"

    print(f"[INFO] Saving daily basic factors to {output_path} with shape={df.shape}")
    # HDF5: 通用存储格式，便于后续调试
    df.to_hdf(output_path, key="data", mode="w")

    # Pickle: 供 Qlib StaticDataLoader 直接读取
    pkl_path = output_root / "result.pkl"
    print(f"[INFO] Saving daily basic factors pickle to {pkl_path}")
    df.to_pickle(pkl_path)

    # Parquet: 供其他分析/调试使用
    parquet_path = output_root / "result.parquet"
    print(f"[INFO] Saving daily basic factors parquet to {parquet_path}")
    df.to_parquet(parquet_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
