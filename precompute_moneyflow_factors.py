import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Pre-compute moneyflow-based capital flow factors into a standard HDF5 table.

    - Input  H5:  <snapshot_root>/moneyflow.h5, <snapshot_root>/daily_pv.h5, optional <snapshot_root>/daily_basic.h5
    - Output H5:  C:/Users/lc999/NewAIstock/AIstock/factors/capital_flow_daily/result.h5

    Index: MultiIndex(datetime, instrument)
    Columns (examples):
        mf_main_net_amt          主力净流入金额（特大单+大单）
        mf_main_net_ratio        主力净流入占当日成交额
        mf_main_net_amt_5d       5 日滚动主力净流入金额
        mf_main_net_ratio_5d     5 日滚动主力净流入占比
        mf_main_net_mv_5d        5 日主力净流入 / 流通市值（如 daily_basic 可用）
    """

    # 默认 snapshot 根目录，可通过环境变量覆盖
    snapshot_root = os.environ.get(
        "AIstock_SNAPSHOT_ROOT",
        r"C:/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_20251209",
    )

    # 在 Windows 下直接使用 C:/... 路径；在 WSL/Linux 下，将 C:/ 前缀转换为 /mnt/c/
    if os.name != "nt" and (snapshot_root.startswith("C:/") or snapshot_root.startswith("c:/")):
        snapshot_root_path = Path("/mnt/c") / Path(snapshot_root[3:])
    else:
        snapshot_root_path = Path(snapshot_root)

    moneyflow_path = snapshot_root_path / "moneyflow.h5"
    daily_pv_path = snapshot_root_path / "daily_pv.h5"
    daily_basic_path = snapshot_root_path / "daily_basic.h5"

    if not moneyflow_path.exists():
        raise FileNotFoundError(f"moneyflow.h5 not found at {moneyflow_path}")

    if not daily_pv_path.exists():
        raise FileNotFoundError(f"daily_pv.h5 not found at {daily_pv_path}")

    # 读 moneyflow 面板
    print(f"[INFO] Loading moneyflow from {moneyflow_path}")
    mf = pd.read_hdf(moneyflow_path, key="data")

    # 读 daily_pv 以获得成交额 amount（如有）
    print(f"[INFO] Loading daily_pv from {daily_pv_path}")
    pv = pd.read_hdf(daily_pv_path, key="data")

    # 对齐索引
    common_index = mf.index.intersection(pv.index)
    mf = mf.loc[common_index].copy()
    pv = pv.loc[common_index].copy()

    # 尽量使用 H5 备忘录中的约定列名
    amount_col_candidates = ["$amount", "amount"]
    amount_col = None
    for c in amount_col_candidates:
        if c in pv.columns:
            amount_col = c
            break
    if amount_col is None:
        raise KeyError(f"Cannot find amount column in daily_pv.h5, tried {amount_col_candidates}")

    # 特/大单买卖金额列（若不存在则置为 0）
    def get_mf_col(name: str) -> pd.Series:
        return mf[name] if name in mf.columns else pd.Series(0.0, index=mf.index)

    elg_buy = get_mf_col("mf_elg_buy_amt")
    elg_sell = get_mf_col("mf_elg_sell_amt")
    lg_buy = get_mf_col("mf_lg_buy_amt")
    lg_sell = get_mf_col("mf_lg_sell_amt")

    # 主力净流入金额（特大 + 大单）
    mf_main_net_amt = (elg_buy + lg_buy) - (elg_sell + lg_sell)

    amount = pv[amount_col].replace(0, np.nan)
    mf_main_net_ratio = mf_main_net_amt / amount

    # 多日滚动指标（简单用 groupby + rolling）
    # 注意：moneyflow 与 daily_pv 索引均为 MultiIndex(datetime, instrument)
    df = pd.DataFrame(index=common_index)
    df["mf_main_net_amt"] = mf_main_net_amt
    df["mf_main_net_ratio"] = mf_main_net_ratio

    # 5 日窗口
    df["mf_main_net_amt_5d"] = (
        df["mf_main_net_amt"].groupby(level="instrument").rolling(5).sum().reset_index(level=0, drop=True)
    )
    df["mf_main_net_ratio_5d"] = (
        df["mf_main_net_ratio"].groupby(level="instrument").rolling(5).sum().reset_index(level=0, drop=True)
    )

    # 如有 daily_basic.h5，则计算相对市值的指标
    if daily_basic_path.exists():
        print(f"[INFO] Loading daily_basic from {daily_basic_path}")
        db = pd.read_hdf(daily_basic_path, key="data")
        db = db.reindex(common_index)
        mv_col_candidates = ["db_circ_mv", "db_total_mv"]
        mv_col = None
        for c in mv_col_candidates:
            if c in db.columns:
                mv_col = c
                break
        if mv_col is not None:
            mv = db[mv_col].replace(0, np.nan)
            df["mf_main_net_mv_5d"] = df["mf_main_net_amt_5d"] / mv
        else:
            print(f"[WARN] No market value column found in daily_basic.h5, skip mf_main_net_mv_5d")
    else:
        print(f"[INFO] daily_basic.h5 not found at {daily_basic_path}, skip relative-to-MV factors")

    # 输出路径：在 Windows 下直接写入 C:/，在 WSL/Linux 下映射到 /mnt/c/
    output_root_str = r"C:/Users/lc999/NewAIstock/AIstock/factors/capital_flow_daily"
    if os.name != "nt" and (output_root_str.startswith("C:/") or output_root_str.startswith("c:/")):
        output_root = Path("/mnt/c") / Path(output_root_str[3:])
    else:
        output_root = Path(output_root_str)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "result.h5"

    print(f"[INFO] Saving capital flow factors to {output_path} with shape={df.shape}")
    # HDF5: 供通用脚本/调试使用
    df.to_hdf(output_path, key="data", mode="w")

    # Pickle: 供 Qlib StaticDataLoader 通过 pickle.load 直接读取
    pkl_path = output_root / "result.pkl"
    print(f"[INFO] Saving capital flow factors pickle to {pkl_path}")
    df.to_pickle(pkl_path)

    # Parquet: 可供其他分析/调试使用
    parquet_path = output_root / "result.parquet"
    print(f"[INFO] Saving capital flow factors parquet to {parquet_path}")
    df.to_parquet(parquet_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
