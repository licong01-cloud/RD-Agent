import pandas as pd
import qlib
from qlib.data import D

"""
检查 Qlib 原始交易日历 与 当前挂载因子表 的日期覆盖情况。

运行方式（在 WSL 中示例）：

    cd /mnt/f/dev/RD-Agent-main
    conda activate rdagent-gpu  # 按你的环境名调整
    python tools/check_qlib_calendar_vs_factors.py

脚本会输出：
- Qlib 原始交易日历的起止日期与总交易日数；
- 每个因子表的起止日期、去重后的日期数；
- 与 Qlib 日历的交集大小、仅在因子表/仅在 Qlib 日历中的日期数及样例。
"""


def init_qlib():
    """根据当前 RD-Agent Qlib 场景的配置初始化 Qlib。"""
    qlib.init(
        provider_uri="/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209",
        region="cn",
    )


def get_qlib_trading_days() -> pd.DatetimeIndex:
    """获取 Qlib 的日频交易日历。"""
    cal = D.calendar(freq="day")
    days = pd.DatetimeIndex(cal)
    print("=== Qlib 原始交易日历 ===")
    print("min:", days.min())
    print("max:", days.max())
    print("总交易日数:", len(days))
    print()
    return days


def check_factor_dates(path: str, qlib_days: pd.DatetimeIndex, name: str | None = None) -> None:
    """对比单个因子表与 Qlib 交易日历的日期覆盖情况。"""
    name = name or path
    print(f"=== 因子表 [{name}] ===")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)

    # 兼容 MultiIndex(datetime, instrument) 和单层 datetime 两种情况
    if isinstance(df.index, pd.MultiIndex):
        if "datetime" in df.index.names:
            factor_days = df.index.get_level_values("datetime").unique()
        else:
            # 没有命名，用第 0 层当作日期
            factor_days = df.index.get_level_values(0).unique()
    else:
        factor_days = pd.DatetimeIndex(df.index).unique()

    factor_days = pd.DatetimeIndex(factor_days).sort_values()

    print("min:", factor_days.min())
    print("max:", factor_days.max())
    print("总日期数（去重后）:", len(factor_days))

    inter = factor_days.intersection(qlib_days)
    only_in_factor = factor_days.difference(qlib_days)
    only_in_qlib = qlib_days.difference(factor_days)

    print("与 Qlib 交易日历的交集天数:", len(inter))
    print("仅出现在因子表、不在 Qlib 日历中的日期数:", len(only_in_factor))
    print("仅出现在 Qlib 日历、不在因子表中的日期数:", len(only_in_qlib))

    if len(only_in_factor) > 0:
        print("  样例 only_in_factor:", list(only_in_factor[:5]))
    if len(only_in_qlib) > 0:
        print("  样例 only_in_qlib:", list(only_in_qlib[:5]))
    print()


def main() -> None:
    init_qlib()
    qlib_days = get_qlib_trading_days()

    # 1) AE 10 日重构误差因子
    check_factor_dates(
        "/mnt/f/Dev/AIstock/factors/ae_recon_error_10d/result.pkl",
        qlib_days,
        name="AE 10d 重构误差",
    )

    # 2) daily_basic 因子表
    check_factor_dates(
        "/mnt/f/Dev/AIstock/factors/daily_basic_factors/result.pkl",
        qlib_days,
        name="daily_basic 因子表",
    )

    # 3) 若存在 combined_factors_df.parquet，可一并检查
    try:
        check_factor_dates(
            "/mnt/f/Dev/AIstock/qlib_rd_workspace/combined_factors_df.parquet",
            qlib_days,
            name="combined_factors_df.parquet",
        )
    except FileNotFoundError:
        print("[提示] 未找到 combined_factors_df.parquet，跳过该表检查。")


if __name__ == "__main__":
    main()
