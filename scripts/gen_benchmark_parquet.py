"""一次性脚本：生成 SH000300 日收益率 parquet 文件，供分钟线回测注入 benchmark。

用法: python gen_benchmark_parquet.py [--start 2024-07-01] [--end 2026-03-10] [--output benchmark_sh000300.parquet]

生成格式: pd.Series(index=DatetimeIndex, values=daily_return, name='bench')
"""
import argparse
from pathlib import Path

import pandas as pd
import qlib
from qlib.data import D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default="2026-03-10")
    parser.add_argument("--output", default="benchmark_sh000300.parquet")
    parser.add_argument("--qlib-dir", default="/home/lc999/data/qlib_bin")
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_dir, region="cn")

    # 获取 000300.sh (沪深300指数) 日收益率
    df = D.features(
        ["000300.sh"],
        ["$close/Ref($close,1)-1"],
        start_time=args.start,
        end_time=args.end,
        freq="day",
    )
    # df: MultiIndex (instrument, datetime) → 取出 Series
    df.columns = ["bench"]
    sr = df["bench"].droplevel("instrument")
    sr.index.name = "datetime"

    # 保存
    out = Path(args.output)
    sr.to_frame().to_parquet(out)
    print(f"Saved {len(sr)} rows to {out.resolve()}")
    print(f"Date range: {sr.index.min()} ~ {sr.index.max()}")
    print(f"Sample:\n{sr.head()}")


if __name__ == "__main__":
    main()
