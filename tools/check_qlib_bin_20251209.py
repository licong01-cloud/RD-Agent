"""Check structure and coverage of qlib_bin_20251209.

This script is **read-only**: it will not modify any files under the bin directory.

Usage (WSL):

    cd /mnt/c/Users/lc999/RD-Agent-main
    python tools/check_qlib_bin_20251209.py
"""

import pprint

import qlib
from qlib.data import D

PROVIDER_URI = "/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209"
REGION = "cn"


def main() -> None:
    print("初始化 Qlib...")
    qlib.init(provider_uri=PROVIDER_URI, region=REGION)
    print(f"provider_uri = {PROVIDER_URI}")

    # 1) 查看 'all' 股票池配置
    print("\n=== 1. stock pool config: 'all' ===")
    pool_cfg = D.instruments("all")
    pprint.pprint(pool_cfg)

    # 2) 展开为真实股票列表（覆盖完整导出区间 2010-01-07 ~ 2025-12-01）
    print("\n=== 2. 展开 'all' 为真实股票列表 (D.list_instruments) ===")
    insts = list(
        D.list_instruments(
            pool_cfg,
            start_time="2010-01-07",
            end_time="2025-12-01",
        )
    )
    print(f"展开后的真实股票数量: {len(insts)}")
    preview_n = min(20, len(insts))
    print(f"前 {preview_n} 个示例:")
    for code in insts[:preview_n]:
        print("  ", code)

    # 如果只有 VERY 少的 instrument（例如 1 个 DAILY_ALL），直接提示并返回
    if len(insts) <= 3:
        print("\n[警告] 当前 bin 在 'all' 股票池下仅包含少量 instrument,\n"
              "      可能是聚合序列(如 DAILY_ALL)，不适合作为横截面研究数据。")
        return

    # 3) 检查完整区间内的交易日历是否正常
    print("\n=== 3. 交易日历检查 (D.calendar) ===")
    cal = D.calendar(start_time="2010-01-07", end_time="2025-12-01", freq="day")
    print(f"2010-01-07 ~ 2025-12-01 交易日数量: {len(cal)}")
    print("前 10 个交易日:")
    for d in cal[:10]:
        print("  ", d)

    # 4) 抽样若干股票，检查特征数据是否存在且结构合理
    print("\n=== 4. 抽样若干股票检查特征数据 (D.features) ===")
    sample_insts = insts[:5]
    fields = ["$close", "$volume"]
    for inst in sample_insts:
        print(f"\n[标的] {inst}")
        try:
            # 某些 Qlib 版本要求 instrument 为 list / Index，而不是单个字符串
            # 这里在完整导出区间内选取前 1 个月的数据做抽样检查
            df = D.features(
                [inst],
                fields,
                start_time="2010-01-07",
                end_time="2010-02-07",
                freq="day",
            )
            print("  样本行数:", df.shape[0])
            print("  列:", list(df.columns))
            if not df.empty:
                print("  前 3 行:")
                print(df.head(3))
        except Exception as e:
            print("  [错误] 读取特征数据失败:", repr(e))


if __name__ == "__main__":
    main()
