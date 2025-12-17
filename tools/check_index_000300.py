"""Check 000300 index data in qlib_bin_20251209.

Usage (WSL):

    cd /mnt/f/dev/RD-Agent-main
    python tools/check_index_000300.py
"""

import qlib
from qlib.data import D

PROVIDER_URI = "/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209"
REGION = "cn"
INDEX_CODE = "000300.SH"  # 按导出规范使用的指数代码


def main() -> None:
    print("初始化 Qlib...")
    qlib.init(provider_uri=PROVIDER_URI, region=REGION)
    print(f"provider_uri = {PROVIDER_URI}")

    # 1) 直接按代码读取指数行情
    print("\n=== 1. 直接读取 000300.SH 日线特征 ===")
    try:
        df = D.features([INDEX_CODE], ["$close", "$volume"],
                        start_time="2016-01-01", end_time="2016-01-15", freq="day")
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print("前 5 行:")
        print(df.head())
    except Exception as e:
        print("[错误] D.features 读取失败:", repr(e))

    # 2) 如果定义了 index 股票池，检查是否包含 000300.SH
    print("\n=== 2. 若存在 'index' 股票池，检查是否包含 000300.SH ===")
    try:
        pool_cfg = D.instruments("index")
        print("index pool config:", pool_cfg)
        insts = list(D.list_instruments(pool_cfg, start_time="2010-01-07", end_time="2025-12-01"))
        print("index 池 instrument 数量:", len(insts))
        print("前 10 个:", insts[:10])
        print("000300.SH 是否在池中:", INDEX_CODE in insts)
    except Exception as e:
        print("[提示] 获取 'index' 股票池失败，可能尚未配置:", repr(e))


if __name__ == "__main__":
    main()
