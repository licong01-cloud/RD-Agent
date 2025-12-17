import qlib
from qlib.data import D

PROVIDER_URI = "/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209"
REGION = "cn"


def main() -> None:
    print("初始化 Qlib...")
    qlib.init(provider_uri=PROVIDER_URI, region=REGION)

    print("列出 instruments('all') 在 2016-01-01 到 2016-01-10 区间内的股票列表...")
    # 1) 先查看 stock pool 配置本身（通常是一个包含 market / filter_pipe 等字段的 dict）
    pool_cfg = D.instruments("all")
    print("stock pool config for 'all':", pool_cfg)

    # 2) 再用 D.list_instruments 将该 stock pool 展开为真实股票列表
    insts = D.list_instruments(pool_cfg, start_time="2016-01-01", end_time="2016-01-10")
    insts = list(insts)

    print(f"展开后的真实股票数量: {len(insts)} 只")
    # 只打印前若干只，避免输出过长
    preview_n = min(50, len(insts))
    print(f"前 {preview_n} 个示例: ")
    for code in insts[:preview_n]:
        print(code)


if __name__ == "__main__":
    main()
