import qlib
from qlib.data import D

PROVIDER_URI = "/mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251206"


def main() -> None:
    print("初始化 Qlib...")
    qlib.init(provider_uri=PROVIDER_URI, region="cn")

    # 1. 检查 market="all" 下的股票池
    print("\n=== list_instruments(instruments=\"all\") ===")
    try:
        insts = D.list_instruments(instruments="all", as_list=True)
        print("总股票数:", len(insts))
        print("前 10 个标的:", insts[:10])
    except Exception as e:
        print("list_instruments 出错:", repr(e))
        return

    # 2. 如果有股票，抽样一些看 features
    if not insts:
        print("market='all' 下没有任何 instrument，后续无法继续测试 features。")
        return

    sample_insts = insts[:50]
    print("\n=== D.features 日线数据 2016-01-01 ~ 2016-03-31 (前 50 只股票) ===")
    try:
        df = D.features(
            instruments=sample_insts,
            fields=["$open", "$high", "$low", "$close", "$volume"],
            start_time="2016-01-01",
            end_time="2016-03-31",
            freq="day",
        )
        print("features shape:", df.shape)
        print("index 前 5 行:")
        print(df.index[:5])
        print("columns:", list(df.columns))
    except Exception as e:
        print("D.features 出错:", repr(e))


if __name__ == "__main__":
    main()
