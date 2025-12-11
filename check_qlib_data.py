import qlib
from qlib.config import C
from qlib.data import D


def main() -> None:
    """Minimal script to verify Qlib bin dataset exported by AIstock.

    - Initializes Qlib with the local bin snapshot.
    - Reads DAILY_ALL instrument OHLCV between 2025-11-01 and 2025-12-01.
    - Prints basic stats and head/tail for manual inspection.
    """

    qlib.init(
        provider_uri="/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209",
        region="cn",
    )
    print("Qlib initialized with provider_uri=/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209")

    # 使用 Qlib 提供的 instruments 定义
    instruments = D.instruments("all")
    print("D.instruments('all') type:", type(instruments))
    print("D.instruments('all') value:", instruments)

    # 将 instruments 直接传给 D.features
    df = D.features(
        instruments,
        ["$open", "$high", "$low", "$close", "$volume"],
        start_time="2025-11-01",
        end_time="2025-12-01",
    )

    print("DataFrame shape:", df.shape)
    # 打印前几只股票的代码方便确认
    if not df.empty:
        instruments = list({idx[0] for idx in df.index[:100]})
        print("sample instruments:", instruments[:10])
    print("=== HEAD ===")
    print(df.head())
    print("=== TAIL ===")
    print(df.tail())


if __name__ == "__main__":
    main()
