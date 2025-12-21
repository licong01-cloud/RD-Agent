import pandas as pd

"""
检查 AE / daily_basic 因子表的索引结构。

使用方式（在 WSL 中示例）：

    cd /mnt/f/dev/RD-Agent-main
    conda activate rdagent-gpu  # 按你的环境名调整
    python tools/check_factor_indexes.py

脚本只读不写：
- 对每个因子表打印 index 类型、层级数、名称和前几行示例；
- 方便确认是否已经是 MultiIndex(['datetime', 'instrument'])。
"""


def main() -> None:
    paths = {
        "ae": "/mnt/f/Dev/AIstock/factors/ae_recon_error_10d/result.pkl",
        "daily_basic": "/mnt/f/Dev/AIstock/factors/daily_basic_factors/result.pkl",
    }

    for name, path in paths.items():
        try:
            df = pd.read_pickle(path)
        except FileNotFoundError:
            print(name, "NOT FOUND:", path)
            continue

        print("==", name, "==")
        print("index type:", type(df.index))
        print("index.nlevels:", df.index.nlevels)
        print("index.names:", df.index.names)
        print("head index sample:", df.index[:3])
        print()


if __name__ == "__main__":
    main()
