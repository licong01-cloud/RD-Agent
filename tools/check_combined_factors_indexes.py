import pandas as pd
from pathlib import Path

"""
搜索并检查所有 combined_factors_df.parquet 的索引结构。

使用方式（在 WSL 中示例）：

    cd /mnt/f/dev/RD-Agent-main
    conda activate rdagent-gpu  # 按你的环境名调整
    python tools/check_combined_factors_indexes.py

脚本行为：
- 在以下目录下递归搜索名为 combined_factors_df.parquet 的文件：
  * /mnt/f/Dev/AIstock
  * /mnt/f/dev/RD-Agent-main
- 对每个找到的文件：
  * 打印路径
  * 加载 DataFrame
  * 打印 index 类型、层级数、名称和前几行示例
- 只读不写，用于定位哪一个 combined_factors_df.parquet 可能导致 Qlib 的 MergeError。
"""


SEARCH_ROOTS = [
    Path("/mnt/f/Dev/AIstock"),
    Path("/mnt/f/dev/RD-Agent-main"),
]


def check_one_file(path: Path) -> None:
    print("==============================")
    print("文件:", path)
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        print("[错误] 读取失败:", e)
        return

    print("index type:", type(df.index))
    print("index.nlevels:", getattr(df.index, "nlevels", "N/A"))
    print("index.names:", getattr(df.index, "names", None))
    try:
        print("head index sample:", list(df.index[:5]))
    except Exception as e:  # noqa: BLE001
        print("[警告] 打印 index 样例失败:", e)
    print()


def main() -> None:
    found_any = False
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        print(f"[信息] 在 {root} 下搜索 combined_factors_df.parquet ...")
        for path in root.rglob("combined_factors_df.parquet"):
            found_any = True
            check_one_file(path)

    if not found_any:
        print("[提示] 未在预设目录下找到任何 combined_factors_df.parquet 文件。")


if __name__ == "__main__":
    main()
