import pandas as pd
from pathlib import Path

"""
修复 combined_factors_df.parquet 的索引为 MultiIndex(datetime, instrument)。

使用方式（在 WSL 中示例）：

    cd /mnt/f/dev/RD-Agent-main
    conda activate rdagent-gpu  # 按你的环境名调整
    python tools/fix_combined_factors_index.py

脚本逻辑：
1. 加载 combined_factors_df.parquet；
2. 打印当前 index 类型 / 层级数 / 名称；
3. 若 index 已经是 MultiIndex(datetime, instrument)，则不改动，仅提示；
4. 否则：
   - 如果存在 'datetime' 和 'instrument' 两列：
       * 将这两列转为 MultiIndex(datetime, instrument)；
       * 按 index 排序；
       * 覆盖写回原 parquet 文件；
   - 如果不存在这两列，则仅打印列名，提示需要人工检查。
"""


def main() -> None:
    path = Path("/mnt/f/Dev/AIstock/qlib_rd_workspace/combined_factors_df.parquet")

    if not path.exists():
        print("[错误] 找不到文件:", path)
        return

    print("[信息] 加载:", path)
    df = pd.read_parquet(path)

    print("=== 修复前索引信息 ===")
    print("index type:", type(df.index))
    print("index.nlevels:", getattr(df.index, "nlevels", "N/A"))
    print("index.names:", getattr(df.index, "names", None))
    print("前几行 index 示例:", list(df.index[:5]))
    print()

    # 如果已经是 MultiIndex(datetime, instrument)，直接返回
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["datetime", "instrument"]:
        print("[提示] 当前索引已经是 MultiIndex(['datetime', 'instrument'])，不需要修复。")
        return

    # 尝试根据列构造 MultiIndex
    cols = set(df.columns)
    if {"datetime", "instrument"}.issubset(cols):
        print("[信息] 检测到 'datetime' 和 'instrument' 两列，将据此重建 MultiIndex...")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index(["datetime", "instrument"]).sort_index()

        print("=== 修复后索引信息 ===")
        print("index type:", type(df.index))
        print("index.nlevels:", df.index.nlevels)
        print("index.names:", df.index.names)
        print("前几行 index 示例:", list(df.index[:5]))

        # 覆盖写回
        df.to_parquet(path)
        print("[成功] 已覆盖写回:", path)
    else:
        print("[警告] DataFrame 中不存在 'datetime' 和 'instrument' 两列，无法自动修复索引。")
        print("当前列名列表:")
        print(sorted(df.columns.tolist()))


if __name__ == "__main__":
    main()
