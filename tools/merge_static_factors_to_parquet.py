import pandas as pd
from pathlib import Path


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(".env", override=True)
    except Exception:
        return

"""
线下合并静态因子表为一个统一的 parquet，供 Qlib 的 StaticDataLoader 使用。

当前合并来源：
- workspace 下的 combined_factors_df.parquet（如果存在）
- AE 因子：/mnt/f/Dev/AIstock/factors/ae_recon_error_10d/result.pkl
- daily_basic 因子：/mnt/f/Dev/AIstock/factors/daily_basic_factors/result.pkl

输出：
- /mnt/f/Dev/AIstock/factors/combined_static_factors.parquet

使用方式（在 WSL 中）：

    cd /mnt/f/dev/RD-Agent-main
    conda activate rdagent-gpu  # 按你的环境名调整
    python tools/merge_static_factors_to_parquet.py

脚本只做线下合并，不改 YAML。后续可以在 conf_combined_factors.yaml 中
用一个 StaticDataLoader 指向该输出文件。
"""


_load_dotenv_if_available()


def _get_factors_root() -> Path:
    import os

    root = (
        os.environ.get("AISTOCK_FACTORS_ROOT", "")
        or os.environ.get("AIstock_FACTORS_ROOT", "")
        or "/mnt/f/Dev/AIstock/factors"
    )
    return Path(root)


def _get_rep_workspace_dirs() -> list[Path]:
    import os

    raw = os.environ.get("MERGE_STATIC_FACTORS_REP_WORKSPACE_DIRS", "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(":") if p.strip()]
        return [Path(p) for p in parts]

    # Fallback: keep previous default (may not exist on every machine)
    return [
        Path("/mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/5223c0ad21f442ddb64c887b2c0a9d09"),
    ]


# 可以通过 env 覆盖：MERGE_STATIC_FACTORS_REP_WORKSPACE_DIRS=/path/ws1:/path/ws2
REP_WORKSPACE_DIRS = _get_rep_workspace_dirs()

FACTORS_ROOT = _get_factors_root()
AE_PATH = FACTORS_ROOT / "ae_recon_error_10d" / "result.pkl"
DAILY_BASIC_PATH = FACTORS_ROOT / "daily_basic_factors" / "result.pkl"
MONEYFLOW_FACTORS_PATH = FACTORS_ROOT / "moneyflow_factors" / "result.pkl"
OUTPUT_PATH = FACTORS_ROOT / "combined_static_factors.parquet"


def find_workspace_combined() -> Path | None:
    for ws in REP_WORKSPACE_DIRS:
        if not ws.exists():
            continue
        cand = ws / "combined_factors_df.parquet"
        if cand.exists():
            return cand
    return None


def _ensure_multiindex(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """确保 DataFrame 使用 MultiIndex(datetime, instrument)。

    - 如果本来就是 MultiIndex，则按索引排序后返回；
    - 如果是单层索引，但同时存在 datetime/instrument 列，则用这两列构造 MultiIndex；
    - 否则，将当前索引视为 instrument，新增一个虚拟 datetime 层（需要人工后续检查）。
    """

    if isinstance(df.index, pd.MultiIndex):
        print(f"[调试] {name}: 已是 MultiIndex, nlevels={df.index.nlevels}, names={df.index.names}")
        return df.sort_index()

    print(f"[调试] {name}: 检测到单层索引, index.name={df.index.name}, 尝试修复为 MultiIndex...")

    # 优先使用显式列
    cols = set(df.columns)
    if {"datetime", "instrument"}.issubset(cols):
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index(["datetime", "instrument"]).sort_index()
        print(f"[调试] {name}: 通过列 datetime/instrument 构造 MultiIndex 完成")
        return df

    # 退而求其次：如果 index 看起来是 datetime，且有 instrument 列
    if "instrument" in cols and (df.index.inferred_type in {"datetime64", "date", "mixed"}):
        df = df.reset_index().rename(columns={df.index.name or "index": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index(["datetime", "instrument"]).sort_index()
        print(f"[调试] {name}: 通过 index+instrument 列构造 MultiIndex 完成")
        return df

    # 最保守：把当前 index 当作 instrument，构造一个虚拟 datetime 层
    df = df.copy()
    df["instrument"] = df.index.astype(str)
    df["datetime"] = pd.Timestamp("1900-01-01")
    df = df.set_index(["datetime", "instrument"]).sort_index()
    print(f"[警告] {name}: 无法从列中恢复标准索引, 使用占位 datetime=1900-01-01 + 原 index 作为 instrument")
    return df


def load_multiindex_df(path: Path, name: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)

    print(f"[调试] 载入 {name} ({path}) 时的原始索引信息: type={type(df.index)}, nlevels={getattr(df.index, 'nlevels', 'N/A')}, names={getattr(df.index, 'names', None)}")
    df = _ensure_multiindex(df, name)
    print(f"[调试] 修复后 {name} 索引: type={type(df.index)}, nlevels={df.index.nlevels}, names={df.index.names}")
    return df


def merge_and_sort(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    print("Merging factor tables with pd.concat(axis=1)...")
    merged = pd.concat(dfs, axis=1)
    print("Merged shape:", merged.shape)
    print("Merged index type:", type(merged.index))
    print("Merged index names:", merged.index.names)
    if isinstance(merged.index, pd.MultiIndex):
        print("Merged index levels:", merged.index.nlevels)
        print("Merged index level names:", merged.index.names)

    # 为避免 pandas 在 MultiIndex 上做区间切片时抛出 UnsortedIndexError，
    # 需要保证索引按 (datetime, instrument) 做 lex 排序。
    print("Sorting merged index by (datetime, instrument)...")
    before_lexsort_depth = getattr(merged.index, "lexsort_depth", None)
    print("  Before sort: lexsort_depth =", before_lexsort_depth)

    try:
        merged = merged.sort_index()
    except Exception as e:
        print("Warning: sort_index() failed, fallback to sort_index(level=[\"datetime\", \"instrument\"]) if possible. Error:", repr(e))
        if isinstance(merged.index, pd.MultiIndex) and set(merged.index.names) >= {"datetime", "instrument"}:
            merged = merged.sort_index(level=["datetime", "instrument"])

    after_lexsort_depth = getattr(merged.index, "lexsort_depth", None)
    print("  After sort: lexsort_depth =", after_lexsort_depth)
    return merged


def main() -> None:
    print("[信息] 查找 workspace 中的 combined_factors_df.parquet ...")
    ws_combined = find_workspace_combined()
    if ws_combined is not None:
        print("[信息] 使用 workspace combined:", ws_combined)
        df_combined = load_multiindex_df(ws_combined, name="workspace_combined")
    else:
        print("[提示] 未找到 workspace combined_factors_df.parquet，将只合并 AE 与 daily_basic。")
        df_combined = None

    dfs: list[pd.DataFrame] = []

    if df_combined is not None:
        print("  - 添加 combined_factors_df，列数:", df_combined.shape[1])
        dfs.append(df_combined)

    if AE_PATH.exists():
        df_ae = load_multiindex_df(AE_PATH, name="ae_factors")
        print("  - 添加 AE 因子，列数:", df_ae.shape[1])
        dfs.append(df_ae)
    else:
        print("[警告] 未找到 AE 因子表:", AE_PATH)

    if DAILY_BASIC_PATH.exists():
        df_db = load_multiindex_df(DAILY_BASIC_PATH, name="daily_basic_factors")
        print("  - 添加 daily_basic 因子，列数:", df_db.shape[1])
        dfs.append(df_db)
    else:
        print("[警告] 未找到 daily_basic 因子表:", DAILY_BASIC_PATH)

    # 个股资金流向因子（moneyflow_factors），由 precompute_moneyflow_factors.py 生成
    if MONEYFLOW_FACTORS_PATH.exists():
        df_mf = load_multiindex_df(MONEYFLOW_FACTORS_PATH, name="moneyflow_factors")
        print("  - 添加 moneyflow 因子，列数:", df_mf.shape[1])
        dfs.append(df_mf)
    else:
        print("[提示] 未找到 moneyflow 因子表:", MONEYFLOW_FACTORS_PATH)

    if not dfs:
        print("[错误] 没有任何可合并的因子表，终止。")
        return

    print("[信息] 开始按索引对齐并合并列（含排序）...")
    df_merged = merge_and_sort(dfs)
    print("[信息] 合并完成，形状:", df_merged.shape)
    print("[信息] 输出路径:", OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(OUTPUT_PATH)
    print("[成功] 已写出:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
