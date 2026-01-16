import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _try_get_columns_from_h5(h5_path: Path, key: str = "data") -> list[str]:
    """Try to get DataFrame columns from an HDF5 written by pandas (format='fixed').

    We avoid loading the full dataset when possible.
    """

    if not h5_path.exists():
        return []

    try:
        with pd.HDFStore(str(h5_path), mode="r") as store:
            h5_key = f"/{key}" if f"/{key}" in store.keys() else key
            if h5_key not in store:
                return []

            storer = store.get_storer(h5_key)

            # For some pandas versions / formats, column names can be obtained from axes.
            # For a DataFrame, axes[0] is typically the columns.
            axes = getattr(storer, "axes", None)
            if axes and len(axes) >= 1:
                try:
                    return [str(c) for c in list(axes[0])]
                except Exception:
                    pass

            # For some storers, pandas exposes non_index_axes.
            # It is typically a list like [(1, ['colA','colB',...])]
            non_index_axes = getattr(storer, "non_index_axes", None)
            if non_index_axes:
                cols = non_index_axes[0][1]
                return [str(c) for c in cols]
    except Exception:
        pass

    # Fallback: load a small file (template/debug) if user points to it.
    try:
        df = pd.read_hdf(str(h5_path), key=key)
        return [str(c) for c in df.columns]
    except Exception:
        return []


def build_daily_pv_schema(prefix: str = "", include_factor: bool = True, include_amount: bool = True) -> dict[str, Any]:
    """Schema derived from docs/AIstock_日频H5导出规范.md.

    Notes on naming:
    - Some Qlib-style exports use "$open/$close/$factor" (with '$' prefix).
    - Some RD-Agent factor data templates use "open/close/volume/amount" (no '$', and typically no factor column).
    This function allows generating schema matching the actual H5 columns.
    """

    open_col = f"{prefix}open"
    high_col = f"{prefix}high"
    low_col = f"{prefix}low"
    close_col = f"{prefix}close"
    vol_col = f"{prefix}volume"
    factor_col = f"{prefix}factor"
    amt_col = f"{prefix}amount"

    cols: list[dict[str, Any]] = [
        {
            "name": open_col,
            "dtype": "float32",
            "meaning": "当日开盘价（前复权）",
            "unit": "元",
            "adjustment": "qfq",
            "missing_policy": "NaN",
            "quality": "authoritative",
            "source_table": "market.kline_daily_raw + market.adj_factor",
            "formula": f"{open_col} = open_li/1000 * {factor_col}" if include_factor else "导出后价格字段为前复权（已处理）",
        },
        {
            "name": high_col,
            "dtype": "float32",
            "meaning": "当日最高价（前复权）",
            "unit": "元",
            "adjustment": "qfq",
            "missing_policy": "NaN",
            "quality": "authoritative",
            "source_table": "market.kline_daily_raw + market.adj_factor",
            "formula": f"{high_col} = high_li/1000 * {factor_col}" if include_factor else "导出后价格字段为前复权（已处理）",
        },
        {
            "name": low_col,
            "dtype": "float32",
            "meaning": "当日最低价（前复权）",
            "unit": "元",
            "adjustment": "qfq",
            "missing_policy": "NaN",
            "quality": "authoritative",
            "source_table": "market.kline_daily_raw + market.adj_factor",
            "formula": f"{low_col} = low_li/1000 * {factor_col}" if include_factor else "导出后价格字段为前复权（已处理）",
        },
        {
            "name": close_col,
            "dtype": "float32",
            "meaning": "当日收盘价（前复权）",
            "unit": "元",
            "adjustment": "qfq",
            "missing_policy": "NaN",
            "quality": "authoritative",
            "source_table": "market.kline_daily_raw + market.adj_factor",
            "formula": f"{close_col} = close_li/1000 * {factor_col}" if include_factor else "导出后价格字段为前复权（已处理）",
        },
        {
            "name": vol_col,
            "dtype": "float32",
            "meaning": "当日成交量（复权调整后）",
            "unit": "股",
            "adjustment": "qfq_inverse",
            "missing_policy": "NaN",
            "quality": "authoritative",
            "source_table": "market.kline_daily_raw + market.adj_factor",
            "formula": f"{vol_col} = (volume_hand*100)/{factor_col}" if include_factor else "导出后 volume 已按复权口径调整（无单独 factor 列）",
        },
    ]

    if include_factor:
        cols.append(
            {
                "name": factor_col,
                "dtype": "float32",
                "meaning": "前复权因子（qfq_factor）",
                "unit": "无",
                "adjustment": "none",
                "missing_policy": "error",
                "quality": "authoritative",
                "source_table": "market.adj_factor",
                "formula": f"{factor_col} = adj_factor/latest_adj_factor",
            }
        )

    if include_amount:
        cols.append(
            {
                "name": amt_col,
                "dtype": "float32",
                "meaning": "当日成交额（现金成交金额；不复权）",
                "unit": "元",
                "adjustment": "none",
                "missing_policy": "NaN_preferred_over_zero",
                "quality": "unknown_or_placeholder",
                "source_table": "market.kline_daily_raw",
                "formula": "建议导出真实成交额；若上游无法提供，H5 侧应使用 NaN 表示缺失（不建议用 0 伪装真实值）",
            }
        )

    assumptions: list[str] = [
        "instrument 使用 Tushare ts_code 格式，如 000001.SZ",
        "数据库价格单位为厘（元*1000），导出时除以1000换算为元",
        "价格字段为前复权（导出时已处理）",
    ]
    if include_factor:
        assumptions = assumptions + [
            f"存在复权因子列 {factor_col}（前复权 qfq_factor）",
            f"volume 已按复权因子做反向调整：raw_volume_shares = {vol_col} * {factor_col}",
        ]
    else:
        assumptions = assumptions + [
            "当前 H5 不包含单独的复权因子列 factor；若需要还原未复权价格/成交量，需要依赖外部 adj_factor 数据",
        ]

    return {
        "dataset": "daily_pv",
        "h5_key": "data",
        "index": {"type": "MultiIndex", "names": ["datetime", "instrument"]},
        "governance": {
            "amount": {
                "should_exist": True,
                "preferred_source": "db_real_amount",
                "adjustment": "none",
                "unit": "元",
                "missing_policy": "H5: NaN; bin/CSV: if forced by schema, 0 is allowed ONLY as placeholder and must be documented",
                "notes": "amount=0 is ambiguous and can silently corrupt amount-based features; prefer NaN when missing.",
            }
        },
        "assumptions": assumptions,
        "columns": cols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily_pv schema (json/csv) for RD-Agent.")
    parser.add_argument(
        "--h5-path",
        default="",
        help="Optional path to daily_pv.h5 for validation (Windows or WSL path).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "docs"),
        help="Output directory for schema files.",
    )
    parser.add_argument(
        "--key",
        default="data",
        help="H5 key (default: data).",
    )
    args = parser.parse_args()

    cols_in_h5: list[str] = []
    if args.h5_path:
        h5_path = Path(args.h5_path)
        cols_in_h5 = _try_get_columns_from_h5(h5_path, key=args.key)

    # Decide naming style from H5 if available.
    prefix = "$" if any(str(c).startswith("$") for c in cols_in_h5) else ""
    include_factor = f"{prefix}factor" in {str(c) for c in cols_in_h5} if cols_in_h5 else True
    include_amount = f"{prefix}amount" in {str(c) for c in cols_in_h5} if cols_in_h5 else True

    schema = build_daily_pv_schema(prefix=prefix, include_factor=include_factor, include_amount=include_amount)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if cols_in_h5:
        schema["columns_in_h5"] = cols_in_h5
        defined = [str(c["name"]) for c in schema["columns"]]
        defined_set = set(defined)
        in_h5_set = {str(c) for c in cols_in_h5}

        extra = sorted(in_h5_set - defined_set)
        missing = sorted(defined_set - in_h5_set)
        if extra:
            schema["warnings"] = schema.get("warnings", []) + [f"H5 contains extra columns not in schema: {extra}"]
        if missing:
            schema["warnings"] = schema.get("warnings", []) + [f"Schema defines columns missing from H5: {missing}"]

        # Notes for optional columns.
        if f"{prefix}amount" not in in_h5_set:
            schema["notes"] = schema.get("notes", []) + [f"{prefix}amount is optional and not present in this H5."]
        if f"{prefix}factor" not in in_h5_set:
            schema["notes"] = schema.get("notes", []) + [
                f"{prefix}factor not present in this H5; cannot directly restore unadjusted price/volume without external adj_factor."
            ]

    json_path = out_dir / "daily_pv_schema.json"
    csv_path = out_dir / "daily_pv_schema.csv"

    json_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(schema["columns"]).to_csv(csv_path, index=False, encoding="utf-8")

    print("[SUCCESS] wrote:", json_path)
    print("[SUCCESS] wrote:", csv_path)


if __name__ == "__main__":
    main()
