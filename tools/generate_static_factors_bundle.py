import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def _to_unix_path(p: Path) -> Path:
    s = str(p).replace("\\", "/")
    if os.name != "nt" and len(s) >= 3 and s[1:3] == ":/":
        drive = s[0].lower()
        return Path("/mnt") / drive / s[3:]
    return Path(s)


def _ensure_multiindex(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        if list(df.index.names) != ["datetime", "instrument"]:
            df.index.set_names(["datetime", "instrument"], inplace=True)
        return _normalize_instrument_index(df, name).sort_index()

    cols = set(df.columns)
    if {"datetime", "instrument"}.issubset(cols):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index(["datetime", "instrument"]).sort_index()
        return _normalize_instrument_index(df, name)

    raise ValueError(f"{name} is not a MultiIndex(datetime, instrument) and has no datetime/instrument columns")


def _normalize_instrument_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize instrument codes to Qlib-style.

    Supported examples:
    - SH600000/SZ000001 -> 600000.SH/000001.SZ
    - Already normalized: 600000.SH/000001.SZ (kept)
    """

    if df is None or df.empty:
        return df

    if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {"datetime", "instrument"}:
        return df

    inst = df.index.get_level_values("instrument").astype(str)
    # SH600000/SZ000001 -> 600000.SH/000001.SZ
    m = inst.str.match(r"^(SH|SZ)(\d{6})$")
    if bool(m.any()):
        exch = inst.str.slice(0, 2)
        code = inst.str.slice(2, 8)
        inst_norm = inst.where(~m, code + "." + exch)
        df = df.copy()
        df.index = pd.MultiIndex.from_arrays(
            [
                df.index.get_level_values("datetime"),
                pd.Index(inst_norm, name="instrument"),
            ],
            names=["datetime", "instrument"],
        )
        # After normalization, duplicated index may appear; keep the last occurrence.
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")]

    return df


def _read_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")

    if path.suffix == ".h5":
        df = pd.read_hdf(path, key="data") if _h5_has_key(path, "data") else pd.read_hdf(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return _ensure_multiindex(df, name)


def _h5_has_key(path: Path, key: str) -> bool:
    try:
        with pd.HDFStore(path) as store:
            return f"/{key}" in store.keys()
    except Exception:
        return False


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_f = numer.astype("float64")
    denom_f = denom.astype("float64").mask(denom.astype("float64") == 0, np.nan)
    return (numer_f / denom_f).astype("float64")


def _rolling_sum_by_instrument(s: pd.Series, window: int) -> pd.Series:
    # Keep MultiIndex(datetime, instrument) aligned.
    out = (
        s.groupby(level="instrument")
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return out.astype("float64")


def _derive_moneyflow_features(df_mf_raw: pd.DataFrame) -> pd.DataFrame:
    """Derive stable, schema-friendly moneyflow features from raw buy/sell columns.

    This makes factor generation more robust by providing commonly-needed
    net inflow and ratio features as precomputed columns.
    """

    df = df_mf_raw.sort_index()

    required = [
        "mf_sm_buy_amt",
        "mf_sm_sell_amt",
        "mf_md_buy_amt",
        "mf_md_sell_amt",
        "mf_lg_buy_amt",
        "mf_lg_sell_amt",
        "mf_elg_buy_amt",
        "mf_elg_sell_amt",
        "mf_sm_buy_vol",
        "mf_sm_sell_vol",
        "mf_md_buy_vol",
        "mf_md_sell_vol",
        "mf_lg_buy_vol",
        "mf_lg_sell_vol",
        "mf_elg_buy_vol",
        "mf_elg_sell_vol",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # If a snapshot has incomplete moneyflow raw columns, skip derivation.
        return pd.DataFrame(index=df.index)

    buy_amt_total = df["mf_sm_buy_amt"] + df["mf_md_buy_amt"] + df["mf_lg_buy_amt"] + df["mf_elg_buy_amt"]
    sell_amt_total = df["mf_sm_sell_amt"] + df["mf_md_sell_amt"] + df["mf_lg_sell_amt"] + df["mf_elg_sell_amt"]
    buy_vol_total = df["mf_sm_buy_vol"] + df["mf_md_buy_vol"] + df["mf_lg_buy_vol"] + df["mf_elg_buy_vol"]
    sell_vol_total = df["mf_sm_sell_vol"] + df["mf_md_sell_vol"] + df["mf_lg_sell_vol"] + df["mf_elg_sell_vol"]

    main_buy_amt = df["mf_lg_buy_amt"] + df["mf_elg_buy_amt"]
    main_sell_amt = df["mf_lg_sell_amt"] + df["mf_elg_sell_amt"]
    main_buy_vol = df["mf_lg_buy_vol"] + df["mf_elg_buy_vol"]
    main_sell_vol = df["mf_lg_sell_vol"] + df["mf_elg_sell_vol"]

    total_turnover_amt = buy_amt_total + sell_amt_total
    total_turnover_vol = buy_vol_total + sell_vol_total
    main_turnover_amt = main_buy_amt + main_sell_amt
    main_turnover_vol = main_buy_vol + main_sell_vol

    total_net_amt = (buy_amt_total - sell_amt_total).astype("float64")
    total_net_vol = (buy_vol_total - sell_vol_total).astype("float64")
    main_net_amt = (main_buy_amt - main_sell_amt).astype("float64")
    main_net_vol = (main_buy_vol - main_sell_vol).astype("float64")
    elg_net_amt = (df["mf_elg_buy_amt"] - df["mf_elg_sell_amt"]).astype("float64")
    elg_net_vol = (df["mf_elg_buy_vol"] - df["mf_elg_sell_vol"]).astype("float64")

    out = pd.DataFrame(index=df.index)
    out["mf_total_net_amt"] = total_net_amt
    out["mf_total_net_vol"] = total_net_vol
    out["mf_total_net_amt_ratio"] = _safe_div(total_net_amt, total_turnover_amt)
    out["mf_total_net_vol_ratio"] = _safe_div(total_net_vol, total_turnover_vol)

    out["mf_main_net_amt"] = main_net_amt
    out["mf_main_net_vol"] = main_net_vol
    out["mf_main_net_amt_ratio"] = _safe_div(main_net_amt, main_turnover_amt)
    out["mf_main_net_vol_ratio"] = _safe_div(main_net_vol, main_turnover_vol)

    out["mf_elg_net_amt"] = elg_net_amt
    out["mf_elg_net_vol"] = elg_net_vol
    out["mf_elg_net_amt_ratio"] = _safe_div(elg_net_amt, (df["mf_elg_buy_amt"] + df["mf_elg_sell_amt"]))
    out["mf_elg_net_vol_ratio"] = _safe_div(elg_net_vol, (df["mf_elg_buy_vol"] + df["mf_elg_sell_vol"]))

    out["mf_elg_share_in_main_amt"] = _safe_div(df["mf_elg_buy_amt"], (df["mf_lg_buy_amt"] + df["mf_elg_buy_amt"]))
    out["mf_elg_share_in_main_vol"] = _safe_div(df["mf_elg_buy_vol"], (df["mf_lg_buy_vol"] + df["mf_elg_buy_vol"]))

    for w in (5, 20):
        out[f"mf_total_net_amt_{w}d"] = _rolling_sum_by_instrument(total_net_amt, w)
        out[f"mf_main_net_amt_{w}d"] = _rolling_sum_by_instrument(main_net_amt, w)
        out[f"mf_elg_net_amt_{w}d"] = _rolling_sum_by_instrument(elg_net_amt, w)

        total_turnover_amt_w = _rolling_sum_by_instrument(total_turnover_amt.astype("float64"), w)
        main_turnover_amt_w = _rolling_sum_by_instrument(main_turnover_amt.astype("float64"), w)
        elg_turnover_amt_w = _rolling_sum_by_instrument((df["mf_elg_buy_amt"] + df["mf_elg_sell_amt"]).astype("float64"), w)

        out[f"mf_total_net_amt_ratio_{w}d"] = _safe_div(out[f"mf_total_net_amt_{w}d"], total_turnover_amt_w)
        out[f"mf_main_net_amt_ratio_{w}d"] = _safe_div(out[f"mf_main_net_amt_{w}d"], main_turnover_amt_w)
        out[f"mf_elg_net_amt_ratio_{w}d"] = _safe_div(out[f"mf_elg_net_amt_{w}d"], elg_turnover_amt_w)

    return out


def _build_schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a machine-readable schema.

    - name: column name
    - dtype: pandas dtype
    - meaning: best-effort mapping; unknown fields will have empty meaning
    - source: inferred from prefix
    """

    meaning_map = {
        # daily_basic common
        "db_pe_ttm": "市盈率TTM",
        "db_pe": "市盈率",
        "db_pb": "市净率",
        "db_circ_mv": "流通市值",
        "db_total_mv": "总市值",
        "db_turnover_rate": "换手率",
        "db_volume_ratio": "量比",
        # moneyflow common (raw)
        "mf_net_amt": "资金净流入金额（买入-卖出）",
        "mf_net_vol": "资金净流入量（买入-卖出）",
        "mf_lg_buy_amt": "大单买入金额",
        "mf_lg_sell_amt": "大单卖出金额",
        "mf_elg_buy_amt": "超大单买入金额",
        "mf_elg_sell_amt": "超大单卖出金额",
        "mf_lg_buy_vol": "大单买入量",
        "mf_lg_sell_vol": "大单卖出量",
        "mf_elg_buy_vol": "超大单买入量",
        "mf_elg_sell_vol": "超大单卖出量",
        "mf_total_net_amt": "全档净流入金额（全买入-全卖出）",
        "mf_total_net_vol": "全档净流入量（全买入-全卖出）",
        "mf_total_net_amt_ratio": "全档净流入强度（净流入/成交额，分母为买卖额之和）",
        "mf_total_net_vol_ratio": "全档净流入强度（净流入/成交量，分母为买卖量之和）",
        "mf_main_net_amt": "主力净流入金额（大单+特大单）",
        "mf_main_net_vol": "主力净流入量（大单+特大单）",
        "mf_main_net_amt_ratio": "主力净流入强度（主力净流入/主力买卖额之和）",
        "mf_main_net_vol_ratio": "主力净流入强度（主力净流入/主力买卖量之和）",
        "mf_elg_net_amt": "特大单净流入金额（特大单买入-卖出）",
        "mf_elg_net_vol": "特大单净流入量（特大单买入-卖出）",
        "mf_elg_net_amt_ratio": "特大单净流入强度（特大单净流入/特大单买卖额之和）",
        "mf_elg_net_vol_ratio": "特大单净流入强度（特大单净流入/特大单买卖量之和）",
        "mf_elg_share_in_main_amt": "特大单买入占主力买入比（特大单买入/(大单+特大单买入)）",
        "mf_elg_share_in_main_vol": "特大单买入量占主力买入量比（特大单买入/(大单+特大单买入)）",
        "mf_total_net_amt_5d": "全档净流入金额5日滚动和",
        "mf_total_net_amt_20d": "全档净流入金额20日滚动和",
        "mf_main_net_amt_5d": "主力净流入金额5日滚动和",
        "mf_main_net_amt_20d": "主力净流入金额20日滚动和",
        "mf_elg_net_amt_5d": "特大单净流入金额5日滚动和",
        "mf_elg_net_amt_20d": "特大单净流入金额20日滚动和",
        "mf_total_net_amt_ratio_5d": "全档净流入强度5日（5日净流入/5日买卖额之和）",
        "mf_total_net_amt_ratio_20d": "全档净流入强度20日（20日净流入/20日买卖额之和）",
        "mf_main_net_amt_ratio_5d": "主力净流入强度5日（5日主力净流入/5日主力买卖额之和）",
        "mf_main_net_amt_ratio_20d": "主力净流入强度20日（20日主力净流入/20日主力买卖额之和）",
        "mf_elg_net_amt_ratio_5d": "特大单净流入强度5日（5日特大单净流入/5日特大单买卖额之和）",
        "mf_elg_net_amt_ratio_20d": "特大单净流入强度20日（20日特大单净流入/20日特大单买卖额之和）",
        # precomputed examples
        "value_pe_inv": "估值因子：1/PE，越大代表越便宜",
        "value_pb_inv": "估值因子：1/PB，越大代表越便宜",
        "size_log_mv": "规模因子：log(市值)",
        "liquidity_turnover": "流动性因子：换手率",
        "liquidity_vol_ratio": "流动性因子：量比",
    }

    schema: list[dict[str, Any]] = []
    for col in df.columns:
        col_str = str(col)
        if col_str.startswith("db_"):
            source = "daily_basic_raw"
        elif col_str.startswith("mf_"):
            source = "moneyflow_raw_or_factor"
        elif col_str.startswith("ae_"):
            source = "ae_factor"
        else:
            source = "precomputed_or_other"

        schema.append(
            {
                "name": col_str,
                "dtype": str(df[col].dtype),
                "meaning": meaning_map.get(col_str, ""),
                "source": source,
            }
        )

    return schema


def _load_field_map(path: Path) -> dict[str, dict[str, Any]]:
    """Load external field mapping exported from AIstock metadata.

    Supported formats:
    - CSV: must contain a column for field name (default: 'name')
    - JSON: either a list of objects or a dict keyed by name

    The returned mapping is {name -> metadata_dict}.
    """

    if not path.exists():
        raise FileNotFoundError(f"field-map not found: {path}")

    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
        if "name" not in df.columns:
            raise ValueError(f"field-map csv must contain 'name' column, got columns={list(df.columns)}")
        out: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            out[name] = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        return out

    if path.suffix.lower() in {".json"}:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if not k:
                    continue
                out[str(k)] = v if isinstance(v, dict) else {"meaning_cn": v}
            return out
        if isinstance(obj, list):
            out = {}
            for item in obj:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                out[name] = item
            return out
        raise ValueError(f"Unsupported JSON schema for field-map: root type={type(obj)}")

    raise ValueError(f"Unsupported field-map format: {path}")


def _apply_field_map(schema: list[dict[str, Any]], field_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge field-map metadata into schema entries by column name.

    Convention (recommended):
    - meaning_cn: Chinese meaning/description
    - unit: unit string
    - source_table: daily_basic/moneyflow/... original table
    - comment: raw DB comment (optional)
    """

    for entry in schema:
        name = entry.get("name")
        if not isinstance(name, str):
            continue

        meta = field_map.get(name)
        if not isinstance(meta, dict):
            continue

        meaning_cn = meta.get("meaning_cn") or meta.get("meaning") or meta.get("comment")
        if isinstance(meaning_cn, str) and meaning_cn.strip():
            entry["meaning"] = meaning_cn.strip()

        unit = meta.get("unit")
        if isinstance(unit, str) and unit.strip():
            entry["unit"] = unit.strip()

        source_table = meta.get("source_table")
        if isinstance(source_table, str) and source_table.strip():
            entry["source_table"] = source_table.strip()

    return schema


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate static_factors.parquet for factor.py by merging daily_basic/moneyflow raw fields and existing precomputed factor tables."
    )

    parser.add_argument(
        "--snapshot-root",
        default=os.environ.get(
            "AIstock_SNAPSHOT_ROOT",
            r"C:/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_20251209",
        ),
        help="Snapshot root containing daily_basic.h5 and moneyflow.h5 (Windows or WSL path).",
    )

    parser.add_argument(
        "--aistock-factors-root",
        default=os.environ.get(
            "AIstock_FACTORS_ROOT",
            r"C:/Users/lc999/NewAIstock/AIstock/factors",
        ),
        help="AIstock factors root containing precomputed factor outputs (daily_basic_factors, moneyflow_factors, ae_recon_error_10d...).",
    )

    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="RD-Agent repo root; used to place outputs under git_ignore_folder.",
    )

    parser.add_argument(
        "--field-map",
        default="",
        help="Optional field mapping file (CSV/JSON) exported from AIstock DB comments/metadata. It will be used to enrich schema meanings without modifying H5.",
    )

    parser.add_argument(
        "--out-rel",
        default=r"git_ignore_folder/factor_implementation_source_data/static_factors.parquet",
        help="Output parquet relative to repo_root.",
    )

    parser.add_argument(
        "--out-rel-debug",
        default=r"git_ignore_folder/factor_implementation_source_data_debug/static_factors.parquet",
        help="Debug output parquet relative to repo_root.",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    snapshot_root = _to_unix_path(Path(args.snapshot_root))
    aistock_factors_root = _to_unix_path(Path(args.aistock_factors_root))

    daily_basic_path = snapshot_root / "daily_basic.h5"
    moneyflow_path = snapshot_root / "moneyflow.h5"

    print("[INFO] snapshot_root:", snapshot_root)
    print("[INFO] factors_root :", aistock_factors_root)

    # Raw tables
    print("[INFO] Loading raw daily_basic.h5 ...")
    df_db_raw = _read_table(daily_basic_path, "daily_basic_raw")

    print("[INFO] Loading raw moneyflow.h5 ...")
    df_mf_raw = _read_table(moneyflow_path, "moneyflow_raw")

    # Optional precomputed factor tables
    dfs: list[pd.DataFrame] = []

    # Keep raw fields, but to avoid name collisions we keep their existing prefixes.
    dfs.append(df_db_raw)
    dfs.append(df_mf_raw)

    cand_tables: list[tuple[str, Path]] = [
        ("daily_basic_factors", aistock_factors_root / "daily_basic_factors" / "result.pkl"),
        ("moneyflow_factors", aistock_factors_root / "moneyflow_factors" / "result.pkl"),
        ("ae_recon_error_10d", aistock_factors_root / "ae_recon_error_10d" / "result.pkl"),
        ("combined_static_factors", aistock_factors_root / "combined_static_factors.parquet"),
    ]

    for name, p in cand_tables:
        if p.exists():
            print(f"[INFO] Loading optional table {name}: {p}")
            try:
                dfs.append(_read_table(p, name))
            except Exception as e:
                print(f"[WARN] Failed to load {name} from {p}: {repr(e)}")
        else:
            print(f"[INFO] Optional table not found: {name} ({p})")

    print("[INFO] Deriving common moneyflow features ...")
    df_mf_derived = _derive_moneyflow_features(df_mf_raw)
    if df_mf_derived.empty:
        for df_opt in dfs[2:]:
            if not isinstance(df_opt, pd.DataFrame) or df_opt.empty:
                continue
            df_try = df_opt[[c for c in df_opt.columns if str(c).startswith("mf_")]]
            if df_try.empty:
                continue
            df_mf_derived = _derive_moneyflow_features(df_try)
            if not df_mf_derived.empty:
                break

    if not df_mf_derived.empty:
        dfs.append(df_mf_derived)

    print("[INFO] Concatenating tables (axis=1) ...")
    df_merged = pd.concat(dfs, axis=1)
    df_merged = df_merged.sort_index()
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated(keep="last")]

    out_path = repo_root / args.out_rel
    out_debug_path = repo_root / args.out_rel_debug

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_debug_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Writing:", out_path)
    df_merged.to_parquet(out_path)

    print("[INFO] Writing:", out_debug_path)
    df_merged.to_parquet(out_debug_path)

    # Schema
    schema_cols = _build_schema(df_merged)

    if args.field_map:
        fm_path = _to_unix_path(Path(args.field_map))
        print("[INFO] Loading field-map:", fm_path)
        field_map = _load_field_map(fm_path)
        schema_cols = _apply_field_map(schema_cols, field_map)

    schema = {
        "index": {
            "type": "MultiIndex",
            "names": ["datetime", "instrument"],
        },
        "columns": schema_cols,
    }

    schema_json_path = out_path.parent / "static_factors_schema.json"
    schema_csv_path = out_path.parent / "static_factors_schema.csv"

    schema_json_debug_path = out_debug_path.parent / "static_factors_schema.json"
    schema_csv_debug_path = out_debug_path.parent / "static_factors_schema.csv"

    print("[INFO] Writing schema:", schema_json_path)
    schema_json_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Writing schema:", schema_json_debug_path)
    schema_json_debug_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Writing schema:", schema_csv_path)
    pd.DataFrame(schema["columns"]).to_csv(schema_csv_path, index=False, encoding="utf-8")

    print("[INFO] Writing schema:", schema_csv_debug_path)
    pd.DataFrame(schema["columns"]).to_csv(schema_csv_debug_path, index=False, encoding="utf-8")

    print("[SUCCESS] static_factors.parquet + schema generated.")


if __name__ == "__main__":
    main()
