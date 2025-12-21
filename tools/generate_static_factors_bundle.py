import argparse
import csv
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


def _pick_amount_volume(df_pv: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    if df_pv is None or df_pv.empty:
        return None

    for amt_col, vol_col in (("amount", "volume"), ("$amount", "$volume")):
        if amt_col in df_pv.columns and vol_col in df_pv.columns:
            amount = df_pv[amt_col].astype("float64").mask(df_pv[amt_col].astype("float64") == 0, np.nan)
            volume = df_pv[vol_col].astype("float64").mask(df_pv[vol_col].astype("float64") == 0, np.nan)
            return amount, volume

    return None


def _derive_moneyflow_features(df_mf_raw: pd.DataFrame, df_pv: pd.DataFrame | None) -> pd.DataFrame:
    """Derive stable, schema-friendly moneyflow features from raw buy/sell columns.

    This makes factor generation more robust by providing commonly-needed
    net inflow and ratio features as precomputed columns.
    """

    df = df_mf_raw.sort_index()

    picked = _pick_amount_volume(df_pv) if df_pv is not None else None
    if picked is None:
        return pd.DataFrame(index=df.index)

    amount, volume = picked

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

    total_net_amt = (buy_amt_total - sell_amt_total).astype("float64")
    total_net_vol = (buy_vol_total - sell_vol_total).astype("float64")
    main_net_amt = (main_buy_amt - main_sell_amt).astype("float64")
    main_net_vol = (main_buy_vol - main_sell_vol).astype("float64")
    elg_net_amt = (df["mf_elg_buy_amt"] - df["mf_elg_sell_amt"]).astype("float64")
    elg_net_vol = (df["mf_elg_buy_vol"] - df["mf_elg_sell_vol"]).astype("float64")

    out = pd.DataFrame(index=df.index)
    out["mf_total_net_amt"] = total_net_amt
    out["mf_total_net_vol"] = total_net_vol
    out["mf_total_net_amt_ratio"] = _safe_div(total_net_amt, amount)
    out["mf_total_net_vol_ratio"] = _safe_div(total_net_vol, volume)

    out["mf_main_net_amt"] = main_net_amt
    out["mf_main_net_vol"] = main_net_vol
    out["mf_main_net_amt_ratio"] = _safe_div(main_net_amt, amount)
    out["mf_main_net_vol_ratio"] = _safe_div(main_net_vol, volume)

    out["mf_elg_net_amt"] = elg_net_amt
    out["mf_elg_net_vol"] = elg_net_vol
    out["mf_elg_net_amt_ratio"] = _safe_div(elg_net_amt, amount)
    out["mf_elg_net_vol_ratio"] = _safe_div(elg_net_vol, volume)

    out["mf_elg_share_in_main_amt"] = _safe_div(df["mf_elg_buy_amt"], (df["mf_lg_buy_amt"] + df["mf_elg_buy_amt"]))
    out["mf_elg_share_in_main_vol"] = _safe_div(df["mf_elg_buy_vol"], (df["mf_lg_buy_vol"] + df["mf_elg_buy_vol"]))

    for w in (5, 20):
        out[f"mf_total_net_amt_{w}d"] = _rolling_sum_by_instrument(total_net_amt, w)
        out[f"mf_main_net_amt_{w}d"] = _rolling_sum_by_instrument(main_net_amt, w)
        out[f"mf_elg_net_amt_{w}d"] = _rolling_sum_by_instrument(elg_net_amt, w)

        amount_w = _rolling_sum_by_instrument(amount, w)
        out[f"mf_total_net_amt_ratio_{w}d"] = _safe_div(out[f"mf_total_net_amt_{w}d"], amount_w)
        out[f"mf_main_net_amt_ratio_{w}d"] = _safe_div(out[f"mf_main_net_amt_{w}d"], amount_w)
        out[f"mf_elg_net_amt_ratio_{w}d"] = _safe_div(out[f"mf_elg_net_amt_{w}d"], amount_w)

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
        "mf_net_amt": "资金净流入金额（买入-卖出）。别名/等价口径：≈ mf_total_net_amt（当 mf_total_net_amt 按全档买卖额重算时）。",
        "mf_net_vol": "资金净流入量（买入-卖出）。别名/等价口径：≈ mf_total_net_vol（当 mf_total_net_vol 按全档买卖量重算时）。",
        "mf_lg_buy_amt": "大单买入金额",
        "mf_lg_sell_amt": "大单卖出金额",
        "mf_elg_buy_amt": "超大单买入金额",
        "mf_elg_sell_amt": "超大单卖出金额",
        "mf_lg_buy_vol": "大单买入量",
        "mf_lg_sell_vol": "大单卖出量",
        "mf_elg_buy_vol": "超大单买入量",
        "mf_elg_sell_vol": "超大单卖出量",
        "mf_total_net_amt": "全档净流入金额（全买入-全卖出）。别名/等价口径：≈ mf_net_amt（上游直接提供净流入时）。",
        "mf_total_net_vol": "全档净流入量（全买入-全卖出）。别名/等价口径：≈ mf_net_vol（上游直接提供净流入时）。",
        "mf_total_net_amt_ratio": "全档净流入强度（净流入/成交额）",
        "mf_total_net_vol_ratio": "全档净流入强度（净流入/成交量）",
        "mf_main_net_amt": "主力净流入金额（大单+特大单）",
        "mf_main_net_vol": "主力净流入量（大单+特大单）",
        "mf_main_net_amt_ratio": "主力净流入强度（主力净流入/成交额）",
        "mf_main_net_vol_ratio": "主力净流入强度（主力净流入/成交量）",
        "mf_elg_net_amt": "特大单净流入金额（特大单买入-卖出）",
        "mf_elg_net_vol": "特大单净流入量（特大单买入-卖出）",
        "mf_elg_net_amt_ratio": "特大单净流入强度（特大单净流入/成交额）：(mf_elg_buy_amt-mf_elg_sell_amt)/amount；amount=0或缺失=>NaN",
        "mf_elg_net_vol_ratio": "特大单净流入强度（特大单净流入/成交量）：(mf_elg_buy_vol-mf_elg_sell_vol)/volume；volume=0或缺失=>NaN",
        "mf_elg_share_in_main_amt": "特大单净流入占主力净流入比：(mf_elg_buy_amt-mf_elg_sell_amt)/((mf_lg_buy_amt+mf_elg_buy_amt)-(mf_lg_sell_amt+mf_elg_sell_amt))；分母为0或缺失=>NaN",
        "mf_elg_share_in_main_vol": "特大单净流入占主力净流入比：(mf_elg_buy_vol-mf_elg_sell_vol)/((mf_lg_buy_vol+mf_elg_buy_vol)-(mf_lg_sell_vol+mf_elg_sell_vol))；分母为0或缺失=>NaN",
        "mf_total_net_amt_5d": "sum_5d(mf_total_net_amt)，全档净流入金额5日滚动和",
        "mf_total_net_amt_20d": "sum_20d(mf_total_net_amt)，全档净流入金额20日滚动和",
        "mf_main_net_amt_5d": "sum_5d(mf_main_net_amt)，主力净流入金额5日滚动和",
        "mf_main_net_amt_20d": "sum_20d(mf_main_net_amt)，主力净流入金额20日滚动和",
        "mf_elg_net_amt_5d": "sum_5d(mf_elg_net_amt)，特大单净流入金额5日滚动和",
        "mf_elg_net_amt_20d": "sum_20d(mf_elg_net_amt)，特大单净流入金额20日滚动和",
        "mf_total_net_amt_ratio_5d": "sum_5d(mf_total_net_amt)/sum_5d(amount)，全档净流入强度5日滚动；分母为0或缺失=>NaN",
        "mf_total_net_amt_ratio_20d": "sum_20d(mf_total_net_amt)/sum_20d(amount)，全档净流入强度20日滚动；分母为0或缺失=>NaN",
        "mf_main_net_amt_ratio_5d": "sum_5d(mf_main_net_amt)/sum_5d(amount)，主力净流入强度5日滚动；分母为0或缺失=>NaN",
        "mf_main_net_amt_ratio_20d": "sum_20d(mf_main_net_amt)/sum_20d(amount)，主力净流入强度20日滚动；分母为0或缺失=>NaN",
        "mf_elg_net_amt_ratio_5d": "sum_5d(mf_elg_net_amt)/sum_5d(amount)，特大单净流入强度5日滚动；分母为0或缺失=>NaN",
        "mf_elg_net_amt_ratio_20d": "sum_20d(mf_elg_net_amt)/sum_20d(amount)，特大单净流入强度20日滚动；分母为0或缺失=>NaN",
        # precomputed (traceable via precompute_daily_basic_factors.py)
        "value_pe_inv": "倒数市盈率（估值因子）：1/db_pe_ttm（优先）或 1/db_pe；分母为0或缺失=>NaN",
        "value_pb_inv": "倒数市净率（估值因子）：1/db_pb；分母为0或缺失=>NaN",
        "size_log_mv": "市值对数（规模因子）：log(db_circ_mv 优先，否则 db_total_mv)；仅对>0取对数，否则=>NaN",
        "liquidity_turnover": "换手率（流动性因子）：db_turnover_rate（缺失=>NaN）。别名/等价口径：liquidity_turnover ≈ db_turnover_rate",
        "liquidity_vol_ratio": "量比（流动性因子）：db_volume_ratio（缺失=>NaN）。别名/等价口径：liquidity_vol_ratio ≈ db_volume_ratio",
        "ae_recon_error_10d": "10日自编码器重构误差（异常度）：依赖预训练模型 AE10D_MODEL_PATH=models/ae_10d.pth（payload 含 window/features/mean/std）；对每个(股票,日期)取过去window天、features列拼接成向量x，归一化后输入AE，误差=mean((x-recon(x))^2)；仅对窗口内全为有限值的样本计算，否则缺失",
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


def _print_untraceable_schema_fields(schema_cols: list[dict[str, Any]], title: str) -> None:
    missing: list[tuple[str, str, str]] = []
    for item in schema_cols:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "")
        dtype = str(item.get("dtype", "") or "")
        source = str(item.get("source", "") or "")
        meaning = item.get("meaning")
        meaning_s = "" if meaning is None else str(meaning).strip()
        if not meaning_s or meaning_s.lower() == "nan":
            if name:
                missing.append((name, dtype, source))

    if not missing:
        print(f"[INFO] {title}: none")
        return

    print(f"[WARN] {title}: {len(missing)} fields")
    # Group by source for readability.
    by_source: dict[str, list[tuple[str, str]]] = {}
    for name, dtype, source in missing:
        by_source.setdefault(source or "(unknown)", []).append((name, dtype))

    for src in sorted(by_source.keys()):
        items = by_source[src]
        print(f"  - source={src}: {len(items)}")
        for name, dtype in items:
            print(f"    - {name} ({dtype})")


def _fill_derived_meanings(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill meanings for derived/precomputed fields by naming conventions.

    This is a fallback when neither the hardcoded meaning_map nor external field-map
    provides meanings (e.g., rolling windows like *_5d/_20d).
    """

    for entry in schema:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(entry.get("meaning"), str) and entry["meaning"].strip():
            continue

        # PriceStrength_10D -> 10日价格强度
        if name.startswith("PriceStrength_") and name.endswith("D"):
            w = name[len("PriceStrength_") : -1]
            if w.isdigit():
                entry["meaning"] = f"{w}日价格强度"
                continue

        # Moneyflow rolling sum fields
        if name.startswith("mf_") and (name.endswith("_5d") or name.endswith("_20d")):
            if name.endswith("_5d"):
                w = "5"
                base = name[: -len("_5d")]
            else:
                w = "20"
                base = name[: -len("_20d")]

            # Ratio rolling fields
            if base.endswith("_ratio"):
                base2 = base[: -len("_ratio")]
                if base2 == "mf_total_net_amt":
                    entry["meaning"] = f"全档净流入强度{w}日（{w}日净流入/{w}日成交额）"
                    continue
                if base2 == "mf_main_net_amt":
                    entry["meaning"] = f"主力净流入强度{w}日（{w}日主力净流入/{w}日成交额）"
                    continue
                if base2 == "mf_elg_net_amt":
                    entry["meaning"] = f"特大单净流入强度{w}日（{w}日特大单净流入/{w}日成交额）"
                    continue
                entry["meaning"] = f"{w}日滚动强度指标"
                continue

            # Non-ratio rolling sums
            if base == "mf_total_net_amt":
                entry["meaning"] = f"全档净流入金额{w}日滚动和"
                continue
            if base == "mf_main_net_amt":
                entry["meaning"] = f"主力净流入金额{w}日滚动和"
                continue
            if base == "mf_elg_net_amt":
                entry["meaning"] = f"特大单净流入金额{w}日滚动和"
                continue

    return schema


def _schema_cols_from_parquet_metadata(path: Path) -> list[dict[str, Any]] | None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None

    try:
        pf = pq.ParquetFile(path)
        arrow_schema = pf.schema_arrow
    except Exception:
        return None

    schema_cols: list[dict[str, Any]] = []
    for field in arrow_schema:
        name = str(field.name)
        dtype = str(field.type)
        if name.startswith("db_"):
            source = "daily_basic_raw"
        elif name.startswith("mf_"):
            source = "moneyflow_raw_or_factor"
        elif name.startswith("ae_"):
            source = "ae_factor"
        else:
            source = "precomputed_or_other"

        schema_cols.append(
            {
                "name": name,
                "dtype": dtype,
                "meaning": "",
                "source": source,
            }
        )
    return schema_cols


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
        # Some exports may omit optional columns (e.g. unit) by writing fewer separators,
        # which shifts later columns left. We normalize rows to the header length.
        df = None
        last_err = None

        def _norm_col(x: Any) -> str:
            s = str(x).replace("\ufeff", "").strip()
            # Some exporters wrap headers in quotes, e.g. '"name"'.
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                s = s[1:-1].strip()
            return s

        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        raise ValueError("field-map csv is empty")

                    # Normalize header names: trim whitespace and remove UTF-8 BOM if present.
                    header = [_norm_col(h) for h in header]
                    expected_n = len(header)
                    header_idx = {h: i for i, h in enumerate(header)}

                    allowed_source_table = {"daily_basic", "moneyflow"}
                    rows: list[list[str]] = []
                    for row in reader:
                        if row is None:
                            continue
                        if len(row) == 0:
                            continue

                        # Trim whitespace but keep empty strings.
                        row = [str(x).strip() for x in row]

                        # Heuristic fix: if 'unit' is omitted (missing comma), then
                        # row length is expected_n-1 and the would-be unit cell is actually source_table.
                        # Example (bad): name,meaning_cn,daily_basic,comment
                        # Expected:      name,meaning_cn,,daily_basic,comment
                        if (
                            "unit" in header_idx
                            and "source_table" in header_idx
                            and len(row) == expected_n - 1
                        ):
                            unit_i = header_idx["unit"]
                            source_i = header_idx["source_table"]
                            if unit_i < len(row) and row[unit_i] in allowed_source_table:
                                row.insert(unit_i, "")

                        # If still shorter, pad to length.
                        if len(row) < expected_n:
                            row = row + [""] * (expected_n - len(row))
                        # If longer, merge extras into the last column.
                        elif len(row) > expected_n:
                            row = row[: expected_n - 1] + [",".join(row[expected_n - 1 :])]

                        rows.append(row)

                df = pd.DataFrame(rows, columns=header)
                # Extra safety: some CSV exports may still retain BOM on the first column.
                df.columns = [_norm_col(c) for c in df.columns]
                last_err = None
                break
            except Exception as e:
                last_err = e
                df = None

        if df is None:
            raise last_err
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
            r"F:/Dev/AIstock/qlib_snapshots/qlib_export_20251209",
        ),
        help="Snapshot root containing daily_basic.h5 and moneyflow.h5 (Windows or WSL path).",
    )

    parser.add_argument(
        "--aistock-factors-root",
        default=os.environ.get(
            "AIstock_FACTORS_ROOT",
            r"F:/Dev/AIstock/factors",
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

    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only (re)generate static_factors_schema.json/csv in the output folder. If the parquet already exists, try to read column metadata from parquet without rewriting it.",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    out_path = repo_root / args.out_rel
    out_debug_path = repo_root / args.out_rel_debug

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_debug_path.parent.mkdir(parents=True, exist_ok=True)

    same_out = False
    try:
        same_out = out_path.resolve() == out_debug_path.resolve()
    except Exception:
        same_out = str(out_path) == str(out_debug_path)

    # Fast path: when schema-only is enabled and the parquet already exists, avoid reading
    # any upstream H5/PKL/parquet factor tables. Just read parquet schema metadata and
    # enrich with field-map.
    if args.schema_only and out_path.exists():
        print("[INFO] schema-only enabled; reading parquet metadata from:", out_path)
        schema_cols = _schema_cols_from_parquet_metadata(out_path)
        if not schema_cols:
            raise RuntimeError(
                "schema-only requires reading parquet metadata. "
                "Failed to read schema from parquet. Please ensure pyarrow is installed, "
                "or re-run without --schema-only to rebuild parquet and schema."
            )

        snapshot_root = _to_unix_path(Path(args.snapshot_root))
        auto_field_map = snapshot_root / "metadata" / "aistock_field_map.csv"
        fm_candidate = _to_unix_path(Path(args.field_map)) if args.field_map else auto_field_map
        if fm_candidate and fm_candidate.exists():
            print("[INFO] Loading field-map:", fm_candidate)
            field_map = _load_field_map(fm_candidate)
            schema_cols = _apply_field_map(schema_cols, field_map)
        schema_cols = _fill_derived_meanings(schema_cols)

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

        if not same_out:
            print("[INFO] Writing schema:", schema_json_debug_path)
            schema_json_debug_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[INFO] Writing schema:", schema_csv_path)
        pd.DataFrame(schema["columns"]).to_csv(schema_csv_path, index=False, encoding="utf-8")

        if not same_out:
            print("[INFO] Writing schema:", schema_csv_debug_path)
            pd.DataFrame(schema["columns"]).to_csv(schema_csv_debug_path, index=False, encoding="utf-8")

        print("[SUCCESS] schema generated (schema-only).")
        return

    snapshot_root = _to_unix_path(Path(args.snapshot_root))
    aistock_factors_root = _to_unix_path(Path(args.aistock_factors_root))

    daily_basic_path = snapshot_root / "daily_basic.h5"
    moneyflow_path = snapshot_root / "moneyflow.h5"
    daily_pv_path = snapshot_root / "daily_pv.h5"

    print("[INFO] snapshot_root:", snapshot_root)
    print("[INFO] factors_root :", aistock_factors_root)

    # Raw tables
    print("[INFO] Loading raw daily_basic.h5 ...")
    df_db_raw = _read_table(daily_basic_path, "daily_basic_raw")

    print("[INFO] Loading raw moneyflow.h5 ...")
    df_mf_raw = _read_table(moneyflow_path, "moneyflow_raw")

    df_pv = None
    if daily_pv_path.exists():
        print("[INFO] Loading raw daily_pv.h5 ...")
        df_pv = _read_table(daily_pv_path, "daily_pv")
    else:
        print(f"[WARN] daily_pv.h5 not found: {daily_pv_path} (mf_*_ratio derived features will be skipped)")

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
    df_mf_derived = _derive_moneyflow_features(df_mf_raw, df_pv)
    if df_mf_derived.empty:
        for df_opt in dfs[2:]:
            if not isinstance(df_opt, pd.DataFrame) or df_opt.empty:
                continue
            df_try = df_opt[[c for c in df_opt.columns if str(c).startswith("mf_")]]
            if df_try.empty:
                continue
            df_mf_derived = _derive_moneyflow_features(df_try, df_pv)
            if not df_mf_derived.empty:
                break

    if not df_mf_derived.empty:
        dfs.append(df_mf_derived)

    print("[INFO] Concatenating tables (axis=1) ...")
    df_merged = pd.concat(dfs, axis=1)
    df_merged = df_merged.sort_index()
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated(keep="last")]

    schema_cols: list[dict[str, Any]]
    if args.schema_only and out_path.exists():
        print("[INFO] schema-only enabled; reading parquet metadata from:", out_path)
        schema_cols = _schema_cols_from_parquet_metadata(out_path) or _build_schema(df_merged)
    else:
        schema_cols = _build_schema(df_merged)

    if not args.schema_only:
        print("[INFO] Writing:", out_path)
        df_merged.to_parquet(out_path)

        if not same_out:
            print("[INFO] Writing:", out_debug_path)
            df_merged.to_parquet(out_debug_path)

    auto_field_map = snapshot_root / "metadata" / "aistock_field_map.csv"
    fm_candidate = _to_unix_path(Path(args.field_map)) if args.field_map else auto_field_map
    if fm_candidate and fm_candidate.exists():
        print("[INFO] Loading field-map:", fm_candidate)
        field_map = _load_field_map(fm_candidate)
        schema_cols = _apply_field_map(schema_cols, field_map)

    schema_cols = _fill_derived_meanings(schema_cols)

    _print_untraceable_schema_fields(schema_cols, title="Untraceable schema fields (meaning missing)")

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

    if not same_out:
        print("[INFO] Writing schema:", schema_json_debug_path)
        schema_json_debug_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Writing schema:", schema_csv_path)
    pd.DataFrame(schema["columns"]).to_csv(schema_csv_path, index=False, encoding="utf-8")

    if not same_out:
        print("[INFO] Writing schema:", schema_csv_debug_path)
        pd.DataFrame(schema["columns"]).to_csv(schema_csv_debug_path, index=False, encoding="utf-8")

    if args.schema_only:
        print("[SUCCESS] schema generated (schema-only).")
    else:
        print("[SUCCESS] static_factors.parquet + schema generated.")


if __name__ == "__main__":
    main()
