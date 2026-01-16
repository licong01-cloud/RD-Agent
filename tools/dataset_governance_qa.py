import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def _json_default(o: Any):
    """Best-effort JSON serializer for pandas/numpy objects."""

    # pandas Timestamp / datetime-like
    try:
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
    except Exception:
        pass

    # numpy scalar
    try:
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass

    # numpy datetime64
    try:
        if isinstance(o, np.datetime64):
            return str(o)
    except Exception:
        pass

    # numpy array
    try:
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass

    # pandas NA
    if o is pd.NA:
        return None

    # pathlib
    try:
        if isinstance(o, Path):
            return str(o)
    except Exception:
        pass

    return str(o)


@dataclass
class DatasetRef:
    dataset_id: str
    kind: str  # h5 | qlib_bin
    path: str
    key: str = "data"  # for h5
    region: str = "cn"  # for qlib


def _to_unix_path(p: str) -> str:
    s = str(p)
    if os.name != "nt":
        return s
    if len(s) >= 3 and s[1:3] == ":\\":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return s.replace("\\", "/")


def _guess_snapshot_root() -> str:
    return os.environ.get("AIstock_SNAPSHOT_ROOT", r"/mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209")


def _guess_provider_uri() -> str:
    return os.environ.get("AIstock_Qlib_PROVIDER_URI", "/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209")


def _safe_ratio(mask: pd.Series) -> float:
    if mask is None or len(mask) == 0:
        return float("nan")
    return float(mask.mean())


def _quantiles(s: pd.Series) -> dict[str, Any]:
    s = s.dropna()
    if len(s) == 0:
        return {"count": 0}
    qs = s.quantile([0.0, 0.01, 0.5, 0.99, 1.0]).to_dict()
    return {
        "count": int(len(s)),
        "min": float(qs.get(0.0)),
        "p01": float(qs.get(0.01)),
        "p50": float(qs.get(0.5)),
        "p99": float(qs.get(0.99)),
        "max": float(qs.get(1.0)),
    }


def _pick_default_key(store: pd.HDFStore) -> str:
    keys = store.keys()
    if not keys:
        raise RuntimeError("No keys found in H5.")
    for k in ("/data", "/daily", "/day", "/features"):
        if k in keys:
            return k
    return max(keys, key=len)


def _read_h5_sample(h5_path: Path, key: str, max_rows: int) -> pd.DataFrame:
    with pd.HDFStore(str(h5_path), mode="r") as store:
        h5_key = key
        if not h5_key.startswith("/"):
            h5_key = f"/{h5_key}"
        if h5_key not in store.keys():
            h5_key = _pick_default_key(store)
        df = store.select(h5_key, start=0, stop=max_rows)
    return df


def _scan_h5_unique_dates_instruments(h5_path: Path, key: str, chunk_size: int = 300000) -> tuple[list[str], list[str]]:
    with pd.HDFStore(str(h5_path), mode="r") as store:
        h5_key = key
        if not h5_key.startswith("/"):
            h5_key = f"/{h5_key}"
        if h5_key not in store.keys():
            h5_key = _pick_default_key(store)

        dates: set[str] = set()
        insts: set[str] = set()

        try:
            it = store.select(h5_key, iterator=True, chunksize=chunk_size)
            for chunk in it:
                dt = _infer_datetime_series(chunk)
                ins = _infer_instrument_series(chunk)
                if dt is not None:
                    dates.update(_normalize_dates(dt))
                if ins is not None:
                    insts.update(set(pd.Series(ins).dropna().astype(str).tolist()))
        except Exception:
            df = store.select(h5_key)
            dt = _infer_datetime_series(df)
            ins = _infer_instrument_series(df)
            if dt is not None:
                dates.update(_normalize_dates(dt))
            if ins is not None:
                insts.update(set(pd.Series(ins).dropna().astype(str).tolist()))

    return (sorted(dates), sorted(insts))


def _infer_datetime_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
        names = list(df.index.names)
        if "datetime" in names:
            return pd.to_datetime(df.index.get_level_values("datetime"), errors="coerce")
        return pd.to_datetime(df.index.get_level_values(0), errors="coerce")

    if df.index.inferred_type in {"datetime64", "date"}:
        return pd.to_datetime(df.index, errors="coerce")

    cand = [c for c in df.columns if isinstance(c, str) and ("date" in c.lower() or "time" in c.lower())]
    if cand:
        return pd.to_datetime(df[cand[0]], errors="coerce")
    return None


def _infer_instrument_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
        names = list(df.index.names)
        if "instrument" in names:
            return df.index.get_level_values("instrument").astype(str)
        return df.index.get_level_values(-1).astype(str)

    if "instrument" in df.columns:
        return df["instrument"].astype(str)
    return None


def _expected_h5_required_cols(h5_name: str) -> list[str]:
    base = h5_name.lower()
    if base == "daily_pv.h5":
        return ["open", "high", "low", "close", "volume", "amount"]
    if base == "daily_basic.h5":
        return ["db_turnover_rate", "db_circ_mv"]
    if base == "moneyflow.h5":
        return ["mf_net_amt"]
    return []


def check_h5_dataset(ds: DatasetRef, max_rows: int = 200000) -> dict[str, Any]:
    path = Path(ds.path)
    out: dict[str, Any] = {
        "dataset_id": ds.dataset_id,
        "kind": ds.kind,
        "path": str(path),
        "status": "pass",
        "checks": {},
    }

    if not path.exists():
        out["status"] = "fail"
        out["error"] = f"file not found: {path}"
        return out

    try:
        df = _read_h5_sample(path, ds.key, max_rows=max_rows)
    except Exception as e:
        out["status"] = "fail"
        out["error"] = f"failed to read h5: {repr(e)}"
        return out

    cols = [str(c) for c in df.columns]
    # Column naming policy differs by dataset:
    # - daily_pv.h5: should NOT use Qlib-style '$' prefixes.
    # - other H5 (daily_basic/moneyflow/...): allow any naming; typically db_/mf_.
    if Path(ds.path).name.lower() == "daily_pv.h5":
        dollar_cols = [c for c in cols if c.startswith("$")]
        out["checks"]["column_name_no_dollar_prefix"] = {
            "status": "fail" if dollar_cols else "pass",
            "dollar_cols_sample": dollar_cols[:50],
            "dollar_cols_count": len(dollar_cols),
        }
        if dollar_cols:
            out["status"] = "fail"
    else:
        out["checks"]["column_name_no_dollar_prefix"] = {
            "status": "skip",
            "message": "Only enforced for daily_pv.h5",
        }

    required = _expected_h5_required_cols(Path(ds.path).name)
    missing_required = [c for c in required if c not in cols]
    out["checks"]["required_columns"] = {
        "status": "fail" if missing_required else "pass",
        "required": required,
        "missing": missing_required,
    }
    if missing_required:
        out["status"] = "fail"

    dt = _infer_datetime_series(df)
    inst = _infer_instrument_series(df)

    date_cov: dict[str, Any] = {"status": "pass"}
    if dt is not None and len(dt) > 0:
        date_cov.update(
            {
                "min": str(pd.to_datetime(dt).min()),
                "max": str(pd.to_datetime(dt).max()),
                "null_ratio": _safe_ratio(pd.isna(dt)),
            }
        )
    else:
        date_cov["status"] = "warn"
        date_cov["message"] = "unable to infer datetime"

    out["checks"]["date_coverage"] = date_cov

    inst_cov: dict[str, Any] = {"status": "pass"}
    if inst is not None and len(inst) > 0:
        inst_cov.update(
            {
                "unique_count": int(pd.Series(inst).nunique(dropna=True)),
                "null_ratio": _safe_ratio(pd.Series(inst).isna()),
                "sample": [x for x in pd.Series(inst).dropna().unique()[:20].tolist()],
            }
        )
        inst_str = pd.Series(inst).dropna().astype(str)
        if len(inst_str) > 0:
            bad_fmt = inst_str[~inst_str.str.match(r"^\d{6}\.(SZ|SH|BJ)$")]
            inst_cov["bad_format_ratio"] = float(len(bad_fmt) / len(inst_str)) if len(inst_str) else float("nan")
            inst_cov["bad_format_sample"] = bad_fmt.unique()[:20].tolist()
            if inst_cov.get("bad_format_ratio", 0.0) > 0:
                inst_cov["status"] = "warn"
    else:
        inst_cov["status"] = "warn"
        inst_cov["message"] = "unable to infer instrument"

    out["checks"]["instrument_format"] = inst_cov

    quality: dict[str, Any] = {"status": "pass", "fields": {}}
    for f in ("amount", "volume", "close"):
        if f in df.columns:
            s_raw = df[f]
            s_num = pd.to_numeric(s_raw, errors="coerce")
            is_num_like = s_num.notna().any() or np.issubdtype(getattr(s_raw, "dtype", object), np.number)
            quality["fields"][f] = {
                "nan_ratio": _safe_ratio(s_raw.isna()),
                "zero_ratio": _safe_ratio((s_num == 0) if is_num_like else pd.Series(False, index=s_raw.index)),
                "negative_count": int((s_num < 0).sum(skipna=True)) if is_num_like else None,
                "quantiles": _quantiles(s_num) if is_num_like else None,
            }

    out["checks"]["key_field_quality"] = quality

    out["schema"] = {
        "index": {
            "type": "MultiIndex" if isinstance(df.index, pd.MultiIndex) else str(type(df.index)),
            "names": list(df.index.names) if isinstance(df.index, pd.MultiIndex) else [str(df.index.name)],
        },
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
    }

    # Provide full unique dates/instruments for alignment checks and to unify counts.
    out["sample"] = {"dates_unique": [], "instruments_unique": []}
    try:
        full_dates, full_insts = _scan_h5_unique_dates_instruments(Path(ds.path), ds.key)
        out["sample"]["dates_unique"] = full_dates
        out["sample"]["instruments_unique"] = full_insts

        # Unify QA check fields to full-scan results (avoid sample/max_rows bias)
        try:
            dc = out.get("checks", {}).get("date_coverage")
            if isinstance(dc, dict) and full_dates:
                dc["min"] = f"{min(full_dates)} 00:00:00"
                dc["max"] = f"{max(full_dates)} 00:00:00"
        except Exception:
            pass

        try:
            ic = out.get("checks", {}).get("instrument_format")
            if isinstance(ic, dict) and full_insts:
                ic["unique_count"] = int(len(full_insts))
                ic["sample"] = list(full_insts[:20])
        except Exception:
            pass
    except Exception:
        if dt is not None:
            try:
                out["sample"]["dates_unique"] = _normalize_dates(dt)
            except Exception:
                out["sample"]["dates_unique"] = []
        if inst is not None:
            try:
                out["sample"]["instruments_unique"] = sorted(pd.Series(inst).dropna().astype(str).unique().tolist())
            except Exception:
                out["sample"]["instruments_unique"] = []

    return out


def _qlib_init(provider_uri: str, region: str) -> None:
    import qlib

    qlib.init(provider_uri=provider_uri, region=region)


def _qlib_features_sample(
    instruments: list[str],
    fields: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    from qlib.data import D

    return D.features(instruments, fields, start_time=start_time, end_time=end_time, freq="day")


def check_qlib_bin_dataset(ds: DatasetRef) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset_id": ds.dataset_id,
        "kind": ds.kind,
        "provider_uri": ds.path,
        "region": ds.region,
        "status": "pass",
        "checks": {},
    }

    try:
        _qlib_init(ds.path, ds.region)
    except Exception as e:
        out["status"] = "fail"
        out["error"] = f"qlib.init failed: {repr(e)}"
        return out

    from qlib.data import D

    try:
        cal = D.calendar(start_time="2010-01-01", end_time="2025-12-31", freq="day")
        out["checks"]["calendar"] = {
            "status": "pass",
            "count": int(len(cal)),
            "min": str(cal[0]) if len(cal) else None,
            "max": str(cal[-1]) if len(cal) else None,
        }
    except Exception as e:
        out["checks"]["calendar"] = {"status": "fail", "error": repr(e)}
        out["status"] = "fail"

    insts: list[str] = []
    pool_summaries: dict[str, Any] = {}

    # Derive available pool names from provider_uri/instruments/*.txt.
    # Qlib typically maps pool name -> <provider_uri>/instruments/<pool_name>.txt
    pool_candidates: list[str] = ["all", "index"]
    try:
        instruments_dir = Path(ds.path) / "instruments"
        if instruments_dir.exists():
            pool_candidates = sorted([p.stem for p in instruments_dir.glob("*.txt")])
            # Ensure common pools are checked first if present.
            for must in ("all", "index"):
                if must in pool_candidates:
                    pool_candidates.remove(must)
                    pool_candidates.insert(0, must)
    except Exception:
        pool_candidates = ["all", "index"]

    for pool_name in pool_candidates:
        try:
            pool_cfg = D.instruments(pool_name)
            pool_insts = list(D.list_instruments(pool_cfg, start_time="2010-01-01", end_time="2025-12-31"))
            pool_summaries[pool_name] = {
                "status": "pass",
                "count": int(len(pool_insts)),
                "sample": pool_insts[:20],
            }
            if pool_name == "all":
                insts = pool_insts
                if len(pool_insts) <= 3:
                    pool_summaries[pool_name]["status"] = "warn"
                    pool_summaries[pool_name]["message"] = "too few instruments under pool 'all'"
        except Exception as e:
            pool_summaries[pool_name] = {"status": "fail", "error": repr(e)}

    out["checks"]["instrument_pools"] = pool_summaries
    if not insts and pool_summaries.get("all", {}).get("status") == "fail":
        out["status"] = "fail"

    fields = ["$open", "$high", "$low", "$close", "$volume"]
    if ds.dataset_id.endswith("daily"):
        fields.append("$amount")

    if insts:
        sample_inst = [insts[0]]
        try:
            df = _qlib_features_sample(sample_inst, fields, start_time="2016-01-01", end_time="2016-01-31")
            out["checks"]["features_sample"] = {
                "status": "pass",
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "columns": [str(c) for c in df.columns],
                "nan_ratio_by_field": {str(c): float(df[c].isna().mean()) for c in df.columns},
            }
        except Exception as e:
            out["checks"]["features_sample"] = {"status": "fail", "error": repr(e)}
            out["status"] = "fail"

    index_code = "000300.SH"
    try:
        df_idx = _qlib_features_sample([index_code], ["$close", "$volume"], start_time="2016-01-01", end_time="2016-01-15")
        out["checks"]["index_000300"] = {
            "status": "pass",
            "shape": [int(df_idx.shape[0]), int(df_idx.shape[1])],
            "columns": [str(c) for c in df_idx.columns],
            "nan_ratio_by_field": {str(c): float(df_idx[c].isna().mean()) for c in df_idx.columns},
        }
    except Exception as e:
        out["checks"]["index_000300"] = {"status": "fail", "error": repr(e)}
        out["status"] = "fail"

    # Explicit pool membership check for 000300.SH: should be in index pool if index exists.
    idx_pool = pool_summaries.get("index") if isinstance(pool_summaries, dict) else None
    if isinstance(idx_pool, dict) and idx_pool.get("status") == "pass":
        idx_sample = idx_pool.get("sample") or []
        out["checks"]["index_pool_contains_000300"] = {
            "status": "pass" if (index_code in idx_sample or idx_pool.get("count", 0) == 1) else "warn",
            "note": "index pool is allowed to contain only index instruments; all pool does not need to include 000300.SH",
            "index_code": index_code,
            "index_pool_count": idx_pool.get("count"),
            "index_pool_sample": idx_sample,
        }
    else:
        out["checks"]["index_pool_contains_000300"] = {
            "status": "skip",
            "note": "index pool not available",
            "index_code": index_code,
        }

    # Provide lightweight sample for downstream cross-dataset alignment checks.
    out["schema"] = {
        "calendar": out.get("checks", {}).get("calendar"),
        "fields_sample": out.get("checks", {}).get("features_sample", {}).get("columns"),
        "instrument_pools": out.get("checks", {}).get("instrument_pools"),
    }

    # Provide full instruments/calendar for consistent, exact comparisons.
    out["sample"] = {"dates_unique": [], "instruments_unique": []}
    try:
        all_txt = Path(ds.path) / "instruments" / "all.txt"
        if all_txt.exists():
            insts_full: list[str] = []
            with all_txt.open("r", encoding="utf-8") as f:
                for line in f:
                    code = line.split("\t")[0].strip()
                    if code:
                        insts_full.append(code)
            out["sample"]["instruments_unique"] = sorted(set(insts_full))
    except Exception:
        pass

    try:
        cal_txt = Path(ds.path) / "calendars" / "day.txt"
        if cal_txt.exists():
            days_full: list[str] = []
            with cal_txt.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        days_full.append(s[:10])
            out["sample"]["dates_unique"] = sorted(set(days_full))
    except Exception:
        pass

    return out


def _overlap_stats(a: Iterable[Any], b: Iterable[Any], max_examples: int = 30) -> dict[str, Any]:
    sa = set(a)
    sb = set(b)
    inter = sa & sb
    only_a = sa - sb
    only_b = sb - sa
    return {
        "a_count": int(len(sa)),
        "b_count": int(len(sb)),
        "intersection_count": int(len(inter)),
        "a_only_count": int(len(only_a)),
        "b_only_count": int(len(only_b)),
        "a_only_sample": sorted(list(only_a))[:max_examples],
        "b_only_sample": sorted(list(only_b))[:max_examples],
    }


def cross_dataset_alignment(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-source alignment checks.

    Notes:
    - This check currently uses sampled unique dates/instruments captured during per-dataset QA.
    - It is intended to catch naming/coding/misalignment regressions early.
    """

    by_id = {d.get("dataset_id"): d for d in datasets if isinstance(d, dict)}
    out: dict[str, Any] = {"status": "pass", "checks": {}}

    h5_ids = [k for k, v in by_id.items() if isinstance(v, dict) and v.get("kind") == "h5"]
    if not h5_ids:
        out["status"] = "warn"
        out["checks"]["h5_presence"] = {"status": "warn", "message": "no h5 datasets found"}
        return out

    # H5 alignment: compare instruments/dates across H5 datasets.
    h5_samples = {
        dsid: {
            "dates": (by_id.get(dsid, {}).get("sample", {}) or {}).get("dates_unique", []),
            "insts": (by_id.get(dsid, {}).get("sample", {}) or {}).get("instruments_unique", []),
        }
        for dsid in h5_ids
    }
    anchor = "daily_pv" if "daily_pv" in h5_samples else h5_ids[0]
    anchor_dates = _normalize_dates(h5_samples.get(anchor, {}).get("dates", []))
    anchor_insts = h5_samples.get(anchor, {}).get("insts", [])

    h5_align: dict[str, Any] = {"status": "pass", "anchor": anchor, "pairs": {}}
    for dsid in h5_ids:
        if dsid == anchor:
            continue
        ds_dates = _normalize_dates(h5_samples.get(dsid, {}).get("dates", []))
        ds_insts = h5_samples.get(dsid, {}).get("insts", [])
        h5_align["pairs"][dsid] = {
            "date_overlap": _overlap_stats(anchor_dates, ds_dates),
            "instrument_overlap": _overlap_stats(anchor_insts, ds_insts),
            "instrument_diff": {
                "anchor_only": _truncate_sorted(set(anchor_insts) - set(ds_insts)),
                "other_only": _truncate_sorted(set(ds_insts) - set(anchor_insts)),
            },
        }
        if h5_align["pairs"][dsid]["instrument_overlap"]["a_only_count"] > 0:
            h5_align["status"] = "warn"
        if h5_align["pairs"][dsid]["date_overlap"]["a_only_count"] > 0:
            h5_align["status"] = "warn"

    # H5 global inconsistent instruments summary
    try:
        h5_insts_by_id: dict[str, set[str]] = {
            d.get("dataset_id"): set(((d.get("sample") or {}).get("instruments_unique") or []))
            for d in datasets
            if isinstance(d, dict) and d.get("dataset_id") and d.get("kind") == "h5"
        }
        if h5_insts_by_id:
            union_set: set[str] = set().union(*h5_insts_by_id.values())
            inter_set: set[str] = set.intersection(*h5_insts_by_id.values())
            not_in_all = union_set - inter_set
            out["h5_instruments"] = {
                "h5_dataset_ids": sorted(h5_insts_by_id.keys()),
                "union_count": len(union_set),
                "intersection_count": len(inter_set),
                "not_in_all_count": len(not_in_all),
                "not_in_all_sample": _truncate_sorted(not_in_all, limit=200),
            }
    except Exception:
        pass

    out["checks"]["h5_alignment"] = h5_align

    # Baseline alignment against Qlib daily bin: instruments/all.txt and calendars/day.txt.
    baseline: dict[str, Any] = {"status": "warn", "baseline": "qlib_bin_daily"}
    qlib_ds = by_id.get("qlib_bin_daily")
    provider_uri = None
    if isinstance(qlib_ds, dict):
        provider_uri = qlib_ds.get("provider_uri") or qlib_ds.get("path")
    q_all: list[str] = []
    q_calendar: list[str] = []
    if isinstance(provider_uri, str) and provider_uri:
        try:
            inst_dir = Path(provider_uri) / "instruments"
            all_txt = inst_dir / "all.txt"
            if all_txt.exists():
                with all_txt.open("r", encoding="utf-8") as f:
                    for line in f:
                        code = line.split("\t")[0].strip()
                        if code:
                            q_all.append(code)
        except Exception:
            q_all = []

        try:
            cal_path = Path(provider_uri) / "calendars" / "day.txt"
            if cal_path.exists():
                with cal_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            q_calendar.append(s[:10])
        except Exception:
            q_calendar = []

    q_all_set = set(q_all)
    q_cal_set = set(q_calendar)
    baseline["instruments_all_count"] = len(q_all_set) if q_all_set else None
    baseline["calendar_count"] = len(q_cal_set) if q_cal_set else None

    if q_all_set and q_cal_set:
        baseline["status"] = "pass"

    per_dataset: dict[str, Any] = {}
    for ds_id, ds in by_id.items():
        if not isinstance(ds, dict):
            continue
        kind = ds.get("kind")
        ds_insts = []
        ds_dates = []
        if kind == "h5":
            ds_insts = ((ds.get("sample") or {}).get("instruments_unique") or [])
            ds_dates = _normalize_dates(((ds.get("sample") or {}).get("dates_unique") or []))
        elif kind == "qlib_bin":
            # Use full all.txt and day.txt as dataset coverage itself.
            ds_insts = list(q_all)
            ds_dates = list(q_calendar)

        ds_inst_set = set([x for x in ds_insts if isinstance(x, str) and x])
        ds_date_set = set([x for x in ds_dates if isinstance(x, str) and x])

        inst_extra = _sorted_full(ds_inst_set - q_all_set) if q_all_set else []
        inst_missing = _sorted_full(q_all_set - ds_inst_set) if q_all_set else []

        # Missing trading days: only within dataset date range (avoid listing 2011-2025 for 2010-only datasets)
        if q_cal_set and ds_date_set:
            ds_min = min(ds_date_set)
            ds_max = max(ds_date_set)
            expected_days = {d for d in q_cal_set if ds_min <= d <= ds_max}
            missing_days = _sorted_full(expected_days - ds_date_set)
            extra_days = _sorted_full(ds_date_set - expected_days)
        else:
            missing_days = []
            extra_days = []

        per_dataset[ds_id] = {
            "kind": kind,
            "instrument_count": len(ds_inst_set) if ds_inst_set else None,
            "date_count": len(ds_date_set) if ds_date_set else None,
            "instrument_extra_vs_bin": inst_extra,
            "instrument_missing_vs_bin": inst_missing,
            "missing_trading_days_vs_bin": missing_days,
            "extra_days_vs_bin": extra_days,
        }

    baseline["per_dataset"] = per_dataset
    out["bin_baseline"] = baseline

    # Backward-compatible: keep a coarse h5_vs_qlib section for summary rendering.
    out["checks"]["h5_vs_qlib"] = {
        "status": baseline.get("status"),
        "baseline": "qlib_bin_daily",
        "notes": "bin_baseline provides full diffs vs qlib bin all.txt + calendars/day.txt",
    }

    # Instrument count summary (best-effort) + H5 alignment vs anchor
    instrument_counts: dict[str, Any] = {}
    for ds_id, ds in by_id.items():
        if not isinstance(ds, dict):
            continue
        cnt, sample = _get_instrument_count_and_sample(ds)
        instrument_counts[ds_id] = {"count": cnt, "sample": sample}
    out["instrument_counts"] = instrument_counts

    return out


def _get_instrument_count_and_sample(ds: dict[str, Any]) -> tuple[Optional[int], list[str]]:
    """Return (count, sample_list) best-effort across dataset kinds."""

    kind = ds.get("kind")
    if kind == "h5":
        inst = ((ds.get("checks") or {}).get("instrument_format") or {})
        cnt = inst.get("unique_count")
        sample = inst.get("sample") or ((ds.get("sample") or {}).get("instruments_unique") or [])
        return (int(cnt) if cnt is not None else None, list(sample)[:50])

    if kind == "qlib_bin":
        pools = ((ds.get("checks") or {}).get("instrument_pools") or {})
        all_pool = pools.get("all") if isinstance(pools, dict) else None
        if isinstance(all_pool, dict):
            cnt = all_pool.get("count")
            sample = all_pool.get("sample") or []
            return (int(cnt) if cnt is not None else None, list(sample)[:50])

    return (None, [])


def _truncate_sorted(items: Iterable[str], limit: int = 200) -> list[str]:
    out = sorted(set([x for x in items if isinstance(x, str) and x]))
    return out[:limit]


def _sorted_full(items: Iterable[str]) -> list[str]:
    return sorted(set([x for x in items if isinstance(x, str) and x]))


def _normalize_dates(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for x in items:
        if x is None:
            continue
        # Accept "YYYY-MM-DD", ISO timestamps, pandas timestamps already stringified.
        s = str(x)
        if len(s) >= 10:
            out.append(s[:10])
    return sorted(set(out))


def build_default_manifest(snapshot_root: str, provider_uri: str, region: str) -> list[DatasetRef]:
    snap = Path(snapshot_root)
    h5s: list[DatasetRef] = []
    if snap.exists():
        for p in sorted(snap.glob("*.h5")):
            if p.stem == "result":
                continue
            h5s.append(DatasetRef(dataset_id=p.stem, kind="h5", path=str(p), key="data"))

    out = h5s + [DatasetRef(dataset_id="qlib_bin_daily", kind="qlib_bin", path=provider_uri, region=region)]
    return out


def _flatten_manifest_rows(manifest: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ds in manifest:
        base = {
            "dataset_id": ds.get("dataset_id"),
            "kind": ds.get("kind"),
            "path": ds.get("path") or ds.get("provider_uri"),
            "status": ds.get("status"),
        }

        # Full-scan counts if available
        sample = ds.get("sample") or {}
        if isinstance(sample, dict):
            insts = sample.get("instruments_unique") or []
            dates = sample.get("dates_unique") or []
            if isinstance(insts, list):
                base["instrument_count_full"] = int(len(insts))
            if isinstance(dates, list):
                base["date_count_full"] = int(len(dates))
                base["date_min_full"] = dates[0] if dates else None
                base["date_max_full"] = dates[-1] if dates else None

        checks = ds.get("checks") or {}
        if isinstance(checks, dict):
            for k, v in checks.items():
                if not isinstance(v, dict):
                    continue
                base[f"check_{k}_status"] = v.get("status")
        rows.append(base)
    return pd.DataFrame(rows)


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Dataset Governance Report")
    lines.append("")
    lines.append(f"- generated_at: {report.get('generated_at')}")
    lines.append(f"- snapshot_root: {report.get('snapshot_root')}")
    lines.append(f"- provider_uri: {report.get('provider_uri')}")
    lines.append("")

    ca = report.get("cross_alignment")
    if isinstance(ca, dict):
        lines.append("## Cross Alignment Summary")
        lines.append("")
        lines.append(f"- status: `{ca.get('status')}`")
        checks = ca.get("checks")
        if isinstance(checks, dict):
            for k in ["h5_alignment", "h5_vs_qlib"]:
                if k in checks and isinstance(checks[k], dict):
                    lines.append(f"- **{k}**: `{checks[k].get('status')}`")
        bb = ca.get("bin_baseline")
        if isinstance(bb, dict):
            lines.append(f"- **bin_baseline**: `{bb.get('status')}`")
            lines.append(f"- **bin_all_count**: `{bb.get('instruments_all_count')}`")
            lines.append(f"- **bin_calendar_count**: `{bb.get('calendar_count')}`")
        inst_counts = ca.get("instrument_counts")
        if isinstance(inst_counts, dict) and inst_counts:
            lines.append("")
            lines.append("### Instrument Counts")
            lines.append("")
            for ds_id in sorted(inst_counts.keys()):
                v = inst_counts.get(ds_id) or {}
                if isinstance(v, dict):
                    lines.append(f"- **{ds_id}**: `{v.get('count')}`")
        h5_inst = ca.get("h5_instruments")
        if isinstance(h5_inst, dict) and h5_inst:
            lines.append("")
            lines.append("### H5 Instrument Inconsistency (sample)")
            lines.append("")
            lines.append(f"- **union_count**: `{h5_inst.get('union_count')}`")
            lines.append(f"- **intersection_count**: `{h5_inst.get('intersection_count')}`")
            lines.append(f"- **not_in_all_count**: `{h5_inst.get('not_in_all_count')}`")
            sample = h5_inst.get("not_in_all_sample") or []
            if sample:
                lines.append(f"- **not_in_all_sample**: `{', '.join(sample[:30])}`")
        lines.append("")

    datasets = report.get("datasets", [])
    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        dsid = ds.get("dataset_id")
        status = ds.get("status")
        lines.append(f"## {dsid} ({status})")
        lines.append("")
        if ds.get("kind") == "h5":
            lines.append(f"- path: `{ds.get('path')}`")
        else:
            lines.append(f"- provider_uri: `{ds.get('provider_uri')}`")

        checks = ds.get("checks") or {}
        if isinstance(checks, dict):
            for k, v in checks.items():
                if not isinstance(v, dict):
                    continue
                st = v.get("status")
                if st is None and k == "instrument_pools":
                    st = "pass" if v else "n/a"
                lines.append(f"- **{k}**: `{st}`")
        if ds.get("error"):
            lines.append(f"- error: `{ds.get('error')}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified governance QA for all H5 datasets (daily_pv/moneyflow/daily_basic/...) and Qlib bin datasets "
            "(daily features and index 000300.SH). Outputs report.json/report.md and datasets_manifest.csv/json."
        )
    )
    parser.add_argument("--snapshot-root", default=_guess_snapshot_root())
    parser.add_argument("--provider-uri", default=_guess_provider_uri())
    parser.add_argument("--region", default=os.environ.get("AIstock_Qlib_REGION", "cn"))
    parser.add_argument("--out-dir", default=os.environ.get("AIstock_GOVERNANCE_OUT", r"/mnt/f/Dev/AIstock/data_governance"))
    parser.add_argument("--max-rows", type=int, default=200000)

    args = parser.parse_args()

    snapshot_root = args.snapshot_root
    provider_uri = args.provider_uri
    region = args.region
    out_dir = Path(args.out_dir)

    manifest = build_default_manifest(snapshot_root=snapshot_root, provider_uri=provider_uri, region=region)

    results: list[dict[str, Any]] = []
    schema_entries: list[dict[str, Any]] = []

    for ds in manifest:
        if ds.kind == "h5":
            res = check_h5_dataset(ds, max_rows=int(args.max_rows))
        else:
            res = check_qlib_bin_dataset(ds)
        results.append(res)

        schema_obj = res.get("schema")
        if isinstance(schema_obj, dict):
            # Add full-scan coverage stats into manifest for consistency
            sample = res.get("sample") or {}
            insts = sample.get("instruments_unique") if isinstance(sample, dict) else None
            dates = sample.get("dates_unique") if isinstance(sample, dict) else None
            schema_entries.append(
                {
                    "dataset_id": res.get("dataset_id"),
                    "kind": res.get("kind"),
                    "path": res.get("path") or res.get("provider_uri"),
                    "schema": schema_obj,
                    "coverage": {
                        "instrument_count_full": int(len(insts)) if isinstance(insts, list) else None,
                        "date_count_full": int(len(dates)) if isinstance(dates, list) else None,
                        "date_min_full": dates[0] if isinstance(dates, list) and dates else None,
                        "date_max_full": dates[-1] if isinstance(dates, list) and dates else None,
                    },
                }
            )

    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "snapshot_root": snapshot_root,
        "provider_uri": provider_uri,
        "region": region,
        "datasets": results,
    }

    report["cross_alignment"] = cross_dataset_alignment(results)

    out_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = out_dir / "governance_report.json"
    report_md_path = out_dir / "governance_report.md"
    manifest_json_path = out_dir / "datasets_manifest.json"
    manifest_csv_path = out_dir / "datasets_manifest.csv"

    report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    report_md_path.write_text(_render_markdown(report), encoding="utf-8")

    datasets_manifest = {
        "generated_at": report["generated_at"],
        "snapshot_root": snapshot_root,
        "provider_uri": provider_uri,
        "region": region,
        "datasets": schema_entries,
    }
    manifest_json_path.write_text(
        json.dumps(datasets_manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _flatten_manifest_rows(results).to_csv(manifest_csv_path, index=False, encoding="utf-8")

    print("[OK] wrote:")
    print(" -", report_json_path)
    print(" -", report_md_path)
    print(" -", manifest_json_path)
    print(" -", manifest_csv_path)


if __name__ == "__main__":
    main()
