#!/usr/bin/env bash
set -euo pipefail

# WSL-only script
# Purpose: Full sanity check for static_factors.parquet
# - schema alignment (optional)
# - index validity (MultiIndex datetime/instrument)
# - instrument format checks
# - missing ratio distribution
# - dtype checks
# - inf/nan checks
# - basic value ranges & suspicious extremes (quantiles)

REPO_ROOT="${REPO_ROOT:-/mnt/f/dev/RD-Agent-main}"
PARQUET_PATH="${PARQUET_PATH:-${REPO_ROOT}/git_ignore_folder/factor_implementation_source_data/static_factors.parquet}"
SCHEMA_JSON="${SCHEMA_JSON:-${REPO_ROOT}/git_ignore_folder/factor_implementation_source_data/static_factors_schema.json}"
SCHEMA_CSV="${SCHEMA_CSV:-${REPO_ROOT}/git_ignore_folder/factor_implementation_source_data/static_factors_schema.csv}"
PY_BIN="${PY_BIN:-python3}"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] PARQUET_PATH=${PARQUET_PATH}"
echo "[INFO] SCHEMA_JSON=${SCHEMA_JSON}"
echo "[INFO] SCHEMA_CSV=${SCHEMA_CSV}"
echo "[INFO] PY_BIN=${PY_BIN}"

if [ ! -f "${PARQUET_PATH}" ]; then
  echo "[ERROR] static_factors.parquet not found: ${PARQUET_PATH}"
  exit 1
fi

${PY_BIN} - <<'PY'
import sys
try:
    import pyarrow  # noqa: F401
    print("[INFO] pyarrow OK")
except Exception as e:
    print("[ERROR] pyarrow not available:", repr(e))
    print("[HINT] In WSL, run: pip install -U pyarrow")
    sys.exit(2)
PY

${PY_BIN} - <<PY
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET_PATH = Path("${PARQUET_PATH}")
SCHEMA_JSON = Path("${SCHEMA_JSON}")
SCHEMA_CSV = Path("${SCHEMA_CSV}")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)

print("=== Load parquet ===")
df = pd.read_parquet(PARQUET_PATH)
print("shape:", df.shape)
print("columns:", len(df.columns))
print("index_type:", type(df.index))
print("index_names:", getattr(df.index, "names", None))

print("\n=== Index checks ===")
# 1) MultiIndex names
if not isinstance(df.index, pd.MultiIndex):
    print("[FAIL] index is not MultiIndex")
else:
    names = list(df.index.names)
    print("index_names:", names)
    if "datetime" not in names or "instrument" not in names:
        print("[FAIL] index names must include ['datetime','instrument']")

# 2) datetime parse
if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names:
    dt_raw = df.index.get_level_values("datetime")
    dt = pd.to_datetime(dt_raw, errors="coerce")
    bad_dt = int(pd.isna(dt).sum())
    print("datetime_bad_count:", bad_dt)
    if bad_dt > 0:
        print("[WARN] datetime contains NaT (invalid timestamps)")
    else:
        print("datetime_range:", str(dt.min()), "->", str(dt.max()))

# 3) instrument format
if isinstance(df.index, pd.MultiIndex) and "instrument" in df.index.names:
    ins = df.index.get_level_values("instrument")
    ins_s = pd.Series(ins.astype(str))
    # allow SH/SZ + 6 digits (AIstock style), also allow 6 digits + .SH/.SZ (qlib style)
    pat_aistock = re.compile(r"^(SH|SZ)\d{6}$")
    pat_qlib = re.compile(r"^\d{6}\.(SH|SZ)$")
    ok = ins_s.map(lambda x: bool(pat_aistock.match(x) or pat_qlib.match(x)))
    bad = ins_s[~ok]
    print("instrument_total:", int(ins_s.shape[0]))
    print("instrument_unique:", int(ins_s.nunique()))
    print("instrument_bad_count:", int(bad.shape[0]))
    if bad.shape[0] > 0:
        print("[WARN] sample_bad_instrument:", bad.head(20).tolist())

# 4) duplicated index
try:
    dup = int(df.index.duplicated().sum())
    print("index_duplicated_count:", dup)
    if dup > 0:
        print("[WARN] duplicated MultiIndex rows exist; downstream joins may be unstable")
except Exception as e:
    print("[WARN] cannot compute index duplicates:", repr(e))

print("\n=== Schema alignment (optional) ===")
# Schema expected columns (from schema json/csv if present)
expected_cols = None
if SCHEMA_JSON.exists():
    try:
        j = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
        # support two common layouts
        if isinstance(j, dict) and "columns" in j and isinstance(j["columns"], list):
            expected_cols = [c.get("name") for c in j["columns"] if isinstance(c, dict) and c.get("name")]
        elif isinstance(j, list):
            expected_cols = [c.get("name") for c in j if isinstance(c, dict) and c.get("name")]
    except Exception as e:
        print("[WARN] failed to parse schema json:", repr(e))

if expected_cols is None and SCHEMA_CSV.exists():
    try:
        sc = pd.read_csv(SCHEMA_CSV)
        if "name" in sc.columns:
            expected_cols = sc["name"].dropna().astype(str).tolist()
    except Exception as e:
        print("[WARN] failed to parse schema csv:", repr(e))

if expected_cols is None:
    print("[WARN] schema not found or not parsed; skip expected-vs-actual columns check")
else:
    actual_cols = [c for c in df.columns if isinstance(c, str)]
    expected_set = set(expected_cols)
    actual_set = set(actual_cols)
    missing = sorted(list(expected_set - actual_set))
    extra = sorted(list(actual_set - expected_set))
    print("expected_cols:", len(expected_cols))
    print("actual_cols:", len(actual_cols))
    print("missing_expected_cols:", len(missing))
    print("extra_unexpected_cols:", len(extra))
    if missing:
        print("[WARN] missing_expected_cols_sample:", missing[:50])
    if extra:
        print("[INFO] extra_unexpected_cols_sample:", extra[:50])

print("\n=== Column dtype / missing / inf checks ===")
# Only numeric-like columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
non_num_cols = [c for c in df.columns if c not in num_cols]
print("numeric_cols:", len(num_cols))
print("non_numeric_cols:", len(non_num_cols))
if non_num_cols:
    print("non_numeric_cols_sample:", list(map(str, non_num_cols[:30])))

# missing ratios
miss = df.isna().mean().sort_values(ascending=False)
print("\nmissing_ratio_summary:")
print("max:", float(miss.iloc[0]))
print("p95:", float(miss.quantile(0.95)))
print("median:", float(miss.median()))
print("min:", float(miss.iloc[-1]))
print("\nhighest_missing_30_cols:")
print(miss.head(30).to_string())

# inf checks for numeric
if num_cols:
    inf_counts = {}
    for c in num_cols:
        s = df[c]
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
            # coerce to float for isfinite
            arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", copy=False)
            inf_counts[c] = int(np.isinf(arr).sum())
    inf_s = pd.Series(inf_counts).sort_values(ascending=False)
    total_inf = int(inf_s.sum())
    print("\ntotal_inf_values:", total_inf)
    top_inf = inf_s[inf_s > 0]
    if not top_inf.empty:
        print("[WARN] columns_with_inf:")
        print(top_inf.head(30).to_string())

print("\n=== Value range spot checks (quantiles) ===")
# To avoid huge compute, sample a subset of rows for quantiles
N = df.shape[0]
sample_n = 200000
sample = df
if N > sample_n:
    sample = df.sample(n=sample_n, random_state=0)
    print(f"[INFO] using sample_n={sample_n} for quantile checks (total_rows={N})")
else:
    print(f"[INFO] using full data for quantile checks (total_rows={N})")

num_cols_s = [c for c in sample.columns if pd.api.types.is_numeric_dtype(sample[c])]
if num_cols_s:
    q = sample[num_cols_s].quantile([0.001, 0.01, 0.5, 0.99, 0.999])
    # flag potential ratio fields outside reasonable bounds
    ratio_cols = [c for c in num_cols_s if isinstance(c, str) and (c.endswith('_ratio') or '_ratio_' in c)]
    if ratio_cols:
        q_ratio = q[ratio_cols]
        out = []
        for c in ratio_cols:
            lo = float(q_ratio.loc[0.001, c])
            hi = float(q_ratio.loc[0.999, c])
            if lo < -5 or hi > 5:
                out.append((c, lo, hi))
        print("ratio_cols:", len(ratio_cols))
        print("ratio_cols_suspicious_count:", len(out))
        if out:
            print("[WARN] suspicious_ratio_cols (q0.1%..q99.9%):")
            for c, lo, hi in out[:50]:
                print(f" - {c}: {lo:.6g} .. {hi:.6g}")

    # show a compact summary for key moneyflow columns if present
    key = [
        'db_circ_mv','db_total_mv','db_turnover_rate',
        'mf_lg_buy_amt','mf_lg_sell_amt','mf_net_amt','mf_total_net_amt','mf_main_net_amt'
    ]
    key_present = [c for c in key if c in sample.columns]
    if key_present:
        print("\nkey_cols_quantiles (sample):")
        print(q[key_present].to_string())

print("\n=== Done. ===")
PY

echo "[DONE] full check finished."
