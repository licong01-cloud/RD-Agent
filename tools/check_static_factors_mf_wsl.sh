#!/usr/bin/env bash
set -euo pipefail

# WSL-only script
# Purpose: Inspect static_factors.parquet moneyflow columns (mf_*) for missing/all-NaN issues.

REPO_ROOT="${REPO_ROOT:-/mnt/f/dev/RD-Agent-main}"
PARQUET_PATH="${PARQUET_PATH:-${REPO_ROOT}/git_ignore_folder/factor_implementation_source_data/static_factors.parquet}"
PY_BIN="${PY_BIN:-python3}"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] PARQUET_PATH=${PARQUET_PATH}"
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
import pandas as pd
from pathlib import Path

p = Path("${PARQUET_PATH}")
df = pd.read_parquet(p)

print("=== static_factors.parquet basic ===")
print("shape:", df.shape)
print("columns:", len(df.columns))
print("index_type:", type(df.index))
print("index_names:", getattr(df.index, "names", None))

cols = list(df.columns)
mf_cols = [c for c in cols if isinstance(c, str) and c.startswith("mf_")]
print("\n=== mf_* columns ===")
print("num_mf_cols:", len(mf_cols))

key_cols = [
    "mf_lg_buy_amt","mf_lg_sell_amt",
    "mf_sm_buy_amt","mf_sm_sell_amt",
    "mf_md_buy_amt","mf_md_sell_amt",
    "mf_elg_buy_amt","mf_elg_sell_amt",
]
present = [c for c in key_cols if c in df.columns]
missing = [c for c in key_cols if c not in df.columns]
print("key_mf_present:", present)
print("key_mf_missing:", missing)

if len(mf_cols) == 0:
    raise SystemExit("[ERROR] No mf_* columns found. Likely the bundle did not include moneyflow fields.")

non_null_ratio = df[mf_cols].notna().mean().sort_values()
all_nan = non_null_ratio[non_null_ratio == 0.0]

print("\n=== mf_* non-null ratio summary ===")
print("mf_all_nan_count:", int(all_nan.shape[0]))
print("min:", float(non_null_ratio.iloc[0]))
print("p10:", float(non_null_ratio.quantile(0.10)))
print("median:", float(non_null_ratio.median()))
print("max:", float(non_null_ratio.iloc[-1]))

print("\n=== lowest_30_mf_cols (non-null ratio) ===")
print(non_null_ratio.head(30).to_string())

if "mf_lg_buy_amt" in df.columns and "datetime" in (getattr(df.index, "names", []) or []):
    dt = pd.to_datetime(df.index.get_level_values("datetime"), errors="coerce")
    s = df["mf_lg_buy_amt"]
    by_day = s.notna().groupby(dt).mean().dropna()
    print("\n=== mf_lg_buy_amt daily non-null ratio ===")
    print("days:", int(by_day.shape[0]), "min:", float(by_day.min()), "median:", float(by_day.median()), "max:", float(by_day.max()))
    print("head3:", dict(list(by_day.head(3).items())))
    print("tail3:", dict(list(by_day.tail(3).items())))
else:
    print("\n[WARN] Cannot compute daily coverage for mf_lg_buy_amt (missing column or index has no datetime).")
PY

echo "[DONE] check finished."
