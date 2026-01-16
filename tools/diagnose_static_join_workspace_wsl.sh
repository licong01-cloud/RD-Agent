#!/usr/bin/env bash
set -euo pipefail

# WSL-only script
# Diagnose whether factor.py join(static_factors.parquet) fails due to index/instrument mismatch.
#
# Usage:
#   cd /mnt/f/dev/RD-Agent-main
#   chmod +x tools/diagnose_static_join_workspace_wsl.sh
#   tools/diagnose_static_join_workspace_wsl.sh /mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/<workspace_id>
#
# Optional env:
#   PY_BIN=python3
#   REPO_ROOT=/mnt/f/dev/RD-Agent-main

WS_PATH="${1:-}"
if [ -z "${WS_PATH}" ]; then
  echo "[ERROR] Missing workspace path argument."
  echo "Example: tools/diagnose_static_join_workspace_wsl.sh /mnt/f/dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/840e6b9c423a4c699c2f6b0b74a65c9c"
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-/mnt/f/dev/RD-Agent-main}"
PY_BIN="${PY_BIN:-python3}"

DAILY_PV="${WS_PATH}/daily_pv.h5"
STATIC_PARQUET="${WS_PATH}/static_factors.parquet"

echo "[INFO] WS_PATH=${WS_PATH}"
echo "[INFO] DAILY_PV=${DAILY_PV}"
echo "[INFO] STATIC_PARQUET=${STATIC_PARQUET}"
echo "[INFO] PY_BIN=${PY_BIN}"

if [ ! -f "${DAILY_PV}" ]; then
  echo "[ERROR] daily_pv.h5 not found in workspace: ${DAILY_PV}"
  exit 2
fi

if [ ! -f "${STATIC_PARQUET}" ]; then
  echo "[ERROR] static_factors.parquet not found in workspace: ${STATIC_PARQUET}"
  echo "[HINT] If factor execution did not link static_factors.parquet into workspace, check FactorFBWorkspace.link_all_files_in_folder_to_workspace."
  exit 3
fi

${PY_BIN} - <<'PY'
import sys
try:
    import pyarrow  # noqa: F401
except Exception as e:
    print("[ERROR] pyarrow not available:", repr(e))
    print("[HINT] In WSL, run: pip install -U pyarrow")
    sys.exit(4)
PY

${PY_BIN} - <<PY
import re
from pathlib import Path

import pandas as pd

ws = Path("${WS_PATH}")
daily_pv_path = ws / "daily_pv.h5"
static_path = ws / "static_factors.parquet"

print("=== Load daily_pv.h5 ===")
df_pv = pd.read_hdf(daily_pv_path, key="data").sort_index()
print("pv_shape:", df_pv.shape)
print("pv_index_type:", type(df_pv.index))
print("pv_index_names:", getattr(df_pv.index, "names", None))

print("\n=== Load static_factors.parquet ===")
df_static = pd.read_parquet(static_path).sort_index()
print("static_shape:", df_static.shape)
print("static_index_type:", type(df_static.index))
print("static_index_names:", getattr(df_static.index, "names", None))

# Basic index name sanity
if not isinstance(df_pv.index, pd.MultiIndex) or "instrument" not in df_pv.index.names or "datetime" not in df_pv.index.names:
    print("[FAIL] daily_pv.h5 index is not MultiIndex(datetime, instrument)")
if not isinstance(df_static.index, pd.MultiIndex) or "instrument" not in df_static.index.names or "datetime" not in df_static.index.names:
    print("[FAIL] static_factors.parquet index is not MultiIndex(datetime, instrument)")

# Show instrument samples and inferred format
pat_aistock = re.compile(r"^(SH|SZ)\d{6}$")
pat_qlib = re.compile(r"^\d{6}\.(SH|SZ)$")

def summarize_instrument(idx, tag: str):
    ins = pd.Index(idx.get_level_values("instrument").astype(str))
    sample = ins.unique()[:10].tolist()
    a = pd.Series(ins.map(lambda x: bool(pat_aistock.match(x))), dtype="float64").mean()
    b = pd.Series(ins.map(lambda x: bool(pat_qlib.match(x))), dtype="float64").mean()
    print(f"{tag}_instrument_unique:", int(ins.nunique()))
    print(f"{tag}_instrument_sample_10:", sample)
    print(f"{tag}_instrument_share_aistock_SHSZxxxxxx:", float(a))
    print(f"{tag}_instrument_share_qlib_xxxxxx.SH:", float(b))

print("\n=== Instrument format summary ===")
summarize_instrument(df_pv.index, "pv")
summarize_instrument(df_static.index, "static")

# Join diagnostics
print("\n=== Join diagnostics ===")
# pick a few key cols that should exist in static
key_cols = [
    "db_turnover_rate",
    "db_circ_mv",
    "mf_lg_buy_amt",
    "mf_elg_buy_amt",
    "mf_lg_sell_amt",
    "mf_elg_sell_amt",
]
key_present = [c for c in key_cols if c in df_static.columns]
print("static_key_cols_present:", key_present)

# 1) Direct join
joined = df_pv.join(df_static[key_present], how="left") if key_present else df_pv.join(df_static, how="left")
for c in key_present:
    nn = float(joined[c].notna().mean())
    print(f"direct_join_non_null_ratio[{c}]:", nn)

# 2) Instrument-normalized join attempts

def normalize_to_qlib_style(index: pd.MultiIndex) -> pd.MultiIndex:
    dt = index.get_level_values("datetime")
    ins = pd.Index(index.get_level_values("instrument").astype(str))
    m = ins.str.match(r"^(SH|SZ)(\d{6})$")
    if bool(m.any()):
        exch = ins.str.slice(0, 2)
        code = ins.str.slice(2, 8)
        ins = ins.where(~m, code + "." + exch)
    return pd.MultiIndex.from_arrays([dt, ins], names=["datetime", "instrument"])


def normalize_to_aistock_style(index: pd.MultiIndex) -> pd.MultiIndex:
    dt = index.get_level_values("datetime")
    ins = pd.Index(index.get_level_values("instrument").astype(str))
    m = ins.str.match(r"^(\d{6})\.(SH|SZ)$")
    if bool(m.any()):
        code = ins.str.extract(r"^(\d{6})\.(SH|SZ)$", expand=True)[0]
        exch = ins.str.extract(r"^(\d{6})\.(SH|SZ)$", expand=True)[1]
        ins = ins.where(~m, exch + code)
    return pd.MultiIndex.from_arrays([dt, ins], names=["datetime", "instrument"])

# Normalize pv -> qlib, join
pv_q = df_pv.copy()
pv_q.index = normalize_to_qlib_style(pv_q.index)
# handle duplicates after normalization
if not pv_q.index.is_unique:
    pv_q = pv_q[~pv_q.index.duplicated(keep="last")]
joined_q = pv_q.join(df_static[key_present], how="left") if key_present else pv_q.join(df_static, how="left")
print("\n[normalized pv->qlib] join non-null ratios:")
for c in key_present:
    print(f"pv_to_qlib_non_null_ratio[{c}]:", float(joined_q[c].notna().mean()))

# Normalize static -> aistock, join
st_a = df_static.copy()
st_a.index = normalize_to_aistock_style(st_a.index)
if not st_a.index.is_unique:
    st_a = st_a[~st_a.index.duplicated(keep="last")]
joined_a = df_pv.join(st_a[key_present], how="left") if key_present else df_pv.join(st_a, how="left")
print("\n[normalized static->aistock] join non-null ratios:")
for c in key_present:
    print(f"static_to_aistock_non_null_ratio[{c}]:", float(joined_a[c].notna().mean()))

# Quick verdict
if key_present:
    direct_ok = max(float(joined[c].notna().mean()) for c in key_present)
    pvq_ok = max(float(joined_q[c].notna().mean()) for c in key_present)
    sta_ok = max(float(joined_a[c].notna().mean()) for c in key_present)
    print("\n=== Verdict ===")
    print("direct_join_best_non_null_ratio:", direct_ok)
    print("pv->qlib_join_best_non_null_ratio:", pvq_ok)
    print("static->aistock_join_best_non_null_ratio:", sta_ok)
    if pvq_ok > direct_ok + 0.1:
        print("[LIKELY_ROOT_CAUSE] daily_pv instrument is SH/SZxxxxxx while static is xxxxxx.SH; normalize daily_pv instrument before join.")
    elif sta_ok > direct_ok + 0.1:
        print("[LIKELY_ROOT_CAUSE] static instrument is xxxxxx.SH while daily_pv is SH/SZxxxxxx; normalize static instrument before join.")
    elif direct_ok < 0.05 and pvq_ok < 0.05 and sta_ok < 0.05:
        print("[LIKELY_ROOT_CAUSE] join keys mismatch beyond simple format. Check datetime alignment, calendar, or instrument universe differences.")
    else:
        print("[INFO] join seems mostly OK; NaN issue may come from formula/rolling/denominator zero rather than join mismatch.")
else:
    print("[WARN] no key cols found in static; cannot assess join quality via key cols.")
PY

echo "[DONE] diagnosis finished."
