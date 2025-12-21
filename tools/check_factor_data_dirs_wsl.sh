#!/usr/bin/env bash
set -euo pipefail

# WSL-only script
# Check whether exported H5 files and RD-Agent factor execution data folders
# use Qlib-style instruments (000001.SZ/600000.SH) or AIstock-style (SH600000/SZ000001).
#
# Usage:
#   cd /mnt/f/dev/RD-Agent-main
#   chmod +x tools/check_factor_data_dirs_wsl.sh
#   tools/check_factor_data_dirs_wsl.sh /mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209
#
# Optional env:
#   PY_BIN=python3

EXPORT_DIR="${1:-}"
if [ -z "${EXPORT_DIR}" ]; then
  echo "[ERROR] Missing EXPORT_DIR argument."
  echo "Example: tools/check_factor_data_dirs_wsl.sh /mnt/f/Dev/AIstock/qlib_snapshots/qlib_export_20251209"
  exit 1
fi

PY_BIN="${PY_BIN:-python3}"

${PY_BIN} - <<PY
import os
import re
from pathlib import Path

import pandas as pd

export_dir = Path("${EXPORT_DIR}").resolve()
files = ["daily_pv.h5", "daily_basic.h5", "moneyflow.h5"]

pat_aistock = re.compile(r"^(SH|SZ)\d{6}$")  # SH600000
pat_qlib = re.compile(r"^\d{6}\.(SH|SZ)$")  # 600000.SH


def _mean_bool(values):
    # values can be Index/ndarray/list of bool
    return float(pd.Series(list(values), dtype="float64").mean())


def summarize_h5(h5_path: Path, tag: str) -> None:
    df = pd.read_hdf(h5_path, key="data").sort_index()

    print(f"\n=== {tag} ===")
    print("path:", str(h5_path))
    print("shape:", df.shape)
    print("index_names:", getattr(df.index, "names", None))

    if not isinstance(df.index, pd.MultiIndex) or "instrument" not in df.index.names:
        print("[FAIL] index is not MultiIndex(datetime, instrument)")
        return

    ins_unique = pd.Index(df.index.get_level_values("instrument").astype(str)).unique()
    a = ins_unique.map(lambda x: bool(pat_aistock.match(x)))
    q = ins_unique.map(lambda x: bool(pat_qlib.match(x)))

    print("instrument_unique:", int(len(ins_unique)))
    print("sample_10:", ins_unique[:10].tolist())
    print("share_aistock_SHSZxxxxxx:", _mean_bool(a))
    print("share_qlib_xxxxxx.SH:", _mean_bool(q))


def check_dir(dir_path: Path, title: str) -> None:
    print("\n" + "=" * 90)
    print(f"[CHECK_DIR] {title}: {dir_path}")
    print("exists:", dir_path.exists())
    if not dir_path.exists():
        return

    for fn in files:
        p = dir_path / fn
        if not p.exists():
            print(f"\n[MISSING] {p}")
            continue
        try:
            summarize_h5(p, f"{title}/{fn}")
        except Exception as e:
            print(f"\n[ERROR] {title}/{fn}: {repr(e)}")


# 1) Check the provided export dir
check_dir(export_dir, "EXPORT_DIR")

# 2) Detect and check RD-Agent factor execution data folders
try:
    from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS

    repo_root = Path(".").resolve()
    data_folder = (repo_root / str(FACTOR_COSTEER_SETTINGS.data_folder)).resolve()
    data_folder_debug = (repo_root / str(FACTOR_COSTEER_SETTINGS.data_folder_debug)).resolve()

    print("\n" + "=" * 90)
    print("[RDAGENT_FACTOR_DATA_FOLDER]")
    print("FACTOR_COSTEER_SETTINGS.data_folder:", FACTOR_COSTEER_SETTINGS.data_folder)
    print("resolved:", data_folder)

    print("\n[RDAGENT_FACTOR_DATA_FOLDER_DEBUG]")
    print("FACTOR_COSTEER_SETTINGS.data_folder_debug:", FACTOR_COSTEER_SETTINGS.data_folder_debug)
    print("resolved:", data_folder_debug)

    print("\n[ENV overrides: FACTOR_CoSTEER_*]")
    for k in sorted(os.environ):
        if k.startswith("FACTOR_CoSTEER_"):
            print(k, "=", os.environ[k])

    check_dir(data_folder, "RDAGENT_data_folder")
    check_dir(data_folder_debug, "RDAGENT_data_folder_debug")

except Exception as e:
    print("\n" + "=" * 90)
    print("[WARN] Cannot import rdagent to resolve FACTOR_COSTEER_SETTINGS:", repr(e))
    print("[HINT] Run this script under the same env where you run RD-Agent (e.g. rdagent-gpu).")
PY
