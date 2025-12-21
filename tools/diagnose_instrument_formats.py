from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _to_unix_path(p: Path) -> Path:
    """Convert a Windows-like C:/ path into WSL /mnt/c path when running on Linux."""

    s = str(p).replace("\\", "/")
    if os.name != "nt" and len(s) >= 3 and s[1:3] == ":/":
        drive = s[0].lower()
        return Path("/mnt") / drive / s[3:]
    return Path(s)


def _load_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".h5":
        try:
            return pd.read_hdf(path, key="data")
        except Exception:
            return pd.read_hdf(path)
    if suf == ".pkl":
        return pd.read_pickle(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def _get_inst_index(obj: pd.DataFrame | pd.Series) -> Optional[pd.Index]:
    if isinstance(obj, pd.Series):
        idx = obj.index
    else:
        idx = obj.index

    if not isinstance(idx, pd.MultiIndex):
        return None
    if "instrument" not in idx.names:
        return None
    return pd.Index(idx.get_level_values("instrument"))


def _pattern_stats(inst: pd.Index) -> dict[str, int]:
    s = inst.astype(str)
    qlib = s.str.match(r"^\d{6}\.(SZ|SH)$")
    pref = s.str.match(r"^(SH|SZ)\d{6}$")
    other = ~(qlib | pref)
    return {
        "rows": int(len(s)),
        "unique_inst": int(s.nunique()),
        "qlib_cnt": int(qlib.sum()),
        "pref_cnt": int(pref.sum()),
        "other_cnt": int(other.sum()),
    }


def _print_one(path: Path, *, focus_col: str | None = None, topn: int = 10) -> None:
    if not path.exists():
        print(f"\n== {path} ==")
        print("[MISS]")
        return

    print(f"\n== {path} ==")
    try:
        df = _load_any(path)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return

    inst = _get_inst_index(df)
    if inst is None:
        print(f"index type={type(df.index)}; index_names={getattr(df.index, 'names', None)}")
        return

    st = _pattern_stats(inst)
    rows = st["rows"]
    q_ratio = st["qlib_cnt"] / rows if rows else 0.0
    p_ratio = st["pref_cnt"] / rows if rows else 0.0

    print(f"rows={rows} unique_inst={st['unique_inst']}")
    print(f"pattern qlib(000001.SZ) cnt={st['qlib_cnt']} ratio={q_ratio:.6f}")
    print(f"pattern pref(SH600000) cnt={st['pref_cnt']} ratio={p_ratio:.6f}")
    print(f"other cnt={st['other_cnt']}")
    print("sample:", inst.astype(str).drop_duplicates()[:topn].tolist())

    if focus_col and isinstance(df, pd.DataFrame) and focus_col in df.columns:
        s = df[focus_col]
        mask = s.notna()
        inst2 = pd.Index(s.index.get_level_values("instrument").astype(str)[mask.values])
        st2 = _pattern_stats(inst2)
        rows2 = st2["rows"]
        q_ratio2 = st2["qlib_cnt"] / rows2 if rows2 else 0.0
        p_ratio2 = st2["pref_cnt"] / rows2 if rows2 else 0.0
        print(f"-- non-NA of {focus_col} --")
        print(f"rows={rows2} unique_inst={st2['unique_inst']}")
        print(f"pattern qlib cnt={st2['qlib_cnt']} ratio={q_ratio2:.6f}")
        print(f"pattern pref cnt={st2['pref_cnt']} ratio={p_ratio2:.6f}")
        print(f"other cnt={st2['other_cnt']}")
        if rows2:
            print("sample:", inst2.drop_duplicates()[:topn].tolist())


def main() -> None:
    snapshot_root = _to_unix_path(Path(os.environ.get(
        "AISTOCK_SNAPSHOT_ROOT",
        r"F:/Dev/AIstock/qlib_snapshots/qlib_export_20251209",
    )))
    factors_root = _to_unix_path(Path(os.environ.get(
        "AISTOCK_FACTORS_ROOT",
        r"F:/Dev/AIstock/factors",
    )))
    repo_root = _to_unix_path(Path(os.environ.get(
        "RDAGENT_REPO_ROOT",
        r"F:/dev/RD-Agent-main",
    )))

    targets: list[Path] = [
        snapshot_root / "daily_pv.h5",
        snapshot_root / "daily_basic.h5",
        snapshot_root / "moneyflow.h5",
        factors_root / "daily_basic_factors" / "result.pkl",
        factors_root / "moneyflow_factors" / "result.pkl",
        factors_root / "combined_static_factors.parquet",
        repo_root / "git_ignore_folder" / "factor_implementation_source_data_debug" / "static_factors.parquet",
        repo_root / "git_ignore_folder" / "factor_implementation_source_data" / "static_factors.parquet",
    ]

    for p in targets:
        _print_one(p, focus_col="mf_main_net_amt_ratio")
        _print_one(p, focus_col="value_pb_inv")


if __name__ == "__main__":
    main()
