import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


_PHYSICAL_UNITS = {"元", "万元", "股", "万股"}


def _to_unix_path(p: Path) -> Path:
    # Keep consistent with other tools that accept Windows or WSL style paths.
    s = str(p)
    if len(s) >= 3 and s[1:3] == ":/":
        drive = s[0].lower()
        return Path("/mnt") / drive / s[3:]
    return p


def _try_get_columns_dtypes_from_h5(h5_path: Path, key: str = "data") -> list[dict[str, Any]]:
    if not h5_path.exists():
        return []

    cols: list[str] = []
    try:
        with pd.HDFStore(str(h5_path), mode="r") as store:
            h5_key = f"/{key}" if f"/{key}" in store.keys() else key
            if h5_key not in store:
                return []

            storer = store.get_storer(h5_key)
            axes = getattr(storer, "axes", None)
            if axes and len(axes) >= 1:
                try:
                    cols = [str(c) for c in list(axes[0])]
                except Exception:
                    cols = []

            if not cols:
                non_index_axes = getattr(storer, "non_index_axes", None)
                if non_index_axes:
                    try:
                        cols = [str(c) for c in list(non_index_axes[0][1])]
                    except Exception:
                        cols = []

    except Exception:
        cols = []

    if not cols:
        # Fallback to loading a small slice.
        try:
            df = pd.read_hdf(str(h5_path), key=key)
            cols = [str(c) for c in df.columns]
        except Exception:
            return []

    # Dtype best-effort: load a small number of rows.
    try:
        df_head = pd.read_hdf(str(h5_path), key=key, stop=1000)
        dtype_map = {str(c): str(df_head[c].dtype) for c in df_head.columns}
    except Exception:
        dtype_map = {}

    return [{"name": c, "dtype": dtype_map.get(c, "")} for c in cols]


def _load_field_map_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip().strip("\ufeff")
            if not name:
                continue
            out[name] = {
                "meaning_cn": (row.get("meaning_cn") or "").strip(),
                "unit": (row.get("unit") or "").strip(),
                "source_table": (row.get("source_table") or "").strip(),
                "comment": (row.get("comment") or "").strip(),
                "dtype_hint": (row.get("dtype_hint") or "").strip(),
            }
    return out


def _extract_physical_unit(text: str) -> str:
    # Only extract physical units. Percentages/ratios/PE should not be treated as units.
    if not text:
        return ""

    # Common patterns like "流通市值(万元)", "总股本(万股)", "特大单买入金额（元）"
    candidates = []

    # Chinese brackets
    for m in re.finditer(r"[\(（]([^\)）]{1,10})[\)）]", text):
        candidates.append(m.group(1).strip())

    for cand in candidates:
        if cand in _PHYSICAL_UNITS:
            return cand

    return ""


def _normalize_unit(unit: str, meaning_cn: str, comment: str) -> str:
    unit = (unit or "").strip()
    if unit in _PHYSICAL_UNITS:
        return unit

    # Try to infer physical unit from meaning/comment.
    inferred = _extract_physical_unit(meaning_cn) or _extract_physical_unit(comment)
    if inferred in _PHYSICAL_UNITS:
        return inferred

    return ""


def _enrich_columns_with_field_map(
    columns: list[dict[str, Any]],
    field_map: dict[str, dict[str, str]],
    dataset_id: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in columns:
        name = str(c.get("name", ""))
        dtype = str(c.get("dtype", ""))
        fm = field_map.get(name, {})

        meaning_cn = fm.get("meaning_cn", "")
        comment = fm.get("comment", "")

        unit = _normalize_unit(fm.get("unit", ""), meaning_cn, comment)

        # Keep source_table as-is if present; otherwise use dataset_id.
        source_table = fm.get("source_table", "") or dataset_id

        if not dtype:
            dtype = fm.get("dtype_hint", "")

        out.append(
            {
                "name": name,
                "dtype": dtype,
                "meaning_cn": meaning_cn,
                "unit": unit,
                "source_table": source_table,
                "comment": comment,
            }
        )

    return out


def _write_schema_outputs(schema: dict[str, Any], out_dirs: list[Path], stem: str) -> None:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        json_path = d / f"{stem}.json"
        csv_path = d / f"{stem}.csv"

        json_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(schema["columns"]).to_csv(csv_path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset schema CSV/JSON for daily_pv/daily_basic/moneyflow using snapshot H5 and snapshot metadata field_map."
    )
    parser.add_argument(
        "--snapshot-root",
        required=True,
        help="Snapshot root containing daily_pv.h5/daily_basic.h5/moneyflow.h5.",
    )
    parser.add_argument(
        "--field-map",
        default="",
        help="CSV like snapshot_root/metadata/aistock_field_map.csv.",
    )
    parser.add_argument(
        "--out-dir-governance",
        default="",
        help="Governance output root, schemas will be placed under <out>/schemas.",
    )
    parser.add_argument(
        "--out-dir-snapshot",
        default="",
        help="Snapshot output root for schemas, default: <snapshot_root>/metadata/schemas.",
    )
    parser.add_argument(
        "--h5-key",
        default="data",
        help="H5 key (default: data).",
    )

    args = parser.parse_args()

    snapshot_root = _to_unix_path(Path(args.snapshot_root)).resolve()

    field_map_path: Path
    if args.field_map:
        field_map_path = _to_unix_path(Path(args.field_map)).resolve()
    else:
        field_map_path = (snapshot_root / "metadata" / "aistock_field_map.csv").resolve()

    out_dirs: list[Path] = []

    # A) snapshot_root/metadata/schemas
    if args.out_dir_snapshot:
        out_dirs.append(_to_unix_path(Path(args.out_dir_snapshot)).resolve())
    else:
        out_dirs.append((snapshot_root / "metadata" / "schemas").resolve())

    # B) governance_out/schemas
    if args.out_dir_governance:
        out_dirs.append((_to_unix_path(Path(args.out_dir_governance)).resolve() / "schemas").resolve())

    field_map = _load_field_map_csv(field_map_path)

    datasets = [
        ("daily_pv", snapshot_root / "daily_pv.h5"),
        ("daily_basic", snapshot_root / "daily_basic.h5"),
        ("moneyflow", snapshot_root / "moneyflow.h5"),
    ]

    for dataset_id, h5_path in datasets:
        cols = _try_get_columns_dtypes_from_h5(h5_path, key=args.h5_key)
        cols = _enrich_columns_with_field_map(cols, field_map, dataset_id=dataset_id)

        schema: dict[str, Any] = {
            "dataset": dataset_id,
            "h5_key": args.h5_key,
            "path": str(h5_path),
            "index": {"type": "MultiIndex", "names": ["datetime", "instrument"]},
            "columns": cols,
            "field_map": str(field_map_path) if field_map_path.exists() else "",
            "unit_policy": "unit is optional and should only contain physical units (e.g., 元/万元/股/万股). Ratios/percentages/PE should have empty unit and be inferred from meaning_cn.",
        }

        _write_schema_outputs(schema, out_dirs, stem=f"{dataset_id}_schema")

    print("[SUCCESS] wrote schemas to:")
    for d in out_dirs:
        print(" -", d)


if __name__ == "__main__":
    main()
