import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _load_workspace_paths(registry_sqlite: Path) -> list[Path]:
    conn = sqlite3.connect(str(registry_sqlite))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT workspace_path
            FROM workspaces
            WHERE workspace_path IS NOT NULL AND workspace_path != ''
            """
        ).fetchall()
        return [Path(str(r["workspace_path"])) for r in rows]
    finally:
        conn.close()


def _extract_factors_from_parquet(parquet_path: Path) -> list[str]:
    import pandas as pd  # type: ignore

    df = pd.read_parquet(parquet_path)
    if df is None or df.empty:
        return []
    # 所有列名都作为因子名暴露
    return [str(c) for c in df.columns]


def _write_factor_meta(ws_root: Path, factors: list[str], overwrite: bool) -> None:
    meta_path = ws_root / "factor_meta.json"
    if meta_path.exists() and not overwrite:
        return

    payload: dict[str, Any] = {
        "version": "v1",
        "source": "backfill_scan",
        "workspace_path": str(ws_root),
        "factors": [
            {"name": name, "source": "rdagent_generated"} for name in sorted(set(factors))
        ],
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(registry_sqlite: Path, *, overwrite_json: bool = False) -> None:
    ws_paths = _load_workspace_paths(registry_sqlite)

    total_ws = 0
    ws_with_parquet = 0
    ws_written_meta = 0
    factor_counter: Counter[str] = Counter()

    for ws_root in ws_paths:
        total_ws += 1
        if not ws_root.exists():
            continue

        parquet_path = ws_root / "combined_factors_df.parquet"
        if not parquet_path.exists():
            continue

        ws_with_parquet += 1
        try:
            factor_names = _extract_factors_from_parquet(parquet_path)
        except Exception:
            continue

        if not factor_names:
            continue

        for name in factor_names:
            factor_counter[name] += 1

        _write_factor_meta(ws_root, factor_names, overwrite=overwrite_json)
        ws_written_meta += 1

    print(f"扫描 workspace 总数: {total_ws}")
    print(f"包含 combined_factors_df.parquet 的 workspace 数: {ws_with_parquet}")
    print(f"写入/覆盖 factor_meta.json 的 workspace 数: {ws_written_meta}")
    print(f"聚合后唯一因子个数: {len(factor_counter)}")

    # 打印前若干个因子及其覆盖的 workspace 数，便于快速 sanity check
    for name, cnt in factor_counter.most_common(20):
        print(f"因子: {name}  出现于 workspace 数: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan all workspaces and backfill factor_meta.json from combined_factors_df.parquet.")
    parser.add_argument(
        "--registry-sqlite",
        required=True,
        help="Path to registry.sqlite",
    )
    parser.add_argument(
        "--overwrite-json",
        action="store_true",
        help="Overwrite existing factor_meta.json if present.",
    )
    args = parser.parse_args()

    registry_sqlite = Path(args.registry_sqlite)
    if not registry_sqlite.exists():
        raise SystemExit(f"registry.sqlite not found: {registry_sqlite}")

    run(registry_sqlite=registry_sqlite, overwrite_json=bool(args.overwrite_json))


if __name__ == "__main__":
    main()
