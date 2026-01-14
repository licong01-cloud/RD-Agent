#!/usr/bin/env python
"""Check latest RD-Agent Qlib model loops for artifacts and registry consistency.

- 读取 registry.sqlite 中最近 N 条 `action='model' AND has_result=1` 的 loops；
- 检查 workspace 下的 meta/summary/manifest JSON 文件是否存在且可解析；
- 检查 artifacts / artifact_files 记录，并验证对应文件是否真实存在。

Usage (from repo root):

    python tools/check_registry_and_artifacts.py \
        --registry-sqlite git_ignore_folder/registry/registry.sqlite \
        --limit 3
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to parse JSON {path}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check latest RD-Agent Qlib experiments' artifacts and registry records.",
    )
    parser.add_argument(
        "--registry-sqlite",
        required=True,
        help="Path to registry.sqlite",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many latest model loops to check",
    )
    args = parser.parse_args()

    db_path = Path(args.registry_sqlite)
    if not db_path.exists():
        raise SystemExit(f"registry.sqlite not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 1. 找最近几条 action='model' AND has_result=1 的 loop
    # 注意：loops 表中只有 started_at_utc / ended_at_utc，没有 created_at_* 字段。
    sql = """
    SELECT
      l.task_run_id,
      l.loop_id,
      l.action,
      l.status,
      l.has_result,
      l.started_at_utc AS created_at,
      w.workspace_path,
      w.meta_path,
      w.summary_path,
      w.manifest_path
    FROM loops l
    JOIN workspaces w
      ON l.task_run_id = w.task_run_id
     AND l.loop_id = w.loop_id
    WHERE
      (l.has_result = 1 OR l.has_result = '1')
    ORDER BY l.started_at_utc DESC
    LIMIT ?;
    """
    rows = conn.execute(sql, (args.limit,)).fetchall()

    if not rows:
        print("[INFO] No model loops with has_result=1 found.")
        return

    print(f"[INFO] Found {len(rows)} latest model loops with has_result=1")

    for idx, row in enumerate(rows, 1):
        task_run_id = row["task_run_id"]
        loop_id = row["loop_id"]
        ws_path = Path(row["workspace_path"]).resolve()
        print("=" * 80)
        print(f"[Loop #{idx}] task_run_id={task_run_id}, loop_id={loop_id}, created_at={row['created_at']}")
        print(f"  workspace: {ws_path}")

        # 2. 检查 JSON 文件是否存在且可解析
        meta_rel = row["meta_path"]
        summary_rel = row["summary_path"]
        manifest_rel = row["manifest_path"]

        if meta_rel is None or summary_rel is None or manifest_rel is None:
            print("  [WARN] workspace meta/summary/manifest paths are NULL in DB, skipping JSON checks for this row.")
            continue

        meta_path = ws_path / meta_rel
        summary_path = ws_path / summary_rel
        manifest_path = ws_path / manifest_rel

        print(f"  meta:     {meta_path}  exists={meta_path.exists()}")
        print(f"  summary:  {summary_path}  exists={summary_path.exists()}")
        print(f"  manifest: {manifest_path}  exists={manifest_path.exists()}")

        meta_json = load_json_if_exists(meta_path)
        summary_json = load_json_if_exists(summary_path)
        manifest_json = load_json_if_exists(manifest_path)

        # 简单校验几个关键字段
        if meta_json is not None:
            print("  [OK] meta.json loaded, keys:", list(meta_json.keys())[:8])
        if summary_json is not None:
            print("  [OK] summary.json loaded, has_result:", summary_json.get("has_result"))
            files = summary_json.get("files", {})
            if isinstance(files, dict):
                print("      summary.files keys:", list(files.keys()))
        if manifest_json is not None:
            print("  [OK] manifest.json loaded, artifacts count:", len(manifest_json.get("artifacts", [])))

        # 3. 校验 artifacts / artifact_files 记录与实际文件的一致性
        # 3.1 查该 loop 下的 artifacts
        art_sql = """
        SELECT
          artifact_id,
          artifact_type,
          name,
          workspace_id,
          entry_path,
          status
        FROM artifacts
        WHERE task_run_id = ? AND loop_id = ?
        ORDER BY artifact_type, name;
        """
        art_rows = conn.execute(art_sql, (task_run_id, loop_id)).fetchall()
        print(f"  [INFO] artifacts in registry: {len(art_rows)}")
        for ar in art_rows:
            print(
                f"    - {ar['artifact_type']}: {ar['name']} "
                f"(status={ar['status']}, entry={ar['entry_path']})",
            )
            entry_abs = ws_path / ar["entry_path"]
            print(f"        entry_abs={entry_abs} exists={entry_abs.exists()}")

            # 查对应 artifact_files
            af_sql = """
            SELECT path, kind, size_bytes
            FROM artifact_files
            WHERE artifact_id = ?
            ORDER BY path;
            """
            af_rows = conn.execute(af_sql, (ar["artifact_id"],)).fetchall()
            print(f"        files in registry: {len(af_rows)}")
            for afr in af_rows:
                fp = ws_path / afr["path"]
                print(
                    f"          * {afr['path']} (kind={afr['kind']}, size={afr['size_bytes']}) "
                    f"exists={fp.exists()}",
                )

    conn.close()


if __name__ == "__main__":
    main()
