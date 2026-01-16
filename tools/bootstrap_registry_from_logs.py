import argparse
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.backfill_registry_artifacts import _collect_log_session_loops, _safe_int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_task_run(cur: sqlite3.Cursor, task_run_id: str, log_path: Path) -> None:
    row = cur.execute(
        "SELECT task_run_id FROM task_runs WHERE task_run_id=?",
        (task_run_id,),
    ).fetchone()
    if row is not None:
        # 尽量补齐 log_trace_path
        cur.execute(
            """
            UPDATE task_runs
            SET log_trace_path=COALESCE(log_trace_path, ?), updated_at_utc=?
            WHERE task_run_id=?
            """,
            (str(log_path), _utc_now_iso(), task_run_id),
        )
        return

    now = _utc_now_iso()
    cur.execute(
        """
        INSERT INTO task_runs
          (task_run_id, scenario, status, created_at_utc, updated_at_utc,
           git_sha, rdagent_version, log_trace_path, params_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_run_id,
            None,
            "unknown",
            now,
            now,
            None,
            None,
            str(log_path),
            None,
        ),
    )


def _ensure_loop(cur: sqlite3.Cursor, task_run_id: str, loop_id: int) -> None:
    row = cur.execute(
        "SELECT task_run_id, loop_id FROM loops WHERE task_run_id=? AND loop_id=?",
        (task_run_id, loop_id),
    ).fetchone()
    if row is not None:
        return

    now = _utc_now_iso()
    cur.execute(
        """
        INSERT INTO loops
          (task_run_id, loop_id, action, status, has_result,
           best_workspace_id, started_at_utc, ended_at_utc,
           error_type, error_message, ic_mean, rank_ic_mean,
           ann_return, mdd, turnover, multi_score, metrics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_run_id,
            loop_id,
            None,
            "unknown",
            0,
            None,
            now,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )


def _stable_workspace_id(task_run_id: str, loop_id: int, workspace_path: str) -> str:
    ns = uuid.UUID("5b9b5f74-0f3e-4e3c-8f86-6d8a9c7d0f11")
    key = f"{task_run_id}|{loop_id}|{workspace_path}"
    return uuid.uuid5(ns, key).hex


def _ensure_workspace(
    cur: sqlite3.Cursor,
    *,
    task_run_id: str,
    loop_id: int,
    workspace_path: str,
    workspace_role: str = "experiment_workspace",
) -> None:
    # 优先按 task_run_id + loop_id + workspace_path 查是否已存在
    row = cur.execute(
        """
        SELECT workspace_id
        FROM workspaces
        WHERE task_run_id=? AND loop_id=? AND workspace_path=?
        """,
        (task_run_id, loop_id, workspace_path),
    ).fetchone()
    if row is not None:
        return

    ws_root = Path(workspace_path)
    status = None
    try:
        if ws_root.exists():
            status = "finished"
        else:
            status = "missing_path"
    except Exception:
        status = None

    ws_id = _stable_workspace_id(task_run_id, loop_id, workspace_path)
    now = _utc_now_iso()
    cur.execute(
        """
        INSERT INTO workspaces
          (workspace_id, task_run_id, loop_id, workspace_role,
           experiment_type, step_name, status, workspace_path,
           meta_path, summary_path, manifest_path,
           created_at_utc, updated_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ws_id,
            task_run_id,
            loop_id,
            workspace_role,
            None,
            None,
            status,
            workspace_path,
            None,
            None,
            None,
            now,
            now,
        ),
    )


def run_one(db_path: Path, task_run_id: str, log_path: Path, dry_run: bool) -> dict[str, Any]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()

        loops_info = _collect_log_session_loops(log_path)
        results: list[dict[str, Any]] = []

        if not loops_info:
            return {
                "task_run_id": task_run_id,
                "log_path": str(log_path),
                "loops": [],
                "message": "no loops found in logs",
            }

        if not dry_run:
            _ensure_task_run(cur, task_run_id, log_path)

        for loop_id_raw, info in loops_info.items():
            loop_id = _safe_int(loop_id_raw, 0)
            workspaces = sorted(list(info.get("workspace_paths") or []))

            # 先确保 skeleton 存在
            if not dry_run:
                _ensure_loop(cur, task_run_id, loop_id)
                for ws_path in workspaces:
                    _ensure_workspace(
                        cur,
                        task_run_id=task_run_id,
                        loop_id=loop_id,
                        workspace_path=ws_path,
                    )

            # 基于 workspace 内文件推断 action / experiment_type / has_result
            inferred_action: str | None = None
            inferred_exp_type: str | None = None
            has_result = False

            for ws_path in workspaces:
                ws_root = Path(ws_path)
                try:
                    qlib_res = ws_root / "qlib_res.csv"
                    ret_pkl = ws_root / "ret.pkl"
                    combined = ws_root / "combined_factors_df.parquet"

                    if qlib_res.exists() and ret_pkl.exists():
                        inferred_action = "model"
                        inferred_exp_type = "model"
                        has_result = True
                        break
                    if combined.exists() and not has_result:
                        inferred_action = "factor"
                        inferred_exp_type = "factor"
                        has_result = True
                except Exception:
                    continue

            if not dry_run and has_result:
                # 更新 loops.has_result 和 action（仅在原值为空/0 时填充）
                try:
                    cur.execute(
                        """
                        UPDATE loops
                        SET
                          has_result = CASE WHEN has_result IS NULL OR has_result=0 OR has_result='0' THEN 1 ELSE has_result END,
                          action = COALESCE(action, ?)
                        WHERE task_run_id=? AND loop_id=?
                        """,
                        (inferred_action, task_run_id, loop_id),
                    )
                except Exception:
                    pass

                # 为该 loop 的 experiment_workspace 补 experiment_type
                if inferred_exp_type and workspaces:
                    try:
                        ws_example = workspaces[0]
                        cur.execute(
                            """
                            UPDATE workspaces
                            SET
                              experiment_type = COALESCE(experiment_type, ?),
                              status = COALESCE(status, 'finished'),
                              updated_at_utc = ?
                            WHERE task_run_id=? AND loop_id=? AND workspace_path=?
                            """,
                            (inferred_exp_type, _utc_now_iso(), task_run_id, loop_id, ws_example),
                        )
                    except Exception:
                        pass

            results.append(
                {
                    "loop_id": loop_id,
                    "workspace_paths": workspaces,
                    "inferred_action": inferred_action,
                    "has_result": has_result,
                }
            )

        if not dry_run:
            con.commit()

        return {
            "task_run_id": task_run_id,
            "log_path": str(log_path),
            "loops": results,
            "dry_run": dry_run,
        }
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap registry task_runs/loops/workspaces from log sessions.")
    ap.add_argument("--db", default="RDagentDB/registry.sqlite", help="Path to registry.sqlite")
    ap.add_argument(
        "--task-run-id",
        help="Task run id to associate with a single log session (used with --log-path). If omitted and --logs-root is provided, each subdirectory name under logs-root will be used as task_run_id.",
    )
    ap.add_argument(
        "--log-path",
        help="Single log session directory (same as task_runs.log_trace_path). Use with --task-run-id.",
    )
    ap.add_argument(
        "--logs-root",
        help="Root directory containing multiple log session subdirectories. Each direct child directory will be treated as one session.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only print planned changes without writing DB")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    import json as _json

    # Mode 1: logs-root 模式，遍历根目录下的所有子目录
    if args.logs_root:
        root = Path(args.logs_root)
        if not root.exists() or not root.is_dir():
            raise SystemExit(f"logs_root not found or not a directory: {root}")

        all_results: list[dict[str, Any]] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            task_run_id = child.name
            payload = run_one(
                db_path=db_path,
                task_run_id=task_run_id,
                log_path=child,
                dry_run=bool(args.dry_run),
            )
            all_results.append(payload)

        print(
            _json.dumps(
                {
                    "mode": "logs_root",
                    "db": str(db_path),
                    "logs_root": str(root),
                    "dry_run": bool(args.dry_run),
                    "sessions": all_results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    # Mode 2: 单一 log-path 模式（向后兼容）
    if not args.log_path or not args.task_run_id:
        raise SystemExit("Either --logs-root or both --log-path and --task-run-id must be provided")

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"log_path not found: {log_path}")

    payload = run_one(db_path=db_path, task_run_id=str(args.task_run_id), log_path=log_path, dry_run=bool(args.dry_run))
    print(_json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
