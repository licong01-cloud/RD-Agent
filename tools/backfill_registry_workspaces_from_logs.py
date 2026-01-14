import argparse
import json
import shutil
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

from tools.backfill_registry_artifacts import _collect_log_session_loops


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _CompatUnpickler(__import__("pickle").Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
            return Path
        if module == "pathlib" and name in {"PurePosixPath", "PureWindowsPath"}:
            from pathlib import PurePath

            return PurePath
        return super().find_class(module, name)


def _compat_pickle_load(p: Path) -> Any:
    with p.open("rb") as f:
        return _CompatUnpickler(f).load()


def _stable_workspace_id(task_run_id: str, loop_id: int, workspace_path: str) -> str:
    ns = uuid.UUID("5b9b5f74-0f3e-4e3c-8f86-6d8a9c7d0f11")
    key = f"{task_run_id}|{loop_id}|{workspace_path}"
    return uuid.uuid5(ns, key).hex


def _normalize_workspace_path(v: object) -> str:
    s = str(v)
    if not s:
        return ""
    first = s.splitlines()[0]
    return first.strip().replace("\r", "").replace("\n", "")


def _infer_task_run_id_from_session(log_path: Path) -> str | None:
    session_dir = log_path / "__session__"
    if not session_dir.exists() or not session_dir.is_dir():
        return None

    # Try load the earliest snapshot (smallest li/si)
    candidates: list[Path] = []
    for li_dir in session_dir.iterdir():
        if not li_dir.is_dir():
            continue
        if not li_dir.name.isdigit():
            continue
        for fp in li_dir.iterdir():
            if fp.is_file():
                candidates.append(fp)
    if not candidates:
        return None

    def _key(p: Path) -> tuple[int, int, str]:
        try:
            li = int(p.parent.name)
        except Exception:
            li = 10**9
        try:
            si = int(p.name.split("_")[0])
        except Exception:
            si = 10**9
        return (li, si, p.name)

    candidates.sort(key=_key)
    for fp in candidates[:20]:
        try:
            obj = _compat_pickle_load(fp)
        except Exception:
            continue

        for attr in ("task_run_id", "task_id", "run_id"):
            v = getattr(obj, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Best-effort: sometimes stored in nested dicts
        v = getattr(obj, "meta", None)
        if isinstance(v, dict):
            for k in ("task_run_id", "task_id", "run_id"):
                vv = v.get(k)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()

    return None


def _build_task_run_to_session_map(*, logs_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for sess in sorted(logs_root.iterdir()):
        if not sess.is_dir():
            continue
        tr_id = _infer_task_run_id_from_session(sess)
        if not tr_id:
            continue
        # Keep the first discovered session for that task_run_id; later duplicates are ignored.
        out.setdefault(tr_id, sess)
    return out


def _ensure_task_run(cur: sqlite3.Cursor, task_run_id: str, log_path: Path) -> None:
    row = cur.execute(
        "SELECT task_run_id FROM task_runs WHERE task_run_id=?",
        (task_run_id,),
    ).fetchone()

    now = _utc_now_iso()
    log_path_str = str(log_path.resolve())
    if row is None:
        cur.execute(
            """
            INSERT INTO task_runs
              (task_run_id, scenario, status, created_at_utc, updated_at_utc,
               git_sha, rdagent_version, log_trace_path, params_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_run_id, None, "unknown", now, now, None, None, log_path_str, None),
        )
    else:
        cur.execute(
            """
            UPDATE task_runs
            SET log_trace_path=COALESCE(log_trace_path, ?), updated_at_utc=?
            WHERE task_run_id=?
            """,
            (log_path_str, now, task_run_id),
        )


def _ensure_loop(cur: sqlite3.Cursor, task_run_id: str, loop_id: int, log_dir: str) -> None:
    row = cur.execute(
        "SELECT task_run_id, loop_id FROM loops WHERE task_run_id=? AND loop_id=?",
        (task_run_id, int(loop_id)),
    ).fetchone()

    now = _utc_now_iso()
    log_dir_str = str(Path(log_dir).resolve())
    if row is None:
        cur.execute(
            """
            INSERT INTO loops
              (task_run_id, loop_id, action, status, has_result,
               best_workspace_id, started_at_utc, ended_at_utc,
               error_type, error_message, ic_mean, rank_ic_mean,
               ann_return, mdd, turnover, multi_score, metrics_json,
               materialization_status, materialization_error, materialization_updated_at_utc,
               log_dir, asset_bundle_id, is_solidified, sync_status, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_run_id,
                int(loop_id),
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
                None,
                None,
                None,
                log_dir_str,
                None,
                None,
                None,
                now,
            ),
        )
    else:
        cur.execute(
            """
            UPDATE loops
            SET log_dir=COALESCE(NULLIF(log_dir,''), ?), updated_at_utc=?
            WHERE task_run_id=? AND loop_id=?
            """,
            (log_dir_str, now, task_run_id, int(loop_id)),
        )


def _upsert_workspace(
    cur: sqlite3.Cursor,
    *,
    task_run_id: str,
    loop_id: int,
    workspace_path: str,
    workspace_role: str = "experiment_workspace",
) -> str:
    row = cur.execute(
        """
        SELECT workspace_id
        FROM workspaces
        WHERE task_run_id=? AND loop_id=? AND workspace_path=?
        """,
        (task_run_id, int(loop_id), workspace_path),
    ).fetchone()
    if row is not None:
        return str(row[0])

    ws_id = _stable_workspace_id(task_run_id, loop_id, workspace_path)
    now = _utc_now_iso()
    status = "finished" if Path(workspace_path).exists() else "missing_path"

    cur.execute(
        """
        INSERT INTO workspaces
          (workspace_id, task_run_id, loop_id, workspace_role,
           experiment_type, step_name, status, workspace_path,
           meta_path, summary_path, manifest_path,
           created_at_utc, updated_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ws_id, task_run_id, int(loop_id), workspace_role, None, None, status, workspace_path, None, None, None, now, now),
    )
    return ws_id


def _upsert_workspace_and_count(
    cur: sqlite3.Cursor,
    *,
    task_run_id: str,
    loop_id: int,
    workspace_path: str,
    workspace_role: str = "experiment_workspace",
) -> tuple[str, bool]:
    row = cur.execute(
        """
        SELECT workspace_id
        FROM workspaces
        WHERE task_run_id=? AND loop_id=? AND workspace_path=?
        """,
        (task_run_id, int(loop_id), workspace_path),
    ).fetchone()
    if row is not None:
        return str(row[0]), False

    ws_id = _upsert_workspace(
        cur,
        task_run_id=task_run_id,
        loop_id=loop_id,
        workspace_path=workspace_path,
        workspace_role=workspace_role,
    )
    return ws_id, True


def _get_candidate_session_for_task_run(
    cur: sqlite3.Cursor,
    *,
    task_run_id: str,
    task_run_to_session: dict[str, Path],
) -> Path | None:
    # 1) Prefer task_runs.log_trace_path
    row = cur.execute(
        "SELECT log_trace_path FROM task_runs WHERE task_run_id=?",
        (task_run_id,),
    ).fetchone()
    if row is not None:
        v = row[0]
        if isinstance(v, str) and v.strip():
            p = Path(v.strip())
            if p.exists() and p.is_dir():
                return p

    # 2) Fallback: if any loop row already has log_dir
    row2 = cur.execute(
        "SELECT log_dir FROM loops WHERE task_run_id=? AND log_dir IS NOT NULL AND log_dir<>'' LIMIT 1",
        (task_run_id,),
    ).fetchone()
    if row2 is not None:
        v2 = row2[0]
        if isinstance(v2, str) and v2.strip():
            p2 = Path(v2.strip())
            if p2.exists() and p2.is_dir():
                return p2

    # 3) Final fallback: prebuilt scan index
    return task_run_to_session.get(task_run_id)


def run(
    *,
    db_path: Path,
    logs_root: Path,
    backup: bool,
    dry_run: bool,
    only_has_result: bool,
) -> dict[str, Any]:
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")
    if not logs_root.exists() or not logs_root.is_dir():
        raise SystemExit(f"logs_root not found or not a directory: {logs_root}")

    backup_path: str | None = None
    if backup and not dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = db_path.with_name(f"{db_path.name}.bak_{ts}")
        shutil.copy2(db_path, backup_file)
        backup_path = str(backup_file)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        task_run_to_session = _build_task_run_to_session_map(logs_root=logs_root)

        loops_updated = 0
        workspaces_inserted = 0
        sessions_indexed = len(task_run_to_session)

        where = ""
        params: list[Any] = []
        if only_has_result:
            where = "WHERE (has_result IS NOT NULL AND (has_result=1 OR has_result='1'))"

        loop_rows = cur.execute(
            f"SELECT task_run_id, loop_id, has_result FROM loops {where}",
            params,
        ).fetchall()

        task_runs_processed: set[str] = set()
        results: list[dict[str, Any]] = []

        for r in loop_rows:
            task_run_id = str(r[0])
            loop_id = int(r[1])

            if task_run_id not in task_runs_processed:
                task_runs_processed.add(task_run_id)

            sess = _get_candidate_session_for_task_run(
                cur,
                task_run_id=task_run_id,
                task_run_to_session=task_run_to_session,
            )
            if sess is None:
                results.append(
                    {
                        "task_run_id": task_run_id,
                        "loop_id": loop_id,
                        "skipped": True,
                        "reason": "no_log_session_found",
                    }
                )
                continue

            loops_info = _collect_log_session_loops(sess)
            info = loops_info.get(loop_id) or {}
            raw_workspace_paths = list(info.get("workspace_paths") or [])
            workspace_paths = sorted(
                [
                    _normalize_workspace_path(p)
                    for p in raw_workspace_paths
                    if _normalize_workspace_path(p)
                ]
            )

            if not workspace_paths:
                results.append(
                    {
                        "task_run_id": task_run_id,
                        "loop_id": loop_id,
                        "session": str(sess),
                        "workspace_count": 0,
                    }
                )
                continue

            if not dry_run:
                _ensure_task_run(cur, task_run_id, sess)
                _ensure_loop(cur, task_run_id, loop_id, str(sess))

            inserted_any = False
            ws_ids: list[str] = []
            for ws_path in workspace_paths:
                ws_path = _normalize_workspace_path(ws_path)
                if not ws_path:
                    continue
                if dry_run:
                    continue
                ws_id, inserted = _upsert_workspace_and_count(
                    cur,
                    task_run_id=task_run_id,
                    loop_id=loop_id,
                    workspace_path=ws_path,
                )
                ws_ids.append(ws_id)
                if inserted:
                    workspaces_inserted += 1
                    inserted_any = True

            if inserted_any:
                loops_updated += 1

            if ws_ids and (not dry_run):
                cur.execute(
                    """
                    UPDATE loops
                    SET best_workspace_id=COALESCE(best_workspace_id, ?), updated_at_utc=?
                    WHERE task_run_id=? AND loop_id=?
                    """,
                    (ws_ids[0], _utc_now_iso(), task_run_id, int(loop_id)),
                )

            results.append(
                {
                    "task_run_id": task_run_id,
                    "loop_id": loop_id,
                    "session": str(sess),
                    "workspace_count": len(workspace_paths),
                    "workspace_paths": workspace_paths,
                }
            )

        if not dry_run:
            con.commit()

        return {
            "db": str(db_path),
            "backup": backup_path,
            "logs_root": str(logs_root),
            "dry_run": bool(dry_run),
            "only_has_result": bool(only_has_result),
            "sessions_indexed": sessions_indexed,
            "loops_scanned": len(loop_rows),
            "loops_updated": loops_updated,
            "workspaces_inserted": workspaces_inserted,
            "results": results,
        }
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill registry.sqlite workspaces from RD-Agent log sessions (authoritative source).")
    ap.add_argument("--db", default="RDagentDB/registry.sqlite", help="Path to registry.sqlite")
    ap.add_argument("--logs-root", default="log", help="Root directory containing RD-Agent log session subdirectories")
    ap.add_argument("--backup", action="store_true", help="Backup registry.sqlite before writing")
    ap.add_argument("--dry-run", action="store_true", help="Only scan and report; do not write DB")
    ap.add_argument(
        "--only-has-result",
        action="store_true",
        help="Only backfill loops where loops.has_result=1 (recommended for '有结论' loops)",
    )
    args = ap.parse_args()

    payload = run(
        db_path=Path(args.db),
        logs_root=Path(args.logs_root),
        backup=bool(args.backup),
        dry_run=bool(args.dry_run),
        only_has_result=bool(args.only_has_result),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
