import argparse
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class WorkspaceRow:
    workspace_id: str
    task_run_id: str
    loop_id: int
    workspace_role: str
    experiment_type: str
    step_name: str | None
    status: str | None
    workspace_path: str
    meta_path: str | None
    summary_path: str | None
    manifest_path: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_ARTIFACT_NS = uuid.UUID("3e36bb8a-2a9b-4c56-8f50-6312f0d89c42")


def _stable_artifact_id(*, task_run_id: str, workspace_id: str, artifact_type: str, name: str) -> str:
    key = f"{task_run_id}|{workspace_id}|{artifact_type}|{name}"
    return uuid.uuid5(_ARTIFACT_NS, key).hex


def _stable_file_id(*, artifact_id: str, workspace_id: str, path_rel: str, kind: str) -> str:
    key = f"{artifact_id}|{workspace_id}|{kind}|{path_rel}"
    return uuid.uuid5(_ARTIFACT_NS, key).hex


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _rel(ws_root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(ws_root))
    except Exception:
        return str(p)


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _best_effort_file_meta(p: Path) -> tuple[int | None, str | None]:
    try:
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        return int(st.st_size), mtime
    except Exception:
        return None, None


def _iter_yaml_confs(ws_root: Path) -> list[Path]:
    out = []
    try:
        out.extend(list(ws_root.glob("*.yaml")))
        out.extend(list(ws_root.glob("*.yml")))
    except Exception:
        return []
    return [p for p in out if p.exists()]


def _read_loop_action(cur: sqlite3.Cursor, task_run_id: str, loop_id: int) -> str | None:
    try:
        row = cur.execute(
            "select action from loops where task_run_id=? and loop_id=?",
            (task_run_id, loop_id),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _update_workspace_paths(
    cur: sqlite3.Cursor,
    ws: WorkspaceRow,
    meta_rel: str,
    summary_rel: str,
    manifest_rel: str,
) -> None:
    cur.execute(
        """
        update workspaces
        set meta_path=?, summary_path=?, manifest_path=?, updated_at_utc=?
        where workspace_id=?
        """,
        (meta_rel, summary_rel, manifest_rel, _utc_now_iso(), ws.workspace_id),
    )


def _insert_artifact(
    cur: sqlite3.Cursor,
    *,
    artifact_id: str,
    ws: WorkspaceRow,
    artifact_type: str,
    name: str,
    status: str,
    primary_flag: bool,
    summary: dict[str, Any],
    entry_path: str,
) -> None:
    cur.execute(
        """
        insert or replace into artifacts
        (artifact_id, task_run_id, loop_id, workspace_id, artifact_type, name, version, status, primary_flag, summary_json, entry_path, created_at_utc, updated_at_utc)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact_id,
            ws.task_run_id,
            ws.loop_id,
            ws.workspace_id,
            artifact_type,
            name,
            "v1",
            status,
            1 if primary_flag else 0,
            json.dumps(summary, ensure_ascii=False, default=str),
            entry_path,
            _utc_now_iso(),
            _utc_now_iso(),
        ),
    )


def _insert_artifact_file(
    cur: sqlite3.Cursor,
    *,
    artifact_id: str,
    ws: WorkspaceRow,
    path_rel: str,
    kind: str,
    size_bytes: int | None,
    mtime_utc: str | None,
) -> None:
    cur.execute(
        """
        insert or replace into artifact_files
        (file_id, artifact_id, workspace_id, path, sha256, size_bytes, mtime_utc, kind)
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _stable_file_id(
                artifact_id=artifact_id,
                workspace_id=ws.workspace_id,
                path_rel=path_rel,
                kind=kind,
            ),
            artifact_id,
            ws.workspace_id,
            path_rel,
            None,
            size_bytes,
            mtime_utc,
            kind,
        ),
    )


def _cleanup_existing_for_workspace(
    cur: sqlite3.Cursor,
    ws: WorkspaceRow,
    *,
    artifact_types: Iterable[str],
) -> None:
    # Remove previously backfilled artifacts for this workspace to avoid duplicates from older runs.
    # We restrict by workspace_id + task_run_id + artifact_type.
    types = list(artifact_types)
    if not types:
        return
    q_marks = ",".join(["?"] * len(types))
    artifact_ids = [
        r[0]
        for r in cur.execute(
            f"select artifact_id from artifacts where task_run_id=? and workspace_id=? and artifact_type in ({q_marks})",
            (ws.task_run_id, ws.workspace_id, *types),
        ).fetchall()
    ]
    if not artifact_ids:
        return
    q_marks2 = ",".join(["?"] * len(artifact_ids))
    cur.execute(
        f"delete from artifact_files where artifact_id in ({q_marks2})",
        artifact_ids,
    )
    cur.execute(
        f"delete from artifacts where artifact_id in ({q_marks2})",
        artifact_ids,
    )


def _maybe_backfill_one_workspace(
    cur: sqlite3.Cursor,
    ws: WorkspaceRow,
    *,
    overwrite_json: bool,
    dry_run: bool,
    cleanup_existing: bool,
) -> dict[str, Any]:
    ws_root = Path(ws.workspace_path)
    if not ws_root.exists():
        return {"workspace_id": ws.workspace_id, "skipped": True, "reason": "workspace_path_not_found"}

    meta_path = ws_root / "workspace_meta.json"
    summary_path = ws_root / "experiment_summary.json"
    manifest_path = ws_root / "manifest.json"

    # Detect files
    qlib_res = ws_root / "qlib_res.csv"
    ret_pkl = ws_root / "ret.pkl"
    ret_schema_parquet = ws_root / "ret_schema.parquet"
    ret_schema_json = ws_root / "ret_schema.json"
    signals_parquet = ws_root / "signals.parquet"
    signals_json = ws_root / "signals.json"
    mlruns = ws_root / "mlruns"
    combined_factors = ws_root / "combined_factors_df.parquet"
    yaml_confs = _iter_yaml_confs(ws_root)

    action = ws.experiment_type or _read_loop_action(cur, ws.task_run_id, ws.loop_id) or "unknown"

    if action == "model":
        has_result = bool(qlib_res.exists() and ret_pkl.exists())
        required_files = ["qlib_res.csv", "ret.pkl"]
    elif action == "factor":
        has_result = bool(combined_factors.exists())
        required_files = ["combined_factors_df.parquet"]
    else:
        has_result = False
        required_files = []

    now_utc = _utc_now_iso()

    task_run_snapshot = {"task_run_id": ws.task_run_id}
    loop_snapshot = {"task_run_id": ws.task_run_id, "loop_id": ws.loop_id, "action": action}

    pointers = {
        "meta_path": _rel(ws_root, meta_path),
        "summary_path": _rel(ws_root, summary_path),
        "manifest_path": _rel(ws_root, manifest_path),
    }

    result_criteria = {"required_files": required_files, "has_result": has_result}

    meta_payload = {
        "task_run_id": ws.task_run_id,
        "loop_id": ws.loop_id,
        "action": action,
        "workspace_id": ws.workspace_id,
        "workspace_role": ws.workspace_role,
        "experiment_type": ws.experiment_type,
        "workspace_path": str(ws_root),
        "status": ws.status,
        "has_result": has_result,
        "generated_at_utc": now_utc,
        "result_criteria": result_criteria,
        "task_run": task_run_snapshot,
        "loop": loop_snapshot,
        "pointers": pointers,
    }

    summary_payload: dict[str, Any] = {
        "task_run_id": ws.task_run_id,
        "loop_id": ws.loop_id,
        "action": action,
        "workspace_id": ws.workspace_id,
        "workspace_role": ws.workspace_role,
        "experiment_type": ws.experiment_type,
        "workspace_path": str(ws_root),
        "status": ws.status,
        "has_result": has_result,
        "generated_at_utc": now_utc,
        "result_criteria": result_criteria,
        "task_run": task_run_snapshot,
        "loop": loop_snapshot,
        "pointers": pointers,
        "files": {
            "qlib_res.csv": _rel(ws_root, qlib_res) if qlib_res.exists() else None,
            "ret.pkl": _rel(ws_root, ret_pkl) if ret_pkl.exists() else None,
            "ret_schema.parquet": _rel(ws_root, ret_schema_parquet) if ret_schema_parquet.exists() else None,
            "ret_schema.json": _rel(ws_root, ret_schema_json) if ret_schema_json.exists() else None,
            "signals.parquet": _rel(ws_root, signals_parquet) if signals_parquet.exists() else None,
            "signals.json": _rel(ws_root, signals_json) if signals_json.exists() else None,
            "combined_factors_df.parquet": _rel(ws_root, combined_factors) if combined_factors.exists() else None,
            "mlruns": _rel(ws_root, mlruns) if mlruns.exists() else None,
        },
        "artifacts": [],
    }

    manifest_payload: dict[str, Any] = {
        "task_run_id": ws.task_run_id,
        "loop_id": ws.loop_id,
        "action": action,
        "workspace_id": ws.workspace_id,
        "workspace_role": ws.workspace_role,
        "experiment_type": ws.experiment_type,
        "workspace_path": str(ws_root),
        "status": ws.status,
        "has_result": has_result,
        "generated_at_utc": now_utc,
        "result_criteria": result_criteria,
        "task_run": task_run_snapshot,
        "loop": loop_snapshot,
        "pointers": pointers,
        "artifacts": [],
    }

    if not dry_run:
        _write_json(meta_path, meta_payload, overwrite=overwrite_json)
        _write_json(summary_path, summary_payload, overwrite=overwrite_json)
        _write_json(manifest_path, manifest_payload, overwrite=overwrite_json)

        _update_workspace_paths(
            cur,
            ws,
            meta_rel=_rel(ws_root, meta_path),
            summary_rel=_rel(ws_root, summary_path),
            manifest_rel=_rel(ws_root, manifest_path),
        )

    # Backfill artifacts
    artifacts_created: list[str] = []

    if ws.workspace_role == "experiment_workspace":
        if cleanup_existing and (not dry_run):
            if action == "model":
                _cleanup_existing_for_workspace(
                    cur,
                    ws,
                    artifact_types=["report", "model", "config_snapshot"],
                )
            elif action == "factor":
                _cleanup_existing_for_workspace(
                    cur,
                    ws,
                    artifact_types=["factor_data"],
                )
        if action == "model":
            report_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="report",
                name="qlib_report",
            )
            report_status = "present" if (qlib_res.exists() and ret_pkl.exists()) else "missing"
            report_files: list[Path] = [
                qlib_res,
                ret_pkl,
                ret_schema_parquet,
                ret_schema_json,
                signals_parquet,
                signals_json,
            ]
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=report_artifact_id,
                    ws=ws,
                    artifact_type="report",
                    name="qlib_report",
                    status=report_status,
                    primary_flag=True,
                    summary={"files": [p.name for p in report_files]},
                    entry_path=_rel(ws_root, qlib_res) if qlib_res.exists() else ".",
                )

                for p in report_files:
                    if not p.exists():
                        continue
                    size_bytes, mtime_utc = _best_effort_file_meta(p)
                    _insert_artifact_file(
                        cur,
                        artifact_id=report_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, p),
                        kind="report",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(report_artifact_id)

            mlruns_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="model",
                name="mlruns",
            )
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=mlruns_artifact_id,
                    ws=ws,
                    artifact_type="model",
                    name="mlruns",
                    status="present" if mlruns.exists() else "missing",
                    primary_flag=False,
                    summary={"path": "mlruns"},
                    entry_path=_rel(ws_root, mlruns) if mlruns.exists() else ".",
                )
                if mlruns.exists():
                    size_bytes, mtime_utc = _best_effort_file_meta(mlruns)
                    _insert_artifact_file(
                        cur,
                        artifact_id=mlruns_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, mlruns),
                        kind="model",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(mlruns_artifact_id)

            cfg_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="config_snapshot",
                name="workspace_configs",
            )
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=cfg_artifact_id,
                    ws=ws,
                    artifact_type="config_snapshot",
                    name="workspace_configs",
                    status="present" if len(yaml_confs) > 0 else "missing",
                    primary_flag=False,
                    summary={"count": len(yaml_confs)},
                    entry_path=".",
                )
                for p in yaml_confs:
                    size_bytes, mtime_utc = _best_effort_file_meta(p)
                    _insert_artifact_file(
                        cur,
                        artifact_id=cfg_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, p),
                        kind="config_snapshot",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(cfg_artifact_id)

        elif action == "factor":
            factor_artifact_id = _stable_artifact_id(
                task_run_id=ws.task_run_id,
                workspace_id=ws.workspace_id,
                artifact_type="factor_data",
                name="combined_factors",
            )
            if not dry_run:
                _insert_artifact(
                    cur,
                    artifact_id=factor_artifact_id,
                    ws=ws,
                    artifact_type="factor_data",
                    name="combined_factors",
                    status="present" if combined_factors.exists() else "missing",
                    primary_flag=True,
                    summary={"files": ["combined_factors_df.parquet"]},
                    entry_path=_rel(ws_root, combined_factors) if combined_factors.exists() else ".",
                )
                if combined_factors.exists():
                    size_bytes, mtime_utc = _best_effort_file_meta(combined_factors)
                    _insert_artifact_file(
                        cur,
                        artifact_id=factor_artifact_id,
                        ws=ws,
                        path_rel=_rel(ws_root, combined_factors),
                        kind="factor_data",
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                    )
            artifacts_created.append(factor_artifact_id)

    return {
        "workspace_id": ws.workspace_id,
        "loop_id": ws.loop_id,
        "action": action,
        "workspace_role": ws.workspace_role,
        "has_result": has_result,
        "artifacts_created": artifacts_created,
        "wrote_json": (not dry_run),
    }


def _load_workspaces(cur: sqlite3.Cursor, task_run_id: str) -> list[WorkspaceRow]:
    rows = cur.execute(
        """
        select
          workspace_id, task_run_id, loop_id, workspace_role, experiment_type, step_name, status,
          workspace_path, meta_path, summary_path, manifest_path
        from workspaces
        where task_run_id=?
        order by loop_id asc, case when workspace_role='experiment_workspace' then 0 else 1 end, workspace_id asc
        """,
        (task_run_id,),
    ).fetchall()

    out: list[WorkspaceRow] = []
    for r in rows:
        out.append(
            WorkspaceRow(
                workspace_id=r[0],
                task_run_id=r[1],
                loop_id=_safe_int(r[2]),
                workspace_role=r[3],
                experiment_type=r[4],
                step_name=r[5],
                status=r[6],
                workspace_path=r[7],
                meta_path=r[8],
                summary_path=r[9],
                manifest_path=r[10],
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="RDagentDB/registry.sqlite")
    ap.add_argument("--task-run-id", required=True)
    ap.add_argument("--overwrite-json", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-experiment-workspace", action="store_true")
    ap.add_argument(
        "--cleanup-existing",
        action="store_true",
        help="Delete existing artifacts/artifact_files for the selected workspaces before backfilling (useful to remove duplicates from previous runs).",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        wss = _load_workspaces(cur, args.task_run_id)
        if args.only_experiment_workspace:
            wss = [w for w in wss if w.workspace_role == "experiment_workspace"]

        print(f"db={db_path}")
        print(f"task_run_id={args.task_run_id}")
        print(f"workspaces={len(wss)}")
        print(f"dry_run={bool(args.dry_run)} overwrite_json={bool(args.overwrite_json)}")

        results: list[dict[str, Any]] = []
        for ws in wss:
            res = _maybe_backfill_one_workspace(
                cur,
                ws,
                overwrite_json=bool(args.overwrite_json),
                dry_run=bool(args.dry_run),
                cleanup_existing=bool(args.cleanup_existing),
            )
            results.append(res)

        if not args.dry_run:
            con.commit()

        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    finally:
        con.close()


if __name__ == "__main__":
    main()
