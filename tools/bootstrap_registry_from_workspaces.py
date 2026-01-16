import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class ExistingWorkspace:
    workspace_path: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_existing_workspace_paths(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT workspace_path
        FROM workspaces
        WHERE workspace_path IS NOT NULL AND workspace_path != ''
        """
    ).fetchall()
    return {str(r["workspace_path"]) for r in rows}


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _detect_results(ws_root: Path) -> dict[str, Any]:
    """检测 workspace 是否有成果文件（因子/模型/策略）。"""

    qlib_res = ws_root / "qlib_res.csv"
    ret_pkl = ws_root / "ret.pkl"
    combined_factors = ws_root / "combined_factors_df.parquet"
    mlruns = ws_root / "mlruns"
    signals_parquet = ws_root / "signals.parquet"
    signals_json = ws_root / "signals.json"

    has_model = qlib_res.exists() and ret_pkl.exists()
    has_factor = combined_factors.exists()
    has_strategy = signals_parquet.exists() or signals_json.exists()

    has_any = has_model or has_factor or has_strategy or mlruns.exists()

    # 粗略推断实验类型
    experiment_type: str | None = None
    if has_model:
        experiment_type = "model"
    elif has_factor:
        experiment_type = "factor"

    return {
        "has_any": has_any,
        "has_model": has_model,
        "has_factor": has_factor,
        "has_strategy": has_strategy,
        "experiment_type": experiment_type,
    }


def _infer_ids_from_meta(ws_root: Path) -> tuple[str, int, str]:
    """尽量从 workspace_meta.json / experiment_summary.json 恢复 task_run_id / loop_id / workspace_id.

    若无法恢复，则用目录名构造 synthetic ID。
    """

    meta = _load_json_if_exists(ws_root / "workspace_meta.json")
    if not meta:
        meta = _load_json_if_exists(ws_root / "experiment_summary.json") or {}

    task_run_id = str(meta.get("task_run_id") or meta.get("task_run", {}).get("task_run_id") or "")
    loop_id = meta.get("loop_id") or meta.get("loop", {}).get("loop_id")
    workspace_id = str(meta.get("workspace_id") or ws_root.name)

    if not task_run_id:
        # 使用 synthetic 前缀 + 目录名，避免与真实 ID 冲突
        task_run_id = f"synthetic_{ws_root.name}"
    try:
        loop_id_int = int(loop_id) if loop_id is not None else 0
    except Exception:
        loop_id_int = 0

    return task_run_id, loop_id_int, workspace_id


def _infer_log_trace_path_from_meta(ws_root: Path) -> str | None:
    meta = _load_json_if_exists(ws_root / "workspace_meta.json") or {}
    tr = meta.get("task_run") or {}
    val = tr.get("log_trace_path") or tr.get("log_path")
    if isinstance(val, str) and val:
        return val
    return None


def _ensure_task_run(
    cur: sqlite3.Cursor,
    task_run_id: str,
    log_trace_path: str | None,
) -> None:
    """确保 task_runs 中存在对应记录，并尽量补齐 log_trace_path。

    注意：表结构需与 bootstrap_registry_from_logs.py 中的 _ensure_task_run 保持一致，
    仅使用 task_runs 已有的列（task_run_id, scenario, status, created_at_utc,
    updated_at_utc, git_sha, rdagent_version, log_trace_path, params_json），
    避免引用不存在的 started_at_utc 等字段。
    """

    row = cur.execute(
        "SELECT task_run_id FROM task_runs WHERE task_run_id=?",
        (task_run_id,),
    ).fetchone()

    now = _utc_now_iso()

    if row is not None:
        # 若已存在，仅在 log_trace_path 为空时补齐，并更新 updated_at_utc
        if log_trace_path:
            try:
                cur.execute(
                    """
                    UPDATE task_runs
                    SET log_trace_path=COALESCE(log_trace_path, ?), updated_at_utc=?
                    WHERE task_run_id=?
                    """,
                    (str(log_trace_path), now, task_run_id),
                )
            except Exception:
                pass
        return

    # 插入一条最小记录，其它字段交由后续 backfill/在线逻辑补齐
    try:
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
                str(log_trace_path) if log_trace_path else None,
                None,
            ),
        )
    except Exception:
        # 若插入失败，不影响后续 loop/workspace 补登记
        pass


def _ensure_loop(
    cur: sqlite3.Cursor,
    task_run_id: str,
    loop_id: int,
    experiment_type: str | None,
    has_result: bool,
) -> None:
    """确保 loops 中存在对应记录，列结构与 bootstrap_registry_from_logs 保持兼容."""

    row = cur.execute(
        "SELECT task_run_id, loop_id FROM loops WHERE task_run_id=? AND loop_id=?",
        (task_run_id, int(loop_id)),
    ).fetchone()
    if row is not None:
        # 已存在则只考虑后续由 backfill/update 来补充 has_result / action
        return

    now = _utc_now_iso()
    action = experiment_type or None

    try:
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
                int(loop_id),
                action,
                "unknown",
                1 if has_result else 0,
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
    except Exception:
        # 插入失败不应中断整个补登记流程
        pass


def _ensure_workspace(
    cur: sqlite3.Cursor,
    workspace_id: str,
    task_run_id: str,
    loop_id: int,
    ws_root: Path,
    experiment_type: str | None,
    has_result: bool,
) -> None:
    status = "finished" if has_result else "unknown"
    cur.execute(
        """
        INSERT OR IGNORE INTO workspaces (
            workspace_id, task_run_id, loop_id, workspace_role,
            experiment_type, workspace_path, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            workspace_id,
            task_run_id,
            int(loop_id),
            "experiment_workspace",
            experiment_type,
            str(ws_root),
            status,
        ),
    )


def run(db_path: Path, workspace_root: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        existing_paths = _load_existing_workspace_paths(conn)

        total_dirs = 0
        considered = 0
        registered = 0

        for entry in workspace_root.iterdir():
            if not entry.is_dir():
                continue
            total_dirs += 1

            ws_root = entry
            ws_path_str = str(ws_root)
            if ws_path_str in existing_paths:
                continue

            info = _detect_results(ws_root)
            if not info["has_any"]:
                continue

            considered += 1

            task_run_id, loop_id, workspace_id = _infer_ids_from_meta(ws_root)
            log_path = _infer_log_trace_path_from_meta(ws_root)

            _ensure_task_run(cur, task_run_id=task_run_id, log_trace_path=log_path)
            _ensure_loop(
                cur,
                task_run_id=task_run_id,
                loop_id=loop_id,
                experiment_type=info["experiment_type"],
                has_result=bool(info["has_any"]),
            )
            _ensure_workspace(
                cur,
                workspace_id=workspace_id,
                task_run_id=task_run_id,
                loop_id=loop_id,
                ws_root=ws_root,
                experiment_type=info["experiment_type"],
                has_result=bool(info["has_any"]),
            )

            registered += 1

        conn.commit()
    finally:
        conn.close()

    print(f"扫描物理 workspace 目录总数: {total_dirs}")
    print(f"其中包含成果文件且尚未入库的 workspace 数: {considered}")
    print(f"成功写入 registry 的新 workspace 数: {registered}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从物理 workspace 目录补登记尚未入库且有成果的 workspace 到 registry.sqlite，"
            "不修改已有记录。"
        )
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to registry.sqlite",
    )
    parser.add_argument(
        "--workspace-root",
        required=True,
        help="Workspace 根目录，例如 /mnt/f/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"registry.sqlite not found: {db_path}")

    workspace_root = Path(args.workspace_root)
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise SystemExit(f"workspace 根目录不存在或不是目录: {workspace_root}")

    run(db_path=db_path, workspace_root=workspace_root)


if __name__ == "__main__":
    main()
