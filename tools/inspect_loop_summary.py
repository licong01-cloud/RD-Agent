import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def inspect_loop(registry_sqlite: Path, task_run_id: str, loop_id: int) -> dict[str, Any]:
    conn = sqlite3.connect(str(registry_sqlite))
    conn.row_factory = sqlite3.Row
    try:
        loop_row = conn.execute(
            """
            SELECT
              l.task_run_id,
              l.loop_id,
              l.action,
              l.status,
              l.has_result,
              l.started_at_utc,
              l.ended_at_utc,
              w.workspace_id,
              w.workspace_path,
              tr.log_trace_path,
              a.model_type,
              a.model_conf_json,
              a.dataset_conf_json,
              a.feature_schema_json
            FROM loops l
            LEFT JOIN workspaces w
              ON l.task_run_id = w.task_run_id
             AND l.loop_id = w.loop_id
            LEFT JOIN task_runs tr
              ON l.task_run_id = tr.task_run_id
            LEFT JOIN artifacts a
              ON l.task_run_id = a.task_run_id
             AND l.loop_id = a.loop_id
             AND w.workspace_id = a.workspace_id
             AND a.artifact_type = 'report'
            WHERE l.task_run_id = ? AND l.loop_id = ?
            """,
            (task_run_id, loop_id),
        ).fetchone()
        if loop_row is None:
            raise SystemExit(f"loop not found: task_run_id={task_run_id}, loop_id={loop_id}")

        ws_path = Path(str(loop_row["workspace_path"])) if loop_row["workspace_path"] else None
        summary_json: dict[str, Any] | None = None
        model_files: list[str] = []
        mlruns_exists = False
        
        # 补充：从 registry 中提取模型元数据
        registry_model_meta = {
            "model_type": loop_row["model_type"],
            "model_conf": json.loads(loop_row["model_conf_json"]) if loop_row["model_conf_json"] else None,
            "dataset_conf": json.loads(loop_row["dataset_conf_json"]) if loop_row["dataset_conf_json"] else None,
            "feature_schema": json.loads(loop_row["feature_schema_json"]) if loop_row["feature_schema_json"] else None,
        }

        if ws_path and ws_path.exists():
            summary_path = ws_path / "experiment_summary.json"
            summary_json = _load_json_if_exists(summary_path) or {}
            # model hints in workspace root
            for name in ("model.pkl", "model.joblib", "model.bin", "model.onnx", "model.pt", "model.pth"):
                p = ws_path / name
                if p.exists():
                    model_files.append(name)
            if (ws_path / "mlruns").exists():
                mlruns_exists = True

        out: dict[str, Any] = {
            "task_run_id": str(loop_row["task_run_id"]),
            "loop_id": int(loop_row["loop_id"]),
            "action": str(loop_row["action"]) if loop_row["action"] is not None else None,
            "status": str(loop_row["status"]) if loop_row["status"] is not None else None,
            "has_result": bool(loop_row["has_result"] in (1, "1", True)),
            "started_at_utc": loop_row["started_at_utc"],
            "ended_at_utc": loop_row["ended_at_utc"],
            "workspace_id": str(loop_row["workspace_id"]) if loop_row["workspace_id"] is not None else None,
            "workspace_path": str(ws_path) if ws_path is not None else None,
            "log_dir": str(loop_row["log_trace_path"]) if loop_row["log_trace_path"] else None,
            "registry_model_meta": registry_model_meta,
            "workspace_summary": summary_json,
            "model_hints": {
                "mlruns_exists": mlruns_exists,
                "model_files": model_files,
            },
        }
        return out
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a single loop: workspace, log_dir, summary, and basic model hints.",
    )
    parser.add_argument("--registry-sqlite", required=True, help="Path to registry.sqlite (WSL/Linux path).")
    parser.add_argument("--task-run-id", required=True, help="Task run id of the loop.")
    parser.add_argument("--loop-id", required=True, type=int, help="Loop id.")
    args = parser.parse_args()

    registry_sqlite = Path(args.registry_sqlite)
    if not registry_sqlite.exists():
        raise SystemExit(f"registry.sqlite not found: {registry_sqlite}")

    info = inspect_loop(registry_sqlite, args.task_run_id, args.loop_id)
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
