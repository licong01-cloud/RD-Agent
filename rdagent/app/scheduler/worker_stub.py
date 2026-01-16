"""
Placeholder worker for RD-Agent task execution.

Intended to be replaced with a real queue/worker (Celery/RQ/etc.).
"""

from __future__ import annotations

import subprocess
import csv
import pickle
from glob import glob
from pathlib import Path
from typing import Optional, Dict, Any

from .task_service import (
    append_task_log,
    update_task_status,
    get_task,
    record_result,
    LOG_DIR,
)
from .config_service import PROJECT_ROOT


def _collect_result_files(workdir: str) -> list:
    """
    Heuristic collection of key result files.
    Checks common names in workdir and subdirs.
    """
    candidates = ["qlib_res.csv", "ret.pkl", "result.h5"]
    found = []
    for fname in candidates:
        for p in glob(str(Path(workdir) / "**" / fname), recursive=True):
            found.append(p)
    return found


def _summarize_results(result_files: list) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"files": []}
    for p in result_files:
        path = Path(p)
        # Basic metadata
        info = {
            "name": path.name,
            "path": str(path),
            "size": path.stat().st_size if path.exists() else None,
            "mtime": path.stat().st_mtime if path.exists() else None,
        }
        summary["files"].append(info)

        # qlib_res.csv: take first row as metrics
        if path.name == "qlib_res.csv":
            try:
                with path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    first_row = next(reader, None)
                    if first_row:
                        summary.setdefault("qlib_res", first_row)
            except Exception as e:  # pragma: no cover - best effort
                summary.setdefault("qlib_res_error", str(e))

        # ret.pkl: capture keys or length for quick glance
        if path.name == "ret.pkl":
            try:
                with path.open("rb") as f:
                    obj = pickle.load(f)
                    if hasattr(obj, "keys"):
                        summary.setdefault("ret_pkl_keys", list(obj.keys()))
                    elif isinstance(obj, (list, tuple)):
                        summary.setdefault("ret_pkl_len", len(obj))
                    else:
                        summary.setdefault("ret_pkl_type", str(type(obj)))
            except Exception as e:  # pragma: no cover
                summary.setdefault("ret_pkl_error", str(e))
    return summary


def run_rdagent_task(task_id: str, workspace: Optional[str] = None) -> int:
    """
    Stub: run rdagent fin_quant for a given task id.
    In production, replace with queue-managed execution and richer logging.
    """
    task = get_task(task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")

    workdir = workspace or task.workspace_path or str(PROJECT_ROOT)
    cmd = [
        "rdagent",
        "fin_quant",
        "--loop-n",
        str(task.loop_n),
        "--all-duration",
        str(task.all_duration),
    ]

    update_task_status(task_id, "running")
    append_task_log(task_id, f"Starting task {task_id}: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.stdout:
            for line in proc.stdout:
                append_task_log(task_id, line.rstrip("\n"))
        proc.wait()
        if proc.returncode == 0:
            update_task_status(task_id, "success")
        else:
            update_task_status(task_id, "fail")
            append_task_log(task_id, f"Task failed with code {proc.returncode}")
        log_path = LOG_DIR / f"{task_id}.log"
        result_files = _collect_result_files(workdir)
        summary = _summarize_results(result_files)
        record_result(
            task_id,
            {
                "returncode": proc.returncode,
                "workdir": workdir,
                "cmd": cmd,
                "log_path": str(log_path),
                "result_files": result_files,
                "summary": summary,
            },
        )
        return proc.returncode
    except Exception as e:
        update_task_status(task_id, "fail")
        append_task_log(task_id, f"Task failed with exception: {e}")
        record_result(
            task_id,
            {
                "returncode": 1,
                "workdir": workdir,
                "cmd": cmd,
                "error": str(e),
                "result_files": [],
            },
        )
        return 1


__all__ = ["run_rdagent_task"]
