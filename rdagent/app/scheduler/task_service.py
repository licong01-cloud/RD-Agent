"""
Local JSONL-based persistence for tasks and datasets (interim implementation).

Notes:
- Intended as a placeholder before integrating a real DB/ORM.
- Provides basic CRUD-like helpers for TaskRecord and DatasetRecord.
- Worker/API layers can call these helpers to manage scheduler state.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import DatasetRecord, TaskRecord
from .config_service import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "scheduler_data"
TASK_FILE = DATA_DIR / "tasks.jsonl"
DATASET_FILE = DATA_DIR / "datasets.jsonl"
LOG_DIR = PROJECT_ROOT / "log" / "scheduler_tasks"
RESULT_FILE = DATA_DIR / "results.jsonl"


def _ensure_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (TASK_FILE, DATASET_FILE, RESULT_FILE):
        if not f.exists():
            f.write_text("", encoding="utf-8")


def _load_jsonl(path: Path):
    _ensure_files()
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def _append_jsonl(path: Path, obj) -> None:
    _ensure_files()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


# Dataset operations
def list_datasets() -> List[DatasetRecord]:
    return [DatasetRecord(**d) for d in _load_jsonl(DATASET_FILE)]


def create_dataset(rec: DatasetRecord) -> DatasetRecord:
    rec.created_at = datetime.utcnow()
    _append_jsonl(DATASET_FILE, asdict(rec))
    return rec


# Task operations
def list_tasks() -> List[TaskRecord]:
    return [TaskRecord(**t) for t in _load_jsonl(TASK_FILE)]


def create_task(rec: TaskRecord) -> TaskRecord:
    rec.created_at = datetime.utcnow()
    rec.updated_at = rec.created_at
    _append_jsonl(TASK_FILE, asdict(rec))
    return rec


def update_task_status(task_id: str, status: str) -> Optional[TaskRecord]:
    tasks = _load_jsonl(TASK_FILE)
    updated = None
    for t in tasks:
        if str(t.get("id")) == str(task_id) or t.get("name") == task_id:
            t["status"] = status
            t["updated_at"] = datetime.utcnow().isoformat()
            updated = TaskRecord(**t)
    # rewrite file
    TASK_FILE.write_text("", encoding="utf-8")
    for t in tasks:
        _append_jsonl(TASK_FILE, t)
    return updated


def get_task(task_id: str) -> Optional[TaskRecord]:
    for t in _load_jsonl(TASK_FILE):
        if str(t.get("id")) == str(task_id) or t.get("name") == task_id:
            return TaskRecord(**t)
    return None


def append_task_log(task_id: str, content: str) -> Path:
    _ensure_files()
    log_path = LOG_DIR / f"{task_id}.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
    return log_path


def read_task_log(task_id: str) -> str:
    _ensure_files()
    log_path = LOG_DIR / f"{task_id}.log"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8")


# Result operations
def record_result(task_id: str, result: dict) -> None:
    payload = {"task_id": task_id, **result}
    _append_jsonl(RESULT_FILE, payload)


def list_results(task_id: Optional[str] = None) -> list:
    items = _load_jsonl(RESULT_FILE)
    if task_id:
        items = [i for i in items if str(i.get("task_id")) == str(task_id)]
    return items


__all__ = [
    "list_datasets",
    "create_dataset",
    "list_tasks",
    "create_task",
    "update_task_status",
    "get_task",
    "append_task_log",
    "read_task_log",
    "list_results",
    "DATA_DIR",
    "LOG_DIR",
]
