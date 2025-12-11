"""
Placeholder API stubs for RD-Agent scheduler.

These functions outline expected behaviors; replace with real web framework handlers.
"""

from __future__ import annotations

from typing import Dict, Optional

from .task_service import (
    append_task_log,
    create_dataset,
    create_task,
    get_task,
    list_datasets,
    list_tasks,
    update_task_status,
    list_results,
    read_task_log,
)
from .models import TaskRecord, DatasetRecord
from .queue_stub import submit_task


# Task APIs
def api_list_tasks() -> Dict:
    tasks = list_tasks()
    return {"items": [t.__dict__ for t in tasks]}


def api_create_task(payload: Dict) -> Dict:
    rec = TaskRecord(
        name=payload.get("name", ""),
        dataset_ids=payload.get("dataset_ids", []),
        loop_n=payload.get("loop_n", 1),
        all_duration=payload.get("all_duration", "1:00:00"),
        evolving_mode=payload.get("evolving_mode", "llm"),
        source_history_id=payload.get("source_history_id"),
    )
    rec = create_task(rec)
    # Submit to in-process queue (placeholder). Replace with real queue in production.
    submit_task(task_id=rec.name or str(rec.id))
    return {"task": rec.__dict__}


def api_get_task(task_id: str) -> Dict:
    t = get_task(task_id)
    return {"task": t.__dict__ if t else None}


def api_update_task_status(task_id: str, status: str) -> Dict:
    t = update_task_status(task_id, status)
    return {"task": t.__dict__ if t else None}


def api_append_log(task_id: str, content: str) -> Dict:
    path = append_task_log(task_id, content)
    return {"log_path": str(path)}


def api_get_log(task_id: str) -> Dict:
    return {"log": read_task_log(task_id)}


# Results
def api_list_results(task_id: Optional[str] = None) -> Dict:
    return {"items": list_results(task_id)}


# Dataset APIs
def api_list_datasets() -> Dict:
    ds = list_datasets()
    return {"items": [d.__dict__ for d in ds]}


def api_create_dataset(payload: Dict) -> Dict:
    rec = DatasetRecord(
        name=payload.get("name", ""),
        provider_uri=payload.get("provider_uri", ""),
        instruments_file=payload.get("instruments_file", "instruments/all.txt"),
        description=payload.get("description"),
    )
    rec = create_dataset(rec)
    return {"dataset": rec.__dict__}


__all__ = [
    "api_list_tasks",
    "api_create_task",
    "api_get_task",
    "api_update_task_status",
    "api_append_log",
    "api_get_log",
    "api_list_results",
    "api_list_datasets",
    "api_create_dataset",
]
