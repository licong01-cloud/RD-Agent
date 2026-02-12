"""
Placeholder API stubs for RD-Agent scheduler.

These functions outline expected behaviors; replace with real web framework handlers.
"""

from __future__ import annotations

from .models import DatasetRecord, TaskRecord
from .queue_stub import submit_task
from .task_service import (
    append_task_log,
    create_dataset,
    create_task,
    get_task,
    list_datasets,
    list_results,
    list_tasks,
    read_task_log,
    update_task_status,
)
from .template_service import (
    activate_template_env,
    apply_template,
    delete_template,
    get_active_env_template,
    get_sync_status,
    get_template_file,
    list_backups,
    list_template_files,
    list_template_history,
    list_templates,
    publish_templates,
    refresh_template_sha256,
    rollback_template,
    save_template_file,
)


# Task APIs
def api_list_tasks() -> dict:
    tasks = list_tasks()
    return {"items": [t.__dict__ for t in tasks]}


def api_create_task(payload: dict) -> dict:
    rec = TaskRecord(
        name=payload.get("name", ""),
        dataset_ids=payload.get("dataset_ids", []),
        loop_n=payload.get("loop_n", 1),
        all_duration=payload.get("all_duration", "1:00:00"),
        evolving_mode=payload.get("evolving_mode", "llm"),
        source_history_id=payload.get("source_history_id"),
        template_version=payload.get("template_version"),
        manifest_hash=payload.get("manifest_hash"),
    )
    rec = create_task(rec)
    # Submit to in-process queue (placeholder). Replace with real queue in production.
    submit_task(task_id=rec.name or str(rec.id))
    return {"task": rec.__dict__}


def api_get_task(task_id: str) -> dict:
    t = get_task(task_id)
    return {"task": t.__dict__ if t else None}


def api_update_task_status(task_id: str, status: str) -> dict:
    t = update_task_status(task_id, status)
    return {"task": t.__dict__ if t else None}


def api_append_log(task_id: str, content: str) -> dict:
    path = append_task_log(task_id, content)
    return {"log_path": str(path)}


def api_get_log(task_id: str) -> dict:
    return {"log": read_task_log(task_id)}


# Results
def api_list_results(task_id: str | None = None) -> dict:
    return {"items": list_results(task_id)}


# Dataset APIs
def api_list_datasets() -> dict:
    ds = list_datasets()
    return {"items": [d.__dict__ for d in ds]}


def api_create_dataset(payload: dict) -> dict:
    rec = DatasetRecord(
        name=payload.get("name", ""),
        provider_uri=payload.get("provider_uri", ""),
        instruments_file=payload.get("instruments_file", "instruments/all.txt"),
        description=payload.get("description"),
    )
    rec = create_dataset(rec)
    return {"dataset": rec.__dict__}


# Template APIs (existing)
def api_publish_templates(payload: dict) -> dict:
    return publish_templates(payload)


def api_list_template_history(scenario: str | None = None, version: str | None = None) -> dict:
    return list_template_history(scenario, version)


def api_rollback_template(payload: dict) -> dict:
    return rollback_template(payload)


# Template APIs (P2 new)
def api_list_templates(scenario: str | None = None) -> dict:
    return list_templates(scenario)


def api_list_template_files(scenario: str, version: str) -> dict:
    return list_template_files(scenario, version)


def api_get_template_file(scenario: str, version: str, rel_path: str) -> dict:
    return get_template_file(scenario, version, rel_path)


def api_save_template_file(scenario: str, version: str, rel_path: str, content: str) -> dict:
    return save_template_file(scenario, version, rel_path, content)


def api_delete_template(scenario: str, version: str) -> dict:
    return delete_template(scenario, version)


def api_apply_template(scenario: str, version: str, force: bool = False, backup: bool = True) -> dict:
    return apply_template(scenario, version, force=force, backup=backup)


def api_get_sync_status() -> dict:
    return get_sync_status()


def api_refresh_template_sha256(scenario: str, version: str) -> dict:
    return refresh_template_sha256(scenario, version)


def api_list_backups() -> dict:
    return list_backups()


def api_rollback_from_backup(backup_id: str) -> dict:
    from .template_service import _rollback_from_backup
    return _rollback_from_backup(backup_id)


# Template APIs (P3 – env-based activation)
def api_activate_template_env(scenario: str, version: str) -> dict:
    return activate_template_env(scenario, version)


def api_get_active_env_template() -> dict:
    return get_active_env_template()


__all__ = [
    "api_activate_template_env",
    "api_append_log",
    "api_apply_template",
    "api_create_dataset",
    "api_create_task",
    "api_delete_template",
    "api_get_active_env_template",
    "api_get_log",
    "api_get_sync_status",
    "api_get_task",
    "api_get_template_file",
    "api_list_backups",
    "api_list_datasets",
    "api_list_results",
    "api_list_tasks",
    "api_list_template_files",
    "api_list_template_history",
    "api_list_templates",
    "api_publish_templates",
    "api_refresh_template_sha256",
    "api_rollback_from_backup",
    "api_rollback_template",
    "api_save_template_file",
    "api_update_task_status",
]
