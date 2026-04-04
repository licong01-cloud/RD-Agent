"""
Lightweight data models for RD-Agent scheduler metadata.

These are placeholders; integrate with real ORM/DB in subsequent steps.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

LOCAL_TZ = timezone(timedelta(hours=8))
_HOSTNAME = socket.gethostname()


@dataclass
class TaskRecord:
    id: int | None = None
    name: str = ""
    status: str = "pending"  # pending/running/success/fail/canceled
    dataset_ids: list[str] = field(default_factory=list)
    loop_n: int = 1
    all_duration: str = "1:00:00"
    evolving_mode: str = "llm"  # bandit/llm/random
    source_history_id: str | None = None
    template_version: str | None = None
    manifest_hash: str | None = None
    workspace_path: str | None = None
    config_hash: str | None = None
    command: str = "fin_quant"  # rdagent 子命令: fin_factor/fin_model/fin_quant
    env_overrides: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(LOCAL_TZ))
    updated_at: datetime = field(default_factory=lambda: datetime.now(LOCAL_TZ))
    source: str = "scheduler"  # scheduler / manual
    rdagent_log_dir: str | None = None  # log/ 下的目录名，用于关联手动任务
    hostname: str = field(default_factory=lambda: _HOSTNAME)


@dataclass
class DatasetRecord:
    id: int | None = None
    name: str = ""
    provider_uri: str = ""
    instruments_file: str = "instruments/all.txt"
    description: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(LOCAL_TZ))


@dataclass
class TemplateHistoryRecord:
    id: int | None = None
    file_name: str = ""
    backup_path: str = ""
    task_id: str | None = None
    user: str | None = None
    hash: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(LOCAL_TZ))
    extra: dict = field(default_factory=dict)


__all__ = ["DatasetRecord", "TaskRecord", "TemplateHistoryRecord"]
