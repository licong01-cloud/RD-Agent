"""
Lightweight data models for RD-Agent scheduler metadata.

These are placeholders; integrate with real ORM/DB in subsequent steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class TaskRecord:
    id: Optional[int] = None
    name: str = ""
    status: str = "pending"  # pending/running/success/fail/canceled
    dataset_ids: List[str] = field(default_factory=list)
    loop_n: int = 1
    all_duration: str = "1:00:00"
    evolving_mode: str = "llm"  # bandit/llm/random
    source_history_id: Optional[str] = None
    workspace_path: Optional[str] = None
    config_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DatasetRecord:
    id: Optional[int] = None
    name: str = ""
    provider_uri: str = ""
    instruments_file: str = "instruments/all.txt"
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TemplateHistoryRecord:
    id: Optional[int] = None
    file_name: str = ""
    backup_path: str = ""
    task_id: Optional[str] = None
    user: Optional[str] = None
    hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    extra: Dict = field(default_factory=dict)


__all__ = ["TaskRecord", "DatasetRecord", "TemplateHistoryRecord"]
