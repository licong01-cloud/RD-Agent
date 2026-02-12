"""
Configuration service with automatic backup for .env and Qlib YAML templates.

Features:
- Centralized path management for .env and Qlib templates.
- Read/write helpers with backup-before-write to history/YYYYMMDD/.
- Timestamp + optional task_id in backup filenames for traceability.

This module is backend-first; APIs/CLI can call these helpers.
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Dict, Optional

import yaml

from .models import TemplateHistoryRecord
from .storage import append_history

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"
HISTORY_ROOT = PROJECT_ROOT / "history"



def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup_path(src: Path, task_id: Optional[str] = None) -> Path:
    date_dir = HISTORY_ROOT / datetime.datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    suffix = task_id if task_id else "manual"
    return date_dir / f"{_timestamp()}_{suffix}_{src.name}"


def backup_file(src: Path, task_id: Optional[str] = None) -> Path:
    """Copy src to history with timestamp/task_id. Returns backup path."""
    if not src.exists():
        raise FileNotFoundError(f"Source file not found for backup: {src}")
    dst = _backup_path(src, task_id)
    shutil.copy2(src, dst)
    return dst


def read_env() -> str:
    return ENV_PATH.read_text(encoding="utf-8")


def write_env(content: str, task_id: Optional[str] = None) -> Path:
    backup = backup_file(ENV_PATH, task_id)
    append_history(
        TemplateHistoryRecord(
            file_name=ENV_PATH.name,
            backup_path=str(backup),
            task_id=task_id,
        )
    )
    ENV_PATH.write_text(content, encoding="utf-8")
    return ENV_PATH


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: Dict, task_id: Optional[str] = None) -> Path:
    backup = backup_file(path, task_id)
    append_history(
        TemplateHistoryRecord(
            file_name=path.name,
            backup_path=str(backup),
            task_id=task_id,
        )
    )
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


__all__ = [
    "PROJECT_ROOT",
    "ENV_PATH",
    "HISTORY_ROOT",
    "read_env",
    "write_env",
    "read_yaml",
    "write_yaml",
    "backup_file",
]
