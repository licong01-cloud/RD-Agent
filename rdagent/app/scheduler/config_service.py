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

# Qlib template files to manage
TEMPLATE_FILES = {
    "factor_baseline": PROJECT_ROOT
    / "rdagent"
    / "scenarios"
    / "qlib"
    / "experiment"
    / "factor_template"
    / "conf_baseline.yaml",
    "factor_combined": PROJECT_ROOT
    / "rdagent"
    / "scenarios"
    / "qlib"
    / "experiment"
    / "factor_template"
    / "conf_combined_factors.yaml",
    "factor_combined_sota": PROJECT_ROOT
    / "rdagent"
    / "scenarios"
    / "qlib"
    / "experiment"
    / "factor_template"
    / "conf_combined_factors_sota_model.yaml",
    "model_baseline": PROJECT_ROOT
    / "rdagent"
    / "scenarios"
    / "qlib"
    / "experiment"
    / "model_template"
    / "conf_baseline_factors_model.yaml",
    "model_sota": PROJECT_ROOT
    / "rdagent"
    / "scenarios"
    / "qlib"
    / "experiment"
    / "model_template"
    / "conf_sota_factors_model.yaml",
}


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


def list_templates() -> Dict[str, Path]:
    """Return template name -> path mapping."""
    return TEMPLATE_FILES


def read_template(name: str) -> Dict:
    path = TEMPLATE_FILES.get(name)
    if path is None:
        raise KeyError(f"Unknown template name: {name}")
    return read_yaml(path)


def write_template(name: str, data: Dict, task_id: Optional[str] = None) -> Path:
    path = TEMPLATE_FILES.get(name)
    if path is None:
        raise KeyError(f"Unknown template name: {name}")
    return write_yaml(path, data, task_id)


# History record hook (placeholder for DB integration)
def record_history(file_name: str, backup_path: Path, task_id: Optional[str] = None, user: Optional[str] = None) -> None:
    """Hook for storing history metadata. To be integrated with DB/ORM."""
    append_history(
        TemplateHistoryRecord(
            file_name=file_name,
            backup_path=str(backup_path),
            task_id=task_id,
            user=user,
        )
    )


__all__ = [
    "ENV_PATH",
    "TEMPLATE_FILES",
    "list_templates",
    "read_env",
    "write_env",
    "read_template",
    "write_template",
    "backup_file",
]
