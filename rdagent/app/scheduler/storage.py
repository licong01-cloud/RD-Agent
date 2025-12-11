"""
Lightweight local storage helpers for template history.

This is an interim implementation using JSON Lines in history/.meta_history.jsonl.
It should be replaced with a real DB/ORM later.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .models import TemplateHistoryRecord
from .config_service import HISTORY_ROOT, backup_file

META_FILE = HISTORY_ROOT / ".meta_history.jsonl"


def _ensure_meta_file() -> None:
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    if not META_FILE.exists():
        META_FILE.write_text("", encoding="utf-8")


def append_history(record: TemplateHistoryRecord) -> None:
    _ensure_meta_file()
    with META_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), default=str) + "\n")


def list_history(file_name: Optional[str] = None) -> List[TemplateHistoryRecord]:
    _ensure_meta_file()
    records: List[TemplateHistoryRecord] = []
    with META_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if file_name and data.get("file_name") != file_name:
                continue
            records.append(TemplateHistoryRecord(**data))
    return records


def rollback_file(src: Path, backup_path: Path) -> Path:
    """Overwrite src with backup content."""
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(backup_path.read_bytes())
    return src


__all__ = ["append_history", "list_history", "rollback_file", "META_FILE"]
