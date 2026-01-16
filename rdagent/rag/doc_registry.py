from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable


@dataclass
class DocRegistryEntry:
    doc_id: str
    enabled: bool
    source_path: str | None = None
    kb_version: str | None = None
    doc_format: str | None = None
    type: str | None = None
    tags: list[str] | None = None


class DocRegistry:
    def __init__(self, registry_path: Path):
        self._path = registry_path
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert(self, entry: DocRegistryEntry) -> None:
        self._data[entry.doc_id] = {
            "enabled": entry.enabled,
            "source_path": entry.source_path,
            "kb_version": entry.kb_version,
            "doc_format": entry.doc_format,
            "type": entry.type,
            "tags": entry.tags,
        }
        self._save()

    def disable(self, doc_id: str) -> None:
        rec = self._data.get(doc_id, {})
        rec["enabled"] = False
        self._data[doc_id] = rec
        self._save()

    def enable(self, doc_id: str) -> None:
        rec = self._data.get(doc_id, {})
        rec["enabled"] = True
        self._data[doc_id] = rec
        self._save()

    def is_enabled(self, doc_id: str) -> bool:
        rec = self._data.get(doc_id)
        if rec is None:
            return True
        return bool(rec.get("enabled", True))

    def enabled_doc_ids(self) -> Iterable[str]:
        for doc_id, rec in self._data.items():
            if bool(rec.get("enabled", True)):
                yield doc_id
