from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    doc_id: str
    source_path: str
    chunk_id: str
    score: float
    type: str
    tags: Sequence[str]
    kb_version: str
    doc_format: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "type": self.type,
            "tags": list(self.tags),
            "kb_version": self.kb_version,
            "doc_format": self.doc_format,
        }
