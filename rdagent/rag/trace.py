from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RagTraceEvent:
    attempt_idx: int
    query: str
    top_k: int
    kb_version: str
    retrieved_contexts: list[Mapping[str, Any]]
    validator_results: list[Mapping[str, Any]]
    decision: str
    timing_ms: Mapping[str, Any]

    def to_jsonl_line(self) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "attempt_idx": self.attempt_idx,
            "query": self.query,
            "top_k": self.top_k,
            "kb_version": self.kb_version,
            "retrieved_contexts": self.retrieved_contexts,
            "validator_results": self.validator_results,
            "decision": self.decision,
            "timing_ms": dict(self.timing_ms),
        }
        return json.dumps(payload, ensure_ascii=False)


class RagTraceWriter:
    def __init__(self, jsonl_path: Path):
        self._path = jsonl_path

    def append(self, event: RagTraceEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(event.to_jsonl_line())
            f.write("\n")
