from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from rdagent.rag.types import RetrievedContext


class RetrieverService(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedContext]:
        raise NotImplementedError

    def disable_doc(self, doc_id: str) -> None:
        raise NotImplementedError

    def enable_doc(self, doc_id: str) -> None:
        raise NotImplementedError
