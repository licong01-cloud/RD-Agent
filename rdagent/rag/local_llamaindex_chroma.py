from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rdagent.rag.doc_registry import DocRegistry
from rdagent.rag.embeddings import embed_query
from rdagent.rag.interfaces import RetrieverService
from rdagent.rag.types import RetrievedContext


class LocalLlamaIndexChromaRetriever(RetrieverService):
    def __init__(
        self,
        *,
        chroma_dir: Path,
        registry: DocRegistry,
        kb_version: str,
    ):
        self._chroma_dir = chroma_dir
        self._registry = registry
        self._kb_version = kb_version

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedContext]:
        try:
            import chromadb
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing RAG optional dependencies. Please install chromadb before using local backend."
            ) from e

        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        collection_name = f"rag_{self._kb_version}"
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Chroma collection '{collection_name}' not found. Build index first (expected at {self._chroma_dir})."
            ) from e

        query_emb = embed_query(query)

        # We intentionally keep the Chroma 'where' filter minimal to avoid backend-specific limitations.
        # P0 filtering is applied after retrieval to guarantee doc_registry enforcement.
        raw = collection.query(
            query_embeddings=[query_emb],
            n_results=max(top_k * 3, top_k),
            include=["documents", "metadatas", "distances"],
        )

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        req_type = None
        req_doc_format = None
        req_tags: set[str] | None = None
        if filters:
            req_type = filters.get("type")
            req_doc_format = filters.get("doc_format")
            if "tags" in filters and filters["tags"] is not None:
                if isinstance(filters["tags"], (list, tuple, set)):
                    req_tags = set([str(x) for x in filters["tags"]])
                else:
                    req_tags = {str(filters["tags"])}

        results: list[RetrievedContext] = []
        for text, meta, dist in zip(docs, metas, dists):
            if not isinstance(meta, dict):
                continue

            doc_id = str(meta.get("doc_id", ""))
            if not doc_id:
                continue

            if not self._registry.is_enabled(doc_id):
                continue

            sem_type = str(meta.get("type", ""))
            doc_format = str(meta.get("doc_format", ""))
            kb_version = str(meta.get("kb_version", self._kb_version))

            tags_raw = meta.get("tags", "")
            if isinstance(tags_raw, str):
                tags = [t for t in tags_raw.split(",") if t]
            elif isinstance(tags_raw, list):
                tags = [str(t) for t in tags_raw]
            else:
                tags = []

            if req_type is not None and sem_type != str(req_type):
                continue
            if req_doc_format is not None and doc_format != str(req_doc_format):
                continue
            if req_tags is not None and not req_tags.issubset(set(tags)):
                continue

            # Chroma returns distances; smaller is closer. Convert to a score for consistency.
            score = float(1.0 / (1.0 + float(dist)))

            results.append(
                RetrievedContext(
                    text=str(text),
                    doc_id=doc_id,
                    source_path=str(meta.get("source_path", "")),
                    chunk_id=str(meta.get("chunk_id", "")),
                    score=score,
                    type=sem_type,
                    tags=tags,
                    kb_version=kb_version,
                    doc_format=doc_format,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def disable_doc(self, doc_id: str) -> None:
        self._registry.disable(doc_id)

    def enable_doc(self, doc_id: str) -> None:
        self._registry.enable(doc_id)
