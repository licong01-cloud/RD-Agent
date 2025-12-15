from __future__ import annotations

from pathlib import Path

from rdagent.rag.doc_registry import DocRegistry
from rdagent.rag.interfaces import RetrieverService
from rdagent.rag.local_llamaindex_chroma import LocalLlamaIndexChromaRetriever
from rdagent.rag.settings import SETTINGS


def get_retriever() -> RetrieverService:
    if not SETTINGS.enabled:
        raise RuntimeError("RAG is disabled by settings.")

    storage = SETTINGS.storage_path()
    registry = DocRegistry(storage / "doc_registry.json")

    backend = SETTINGS.backend.lower()
    if backend == "local":
        return LocalLlamaIndexChromaRetriever(
            chroma_dir=storage / "chroma",
            registry=registry,
            kb_version=SETTINGS.kb_version,
        )

    raise ValueError(f"Unsupported RAG backend: {SETTINGS.backend}")
