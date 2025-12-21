from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from rdagent.rag.doc_registry import DocRegistry, DocRegistryEntry
from rdagent.rag.embeddings import embed_texts
from rdagent.utils import md5_hash


@dataclass(frozen=True)
class SourceDocument:
    source_path: str
    text: str
    type: str
    tags: list[str]
    kb_version: str
    doc_format: str


def _iter_files(root: Path, *, include_exts: set[str]) -> Iterator[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in include_exts:
            yield p


def _safe_relative_path(repo_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(repo_root)
        return str(rel).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


class IndexBuilder:
    def __init__(
        self,
        *,
        repo_root: Path,
        chroma_dir: Path,
        registry: DocRegistry,
        kb_version: str,
        collection_name: str | None = None,
        include_exts: Sequence[str] = ("md", "yaml", "yml"),
        chunk_size: int = 1200,
        overlap: int = 200,
    ):
        self._repo_root = repo_root
        self._chroma_dir = chroma_dir
        self._registry = registry
        self._kb_version = kb_version
        self._collection_name = collection_name or f"rag_{kb_version}"
        self._include_exts = set([e.lower().lstrip(".") for e in include_exts])
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def iter_default_source_docs(self) -> Iterable[SourceDocument]:
        # Default behavior is intentionally conservative to avoid ingesting large memos.
        # Only index curated knowledge cards under RAG/ unless caller explicitly adds more sources.
        rag_root = self._repo_root / "RAG"
        if not rag_root.exists():
            return

        yield from self.iter_source_docs_from_dirs([rag_root])

    def iter_source_docs_from_dirs(self, dirs: Sequence[Path]) -> Iterable[SourceDocument]:
        for root in dirs:
            if not root.exists() or not root.is_dir():
                continue
            files = list(_iter_files(root, include_exts=self._include_exts))
            yield from self.iter_source_docs_from_files(files)

    def iter_source_docs_from_files(self, files: Sequence[Path]) -> Iterable[SourceDocument]:
        for fp in files:
            if not fp.exists() or not fp.is_file():
                continue

            rel = _safe_relative_path(self._repo_root, fp)
            doc_format = fp.suffix.lower().lstrip(".") or "unknown"
            text = fp.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue

            sem_type = self._guess_semantic_type(rel)
            tags = self._guess_tags(rel)

            yield SourceDocument(
                source_path=rel,
                text=text,
                type=sem_type,
                tags=tags,
                kb_version=self._kb_version,
                doc_format=doc_format,
            )

    def build(self, docs: Iterable[SourceDocument] | None = None) -> None:
        if docs is None:
            docs = self.iter_default_source_docs()

        try:
            import chromadb
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing optional dependency chromadb") from e

        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        collection = client.get_or_create_collection(name=self._collection_name)

        ids: list[str] = []
        metadatas: list[dict] = []
        documents: list[str] = []

        for doc in docs:
            doc_id = md5_hash(f"{doc.kb_version}:{doc.source_path}")

            self._registry.upsert(
                DocRegistryEntry(
                    doc_id=doc_id,
                    enabled=True,
                    source_path=doc.source_path,
                    kb_version=doc.kb_version,
                    doc_format=doc.doc_format,
                    type=doc.type,
                    tags=doc.tags,
                )
            )

            chunks = _chunk_text(doc.text, chunk_size=self._chunk_size, overlap=self._overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}#chunk{i}" 
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "doc_id": doc_id,
                        "source_path": doc.source_path,
                        "chunk_id": chunk_id,
                        "kb_version": doc.kb_version,
                        "type": doc.type,
                        "tags": ",".join(doc.tags),
                        "doc_format": doc.doc_format,
                    }
                )

        if not ids:
            return

        embeddings = embed_texts(documents)

        # Upsert: add if missing, update if exists.
        collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def _guess_semantic_type(self, rel_path: str) -> str:
        p = rel_path.lower()
        if p.startswith("rag/"):
            if "hard_rules" in p or p.startswith("rag/01_"):
                return "rule"
            if "fields" in p or p.startswith("rag/02_"):
                return "schema"
            if "contract" in p or p.startswith("rag/03_"):
                return "contract"
            if "failure" in p or p.startswith("rag/05_"):
                return "failure_case"
            return "memo"
        if p.endswith("prompts.yaml") or p.endswith("prompts.yml"):
            return "prompt"
        return "memo"

    def _guess_tags(self, rel_path: str) -> list[str]:
        tags: list[str] = []
        p = rel_path.lower()
        if p.startswith("rag/"):
            tags.append("rag_card")
        if "qlib" in p:
            tags.append("qlib")
        if "factor" in p:
            tags.append("factor")
        if "money" in p or "moneyflow" in p:
            tags.append("moneyflow")
        return tags
