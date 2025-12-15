from __future__ import annotations

from typing import Sequence

from rdagent.rag.types import RetrievedContext


def format_knowledge_context(contexts: Sequence[RetrievedContext], *, max_chars: int = 8000) -> str:
    parts: list[str] = []
    total = 0
    for i, c in enumerate(contexts):
        header = (
            f"[ctx#{i}] doc_id={c.doc_id} source={c.source_path} chunk_id={c.chunk_id} "
            f"score={c.score:.4f} type={c.type} format={c.doc_format} tags={','.join(list(c.tags))}"
        )
        body = c.text.strip()
        block = header + "\n" + body
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)
