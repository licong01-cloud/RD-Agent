from __future__ import annotations

from typing import Sequence

from rdagent.rag.settings import SETTINGS


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'openai'. Please install it before using RAG embeddings.") from e

    if not SETTINGS.embedding_api_key:
        raise RuntimeError("RAG embedding api key is empty. Please set RAG_EMBEDDING_API_KEY in .env")

    client = OpenAI(
        api_key=SETTINGS.embedding_api_key,
        base_url=SETTINGS.embedding_api_base or None,
        timeout=SETTINGS.embedding_timeout,
    )

    resp = client.embeddings.create(model=SETTINGS.embedding_model, input=list(texts))
    # Keep order consistent with input.
    data_sorted = sorted(resp.data, key=lambda x: x.index)
    return [d.embedding for d in data_sorted]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
