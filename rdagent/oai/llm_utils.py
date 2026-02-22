from __future__ import annotations

from typing import Any, Type

import numpy as np

from rdagent.core.utils import import_class
from rdagent.oai.backend.base import APIBackend as BaseAPIBackend
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.utils import md5_hash  # for compatible with previous import


def calculate_embedding_distance_between_str_list(
    source_str_list: list[str],
    target_str_list: list[str],
    batch_size: int = 50,
) -> list[list[float]]:
    if not source_str_list or not target_str_list:
        return [[]]

    all_strings = source_str_list + target_str_list
    embeddings = []
    api_backend = APIBackend()
    
    # 分批调用 embedding 接口，避免超过提供商的单次 input batch size 限制
    for i in range(0, len(all_strings), batch_size):
        batch_strs = all_strings[i:i + batch_size]
        batch_emb = api_backend.create_embedding(batch_strs)
        embeddings.extend(batch_emb)

    source_embeddings = embeddings[: len(source_str_list)]
    target_embeddings = embeddings[len(source_str_list) :]

    source_embeddings_np = np.array(source_embeddings)
    target_embeddings_np = np.array(target_embeddings)

    source_embeddings_np = source_embeddings_np / np.linalg.norm(source_embeddings_np, axis=1, keepdims=True)
    target_embeddings_np = target_embeddings_np / np.linalg.norm(target_embeddings_np, axis=1, keepdims=True)
    similarity_matrix = np.dot(source_embeddings_np, target_embeddings_np.T)

    return similarity_matrix.tolist()  # type: ignore[no-any-return]


def get_api_backend(*args: Any, **kwargs: Any) -> BaseAPIBackend:  # TODO: import it from base.py
    """
    get llm api backend based on settings dynamically.
    """
    api_backend_cls: Type[BaseAPIBackend] = import_class(LLM_SETTINGS.backend)
    return api_backend_cls(*args, **kwargs)


# Alias
APIBackend = get_api_backend
