from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    enabled: bool = True
    backend: str = "local"
    storage_dir: str = "git_ignore_folder/rag_storage"
    kb_version: str = "default"
    top_k: int = 6
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_timeout: int = 60

    model_config = SettingsConfigDict(env_prefix="RAG_")

    def storage_path(self) -> Path:
        return Path(self.storage_dir)


SETTINGS = RagSettings()
