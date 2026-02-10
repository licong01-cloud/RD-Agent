"""Environment file (.env) management for LLM configuration."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EnvManager:
    """Manages .env file operations for LLM configuration."""

    # 已知的API凭证环境变量映射
    KNOWN_API_CREDENTIALS = {
        # DeepSeek
        "DEEPSEEK_API_KEY": "deepseek",
        "DEEPSEEK_API_BASE": "deepseek",
        # Anthropic/Claude
        "ANTHROPIC_API_KEY": "anthropic",
        "ANTHROPIC_BASE_URL": "anthropic",
        # OpenAI
        "OPENAI_API_KEY": "openai",
        "OPENAI_API_BASE": "openai",
        "CHAT_OPENAI_API_KEY": "openai",
        "CHAT_OPENAI_API_BASE": "openai",
        # SiliconFlow/LiteLLM Proxy
        "LITELLM_PROXY_API_KEY": "litellm_proxy",
        "LITELLM_PROXY_API_BASE": "litellm_proxy",
        # Embedding
        "EMBEDDING_API_KEY": "embedding",
        "EMBEDDING_API_BASE": "embedding",
        "RAG_EMBEDDING_API_KEY": "rag_embedding",
        "RAG_EMBEDDING_API_BASE": "rag_embedding",
    }

    def __init__(self, env_path: str | Path | None = None):
        """Initialize EnvManager with .env file path."""
        if env_path is None:
            # Default to RD-Agent root .env
            self.env_path = Path(__file__).resolve().parents[3] / ".env"
        else:
            self.env_path = Path(env_path)

        self.backup_dir = Path(__file__).resolve().parents[3] / "git_ignore_folder" / "env_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def read_env(self) -> dict[str, str]:
        """Read .env file and return as dictionary."""
        if not self.env_path.exists():
            return {}

        env_dict: dict[str, str] = {}
        with open(self.env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_dict[key.strip()] = value.strip()
        return env_dict

    def parse_current_config(self) -> dict[str, Any]:
        """Parse current .env configuration for LLM settings.
        
        Returns comprehensive config including:
        - LITELLM_CHAT_MODEL_MAP (stage mappings)
        - LITELLM_EMBEDDING_MODEL
        - All API credentials (with masked keys for security)
        - Other LLM-related settings
        """
        env_dict = self.read_env()

        config: dict[str, Any] = {
            "backend": env_dict.get("BACKEND", ""),
            "chat_model": env_dict.get("CHAT_MODEL", ""),
            "chat_temperature": env_dict.get("CHAT_TEMPERATURE", ""),
            "chat_stream": env_dict.get("CHAT_STREAM", ""),
            "stage_mappings": {},
            "embedding_config": {},
            "api_credentials": {},
            "last_updated": None,
        }

        # Parse LITELLM_CHAT_MODEL_MAP
        model_map_str = env_dict.get("LITELLM_CHAT_MODEL_MAP", "{}")
        try:
            model_map = json.loads(model_map_str)
            config["stage_mappings"] = model_map
        except json.JSONDecodeError:
            pass

        # Parse LITELLM_EMBEDDING_MODEL
        litellm_embedding = env_dict.get("LITELLM_EMBEDDING_MODEL", "")
        if litellm_embedding:
            config["embedding_config"]["litellm_embedding_model"] = litellm_embedding

        # Parse standard embedding config
        standard_embedding = env_dict.get("EMBEDDING_MODEL", "")
        if standard_embedding:
            config["embedding_config"]["embedding_model"] = standard_embedding
        if env_dict.get("EMBEDDING_API_BASE"):
            config["embedding_config"]["embedding_api_base"] = env_dict.get("EMBEDDING_API_BASE")
        if env_dict.get("EMBEDDING_API_KEY"):
            config["embedding_config"]["embedding_api_key_set"] = bool(env_dict.get("EMBEDDING_API_KEY"))

        # Parse RAG embedding config
        rag_embedding = env_dict.get("RAG_EMBEDDING_MODEL", "")
        if rag_embedding:
            config["embedding_config"]["rag_embedding_model"] = rag_embedding

        # Collect all API credentials (with security masking)
        for env_var, provider in self.KNOWN_API_CREDENTIALS.items():
            value = env_dict.get(env_var)
            if value:
                if "KEY" in env_var:
                    # Mask API keys for security - only show first/last 4 chars
                    masked = f"{value[:4]}****{value[-4:]}" if len(value) > 8 else "****"
                    config["api_credentials"][env_var] = {
                        "provider": provider,
                        "value": masked,
                        "is_set": True,
                    }
                else:
                    # API_BASE URLs can be shown fully
                    config["api_credentials"][env_var] = {
                        "provider": provider,
                        "value": value,
                        "is_set": True,
                    }

        # Get file modification time
        if self.env_path.exists():
            mtime = self.env_path.stat().st_mtime
            config["last_updated"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        return config

    def create_backup(self, reason: str = "manual") -> dict[str, str]:
        """Create a timestamped backup of .env file."""
        if not self.env_path.exists():
            raise FileNotFoundError(f".env file not found: {self.env_path}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_filename = f".env.backup.{timestamp}"
        backup_path = self.backup_dir / backup_filename

        shutil.copy2(self.env_path, backup_path)

        return {
            "backup_filename": backup_filename,
            "backup_path": str(backup_path),
            "backup_size": backup_path.stat().st_size,
            "backup_reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def update_env(self, updates: dict[str, str]) -> list[str]:
        """Update .env file with new values."""
        if not self.env_path.exists():
            raise FileNotFoundError(f".env file not found: {self.env_path}")

        # Read current content
        with open(self.env_path, encoding="utf-8") as f:
            lines = f.readlines()

        updated_keys: list[str] = []
        keys_to_update = set(updates.keys())

        # Update existing keys
        new_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue

            if "=" in stripped:
                key, _ = stripped.split("=", 1)
                key = key.strip()
                if key in keys_to_update:
                    new_lines.append(f"{key}={updates[key]}\n")
                    updated_keys.append(key)
                    keys_to_update.remove(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Append new keys that weren't found
        if keys_to_update:
            new_lines.append("\n# Added by LLM Config Manager\n")
            for key in keys_to_update:
                new_lines.append(f"{key}={updates[key]}\n")
                updated_keys.append(key)

        # Write back
        with open(self.env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        return updated_keys

    def update_or_add_env_var(self, content: str, var_name: str, var_value: str) -> str:
        """Update or add environment variable to .env file content.
        
        If variable exists, update it; if not, append to file end.
        """
        # Try to match existing variable (including commented)
        pattern = rf'^\s*#?\s*{re.escape(var_name)}=.*$'

        if re.search(pattern, content, re.MULTILINE):
            # Variable exists, update it (uncomment and update value)
            new_line = f"{var_name}={var_value}"
            content = re.sub(pattern, new_line, content, flags=re.MULTILINE)
        else:
            # Variable doesn't exist, append to file end
            if not content.endswith('\n'):
                content += '\n'
            content += f"\n# Auto-generated by LLM Config Manager\n{var_name}={var_value}\n"

        return content

    def restore_backup(self, backup_path: str | Path) -> bool:
        """Restore .env from a backup file."""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        shutil.copy2(backup_path, self.env_path)
        return True

    def build_stage_mapping_updates(
        self,
        stage_mappings: list[dict[str, Any]],
        api_credentials: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Build .env updates from stage mappings and API credentials.
        
        Args:
            stage_mappings: List of stage mapping dicts with keys:
                - stage_name: str
                - model_id/full_model_id: str
                - temperature: float (optional)
                - max_tokens: int (optional)
            api_credentials: Optional dict of API credentials to update
                Keys should be env var names like "DEEPSEEK_API_KEY", "OPENAI_API_BASE", etc.
        
        Returns:
            Dict of env var updates
        """
        updates: dict[str, str] = {}

        # Build LITELLM_CHAT_MODEL_MAP
        model_map: dict[str, dict[str, Any]] = {}
        embedding_model = None

        for mapping in stage_mappings:
            stage_name = mapping.get("stage_name")
            # Support both model_id and full_model_id
            model_id = mapping.get("full_model_id") or mapping.get("model_id")
            
            if not stage_name or not model_id:
                continue

            if stage_name == "embedding":
                embedding_model = model_id
            else:
                model_map[stage_name] = {"model": model_id}
                if mapping.get("temperature") is not None:
                    model_map[stage_name]["temperature"] = str(mapping["temperature"])
                if mapping.get("max_tokens") is not None:
                    model_map[stage_name]["max_tokens"] = str(mapping["max_tokens"])

        if model_map:
            # Use compact JSON format like in the actual .env
            updates["LITELLM_CHAT_MODEL_MAP"] = json.dumps(
                model_map, ensure_ascii=False, separators=(',', ':')
            )

        if embedding_model:
            updates["LITELLM_EMBEDDING_MODEL"] = embedding_model

        # Add API credentials if provided
        if api_credentials:
            for key, value in api_credentials.items():
                if value is not None:
                    updates[key] = value

        return updates
