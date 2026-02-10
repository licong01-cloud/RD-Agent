"""Pydantic models for LLM configuration API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    """LLM provider information."""

    provider_name: str = Field(..., description="Provider identifier (e.g., deepseek, siliconflow)")
    display_name: str = Field(..., description="Display name for UI")
    api_base_url: str = Field(..., description="API base URL")
    litellm_prefix: str = Field(..., description="LiteLLM prefix (e.g., deepseek/, openai/)")
    supports_chat: bool = Field(default=True)
    supports_embedding: bool = Field(default=False)
    supports_reasoner: bool = Field(default=False)


class ModelInfo(BaseModel):
    """LLM model information."""

    model_name: str = Field(..., description="Model name")
    display_name: str = Field(..., description="Display name for UI")
    full_model_id: str = Field(..., description="Full model ID with prefix")
    model_type: str = Field(..., description="Model type: chat, embedding, reasoner")
    model_category: Optional[str] = Field(None, description="对话/Coding, 推理模型, 嵌入式模型")
    description: Optional[str] = Field(None, max_length=100, description="Model description")
    provider_name: str = Field(..., description="Provider name")
    is_verified: bool = Field(default=False)
    last_verified_at: Optional[str] = None


class StageMappingInfo(BaseModel):
    """Stage to model mapping information."""

    stage_name: str = Field(..., description="Stage name: coding, direct_exp_gen, feedback, default, etc.")
    model_id: str = Field(..., description="Full model ID (e.g., deepseek/deepseek-chat)")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature 0-2")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")


class ApiCredentialInfo(BaseModel):
    """API credential information (with security masking for keys)."""

    provider: str = Field(..., description="Provider identifier")
    env_var_name: str = Field(..., description="Environment variable name")
    value: str = Field(..., description="Value (masked for keys, full for base URLs)")
    is_set: bool = Field(default=True, description="Whether the credential is set")
    is_key: bool = Field(default=False, description="Whether this is an API key (vs base URL)")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    litellm_embedding_model: Optional[str] = Field(None, description="LiteLLM embedding model ID")
    embedding_model: Optional[str] = Field(None, description="Standard embedding model ID")
    embedding_api_base: Optional[str] = Field(None, description="Embedding API base URL")
    embedding_api_key_set: bool = Field(default=False, description="Whether embedding API key is set")
    rag_embedding_model: Optional[str] = Field(None, description="RAG-specific embedding model")


class CurrentEnvConfig(BaseModel):
    """Current .env configuration - comprehensive format matching AIstock expectations."""

    backend: str = Field(default="", description="Backend class path")
    chat_model: str = Field(default="", description="Default chat model")
    chat_temperature: str = Field(default="", description="Default temperature")
    chat_stream: str = Field(default="", description="Whether to use streaming")
    stage_mappings: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="LITELLM_CHAT_MODEL_MAP content - stage to model mapping"
    )
    embedding_config: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding model configuration"
    )
    api_credentials: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="API credentials (keys masked for security)"
    )
    last_updated: Optional[str] = Field(None, description="ISO timestamp of last update")


class VerificationRequest(BaseModel):
    """Model verification request."""

    provider_name: str
    model_name: str
    full_model_id: str
    api_key: str
    model_type: str
    api_base: Optional[str] = Field(None, description="API base URL for verification")


class VerificationResult(BaseModel):
    """Model verification result."""

    is_verified: bool
    verified_at: Optional[str] = None
    model_type_detected: Optional[str] = None
    error_message: Optional[str] = None
    test_response: Optional[dict[str, Any]] = None


class UpdateEnvRequest(BaseModel):
    """Request to update .env configuration - supports full format including API credentials."""

    stage_mappings: list[StageMappingInfo] = Field(
        ...,
        description="Stage to model mappings"
    )
    api_credentials: Optional[dict[str, str]] = Field(
        None,
        description="Optional API credentials to update (e.g., {'DEEPSEEK_API_KEY': 'sk-xxx'})"
    )
    change_reason: Optional[str] = Field(None, description="Reason for the change")
    backup_reason: str = Field(default="llm_config_update", description="Reason for creating backup")


class UpdateEnvResponse(BaseModel):
    """Response after updating .env configuration."""

    ok: bool
    message: str
    backup_path: Optional[str] = None
    updated_keys: list[str] = Field(default_factory=list, description="List of updated environment variables")
    verification_passed: bool = Field(default=False, description="Whether post-update verification passed")
