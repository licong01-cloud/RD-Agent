"""FastAPI routes for LLM configuration management."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from .models import (
    CurrentEnvConfig,
    UpdateEnvRequest,
    UpdateEnvResponse,
    VerificationRequest,
    VerificationResult,
)
from .service import LLMConfigService

router = APIRouter(prefix="/llm-config", tags=["llm-config"])
service = LLMConfigService()

DEFAULT_BODY = Body(default_factory=dict)


@router.get("/current-config", response_model=CurrentEnvConfig)
def get_current_config() -> CurrentEnvConfig:
    """Get current LLM configuration from .env file."""
    return service.get_current_config()


@router.post("/verify-model", response_model=VerificationResult)
async def verify_model(request: VerificationRequest) -> VerificationResult:
    """Verify model API key and availability."""
    return await service.verify_model(request)


@router.post("/update-config", response_model=UpdateEnvResponse)
def update_config(request: UpdateEnvRequest) -> UpdateEnvResponse:
    """Update .env configuration with new stage mappings."""
    result = service.update_env_config(request)
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.message)
    return result


@router.post("/verify-rdagent")
def verify_rdagent() -> dict:
    """Verify RD-Agent can work with current configuration."""
    return service.verify_rdagent_integration()


@router.post("/rollback")
def rollback(payload: Annotated[dict, DEFAULT_BODY]) -> dict:
    """Rollback .env to a previous backup."""
    backup_path = payload.get("backup_path", "")
    if not backup_path:
        raise HTTPException(status_code=400, detail="backup_path is required")
    
    result = service.rollback_to_backup(backup_path)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("message", "Rollback failed"))
    return result


@router.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "llm-config"}
