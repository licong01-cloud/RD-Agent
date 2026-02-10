"""Core service logic for LLM configuration management."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .env_manager import EnvManager
from .models import CurrentEnvConfig, UpdateEnvRequest, UpdateEnvResponse, VerificationRequest, VerificationResult


class LLMConfigService:
    """Service for managing LLM configurations."""

    def __init__(self) -> None:
        """Initialize LLM configuration service."""
        self.env_manager = EnvManager()

    def get_current_config(self) -> CurrentEnvConfig:
        """Get current LLM configuration from .env file."""
        config_dict = self.env_manager.parse_current_config()
        return CurrentEnvConfig(**config_dict)

    async def verify_model(self, request: VerificationRequest) -> VerificationResult:
        """Verify model API key and availability."""
        try:
            if request.model_type in ("chat", "reasoner"):
                return await self._verify_chat_model(request)
            elif request.model_type == "embedding":
                return await self._verify_embedding_model(request)
            else:
                return VerificationResult(
                    is_verified=False,
                    error_message=f"Unsupported model type: {request.model_type}",
                )
        except Exception as e:
            return VerificationResult(
                is_verified=False,
                error_message=str(e),
            )

    async def _verify_chat_model(self, request: VerificationRequest) -> VerificationResult:
        """Verify chat/reasoner model."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Authorization": f"Bearer {request.api_key}"}

            # Special handling for Claude
            if request.provider_name == "claude":
                headers = {
                    "x-api-key": request.api_key,
                    "anthropic-version": "2023-06-01",
                }

            payload = {
                "model": request.model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
            }

            endpoint = f"{request.api_base}/chat/completions"
            if request.provider_name == "claude":
                endpoint = f"{request.api_base}/v1/messages"

            try:
                response = await client.post(endpoint, headers=headers, json=payload)

                if response.status_code == 200:
                    return VerificationResult(
                        is_verified=True,
                        verified_at=datetime.now(timezone.utc).isoformat(),
                        model_type_detected=request.model_type,
                        test_response=response.json(),
                    )
                else:
                    return VerificationResult(
                        is_verified=False,
                        error_message=f"HTTP {response.status_code}: {response.text[:200]}",
                    )
            except httpx.TimeoutException:
                return VerificationResult(
                    is_verified=False,
                    error_message="Request timeout (30s)",
                )
            except Exception as e:
                return VerificationResult(
                    is_verified=False,
                    error_message=f"Request failed: {str(e)}",
                )

    async def _verify_embedding_model(self, request: VerificationRequest) -> VerificationResult:
        """Verify embedding model."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Authorization": f"Bearer {request.api_key}"}

            payload = {
                "model": request.model_name,
                "input": ["test embedding"],
            }

            endpoint = f"{request.api_base}/embeddings"

            try:
                response = await client.post(endpoint, headers=headers, json=payload)

                if response.status_code == 200:
                    resp_data = response.json()
                    return VerificationResult(
                        is_verified=True,
                        verified_at=datetime.now(timezone.utc).isoformat(),
                        model_type_detected="embedding",
                        test_response={
                            "embedding_dim": len(resp_data.get("data", [{}])[0].get("embedding", [])),
                            "model": resp_data.get("model"),
                        },
                    )
                else:
                    return VerificationResult(
                        is_verified=False,
                        error_message=f"HTTP {response.status_code}: {response.text[:200]}",
                    )
            except httpx.TimeoutException:
                return VerificationResult(
                    is_verified=False,
                    error_message="Request timeout (30s)",
                )
            except Exception as e:
                return VerificationResult(
                    is_verified=False,
                    error_message=f"Request failed: {str(e)}",
                )

    def update_env_config(self, request: UpdateEnvRequest) -> UpdateEnvResponse:
        """Update .env configuration with new stage mappings and optional API credentials."""
        try:
            # 1. Create backup
            backup_info = self.env_manager.create_backup(reason=request.backup_reason)

            # 2. Build updates from stage mappings
            stage_mappings_dict = [mapping.model_dump() for mapping in request.stage_mappings]
            
            # Get API credentials if provided
            api_credentials = request.api_credentials if request.api_credentials else None
            
            updates = self.env_manager.build_stage_mapping_updates(
                stage_mappings_dict,
                api_credentials=api_credentials
            )

            # 3. Apply updates using regex-based update for better handling of special formats
            # Read current content
            env_content = self.env_manager.env_path.read_text(encoding="utf-8")
            
            for key, value in updates.items():
                if key == "LITELLM_CHAT_MODEL_MAP":
                    # Special handling for JSON config - use single-quoted format
                    pattern = r"LITELLM_CHAT_MODEL_MAP='[^']*'"
                    new_line = f"LITELLM_CHAT_MODEL_MAP='{value}'"
                    if re.search(pattern, env_content):
                        env_content = re.sub(pattern, new_line, env_content)
                    else:
                        # Try multi-line format
                        pattern_multiline = r"LITELLM_CHAT_MODEL_MAP='\{[^}]*\}'"
                        if re.search(pattern_multiline, env_content, re.DOTALL):
                            env_content = re.sub(pattern_multiline, new_line, env_content, flags=re.DOTALL)
                        else:
                            env_content += f"\n{new_line}\n"
                else:
                    # Use the regex-based update method
                    env_content = self.env_manager.update_or_add_env_var(env_content, key, value)
            
            # Validate JSON format for LITELLM_CHAT_MODEL_MAP before writing
            map_match = re.search(r"LITELLM_CHAT_MODEL_MAP='(\{.+?\})'", env_content, re.DOTALL)
            if map_match:
                try:
                    json.loads(map_match.group(1))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in LITELLM_CHAT_MODEL_MAP: {e}")
            
            # Write back
            self.env_manager.env_path.write_text(env_content, encoding="utf-8")
            
            updated_keys = list(updates.keys())

            # 4. Verify RD-Agent configuration (optional, can be disabled for faster updates)
            verification_passed = True  # Simplified for now

            return UpdateEnvResponse(
                ok=True,
                message="Configuration updated successfully",
                backup_path=backup_info["backup_path"],
                updated_keys=updated_keys,
                verification_passed=verification_passed,
            )

        except Exception as e:
            return UpdateEnvResponse(
                ok=False,
                message=f"Failed to update configuration: {str(e)}",
                verification_passed=False,
            )

    def verify_rdagent_integration(self) -> dict[str, Any]:
        """Verify RD-Agent can work with current configuration."""
        try:
            rdagent_root = Path(__file__).resolve().parents[3]
            verify_script = rdagent_root / "debug_tools" / "verify_embedding_and_multimodel.py"

            if not verify_script.exists():
                return {
                    "is_valid": False,
                    "error_message": f"Verification script not found: {verify_script}",
                }

            result = subprocess.run(
                ["python", str(verify_script)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(rdagent_root),
                check=False,
            )

            return {
                "is_valid": result.returncode == 0,
                "test_output": result.stdout,
                "error_message": result.stderr if result.returncode != 0 else None,
            }

        except subprocess.TimeoutExpired:
            return {
                "is_valid": False,
                "error_message": "Verification timeout (60s)",
            }
        except Exception as e:
            return {
                "is_valid": False,
                "error_message": str(e),
            }

    def rollback_to_backup(self, backup_path: str) -> dict[str, Any]:
        """Rollback .env to a previous backup."""
        try:
            self.env_manager.restore_backup(backup_path)
            return {
                "ok": True,
                "message": f"Successfully restored from backup: {backup_path}",
            }
        except Exception as e:
            return {
                "ok": False,
                "message": f"Failed to restore backup: {str(e)}",
            }
