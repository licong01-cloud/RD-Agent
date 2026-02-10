"""健康检查和诊断端点"""
import logging
import os
from pathlib import Path

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["Health"])


@router.get("/ping")
async def ping():
    """简单的ping端点"""
    return {"status": "ok", "message": "pong"}


@router.get("/log-root")
async def check_log_root():
    """检查log目录配置"""
    env_val = os.environ.get("RDAGENT_LOG_ROOT")
    repo_root = Path(__file__).resolve().parents[3]
    log_root = Path(env_val) if env_val else repo_root / "log"
    
    return {
        "env_rdagent_log_root": env_val,
        "computed_log_root": str(log_root),
        "log_root_exists": log_root.exists(),
        "repo_root": str(repo_root)
    }
