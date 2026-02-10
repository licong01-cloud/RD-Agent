"""
Alpha基线因子API端点

提供以下接口：
1. GET /api/extractors/alpha_baseline - 获取API说明（需要提供task_id才能获取真实因子）
2. GET /api/extractors/alpha_baseline/{task_id} - 获取task的真实Alpha基线因子信息（从本地model_meta.json读取）
3. POST /api/extractors/alpha_baseline/{task_id} - 通过上传model_meta.json获取Alpha基线因子信息（推荐，避免文件系统依赖）
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..extractors import AlphaBaselineExtractor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/extractors/alpha_baseline", tags=["Alpha Baseline"])

def _get_log_root() -> Path:
    """获取RD-Agent log目录"""
    env_val = os.environ.get("RDAGENT_LOG_ROOT")
    if env_val:
        return Path(env_val)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "log"

RDAGENT_LOG_ROOT = str(_get_log_root())


class AlphaBaselineResponse(BaseModel):
    """Alpha基线因子响应"""
    task_id: Optional[str] = None
    success: bool
    alpha_baseline_factors: List[str]
    factor_count: int
    model_meta: Optional[Dict] = None
    source: Optional[str] = None
    error: Optional[str] = None


class AlphaBaselineRequest(BaseModel):
    """Alpha基线因子请求（通过POST上传model_meta）"""
    model_meta: Dict
    task_id: Optional[str] = None


@router.get("/", response_model=AlphaBaselineResponse)
async def get_alpha_baseline_info():
    """
    获取Alpha基线因子API说明
    
    注意：本API不再返回固定的Alpha因子列表。
    请使用 /api/extractors/alpha_baseline/{task_id} 端点获取特定task的真实Alpha158因子。
    真实因子从model_meta.json的FilterCol配置中提取。
    
    Returns:
        API说明信息
    """
    return AlphaBaselineResponse(
        success=False,
        alpha_baseline_factors=[],
        factor_count=0,
        error="请使用 /api/extractors/alpha_baseline/{task_id} 端点获取特定task的真实Alpha158因子（从model_meta.json的FilterCol配置中提取）"
    )


@router.get("/{task_id}", response_model=AlphaBaselineResponse)
async def get_task_alpha_baseline(task_id: str):
    """
    获取task的真实Alpha基线因子信息（从本地文件系统读取）
    
    从model_meta.json的dataset_conf.kwargs.handler.kwargs.infer_processors.FilterCol.col_list
    提取模型训练时真实使用的Alpha158因子。
    
    注意：如果RD-Agent在WSL中运行而文件在Windows上，可能无法找到文件。
    建议使用POST端点直接上传model_meta数据。
    
    Args:
        task_id: task ID
        
    Returns:
        Alpha基线因子信息（包含真实的FilterCol.col_list）
    """
    try:
        extractor = AlphaBaselineExtractor(RDAGENT_LOG_ROOT)
        result = extractor.extract_task_alpha_baseline(task_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error", "提取失败"))
        
        return AlphaBaselineResponse(**result)
        
    except Exception as e:
        logger.error(f"获取task {task_id} Alpha基线因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{task_id}", response_model=AlphaBaselineResponse)
async def post_task_alpha_baseline(task_id: str, request: AlphaBaselineRequest):
    """
    通过上传model_meta.json获取Alpha基线因子信息（推荐）
    
    从上传的model_meta数据中的FilterCol配置提取Alpha158因子。
    此方法避免了对本地文件系统的依赖，适用于RD-Agent和AIstock分离部署的场景。
    
    Args:
        task_id: task ID
        request: 包含model_meta的请求体
        
    Returns:
        Alpha基线因子信息（包含真实的FilterCol.col_list）
    """
    try:
        extractor = AlphaBaselineExtractor(RDAGENT_LOG_ROOT)
        
        # 直接使用上传的model_meta数据
        model_meta = request.model_meta
        if not model_meta:
            raise HTTPException(status_code=400, detail="请求体中缺少model_meta字段")
        
        # 从上传的model_meta中提取Alpha158因子
        try:
            alpha_factors = extractor.extract_alpha_factors_from_model_meta(model_meta)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"从model_meta提取Alpha158因子失败: {e}")
        
        result = {
            "task_id": task_id,
            "success": True,
            "alpha_baseline_factors": alpha_factors,
            "factor_count": len(alpha_factors),
            "model_meta": model_meta,
            "source": "uploaded_model_meta",
            "error": None
        }
        
        logger.info(f"[POST] task {task_id} 成功提取{len(alpha_factors)}个Alpha158因子")
        return AlphaBaselineResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[POST] 获取task {task_id} Alpha基线因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
