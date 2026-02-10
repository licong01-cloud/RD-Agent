"""
SOTA因子API端点

提供以下接口：
1. GET /api/extractors/sota_factors/{task_id} - 获取单个task的SOTA因子(V1旧接口)
2. POST /api/extractors/sota_factors/batch - 批量获取多个task的SOTA因子(V1旧接口)
3. GET /api/extractors/sota_factors/v2/{task_id} - V2: 基于based_experiments链对齐parquet的SOTA因子
4. POST /api/extractors/sota_factors/v2/batch - V2: 批量获取对齐的SOTA因子
5. GET /api/extractors/sota_factors/v2/{task_id}/aligned - V2: 获取完整对齐因子(含源代码+回测数据)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..extractors import SOTAFactorsExtractor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/extractors/sota_factors", tags=["SOTA Factors"])

# 获取RD-Agent log目录
def _get_log_root() -> Path:
    """获取RD-Agent log目录"""
    env_val = os.environ.get("RDAGENT_LOG_ROOT")
    if env_val:
        return Path(env_val)
    # 使用相对于当前文件的路径
    # rdagent/app/api_endpoints/sota_factors_api.py -> repo_root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "log"

RDAGENT_LOG_ROOT = str(_get_log_root())


class BatchSOTAFactorsRequest(BaseModel):
    """批量获取SOTA因子请求"""
    task_ids: List[str]


class FactorMetrics(BaseModel):
    """因子回测指标"""
    ic: Optional[float] = None
    annualized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    information_ratio: Optional[float] = None
    sharpe: Optional[float] = None


class FactorMetricsV2(BaseModel):
    """V2因子回测指标(更完整)"""
    ic: Optional[float] = None
    icir: Optional[float] = None
    rank_ic: Optional[float] = None
    rank_icir: Optional[float] = None
    annualized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    information_ratio: Optional[float] = None
    sharpe: Optional[float] = None
    annualized_return_with_cost: Optional[float] = None
    max_drawdown_with_cost: Optional[float] = None
    information_ratio_with_cost: Optional[float] = None


class FactorDetailItem(BaseModel):
    """因子详情项"""
    factor_name: str
    factor_formulation: str
    factor_description: str
    source_loop_id: int
    source_loop_tag: str
    metrics: FactorMetrics


class FactorDetailItemV2(BaseModel):
    """V2因子详情项"""
    factor_name: str
    factor_formulation: str
    factor_description: str
    source_loop_id: int
    source_loop_tag: str
    metrics: FactorMetricsV2


class SOTAFactorResponse(BaseModel):
    """SOTA因子响应"""
    task_id: str
    success: bool
    sota_factors: List[str]
    factor_codes: Dict[str, str]
    factor_details: Dict[str, FactorDetailItem]
    factor_count: int
    error: Optional[str] = None


class SOTAFactorResponseV2(BaseModel):
    """V2 SOTA因子响应(兼容旧格式)"""
    task_id: str
    success: bool
    sota_factors: List[str]
    factor_codes: Dict[str, str]
    factor_details: Dict[str, FactorDetailItemV2]
    factor_count: int
    error: Optional[str] = None


class AlignedFactorItem(BaseModel):
    """对齐因子详情"""
    factor_name: str
    original_name: str
    source_code: Optional[str] = None
    source_code_length: int = 0
    source_from: str = ""
    factor_formulation: str = ""
    factor_description: str = ""
    variables: Optional[Dict] = None
    metrics: Dict[str, Any] = {}


class AlignedSOTAFactorsResponse(BaseModel):
    """V2对齐SOTA因子完整响应"""
    task_id: str
    success: bool
    aligned_factors: Dict[str, AlignedFactorItem]
    factor_names: List[str]
    factor_count: int
    last_sota_hist_index: Optional[int] = None
    based_experiments_count: int = 0
    error: Optional[str] = None


class BatchSOTAFactorsResponse(BaseModel):
    """批量SOTA因子响应"""
    total: int
    success_count: int
    results: List[SOTAFactorResponse]


class BatchSOTAFactorsResponseV2(BaseModel):
    """V2批量SOTA因子响应"""
    total: int
    success_count: int
    results: List[SOTAFactorResponseV2]


# ================================================================
# V2 新接口 (基于based_experiments链对齐parquet)
# 注意: V2固定路径路由必须在V1通配符路由/{task_id}之前注册
# ================================================================

@router.get("/v2/{task_id}/aligned", response_model=AlignedSOTAFactorsResponse)
async def get_task_aligned_sota_factors(task_id: str):
    """
    V2: 获取与parquet完全对齐的SOTA因子完整信息
    
    基于based_experiments链提取, 包含每个因子的:
    - 源代码(factor.py)
    - 进入SOTA时的回测数据(exp.result)
    - 因子详情(formulation, description, variables)
    
    Args:
        task_id: task ID
        
    Returns:
        对齐的SOTA因子完整信息
    """
    try:
        extractor = SOTAFactorsExtractor(RDAGENT_LOG_ROOT)
        result = extractor.extract_aligned_sota_factors(task_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error", "提取失败"))
        
        # 转换aligned_factors中的dict为AlignedFactorItem
        aligned_items = {}
        for fname, info in result["aligned_factors"].items():
            aligned_items[fname] = AlignedFactorItem(**info)
        
        return AlignedSOTAFactorsResponse(
            task_id=result["task_id"],
            success=result["success"],
            aligned_factors=aligned_items,
            factor_names=result["factor_names"],
            factor_count=result["factor_count"],
            last_sota_hist_index=result.get("last_sota_hist_index"),
            based_experiments_count=result.get("based_experiments_count", 0),
            error=result.get("error"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2获取task {task_id} 对齐SOTA因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/{task_id}", response_model=SOTAFactorResponseV2)
async def get_task_sota_factors_v2(task_id: str):
    """
    V2: 获取单个task的SOTA因子(兼容V1格式, 内部使用based_experiments链对齐逻辑)
    
    Args:
        task_id: task ID
        
    Returns:
        SOTA因子信息(V2格式)
    """
    try:
        extractor = SOTAFactorsExtractor(RDAGENT_LOG_ROOT)
        result = extractor.extract_task_sota_factors_v2(task_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error", "提取失败"))
        
        return SOTAFactorResponseV2(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2获取task {task_id} SOTA因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/batch", response_model=BatchSOTAFactorsResponseV2)
async def get_batch_sota_factors_v2(request: BatchSOTAFactorsRequest):
    """
    V2: 批量获取多个task的SOTA因子(基于based_experiments链对齐)
    
    Args:
        request: 包含task_ids列表的请求
        
    Returns:
        批量SOTA因子信息(V2格式)
    """
    try:
        extractor = SOTAFactorsExtractor(RDAGENT_LOG_ROOT)
        results = []
        success_count = 0
        
        for task_id in request.task_ids:
            result = extractor.extract_task_sota_factors_v2(task_id)
            results.append(SOTAFactorResponseV2(**result))
            if result["success"]:
                success_count += 1
        
        return BatchSOTAFactorsResponseV2(
            total=len(request.task_ids),
            success_count=success_count,
            results=results
        )
        
    except Exception as e:
        logger.error(f"V2批量获取SOTA因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# V1 旧接口 (保持向后兼容)
# 注意: /{task_id}通配符路由必须在所有固定路径路由之后注册
# ================================================================

@router.post("/batch", response_model=BatchSOTAFactorsResponse)
async def get_batch_sota_factors(request: BatchSOTAFactorsRequest):
    """
    批量获取多个task的SOTA因子(V1旧接口)
    
    Args:
        request: 包含task_ids列表的请求
        
    Returns:
        批量SOTA因子信息
    """
    try:
        extractor = SOTAFactorsExtractor(RDAGENT_LOG_ROOT)
        results = []
        success_count = 0
        
        for task_id in request.task_ids:
            result = extractor.extract_task_sota_factors(task_id)
            results.append(SOTAFactorResponse(**result))
            if result["success"]:
                success_count += 1
        
        return BatchSOTAFactorsResponse(
            total=len(request.task_ids),
            success_count=success_count,
            results=results
        )
        
    except Exception as e:
        logger.error(f"批量获取SOTA因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}", response_model=SOTAFactorResponse)
async def get_task_sota_factors(task_id: str):
    """
    获取单个task的SOTA因子(V1旧接口)
    
    Args:
        task_id: task ID
        
    Returns:
        SOTA因子信息
    """
    try:
        extractor = SOTAFactorsExtractor(RDAGENT_LOG_ROOT)
        result = extractor.extract_task_sota_factors(task_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error", "提取失败"))
        
        return SOTAFactorResponse(**result)
        
    except Exception as e:
        logger.error(f"获取task {task_id} SOTA因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
