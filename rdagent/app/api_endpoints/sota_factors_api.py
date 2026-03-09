"""
SOTA因子API端点

提供以下接口：
1. GET /api/extractors/sota_factors/{task_id} - 获取单个task的SOTA因子(V1旧接口)
2. POST /api/extractors/sota_factors/batch - 批量获取多个task的SOTA因子(V1旧接口)
3. GET /api/extractors/sota_factors/v2/{task_id} - V2: 基于based_experiments链对齐parquet的SOTA因子
4. POST /api/extractors/sota_factors/v2/batch - V2: 批量获取对齐的SOTA因子
5. GET /api/extractors/sota_factors/v2/{task_id}/aligned - V2: 获取完整对齐因子(含源代码+回测数据)
6. GET /api/extractors/sota_factors/v2/{task_id}/factor_metrics - 计算单个task的17项单因子指标
7. POST /api/extractors/sota_factors/v2/batch/factor_metrics - 批量计算多个task的因子指标
"""

import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..extractors import SOTAFactorsExtractor
from ..path_utils import normalize_path

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

# 因子指标计算专用进程池，隔离 GIL 避免阻塞其他请求
_metrics_process_pool = ProcessPoolExecutor(max_workers=8)


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
# 单因子指标 (17项) 请求/响应模型
# ================================================================

class SingleFactorMetricsItem(BaseModel):
    """单因子单窗口的17项指标"""
    factor_name: str
    eval_window: str
    data_start: str
    data_end: str
    return_horizon: str = "T2T1"
    universe: str = "all"
    calculated_at: str
    calc_batch_id: Optional[str] = None
    # Signal quality
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    rank_ic_std: Optional[float] = None
    icir: Optional[float] = None
    rank_icir: Optional[float] = None
    ic_positive_ratio: Optional[float] = None
    ic_csz_mean: Optional[float] = None
    # Long-only portfolio (A股纯多头)
    top_annual_return: Optional[float] = None
    top_excess_annual_return: Optional[float] = None
    top_sharpe: Optional[float] = None
    top_max_drawdown: Optional[float] = None
    top_excess_sharpe: Optional[float] = None
    benchmark_annual_return: Optional[float] = None
    # Other
    group_return_monotonicity: Optional[float] = None
    turnover: Optional[float] = None
    ic_decay_half_life: Optional[float] = None
    coverage: Optional[float] = None
    n_trading_days: Optional[int] = None


class FactorReportItem(BaseModel):
    """单因子单窗口的计算状态报告"""
    factor_name: str
    eval_window: str                         # full/out_sample/recent_6m/recent_3m
    status: str                              # ok/skipped/error
    error_message: Optional[str] = None
    n_trading_days: Optional[int] = None
    required_days: Optional[int] = None
    data_start: Optional[str] = None
    data_end: Optional[str] = None
    data_source: str = "parquet"
    calc_engine: str = "rdagent"
    calculated_at: Optional[str] = None


class FactorMetricsResponse(BaseModel):
    """单个task的因子指标响应"""
    task_id: str
    success: bool
    factor_count: int = 0
    metrics_count: int = 0
    parquet_path: Optional[str] = None
    metrics: List[SingleFactorMetricsItem] = []
    factor_reports: List[FactorReportItem] = []
    calc_batch_id: Optional[str] = None
    error: Optional[str] = None


class BatchFactorMetricsRequest(BaseModel):
    """批量因子指标计算请求"""
    task_ids: List[str]


class BatchFactorMetricsResponse(BaseModel):
    """批量因子指标响应"""
    total: int
    success_count: int
    results: List[FactorMetricsResponse]


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
        result = await asyncio.to_thread(extractor.extract_aligned_sota_factors, task_id)

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


@router.get("/v2/{task_id}/loops/{loop_index}/factors")
async def get_loop_factors(task_id: str, loop_index: int):
    """提取指定 Loop 的所有 final_decision=True 的因子代码和指标。

    Args:
        task_id: task ID
        loop_index: Loop 索引 (0-based)

    Returns:
        包含因子列表和 Loop 整体回测指标的字典
    """
    try:
        extractor = SOTAFactorsExtractor(RDAGENT_LOG_ROOT)
        result = await asyncio.to_thread(extractor.extract_loop_factors, task_id, loop_index)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "未知错误"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取 task {task_id} loop {loop_index} 因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/{task_id}/loops/{loop_index}/factor_metrics", response_model=FactorMetricsResponse)
async def get_loop_factor_metrics(task_id: str, loop_index: int):
    """计算指定 Loop 的所有因子的 17 项独立单因子指标。

    从该 Loop 的 experiment_workspace 中读取 combined_factors_df.parquet，
    使用与 SOTA factor_metrics 完全相同的计算引擎。

    适用于非 SOTA Loop 的因子独立指标计算。

    Args:
        task_id: RD-Agent task ID
        loop_index: Loop 索引 (0-based)
    """
    try:
        extractor = SOTAFactorsExtractor(RDAGENT_LOG_ROOT)
        parquet_path = await asyncio.to_thread(
            extractor.get_loop_parquet_path, task_id, loop_index
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"定位 task {task_id} loop {loop_index} parquet 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _metrics_process_pool, _compute_factor_metrics_from_parquet, str(parquet_path), task_id
    )
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return result


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
        result = await asyncio.to_thread(extractor.extract_task_sota_factors_v2, task_id)

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
            result = await asyncio.to_thread(extractor.extract_task_sota_factors_v2, task_id)
            results.append(SOTAFactorResponseV2(**result))
            if result["success"]:
                success_count += 1
        
        return BatchSOTAFactorsResponseV2(
            total=len(request.task_ids),
            success_count=success_count,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2批量获取SOTA因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# 单因子指标计算接口 (17项指标)
# ================================================================

def _resolve_parquet_path(task_id: str) -> Path:
    """从task_id精确定位combined_factors_df.parquet路径。

    复用SOTA因子提取的定位逻辑：找到hist中最后一个decision=True的因子实验，
    该实验的experiment_workspace即为包含parquet的回测workspace。
    """
    extractor = SOTAFactorsExtractor(RDAGENT_LOG_ROOT)
    session = extractor.load_session(task_id)
    if session is None:
        raise FileNotFoundError(f"无法加载task {task_id} 的session")

    if not hasattr(session, 'trace') or not hasattr(session.trace, 'hist'):
        raise ValueError(f"task {task_id} 的session缺少trace.hist")

    hist = session.trace.hist

    # 与extract_aligned_sota_factors相同逻辑：从后往前找最后一个decision=True的因子实验
    last_sota_idx = None
    for i in range(len(hist) - 1, -1, -1):
        exp, feedback = hist[i]
        if not extractor._is_factor_experiment(exp):
            continue
        decision = getattr(feedback, "decision", None) if feedback else None
        if decision is True:
            last_sota_idx = i
            break

    if last_sota_idx is None:
        raise FileNotFoundError(
            f"task {task_id} 未找到decision=True的因子实验，无法定位parquet"
        )

    sota_exp, _ = hist[last_sota_idx]
    if not hasattr(sota_exp, 'experiment_workspace'):
        raise ValueError(f"task {task_id} loop {last_sota_idx} 没有experiment_workspace")

    ws = sota_exp.experiment_workspace
    if not hasattr(ws, 'workspace_path') or not ws.workspace_path:
        raise ValueError(f"task {task_id} loop {last_sota_idx} 的workspace_path为空")

    ws_path = normalize_path(ws.workspace_path)
    parquet_file = ws_path / "combined_factors_df.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(
            f"task {task_id} SOTA实验(loop {last_sota_idx})的parquet不存在: {parquet_file}"
        )

    return parquet_file


def _compute_task_factor_metrics(task_id: str) -> FactorMetricsResponse:
    """计算单个task的所有因子指标，返回响应对象。"""
    try:
        parquet_path = _resolve_parquet_path(task_id)
    except Exception as e:
        logger.warning(f"resolve parquet for {task_id} failed: {e}")
        return FactorMetricsResponse(
            task_id=task_id, success=False, error=str(e)
        )

    try:
        from ..factor_metrics.engine import compute_all_factors_metrics
        engine_result = compute_all_factors_metrics(parquet_path)

        metrics_list = engine_result["metrics"]
        factor_reports_list = engine_result.get("factor_reports", [])
        calc_batch_id = engine_result.get("calc_batch_id")

        metrics_items = [SingleFactorMetricsItem(**r) for r in metrics_list]
        report_items = [FactorReportItem(**r) for r in factor_reports_list]
        factor_names = set(r["factor_name"] for r in metrics_list)

        return FactorMetricsResponse(
            task_id=task_id,
            success=True,
            factor_count=len(factor_names),
            metrics_count=len(metrics_items),
            parquet_path=str(parquet_path),
            metrics=metrics_items,
            factor_reports=report_items,
            calc_batch_id=calc_batch_id,
        )
    except Exception as e:
        logger.error(f"计算task {task_id} 因子指标失败: {e}", exc_info=True)
        return FactorMetricsResponse(
            task_id=task_id, success=False, error=str(e)
        )


def _compute_factor_metrics_from_parquet(parquet_path_str: str, task_id: str) -> FactorMetricsResponse:
    """从指定 parquet 路径计算因子指标（进程池中执行）。

    与 _compute_task_factor_metrics 逻辑相同，但跳过 parquet 定位步骤，
    直接使用传入的路径。适用于 Loop 级指标计算。
    """
    from pathlib import Path
    parquet_path = Path(parquet_path_str)

    try:
        from ..factor_metrics.engine import compute_all_factors_metrics
        engine_result = compute_all_factors_metrics(parquet_path)

        metrics_list = engine_result["metrics"]
        factor_reports_list = engine_result.get("factor_reports", [])
        calc_batch_id = engine_result.get("calc_batch_id")

        metrics_items = [SingleFactorMetricsItem(**r) for r in metrics_list]
        report_items = [FactorReportItem(**r) for r in factor_reports_list]
        factor_names = set(r["factor_name"] for r in metrics_list)

        return FactorMetricsResponse(
            task_id=task_id,
            success=True,
            factor_count=len(factor_names),
            metrics_count=len(metrics_items),
            parquet_path=parquet_path_str,
            metrics=metrics_items,
            factor_reports=report_items,
            calc_batch_id=calc_batch_id,
        )
    except Exception as e:
        logger.error(f"从 parquet {parquet_path_str} 计算因子指标失败: {e}", exc_info=True)
        return FactorMetricsResponse(
            task_id=task_id, success=False, error=str(e)
        )


@router.get("/v2/{task_id}/factor_metrics", response_model=FactorMetricsResponse)
async def get_task_factor_metrics(task_id: str):
    """
    计算单个task的所有SOTA因子的17项单因子指标。

    从task的workspace中读取combined_factors_df.parquet，
    结合qlib_bin收盘价，计算IC/RankIC/ICIR/多空收益/夏普等17项指标。
    每个因子在4个评估窗口(full/out_sample/recent_6m/recent_3m)分别计算。

    Args:
        task_id: RD-Agent task ID

    Returns:
        所有因子在各窗口的指标列表
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_metrics_process_pool, _compute_task_factor_metrics, task_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return result


@router.post("/v2/batch/factor_metrics", response_model=BatchFactorMetricsResponse)
async def get_batch_factor_metrics(request: BatchFactorMetricsRequest):
    """
    批量计算多个task的因子指标。

    Args:
        request: 包含task_ids列表的请求

    Returns:
        每个task的因子指标计算结果
    """
    results = []
    success_count = 0

    loop = asyncio.get_running_loop()
    # 并行提交所有 task 到进程池（ProcessPoolExecutor max_workers=8 控制上限）
    futures = [
        loop.run_in_executor(_metrics_process_pool, _compute_task_factor_metrics, task_id)
        for task_id in request.task_ids
    ]
    results = await asyncio.gather(*futures, return_exceptions=True)

    # 处理异常结果
    final_results = []
    for task_id, result in zip(request.task_ids, results):
        if isinstance(result, Exception):
            final_results.append(FactorMetricsResponse(
                task_id=task_id,
                success=False,
                error=str(result),
                factor_count=0,
                metrics=[],
            ))
        else:
            final_results.append(result)
            if result.success:
                success_count += 1

    return BatchFactorMetricsResponse(
        total=len(request.task_ids),
        success_count=success_count,
        results=final_results,
    )


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
            result = await asyncio.to_thread(extractor.extract_task_sota_factors, task_id)
            results.append(SOTAFactorResponse(**result))
            if result["success"]:
                success_count += 1
        
        return BatchSOTAFactorsResponse(
            total=len(request.task_ids),
            success_count=success_count,
            results=results
        )
        
    except HTTPException:
        raise
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
        result = await asyncio.to_thread(extractor.extract_task_sota_factors, task_id)

        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error", "提取失败"))
        
        return SOTAFactorResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取task {task_id} SOTA因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
