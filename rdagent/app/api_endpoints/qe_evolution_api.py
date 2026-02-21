"""
QE (QuantEvolver) 演进 API 端点

提供以下接口：
1. POST /api/v1/qe_workspace/tasks/{task_id}/loops - 触发新 LOOP 的回测执行
2. GET /api/v1/qe_workspace/loops/{loop_id}/status - 查询 LOOP 状态
3. GET /api/v1/qe_workspace/loops/{loop_id}/metrics - 获取 LOOP 回测指标
4. GET /api/v1/qe_workspace/loops/{loop_id}/assets/download - 打包下载模型资产
5. DELETE /api/v1/qe_workspace/tasks/{task_id} - 清理任务工作区
"""

import logging
import os
import shutil
import zipfile
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/qe_workspace", tags=["qe_evolution"])

# Workspace base directory for evolution tasks
WORKSPACE_BASE = Path(os.environ.get("QE_WORKSPACE_BASE", "/tmp/qe_evolution_workspace"))

class LoopRunRequest(BaseModel):
    loop_index: int
    config: Dict[str, Any]

class LoopRunResponse(BaseModel):
    loop_id: str
    status: str
    message: str

def _get_task_dir(task_id: str) -> Path:
    return WORKSPACE_BASE / task_id

def _get_loop_dir(task_id: str, loop_id: str) -> Path:
    return _get_task_dir(task_id) / loop_id

async def _run_qlib_backtest(task_id: str, loop_id: str, config: Dict[str, Any]):
    """
    后台任务：执行 QLib 回测
    （实际应调用 rdagent 的核心逻辑生成代码并跑实验，这里先做 Mock 框架）
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    os.makedirs(loop_dir, exist_ok=True)
    
    # 记录状态为 running
    status_file = loop_dir / "status.txt"
    status_file.write_text("running")
    
    # 保存配置
    config_file = loop_dir / "config.json"
    with open(config_file, "w") as f:
        import json
        json.dump(config, f)
        
    try:
        # TODO: 集成真正的 RDAgent 代码生成和回测逻辑
        logger.info(f"Starting QLib backtest for {loop_id} with config: {config}")
        await asyncio.sleep(5)  # 模拟执行耗时
        
        # 模拟生成指标文件
        metrics = {
            "IC": 0.054 + (0.001 * hash(loop_id) % 10),
            "ICIR": 0.68 + (0.01 * hash(loop_id) % 10),
            "Annualized Return": 0.15,
            "Max Drawdown": -0.124
        }
        with open(loop_dir / "qlib_res.json", "w") as f:
            json.dump(metrics, f)
            
        # 模拟生成模型资产
        models_dir = loop_dir / "models"
        os.makedirs(models_dir, exist_ok=True)
        (models_dir / "model.pkl").write_text("mock model content")
        (loop_dir / "features_order.txt").write_text("factor_1\nfactor_2")
        
        # 记录状态为 completed
        status_file.write_text("completed")
        logger.info(f"Completed QLib backtest for {loop_id}")
        
    except Exception as e:
        logger.error(f"Backtest failed for {loop_id}: {e}")
        status_file.write_text("failed")
        (loop_dir / "error.log").write_text(str(e))

@router.post("/tasks/{task_id}/loops", response_model=LoopRunResponse)
async def create_and_run_loop(task_id: str, request: LoopRunRequest, background_tasks: BackgroundTasks):
    """
    接收演进配置并触发 QLib 回测
    """
    loop_id = f"{task_id}_L{request.loop_index}"
    
    try:
        # 启动后台回测任务
        background_tasks.add_task(_run_qlib_backtest, task_id, loop_id, request.config)
        
        return LoopRunResponse(
            loop_id=loop_id,
            status="accepted",
            message=f"Loop {loop_id} accepted and running in background"
        )
    except Exception as e:
        logger.error(f"Failed to trigger loop {loop_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loops/{loop_id}/status")
async def get_loop_status(loop_id: str):
    """
    查询 LOOP 状态
    """
    # 从 loop_id 中解析 task_id (如 Evo_1234_L0)
    parts = loop_id.rsplit("_L", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid loop_id format")
    
    task_id = parts[0]
    loop_dir = _get_loop_dir(task_id, loop_id)
    status_file = loop_dir / "status.txt"
    
    if not status_file.exists():
        return {"status": "not_found"}
        
    status = status_file.read_text().strip()
    return {"status": status}

@router.get("/loops/{loop_id}/metrics")
async def get_loop_metrics(loop_id: str):
    """
    获取 LOOP 回测指标
    """
    parts = loop_id.rsplit("_L", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid loop_id format")
    
    task_id = parts[0]
    loop_dir = _get_loop_dir(task_id, loop_id)
    res_file = loop_dir / "qlib_res.json"
    
    if not res_file.exists():
        # Fallback to Mock if not finished or using mock flow
        return {
            "IC": 0.054,
            "ICIR": 0.68,
            "Annualized Return": 0.15,
            "Max Drawdown": -0.124
        }
        
    import json
    with open(res_file, "r") as f:
        return json.load(f)

@router.get("/loops/{loop_id}/assets/download")
async def download_loop_assets(loop_id: str):
    """
    模型资产打包(ZIP)下载
    """
    parts = loop_id.rsplit("_L", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid loop_id format")
        
    task_id = parts[0]
    loop_dir = _get_loop_dir(task_id, loop_id)
    
    if not loop_dir.exists():
        raise HTTPException(status_code=404, detail="Loop workspace not found")
        
    zip_path = loop_dir / f"{loop_id}_assets.zip"
    
    # Create zip file containing models/ and features_order.txt
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            models_dir = loop_dir / "models"
            if models_dir.exists():
                for root, _, files in os.walk(models_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, loop_dir)
                        zipf.write(file_path, arcname)
                        
            features_file = loop_dir / "features_order.txt"
            if features_file.exists():
                zipf.write(features_file, "features_order.txt")
                
    except Exception as e:
        logger.error(f"Failed to create assets zip for {loop_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create zip package")
        
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="No assets found to download")
        
    return FileResponse(
        path=zip_path,
        filename=f"{loop_id}_assets.zip",
        media_type="application/zip"
    )

@router.delete("/tasks/{task_id}")
async def cleanup_task_workspace(task_id: str):
    """
    彻底删除任务工作区
    """
    task_dir = _get_task_dir(task_id)
    if task_dir.exists():
        try:
            shutil.rmtree(task_dir)
            return {"status": "success", "message": f"Workspace {task_id} cleaned up"}
        except Exception as e:
            logger.error(f"Failed to clean up workspace {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "message": f"Workspace {task_id} not found, assumed clean"}
