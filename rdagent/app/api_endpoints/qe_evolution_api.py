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
import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/qe_workspace", tags=["qe_evolution"])

# Workspace base directory for evolution tasks
WORKSPACE_BASE = Path(os.environ.get("QE_WORKSPACE_BASE", "/tmp/qe_evolution_workspace"))

class LoopRunRequest(BaseModel):
    loop_index: int
    config: Dict[str, Any]
    experiment_files: Optional[Dict[str, str]] = None
    wsl_command: Optional[str] = None

class LoopRunResponse(BaseModel):
    loop_id: str
    status: str
    message: str

def _get_task_dir(task_id: str) -> Path:
    return WORKSPACE_BASE / task_id

def _get_loop_dir(task_id: str, loop_id: str) -> Path:
    return _get_task_dir(task_id) / loop_id

def _append_log(loop_dir: Path, message: str):
    os.makedirs(loop_dir, exist_ok=True)
    log_file = loop_dir / "run.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

async def _run_qlib_backtest(task_id: str, loop_id: str, config: Dict[str, Any], experiment_files: Optional[Dict[str, str]], wsl_command: Optional[str]):
    """
    后台任务：执行 QLib 回测
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    os.makedirs(loop_dir, exist_ok=True)
    
    # 记录状态为 running
    status_file = loop_dir / "status.txt"
    status_file.write_text("running")
    _append_log(loop_dir, f"[START] loop={loop_id} status=running")
    
    # 保存配置
    config_file = loop_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f)
        
    try:
        logger.info(f"Starting QLib backtest for {loop_id} with config: {config}")
        _append_log(loop_dir, f"[INFO] Starting QLib backtest with config={json.dumps(config, ensure_ascii=False)}")
        
        # 写入实验文件
        if experiment_files:
            for rel_path, content in experiment_files.items():
                file_path = loop_dir / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                _append_log(loop_dir, f"[INFO] Wrote experiment file: {rel_path}")

        # 构造执行命令
        # 由于在 RDAgent 中运行，可以直接在 loop_dir 下执行
        cmd_parts = [f"cd {loop_dir}"]
        
        if (loop_dir / "prepare_factors.py").exists():
            cmd_parts.append("python prepare_factors.py")
            
        cmd_parts.append("qrun conf.yaml")
        
        # 将环境变量注入
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{loop_dir}:{env.get('PYTHONPATH', '')}"
        
        final_cmd = " && ".join(cmd_parts)
        _append_log(loop_dir, f"[INFO] Executing command: {final_cmd}")
        
        # 执行子进程
        process = await asyncio.create_subprocess_shell(
            final_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=loop_dir
        )
        
        # 读取输出到日志
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            _append_log(loop_dir, line.decode('utf-8', errors='replace').rstrip())
            
        await process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"QLib backtest failed with return code {process.returncode}")
        
        # qlib_res.json 和图表分析脚本会由 read_exp_res.py 之类的脚本生成？
        # 如果需要，我们可以主动跑一下 read_exp_res.py
        if (loop_dir / "read_exp_res.py").exists():
            _append_log(loop_dir, "[INFO] Running read_exp_res.py to generate metrics...")
            res_process = await asyncio.create_subprocess_shell(
                "python read_exp_res.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=loop_dir
            )
            while True:
                line = await res_process.stdout.readline()
                if not line:
                    break
                _append_log(loop_dir, line.decode('utf-8', errors='replace').rstrip())
            await res_process.wait()
            
        # 记录状态为 completed
        status_file.write_text("completed")
        _append_log(loop_dir, f"[DONE] loop={loop_id} status=completed")
        logger.info(f"Completed QLib backtest for {loop_id}")
        
    except Exception as e:
        logger.error(f"Backtest failed for {loop_id}: {e}")
        status_file.write_text("failed")
        (loop_dir / "error.log").write_text(str(e))
        _append_log(loop_dir, f"[ERROR] loop={loop_id} status=failed error={str(e)}")

@router.get("/tasks/{task_id}/logs")
async def stream_task_logs(task_id: str):
    """
    输出任务日志流（SSE），供 AIstock 侧转发展示。
    """
    task_dir = _get_task_dir(task_id)

    async def event_generator():
        seen_offsets: Dict[str, int] = {}
        while True:
            if not task_dir.exists():
                payload = {"status": "waiting", "logs": [f"Task directory not found yet: {task_id}"]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(1)
                continue

            loop_dirs = sorted([p for p in task_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
            for loop_dir in loop_dirs:
                log_file = loop_dir / "run.log"
                if not log_file.exists():
                    continue

                file_key = str(log_file)
                offset = seen_offsets.get(file_key, 0)
                with open(log_file, "r", encoding="utf-8") as f:
                    f.seek(offset)
                    new_lines = [line.rstrip("\n") for line in f.readlines()]
                    seen_offsets[file_key] = f.tell()

                if new_lines:
                    payload = {"status": "running", "logs": [f"[{loop_dir.name}] {line}" for line in new_lines]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
        raise HTTPException(status_code=404, detail="Metrics not ready")
        
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
