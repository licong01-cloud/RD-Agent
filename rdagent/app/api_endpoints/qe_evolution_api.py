"""
QE (QuantEvolver) 演进 API 端点

提供以下接口（双参数嵌套路由）：
1. POST /api/v1/qe_workspace/tasks/{task_id}/loops - 触发新 LOOP 的回测执行
2. GET /api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/status - 查询 LOOP 状态
3. GET /api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/metrics - 获取 LOOP 回测指标
4. GET /api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/assets/download - 打包下载模型资产
5. DELETE /api/v1/qe_workspace/tasks/{task_id} - 清理任务工作区
6. GET /api/v1/qe_workspace/tasks/{task_id}/logs - 任务日志流（SSE）
7. GET /api/v1/qe_workspace/config - 工作区配置信息
"""

import logging
import os
import shutil
import zipfile
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 确保 .env 文件中的环境变量在模块加载时可用（热重载安全）
# override=False：不覆盖已存在的 shell 环境变量
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    if load_dotenv is not None:
        load_dotenv(_env_path, override=False)
    else:
        # python-dotenv 未安装时，手动解析 .env（仅补充缺失的环境变量）
        with open(_env_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip("'\"")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

router = APIRouter(prefix="/api/v1/qe_workspace", tags=["qe_evolution"])

# ── QE 专属配置 ──
# QE 与 RDAgent 主程序完全隔离，路径配置通过 RDAgent .env 环境变量管理。
# 未来部署到独立服务器时，通过启动命令或 .env 设置环境变量即可。
WORKSPACE_BASE = Path(os.environ.get("QE_WORKSPACE_WSL", ""))
if not str(WORKSPACE_BASE) or str(WORKSPACE_BASE) == ".":
    logger.warning("QE_WORKSPACE_WSL 环境变量未设置，QE API 的 workspace 功能将不可用")

class LoopRunRequest(BaseModel):
    loop_index: int
    config: Dict[str, Any]
    experiment_files: Optional[Dict[str, str]] = None
    wsl_command: Optional[str] = None
    callback_url: Optional[str] = None
    model_source: Optional[Dict[str, Any]] = None

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

async def _run_qlib_backtest(task_id: str, loop_id: str, config: Dict[str, Any], experiment_files: Optional[Dict[str, str]], wsl_command: Optional[str] = None, callback_url: Optional[str] = None, model_source: Optional[Dict[str, Any]] = None):
    """
    后台任务：执行 QLib 回测。

    关键设计：
    - start_new_session=True: 子进程在独立会话中运行，uvicorn --reload 不会杀死它
    - stdout 重定向到日志文件（非 PIPE）：父进程死后子进程不会因管道断裂收到 SIGPIPE
    - pid.txt 记录子进程真实 PID（非 FastAPI worker PID）：健康检查准确判断
    - read_exp_res.py 集成到命令链：父进程死后仍会自动执行生成结果文件
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
            import base64
            for rel_path, content in experiment_files.items():
                if rel_path.endswith(".b64"):
                    # base64 编码的二进制文件（如 benchmark_sh000300.parquet.b64）
                    actual_path = loop_dir / rel_path[:-4]  # 去掉 .b64 后缀
                    actual_path.parent.mkdir(parents=True, exist_ok=True)
                    actual_path.write_bytes(base64.b64decode(content))
                    _append_log(loop_dir, f"[INFO] Wrote binary file: {rel_path[:-4]} (decoded from b64)")
                else:
                    file_path = loop_dir / rel_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content, encoding="utf-8")
                    _append_log(loop_dir, f"[INFO] Wrote experiment file: {rel_path}")

        # 将环境变量注入，确保 model.py 等模块可被 qrun 导入
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{loop_dir}:{env.get('PYTHONPATH', '')}"
        env.setdefault("PYTHONUNBUFFERED", "1")

        # model_source mlruns 符号链接（策略演进复复用源任务训练的模型）
        if model_source:
            cross_node = model_source.get("cross_node", False)
            dst_mlruns = loop_dir / "mlruns"

            if cross_node:
                # 跨节点：从 experiment_files 中解压 mlruns_params.tar.gz
                tar_b64_file = loop_dir / "mlruns_params.tar.gz"
                if tar_b64_file.exists():
                    import tarfile, io
                    tar_data = tar_b64_file.read_bytes()
                    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
                        tar.extractall(path=loop_dir)
                    tar_b64_file.unlink()
                    _append_log(loop_dir, f"[INFO] Cross-node mlruns extracted to {dst_mlruns}")
                else:
                    msg = "Cross-node mode requires mlruns_params.tar.gz, but it was not provided"
                    status_file.write_text("failed")
                    _append_log(loop_dir, f"[ERROR] {msg}")
                    raise RuntimeError(msg)
            else:
                # 同节点：符号链接
                # qlib task_train 将 params.pkl 保存在 workspace 根级 mlruns/ 下，
                # 而 Loop*/mlruns/ 仅包含 QE 自身记录的元数据（无模型权重）。
                # 因此优先链接根级 mlruns，fallback 到 loop 级。
                src_task = model_source.get("source_task_id", "")
                src_loop = model_source.get("source_loop", "")
                if src_task and src_loop:
                    src_mlruns_root = WORKSPACE_BASE / src_task / "mlruns"
                    src_mlruns_loop = WORKSPACE_BASE / src_task / src_loop / "mlruns"
                    src_mlruns = src_mlruns_root if src_mlruns_root.exists() else src_mlruns_loop
                    if src_mlruns.exists() and not dst_mlruns.exists():
                        os.symlink(str(src_mlruns), str(dst_mlruns))
                        _append_log(loop_dir, f"[INFO] Symlink mlruns: {src_mlruns} → {dst_mlruns}")

        # 构造执行命令
        if wsl_command:
            # AIstock 传入的自定义 WSL 命令（auto 模式已包含 read_exp_res.py）
            final_cmd = wsl_command
            _append_log(loop_dir, f"[INFO] Using wsl_command: {final_cmd}")
        else:
            # 默认命令链：cd → prepare_factors → qrun → read_exp_res
            cmd_parts = [f"cd {loop_dir}"]
            if (loop_dir / "prepare_factors.py").exists():
                cmd_parts.append("python prepare_factors.py")
            cmd_parts.append("qrun conf.yaml")
            # 将 read_exp_res.py 集成到命令链，确保即使父进程重启也能生成结果文件
            if (loop_dir / "read_exp_res.py").exists():
                cmd_parts.append("python read_exp_res.py")
            final_cmd = " && ".join(cmd_parts)
            _append_log(loop_dir, f"[INFO] Executing command: {final_cmd}")

        # 将子进程输出重定向到日志文件（非 PIPE），确保子进程与父进程解耦
        log_file_path = loop_dir / "run.log"
        log_fd = os.open(str(log_file_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)

        # 使用 subprocess.Popen 而非 asyncio.create_subprocess_shell:
        # 显式 close_fds=True 防止 uvicorn server socket (fd) 泄漏到子进程，
        # 否则子进程持有端口导致 API 重启时 "Address already in use"
        import subprocess as _subprocess
        _proc = _subprocess.Popen(
            final_cmd,
            shell=True,
            stdout=log_fd,
            stderr=_subprocess.STDOUT,
            env=env,
            cwd=str(loop_dir),
            start_new_session=True,
            close_fds=True,
        )
        os.close(log_fd)  # 父进程关闭 fd 副本，子进程保留自己的副本

        # 写入子进程的真实 PID（而非 FastAPI worker PID）
        (loop_dir / "pid.txt").write_text(str(_proc.pid))
        _append_log(loop_dir, f"[INFO] Subprocess started, pid={_proc.pid}")

        # 在线程池中等待子进程完成（不阻塞事件循环）
        # 如果 uvicorn reload 导致父进程重启，此 await 会被取消，但子进程继续运行
        loop = asyncio.get_event_loop()
        returncode = await loop.run_in_executor(None, _proc.wait)

        if returncode != 0:
            raise RuntimeError(f"QLib backtest failed with return code {returncode}")

        # 记录状态为 completed
        status_file.write_text("completed")
        _append_log(loop_dir, f"[DONE] loop={loop_id} status=completed")
        logger.info(f"Completed QLib backtest for {loop_id}")

        # 通知 AIstock backend（Loop 完成回调）
        if callback_url:
            try:
                import httpx as _httpx
                _cb_payload = {
                    "loop_id": f"{task_id}_{loop_id}",
                    "task_id": task_id,
                    "status": "completed",
                }
                async with httpx.AsyncClient(timeout=10) as _cb:
                    await _cb.post(callback_url, json=_cb_payload)
                logger.info(f"Callback sent to {callback_url} for {loop_id}")
            except Exception as cb_err:
                logger.warning(f"Callback failed for {loop_id}: {cb_err}")

    except Exception as e:
        logger.error(f"Backtest failed for {loop_id}: {e}")
        # 仅在状态未被健康检查更新时标记为 failed
        try:
            current = status_file.read_text().strip() if status_file.exists() else ""
        except Exception:
            current = ""
        if current not in ("completed", "failed"):
            status_file.write_text("failed")
            (loop_dir / "error.log").write_text(str(e))
        _append_log(loop_dir, f"[ERROR] loop={loop_id} error={str(e)}")

@router.get("/tasks/{task_id}/loops/{loop_id}/mlruns-params")
async def download_mlruns_params(task_id: str, loop_id: str):
    """打包 loop 的 mlruns 中 params.pkl 及其目录结构，返回 tar.gz。"""
    import tarfile, io, glob as _glob
    loop_dir = _get_loop_dir(task_id, loop_id)
    mlruns_dir = loop_dir / "mlruns"
    if not mlruns_dir.exists():
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "mlruns not found"})
    params_files = _glob.glob(str(mlruns_dir / "**" / "params.pkl"), recursive=True)
    if not params_files:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "params.pkl not found"})
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for pf in params_files:
            arcname = os.path.relpath(pf, loop_dir)
            tar.add(pf, arcname=arcname)
    buf.seek(0)
    from fastapi.responses import Response
    return Response(content=buf.read(), media_type="application/gzip")

@router.get("/tasks/{task_id}/logs")
async def stream_task_logs(task_id: str):
    """
    输出任务日志流（SSE），供 AIstock 侧转发展示。
    """
    task_dir = _get_task_dir(task_id)

    async def event_generator():
        seen_offsets: Dict[str, int] = {}
        idle_count = 0
        _MAX_IDLE = 300  # 300秒无新日志则终止 SSE
        while True:
            if not task_dir.exists():
                payload = {"status": "waiting", "logs": [f"Task directory not found yet: {task_id}"]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(1)
                idle_count += 1
                if idle_count >= _MAX_IDLE:
                    payload = {"status": "timeout", "logs": ["SSE stream timeout: task directory not found"]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return
                continue

            had_new_lines = False
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
                    had_new_lines = True
                    payload = {"status": "running", "logs": [f"[{loop_dir.name}] {line}" for line in new_lines]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # 检查是否所有 loop 都到达终态（支持并行策略演进场景）
            if loop_dirs:
                all_terminal = True
                any_loop_has_status = False
                for ld in loop_dirs:
                    sf = ld / "status.txt"
                    if sf.exists():
                        any_loop_has_status = True
                        st = sf.read_text().strip()
                        if st not in ("completed", "failed"):
                            all_terminal = False
                            break
                    else:
                        all_terminal = False
                        break
                if any_loop_has_status and all_terminal:
                    final_status = "completed"
                    for ld in loop_dirs:
                        sf = ld / "status.txt"
                        if sf.exists() and sf.read_text().strip() == "failed":
                            final_status = "failed"
                            break
                    payload = {"status": final_status, "logs": [f"All loops finished with status: {final_status}"]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return

            if had_new_lines:
                idle_count = 0
            else:
                idle_count += 1
                if idle_count >= _MAX_IDLE:
                    payload = {"status": "timeout", "logs": ["SSE stream timeout: no new logs"]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/tasks/{task_id}/loops", response_model=LoopRunResponse)
async def create_and_run_loop(task_id: str, request: LoopRunRequest, background_tasks: BackgroundTasks):
    """
    接收演进配置并触发 QLib 回测
    """
    loop_id = f"Loop{request.loop_index}"
    
    try:
        # 启动后台回测任务
        background_tasks.add_task(
            _run_qlib_backtest, task_id, loop_id, request.config,
            request.experiment_files, request.wsl_command,
            request.callback_url, request.model_source,
        )
        
        return LoopRunResponse(
            loop_id=loop_id,
            status="accepted",
            message=f"Loop {loop_id} accepted and running in background"
        )
    except Exception as e:
        logger.error(f"Failed to trigger loop {loop_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}/loops/{loop_id}/status")
async def get_loop_status(task_id: str, loop_id: str):
    """
    查询 LOOP 状态。

    状态判断逻辑（准确判断，不做推测）：
    1. status.txt 存在且为 completed/failed → 直接返回（终态）
    2. status.txt 为 running → 检查 PID 是否存活：
       - PID 存活 → running
       - PID 不存活 + 结果文件存在 → completed（进程正常退出但 status.txt 未更新）
       - PID 不存活 + 无结果文件 → failed（进程异常终止）
    3. status.txt 不存在 → not_found
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    task_dir = _get_task_dir(task_id)
    status_file = loop_dir / "status.txt"

    if not status_file.exists():
        return {"status": "not_found"}

    status = status_file.read_text().strip()

    # 终态直接返回
    if status in ("completed", "failed"):
        return {"status": status}

    if status == "running":
        pid_file = loop_dir / "pid.txt"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)  # 不发信号，仅检查进程是否存在
                return {"status": "running"}
            except (ProcessLookupError, OSError):
                # PID 不存活，检查结果文件确定最终状态（loop 目录优先，兼容 task 目录）
                result_file = loop_dir / "qlib_results_enhanced.json"
                if not result_file.exists():
                    result_file = task_dir / "qlib_results_enhanced.json"
                if result_file.exists():
                    status = "completed"
                else:
                    status = "failed"
                status_file.write_text(status)
                _append_log(loop_dir, f"[DETECT] pid={pid} no longer alive, result_file_exists={result_file.exists()}, marking as {status}")
                return {"status": status}
        else:
            # 无 pid.txt，无法判断进程状态，保持当前状态
            return {"status": status}

    # 其他未知状态（如 interrupted），直接返回
    return {"status": status}

@router.post("/tasks/{task_id}/loops/{loop_id}/kill")
async def kill_loop(task_id: str, loop_id: str):
    """
    终止正在运行的 Loop 进程。

    通过 pid.txt 获取子进程 PID，发送 SIGTERM 给进程组（优雅终止）。
    如果进程未在 5 秒内退出，发送 SIGKILL（强制终止）。
    使用 asyncio.sleep 避免阻塞事件循环。
    """
    import signal
    loop_dir = _get_loop_dir(task_id, loop_id)
    pid_file = loop_dir / "pid.txt"
    status_file = loop_dir / "status.txt"

    if not pid_file.exists():
        raise HTTPException(status_code=404, detail=f"No pid.txt found for {task_id}/{loop_id}")

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid pid.txt: {e}")

    if pid <= 0:
        raise HTTPException(status_code=400, detail=f"Invalid PID value: {pid}")

    killed = False
    error_detail = None
    try:
        os.kill(pid, 0)  # Check if alive

        # 尝试杀进程组（包括子进程），失败则回退到单进程
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            _append_log(loop_dir, f"[KILL] Sent SIGTERM to process group pgid={pgid} (pid={pid})")
        except (ProcessLookupError, OSError):
            os.kill(pid, signal.SIGTERM)
            _append_log(loop_dir, f"[KILL] Sent SIGTERM to pid={pid} (pgid fallback)")

        # 非阻塞等待进程退出
        for _ in range(10):
            await asyncio.sleep(0.5)
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, OSError):
                killed = True
                break

        if not killed:
            # Force kill
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            _append_log(loop_dir, f"[KILL] Sent SIGKILL to pid={pid}")
            killed = True

    except (ProcessLookupError, OSError):
        # Process already dead
        killed = True
        _append_log(loop_dir, f"[KILL] pid={pid} already dead")
    except Exception as e:
        error_detail = str(e)
        _append_log(loop_dir, f"[KILL] Unexpected error killing pid={pid}: {e}")
        logger.error(f"Kill loop {task_id}/{loop_id} pid={pid} error: {e}")

    # Mark status as cancelled
    if status_file.exists():
        current = status_file.read_text().strip()
        if current == "running":
            status_file.write_text("cancelled")
            _append_log(loop_dir, f"[KILL] Marked status as cancelled")

    result = {"killed": killed, "pid": pid, "status": "cancelled"}
    if error_detail:
        result["error"] = error_detail
    return result

@router.get("/tasks/{task_id}/loops/{loop_id}/metrics")
async def get_loop_metrics(task_id: str, loop_id: str):
    """
    获取 LOOP 回测指标。

    read_exp_res.py 输出文件固定为：
    - qlib_results_enhanced.json（完整增强指标，含 summary 字段）
    搜索顺序：loop 目录（wsl_command cd 目标） → task 目录（兼容旧路径）。
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    task_dir = _get_task_dir(task_id)

    enhanced_file = None
    for search_dir in [loop_dir, task_dir]:
        candidate = search_dir / "qlib_results_enhanced.json"
        if candidate.exists():
            enhanced_file = candidate
            break

    if enhanced_file is None:
        raise HTTPException(status_code=404, detail="Metrics not ready: qlib_results_enhanced.json not found")

    with open(enhanced_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("summary", data)

@router.get("/tasks/{task_id}/loops/{loop_id}/enhanced-metrics")
async def get_loop_enhanced_metrics(task_id: str, loop_id: str):
    """
    获取 LOOP 增强诊断指标（IC 时间序列、训练过程、收益曲线等）。
    从 qlib_results_enhanced.json 提取诊断子段，文件不存在则 404。
    """
    loop_dir = _get_loop_dir(task_id, loop_id)
    task_dir = _get_task_dir(task_id)

    enhanced_file = None
    for search_dir in [loop_dir, task_dir]:
        candidate = search_dir / "qlib_results_enhanced.json"
        if candidate.exists():
            enhanced_file = candidate
            break

    if enhanced_file is None:
        raise HTTPException(status_code=404, detail=f"qlib_results_enhanced.json not found in {loop_dir} or {task_dir}")

    with open(enhanced_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    _SECTION_KEYS = [
        "summary",
        "ic_diagnostics", "return_curves", "training_diagnostics",
        "trade_diagnostics", "prediction_diagnostics",
        "feature_importance", "top_stocks", "bottom_stocks", "stock_trades",
        "factor_analysis", "absolute_returns", "all_stocks", "stock_pnl_summary",
        "limit_analysis",
    ]
    result = {}
    for key in _SECTION_KEYS:
        if key in full_data:
            result[key] = full_data[key]

    # IC key 标准化: ic_dates → dates (前端期望 dates)
    ic = result.get("ic_diagnostics")
    if ic and "ic_dates" in ic and "dates" not in ic:
        ic["dates"] = ic.pop("ic_dates")

    if not result:
        raise HTTPException(status_code=500, detail=f"qlib_results_enhanced.json exists but contains no diagnostic sections")

    return result


@router.get("/tasks/{task_id}/loops/{loop_id}/assets/download")
async def download_loop_assets(task_id: str, loop_id: str):
    """
    模型资产打包(ZIP)下载
    """
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
            return {"ok": True, "task_id": task_id}
        except Exception as e:
            logger.error(f"Failed to clean up workspace {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "task_id": task_id}


@router.delete("/tasks/{task_id}/loops/{loop_id}")
async def cleanup_loop_workspace(task_id: str, loop_id: str):
    """
    Delete a single Loop workspace under one task.

    This endpoint is intentionally scoped to task_id/loop_id so AIstock rerun
    can remove stale Loop artifacts without deleting sibling Loop results.
    """
    task_dir = _get_task_dir(task_id).resolve()
    loop_dir = (_get_loop_dir(task_id, loop_id)).resolve()
    try:
        loop_dir.relative_to(task_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="loop_id resolves outside task workspace") from exc

    if not loop_dir.exists():
        return {"ok": True, "task_id": task_id, "loop_id": loop_id, "existed": False}
    if not loop_dir.is_dir():
        raise HTTPException(status_code=500, detail=f"Loop workspace is not a directory: {loop_id}")

    try:
        shutil.rmtree(loop_dir)
        return {"ok": True, "task_id": task_id, "loop_id": loop_id, "existed": True}
    except Exception as e:
        logger.error(f"Failed to clean up loop workspace {task_id}/{loop_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_workspace_config():
    """
    返回 QE 工作区配置信息，供 AIstock 动态获取路径。
    所有路径直接从环境变量读取（RDAgent .env）。
    """
    return {
        "workspace_base": str(WORKSPACE_BASE),
        "factor_data_dir": os.environ.get("RDAGENT_FACTOR_DATA_WSL", ""),
        "qlib_data_path": os.environ.get("QLIB_DATA_PATH_WSL", ""),
    }
