"""
系统指标 API — 轻量端点，供 AIstock 心跳检查用。

GET /system/metrics → CPU/RAM/GPU/磁盘快照 + 运行中任务列表 + 版本信息
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


def _get_gpu_info() -> dict:
    """通过 nvidia-smi 获取 GPU 指标。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5,
            text=True,
        ).strip()
        if out:
            parts = [p.strip() for p in out.split(",")]
            return {
                "gpu_util_pct": int(parts[0]),
                "gpu_memory_used_mb": int(parts[1]),
                "gpu_memory_total_mb": int(parts[2]),
                "gpu_temperature_c": int(parts[3]),
            }
    except FileNotFoundError:
        # nvidia-smi 不存在（无 GPU 或未安装驱动）
        pass
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi 执行超时")
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        logger.warning("nvidia-smi 解析失败: %s", e)
    return {}


def _get_cpu_ram_disk() -> dict:
    """通过 psutil 获取 CPU/RAM/磁盘指标。"""
    try:
        import psutil
        disk = psutil.disk_usage("/")
        return {
            "cpu_util_pct": round(psutil.cpu_percent(interval=0.5), 1),
            "ram_used_pct": round(psutil.virtual_memory().percent, 1),
            "disk_free_gb": round(disk.free / (1024 ** 3), 1),
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
        }
    except ImportError:
        return {}


def _get_running_tasks() -> list[str]:
    """获取当前 scheduler 中运行的任务 ID。"""
    try:
        from rdagent.app.scheduler.task_service import list_tasks
        return [str(t.id) for t in list_tasks() if t.status == "running"]
    except ImportError:
        # scheduler 模块未加载（独立运行 results_api 时）
        return []
    except Exception as e:
        logger.warning("获取运行中任务列表失败: %s", e)
        raise


def _get_versions() -> dict:
    """获取版本信息，供多节点一致性检查。"""
    repo_root = Path(__file__).resolve().parents[3]
    info: dict = {}

    # rdagent 代码版本 — 最新文件修改时间
    try:
        import time
        src_dir = repo_root / "rdagent"
        latest_mtime = 0.0
        for f in src_dir.rglob("*.py"):
            mt = f.stat().st_mtime
            if mt > latest_mtime:
                latest_mtime = mt
        if latest_mtime > 0:
            info["rdagent_last_modified"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(latest_mtime))
    except Exception:
        pass

    # git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root), timeout=3, text=True,
        ).strip()
        info["rdagent_commit"] = commit
    except Exception:
        info["rdagent_commit"] = "unknown"

    # Python/PyTorch/qlib 版本
    import sys
    info["python_version"] = sys.version.split()[0]
    try:
        import torch
        info["pytorch_version"] = torch.__version__
    except ImportError:
        pass
    try:
        import qlib
        info["qlib_version"] = getattr(qlib, "__version__", "unknown")
    except ImportError:
        pass

    return info


@router.get("/system/metrics")
def get_system_metrics():
    gpu = _get_gpu_info()
    sys_info = _get_cpu_ram_disk()
    tasks = _get_running_tasks()
    versions = _get_versions()
    return {
        **gpu,
        **sys_info,
        "running_tasks": tasks,
        "versions": versions,
    }
