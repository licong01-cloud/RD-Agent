"""
Worker for RD-Agent task execution.

架构核心：进程生命周期与 API 服务器解耦。
- 子进程 stdout/stderr 写入日志文件（非 PIPE），进程不依赖父进程 I/O
- PID + 元数据持久化到 .pid.json 文件
- API 重启后通过 PID 文件发现并恢复追踪存活的任务进程
- 后台 tail 线程从日志文件读取新行，更新进度和缓存
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import csv
import pickle
import threading
import time
from collections import deque
from glob import glob
from pathlib import Path
from typing import Optional, Dict, Any

from .task_service import (
    append_task_log,
    update_task_status,
    get_task,
    record_result,
    LOG_DIR,
    RDAGENT_LOG_ROOT,
    _LOG_DIR_RE,
    _infer_task_status,
    _extract_pid_from_log_dir,
)
from .config_service import PROJECT_ROOT

# ── 持久化目录 ──
_PID_DIR = LOG_DIR / "_pids"

# ── 进程追踪（内存） ──
_running_processes: Dict[str, subprocess.Popen] = {}
_running_pids: Dict[str, int] = {}  # task_id -> pid（含恢复的孤儿进程）
_tail_threads: Dict[str, threading.Thread] = {}

# ── 进度追踪 ──
_task_progress: Dict[str, Dict[str, Any]] = {}

# ── 日志尾部缓存 ──
_log_tail: Dict[str, deque] = {}
_LOG_TAIL_SIZE = 50

# Loop 进度正则
_LOOP_RE = re.compile(r"(?:Starting|Begin|Enter)\s+Loop[_ ]?(\d+)", re.IGNORECASE)
_STEP_RE = re.compile(r"(?:Step|Phase|Stage)\s*[:\s]+(\w+)", re.IGNORECASE)
_TOTAL_LOOPS_RE = re.compile(r"total[_ ]?loops?\s*[=:]\s*(\d+)", re.IGNORECASE)


def _ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_DIR.mkdir(parents=True, exist_ok=True)


def _save_pid_file(task_id: str, pid: int, cmd: list, workdir: str) -> None:
    """将 PID 和任务元数据持久化到文件。"""
    _ensure_dirs()
    pid_file = _PID_DIR / f"{task_id}.pid.json"
    data = {
        "task_id": task_id,
        "pid": pid,
        "cmd": cmd,
        "workdir": workdir,
        "started_at": time.time(),
    }
    pid_file.write_text(json.dumps(data), encoding="utf-8")


def _remove_pid_file(task_id: str) -> None:
    pid_file = _PID_DIR / f"{task_id}.pid.json"
    pid_file.unlink(missing_ok=True)


def _load_pid_files() -> list[dict]:
    """扫描所有 PID 文件。"""
    _ensure_dirs()
    results = []
    for f in _PID_DIR.glob("*.pid.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(data)
        except Exception:
            pass
    return results


def _is_pid_alive(pid: int) -> bool:
    """检查 PID 对应的进程是否存活。"""
    try:
        os.kill(pid, 0)  # signal 0 不杀进程，仅检查
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def get_running_process(task_id: str) -> Optional[subprocess.Popen]:
    """获取任务对应的运行中 Popen 对象（仅本次启动创建的进程）。"""
    proc = _running_processes.get(task_id)
    if proc is not None and proc.poll() is not None:
        _running_processes.pop(task_id, None)
        return None
    return proc


def get_running_pid(task_id: str) -> Optional[int]:
    """获取任务的运行 PID（包括恢复的孤儿进程）。"""
    pid = _running_pids.get(task_id)
    if pid and _is_pid_alive(pid):
        return pid
    _running_pids.pop(task_id, None)
    # fallback 到 Popen 对象
    proc = get_running_process(task_id)
    return proc.pid if proc else None


def _rebuild_progress_from_fs(task: "TaskRecord") -> Dict[str, Any]:
    """当内存进度缺失时（API重启等），从文件系统重建进度信息。"""
    log_dir_name = task.rdagent_log_dir
    if not log_dir_name:
        return {}
    log_dir = RDAGENT_LOG_ROOT / log_dir_name
    if not log_dir.exists():
        return {}

    # 统计已完成的 Loop 数（有 feedback 目录的）
    loop_dirs = sorted(log_dir.glob("Loop_*"), key=lambda p: p.name)
    completed_loops = sum(1 for d in loop_dirs if (d / "feedback").exists())
    current_loop = completed_loops

    # 推断状态
    task_id_str = str(task.id)
    pid = _extract_pid_from_log_dir(log_dir)
    inferred_status = _infer_task_status(log_dir, pid=pid if (pid and _is_pid_alive(pid)) else None)

    # total_loops：优先用 task.loop_n，fallback 到已观察到的 Loop 目录数
    total_loops = task.loop_n or len(loop_dirs) or 0

    pct = (current_loop / total_loops * 100) if total_loops > 0 else 0
    return {
        "current_loop": current_loop,
        "total_loops": total_loops,
        "progress_pct": round(pct, 1),
        "inferred_status": inferred_status,
    }


def get_task_progress_info(task_id: str) -> Dict[str, Any]:
    """获取任务结构化进度信息。内存数据缺失时从文件系统重建。"""
    task = get_task(task_id)
    if task is None:
        return {"error": f"Task not found: {task_id}"}

    progress = _task_progress.get(task_id, {})
    has_memory_data = bool(progress)

    current_loop = progress.get("current_loop", 0)
    total_loops = progress.get("total_loops", task.loop_n or 0)

    # 内存无进度数据时（API重启后），从文件系统重建
    if not has_memory_data and task.rdagent_log_dir:
        fs = _rebuild_progress_from_fs(task)
        if fs:
            current_loop = fs["current_loop"]
            total_loops = fs["total_loops"]
            # 若文件系统推断状态与DB不一致（DB仍为running但进程已结束），以文件系统为准
            inferred = fs.get("inferred_status", "")
            if task.status == "running" and inferred in ("success", "fail"):
                update_task_status(task_id, inferred)
                task = get_task(task_id) or task

    pct = (current_loop / total_loops * 100) if total_loops > 0 else 0
    tail = list(_log_tail.get(task_id, []))

    return {
        "task_id": task_id,
        "status": task.status,
        "current_loop": current_loop,
        "total_loops": total_loops,
        "current_step": progress.get("current_step", ""),
        "progress_pct": round(pct, 1),
        "log_tail": "\n".join(tail) if tail else "",
    }


def _parse_progress(task_id: str, line: str) -> None:
    """从日志行中解析进度信息。"""
    if task_id not in _task_progress:
        _task_progress[task_id] = {}

    m = _LOOP_RE.search(line)
    if m:
        _task_progress[task_id]["current_loop"] = int(m.group(1))

    m = _STEP_RE.search(line)
    if m:
        _task_progress[task_id]["current_step"] = m.group(1)

    m = _TOTAL_LOOPS_RE.search(line)
    if m:
        _task_progress[task_id]["total_loops"] = int(m.group(1))


def _tail_log_file(task_id: str, log_path: Path, pid: int) -> None:
    """后台线程：持续 tail 日志文件，更新进度和缓存。当进程退出时结束。"""
    if task_id not in _log_tail:
        _log_tail[task_id] = deque(maxlen=_LOG_TAIL_SIZE)
    if task_id not in _task_progress:
        _task_progress[task_id] = {}

    pos = 0
    while True:
        # 检查进程是否存活
        if not _is_pid_alive(pid):
            # 进程退出，读取剩余日志
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        for line in f:
                            stripped = line.rstrip("\n")
                            _parse_progress(task_id, stripped)
                            _log_tail[task_id].append(stripped)
                except Exception:
                    pass
            break

        # 读取新行
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    new_lines = f.readlines()
                    pos = f.tell()
                for line in new_lines:
                    stripped = line.rstrip("\n")
                    _parse_progress(task_id, stripped)
                    _log_tail[task_id].append(stripped)
            except Exception:
                pass

        time.sleep(1)

    # 进程结束，清理
    _running_pids.pop(task_id, None)
    _running_processes.pop(task_id, None)
    _tail_threads.pop(task_id, None)
    _remove_pid_file(task_id)

    # 根据文件系统推断最终状态并写回 JSONL
    try:
        from .task_service import get_task, update_task_status, _infer_task_status, RDAGENT_LOG_ROOT
        task = get_task(task_id)
        if task and task.status == "running":
            final_status = "fail"
            if task.rdagent_log_dir:
                log_dir = RDAGENT_LOG_ROOT / task.rdagent_log_dir
                if log_dir.exists():
                    final_status = _infer_task_status(log_dir, pid=None)
            update_task_status(task_id, final_status)
    except Exception as _e:
        pass

    # 延迟清理缓存
    def _deferred_cleanup():
        _task_progress.pop(task_id, None)
        _log_tail.pop(task_id, None)

    threading.Timer(60, _deferred_cleanup).start()


def _start_tail_thread(task_id: str, log_path: Path, pid: int) -> None:
    """启动日志 tail 线程（带去重）。"""
    existing = _tail_threads.get(task_id)
    if existing and existing.is_alive():
        return
    t = threading.Thread(target=_tail_log_file, args=(task_id, log_path, pid), daemon=True)
    t.start()
    _tail_threads[task_id] = t


def _detect_rdagent_log_dir(task_id: str, before_dirs: set[str]) -> str | None:
    """检测 rdagent 进程新创建的 log 目录并关联到 scheduler 任务。

    在 Popen 启动后调用，比较启动前后的目录差异。
    找到后更新任务的 rdagent_log_dir 字段，使 discover_unregistered_tasks 能识别
    该目录已属于 scheduler 创建的任务，避免重复注册为 manual。
    """
    if not RDAGENT_LOG_ROOT.exists():
        return None
    current_dirs = {
        d.name for d in RDAGENT_LOG_ROOT.iterdir()
        if d.is_dir() and _LOG_DIR_RE.match(d.name)
    }
    new_dirs = current_dirs - before_dirs
    if len(new_dirs) == 1:
        log_dir_name = new_dirs.pop()
        from .task_service import _update_task_field
        _update_task_field(task_id, "rdagent_log_dir", log_dir_name)
        append_task_log(task_id, f"[worker] detected log directory: {log_dir_name}")
        return log_dir_name
    elif len(new_dirs) > 1:
        # 多个新目录（极罕见），取最新的
        newest = max(new_dirs)
        from .task_service import _update_task_field
        _update_task_field(task_id, "rdagent_log_dir", newest)
        append_task_log(task_id, f"[worker] detected log directory (newest of {len(new_dirs)}): {newest}")
        return newest
    return None


def _collect_result_files(workdir: str) -> list:
    candidates = ["qlib_res.csv", "ret.pkl", "result.h5"]
    found = []
    for fname in candidates:
        for p in glob(str(Path(workdir) / "**" / fname), recursive=True):
            found.append(p)
    return found


def _summarize_results(result_files: list) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"files": []}
    for p in result_files:
        path = Path(p)
        info = {
            "name": path.name,
            "path": str(path),
            "size": path.stat().st_size if path.exists() else None,
            "mtime": path.stat().st_mtime if path.exists() else None,
        }
        summary["files"].append(info)
        if path.name == "qlib_res.csv":
            try:
                with path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    first_row = next(reader, None)
                    if first_row:
                        summary.setdefault("qlib_res", first_row)
            except Exception as e:
                summary.setdefault("qlib_res_error", str(e))
        if path.name == "ret.pkl":
            try:
                with path.open("rb") as f:
                    obj = pickle.load(f)
                    if hasattr(obj, "keys"):
                        summary.setdefault("ret_pkl_keys", list(obj.keys()))
                    elif isinstance(obj, (list, tuple)):
                        summary.setdefault("ret_pkl_len", len(obj))
                    else:
                        summary.setdefault("ret_pkl_type", str(type(obj)))
            except Exception as e:
                summary.setdefault("ret_pkl_error", str(e))
    return summary


def run_rdagent_task(task_id: str, workspace: Optional[str] = None) -> int:
    """
    启动 rdagent 任务。子进程输出写入日志文件（非 PIPE），
    进程生命周期不依赖 API 服务器。
    """
    _ensure_dirs()
    task = get_task(task_id)
    if task is None:
        raise ValueError(f"Task not found: {task_id}")

    workdir = workspace or task.workspace_path or str(PROJECT_ROOT)
    env_overrides = task.env_overrides if hasattr(task, 'env_overrides') else {}
    rdagent_command = getattr(task, 'command', None) or 'fin_quant'
    cmd = [
        "rdagent",
        rdagent_command,
        "--loop-n",
        str(task.loop_n),
        "--all-duration",
        str(task.all_duration),
    ]

    update_task_status(task_id, "running")
    append_task_log(task_id, f"Starting task {task_id}: {' '.join(cmd)}")

    # 初始化进度和日志缓存
    _task_progress[task_id] = {"current_loop": 0, "total_loops": task.loop_n or 0, "current_step": ""}
    _log_tail[task_id] = deque(maxlen=_LOG_TAIL_SIZE)

    log_path = LOG_DIR / f"{task_id}.log"

    # 快照启动前的 log 目录列表（用于启动后检测新建的 rdagent log dir）
    _before_log_dirs: set[str] = set()
    if RDAGENT_LOG_ROOT.exists():
        _before_log_dirs = {
            d.name for d in RDAGENT_LOG_ROOT.iterdir()
            if d.is_dir() and _LOG_DIR_RE.match(d.name)
        }

    try:
        env = {**os.environ, **env_overrides} if env_overrides else None
        # 子进程输出直接写入日志文件，不用 PIPE
        log_fd = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # 独立会话，API 重启不影响
            close_fds=True,  # 防止 uvicorn socket fd 泄漏导致端口占用
            env=env,
        )
        log_fd.close()  # Popen 已经 dup 了 fd，父进程可以关闭

        # 追踪进程
        _running_processes[task_id] = proc
        _running_pids[task_id] = proc.pid

        # 持久化 PID 文件
        _save_pid_file(task_id, proc.pid, cmd, workdir)

        # 启动日志 tail 线程
        _start_tail_thread(task_id, log_path, proc.pid)

        # 检测 rdagent 新建的 log 目录并关联到任务（防止 discover 重复注册）
        # 延迟几秒等 rdagent 创建目录
        time.sleep(5)
        _detect_rdagent_log_dir(task_id, _before_log_dirs)

        # 等待进程结束（在 worker 线程中阻塞）
        proc.wait()

        # 清理
        _running_processes.pop(task_id, None)
        _running_pids.pop(task_id, None)
        _remove_pid_file(task_id)

        if proc.returncode == 0:
            update_task_status(task_id, "success")
        else:
            update_task_status(task_id, "fail")
            append_task_log(task_id, f"Task failed with code {proc.returncode}")

        result_files = _collect_result_files(workdir)
        summary = _summarize_results(result_files)
        record_result(
            task_id,
            {
                "returncode": proc.returncode,
                "workdir": workdir,
                "cmd": cmd,
                "log_path": str(log_path),
                "result_files": result_files,
                "summary": summary,
            },
        )
        return proc.returncode
    except Exception as e:
        _running_processes.pop(task_id, None)
        _running_pids.pop(task_id, None)
        _remove_pid_file(task_id)
        update_task_status(task_id, "fail")
        append_task_log(task_id, f"Task failed with exception: {e}")
        record_result(
            task_id,
            {
                "returncode": 1,
                "workdir": workdir,
                "cmd": cmd,
                "error": str(e),
                "result_files": [],
            },
        )
        return 1


def recover_running_tasks() -> int:
    """
    API 重启后调用：扫描 PID 文件，恢复对存活任务的追踪。
    - 进程存活 → 恢复 tail 线程 + 内存追踪
    - 进程已死 → 标记任务为 fail，清理 PID 文件
    返回恢复的任务数量。
    """
    _ensure_dirs()
    recovered = 0
    for pid_data in _load_pid_files():
        task_id = pid_data.get("task_id")
        pid = pid_data.get("pid")
        if not task_id or not pid:
            continue

        log_path = LOG_DIR / f"{task_id}.log"

        if _is_pid_alive(pid):
            # 进程存活，恢复追踪
            _running_pids[task_id] = pid
            _task_progress[task_id] = {"current_loop": 0, "total_loops": 0, "current_step": ""}
            _log_tail[task_id] = deque(maxlen=_LOG_TAIL_SIZE)
            _start_tail_thread(task_id, log_path, pid)
            append_task_log(task_id, f"[recovery] API restarted, reconnected to running process (PID={pid})")
            recovered += 1
        else:
            # 进程已死
            _remove_pid_file(task_id)
            task = get_task(task_id)
            if task and task.status == "running":
                final_status = "fail"
                if task.rdagent_log_dir:
                    log_dir = RDAGENT_LOG_ROOT / task.rdagent_log_dir
                    if log_dir.exists():
                        final_status = _infer_task_status(log_dir, pid=None)
                update_task_status(task_id, final_status)
                append_task_log(task_id, f"[recovery] Process (PID={pid}) no longer alive after API restart, marking as {final_status}")

    return recovered


# ── 手动任务领养 ──

# 步骤名称到友好名称的映射
_STEP_NAMES = {
    "direct_exp_gen": "hypothesis→experiment",
    "coding": "coding (CoSTEER)",
    "running": "training",
    "feedback": "feedback",
}


def _scan_dir_progress(task_id: str, rdagent_log_dir: Path, pid: int) -> None:
    """后台线程：通过扫描日志目录结构监控手动任务进度。"""
    if task_id not in _task_progress:
        _task_progress[task_id] = {}
    if task_id not in _log_tail:
        _log_tail[task_id] = deque(maxlen=_LOG_TAIL_SIZE)

    while True:
        if not _is_pid_alive(pid):
            break

        # 扫描 Loop_N 目录
        loop_dirs = sorted(rdagent_log_dir.glob("Loop_*"), key=lambda p: p.name)
        # 已完成的 loop 数（有 feedback 目录）与 _rebuild_progress_from_fs 保持一致
        current_loop = sum(1 for d in loop_dirs if (d / "feedback").exists())
        current_step = ""

        if loop_dirs:
            last_loop = loop_dirs[-1]
            # 找最新的步骤子目录
            step_dirs = [
                d for d in last_loop.iterdir()
                if d.is_dir() and d.name in _STEP_NAMES
            ]
            if step_dirs:
                # 按修改时间取最新
                latest_step = max(step_dirs, key=lambda d: d.stat().st_mtime)
                current_step = _STEP_NAMES.get(latest_step.name, latest_step.name)

        _task_progress[task_id]["current_loop"] = current_loop
        _task_progress[task_id]["current_step"] = current_step

        time.sleep(5)

    # 进程退出 — 最终扫描
    loop_dirs = sorted(rdagent_log_dir.glob("Loop_*"), key=lambda p: p.name)
    final_loop_count = sum(1 for d in loop_dirs if (d / "feedback").exists())
    _task_progress[task_id]["current_loop"] = final_loop_count

    # 判断最终状态（用文件系统推断，避免 total_loops 不准导致误判）
    final_status = _infer_task_status(rdagent_log_dir, pid=None)
    update_task_status(task_id, final_status)
    total_loops = _task_progress[task_id].get("total_loops", 0)
    _log_tail[task_id].append(f"[monitor] Process exited, status={final_status} (loops={final_loop_count}/{total_loops})")

    # 清理
    _running_pids.pop(task_id, None)
    _tail_threads.pop(task_id, None)

    def _deferred_cleanup():
        _task_progress.pop(task_id, None)
        _log_tail.pop(task_id, None)
    threading.Timer(60, _deferred_cleanup).start()


def adopt_manual_task(task_id: str, rdagent_log_dir: Path, pid: int, total_loops: int) -> None:
    """领养一个手动启动的 rdagent 任务，开始追踪其进度。

    - task_id: Scheduler 中的任务标识（TaskRecord.name = log dir name）
    - rdagent_log_dir: rdagent log/ 下的完整目录路径
    - pid: 进程 PID
    - total_loops: 目标 loop 数（从 cmdline 推断）
    """
    if not _is_pid_alive(pid):
        return

    # 避免重复领养
    existing = _tail_threads.get(task_id)
    if existing and existing.is_alive():
        return

    _running_pids[task_id] = pid
    _task_progress[task_id] = {
        "current_loop": 0,
        "total_loops": total_loops,
        "current_step": "",
    }
    _log_tail[task_id] = deque(maxlen=_LOG_TAIL_SIZE)

    t = threading.Thread(
        target=_scan_dir_progress,
        args=(task_id, rdagent_log_dir, pid),
        daemon=True,
    )
    t.start()
    _tail_threads[task_id] = t
    print(f"[adopt] monitoring manual task {task_id} (PID={pid}, loops={total_loops})")


__all__ = [
    "run_rdagent_task",
    "get_running_process",
    "get_running_pid",
    "get_task_progress_info",
    "recover_running_tasks",
    "adopt_manual_task",
]
