"""
Lightweight in-process task queue (placeholder).

This is a temporary async runner using threading + queue.
Replace with a real queue (Celery/RQ/etc.) for production.
"""

from __future__ import annotations

import threading
import queue
from typing import Optional, List

from .worker_stub import run_rdagent_task, recover_running_tasks, adopt_manual_task
from .task_service import append_task_log

from pathlib import Path


_task_queue: Optional[queue.Queue] = None
_worker_threads: List[threading.Thread] = []
_stop_event: Optional[threading.Event] = None
WORKER_COUNT = 1  # single worker (no parallel tasks for now)


_recovered = False


def _ensure_queue():
    global _task_queue, _worker_threads, _stop_event, _recovered
    if _task_queue is None:
        _task_queue = queue.Queue()
    if _stop_event is None:
        _stop_event = threading.Event()
    # 首次初始化时恢复 API 重启前存活的任务
    if not _recovered:
        _recovered = True
        # 恢复 Scheduler 创建的任务（PID 文件，轻量操作）
        try:
            n = recover_running_tasks()
            if n > 0:
                print(f"[queue] recovered {n} running task(s) from previous API session")
        except Exception as e:
            print(f"[queue] recover_running_tasks failed: {e}")
    # Start up to WORKER_COUNT threads
    alive_threads = [t for t in _worker_threads if t.is_alive()]
    _worker_threads = alive_threads
    missing = WORKER_COUNT - len(_worker_threads)
    for _ in range(missing):
        t = threading.Thread(target=_worker_loop, daemon=True)
        t.start()
        _worker_threads.append(t)


def _worker_loop():
    assert _task_queue is not None
    assert _stop_event is not None
    while not _stop_event.is_set():
        try:
            task_id = _task_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if task_id is None:
            continue
        try:
            run_rdagent_task(task_id)
        except Exception as e:  # pragma: no cover - placeholder
            append_task_log(task_id, f"[queue] task failed: {e}")
        finally:
            _task_queue.task_done()


def _discover_and_adopt() -> None:
    """发现手动启动的 rdagent 任务并领养运行中的进程。

    同时修正已过时的 "running" 状态：如果进程已不存在，重新推断最终状态。
    """
    from .task_service import (
        discover_unregistered_tasks,
        list_tasks,
        RDAGENT_LOG_ROOT,
        _extract_pid_from_log_dir,
        _is_pid_alive,
        _infer_loop_n_from_cmdline,
        _infer_task_status,
        update_task_status,
    )

    # 发现并注册
    n = discover_unregistered_tasks()
    if n > 0:
        print(f"[queue] discovered {n} manual task(s)")

    # 领养正在运行的手动任务 / 修正过时状态
    for task in list_tasks():
        if task.source != "manual":
            continue
        if task.status != "running":
            continue
        if not task.rdagent_log_dir:
            continue

        log_dir = RDAGENT_LOG_ROOT / task.rdagent_log_dir
        if not log_dir.exists():
            continue

        pid = _extract_pid_from_log_dir(log_dir)
        if pid and _is_pid_alive(pid):
            # 进程存活 → 领养
            total_loops = _infer_loop_n_from_cmdline(pid) or task.loop_n or 0
            task_key = task.rdagent_log_dir
            adopt_manual_task(task_key, log_dir, pid, total_loops)
        else:
            # 进程已退出但状态仍为 running → 修正为最终状态
            final_status = _infer_task_status(log_dir, pid=None)
            if final_status != "running":
                task_key = task.rdagent_log_dir
                update_task_status(task_key, final_status)
                print(f"[discover] fixed stale status: {task_key} running → {final_status}")


def submit_task(task_id: str) -> None:
    _ensure_queue()
    _task_queue.put(task_id)


def stop_queue():
    if _stop_event:
        _stop_event.set()
    if _task_queue:
        _task_queue.put(None)


__all__ = ["submit_task", "stop_queue"]
