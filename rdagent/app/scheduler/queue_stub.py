"""
Lightweight in-process task queue (placeholder).

This is a temporary async runner using threading + queue.
Replace with a real queue (Celery/RQ/etc.) for production.
"""

from __future__ import annotations

import threading
import queue
from typing import Optional, List

from .worker_stub import run_rdagent_task
from .task_service import append_task_log

_task_queue: Optional[queue.Queue] = None
_worker_threads: List[threading.Thread] = []
_stop_event: Optional[threading.Event] = None
WORKER_COUNT = 1  # single worker (no parallel tasks for now)


def _ensure_queue():
    global _task_queue, _worker_threads, _stop_event
    if _task_queue is None:
        _task_queue = queue.Queue()
    if _stop_event is None:
        _stop_event = threading.Event()
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


def submit_task(task_id: str) -> None:
    _ensure_queue()
    _task_queue.put(task_id)


def stop_queue():
    if _stop_event:
        _stop_event.set()
    if _task_queue:
        _task_queue.put(None)


__all__ = ["submit_task", "stop_queue"]
