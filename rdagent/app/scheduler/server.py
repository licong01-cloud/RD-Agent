"""
FastAPI server wiring scheduler API stubs.
Single-process, single-worker mode (queue_stub WORKER_COUNT=1).
"""

from __future__ import annotations

from typing import Annotated

import asyncio
import json
import os
import re
import signal
import subprocess

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import api_stub
from .config_service import read_env, write_env
from .worker_stub import get_running_process, get_running_pid, get_task_progress_info

DEFAULT_BODY = Body(default_factory=dict)


def _register_task_routes(app: FastAPI) -> None:
    @app.get("/tasks")
    def list_tasks() -> dict:
        return api_stub.api_list_tasks()

    @app.post("/tasks")
    def create_task(payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        return api_stub.api_create_task(payload)

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str) -> dict:
        return api_stub.api_get_task(task_id)

    @app.post("/tasks/{task_id}/status")
    def update_task_status(task_id: str, payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        status = payload.get("status", "")
        return api_stub.api_update_task_status(task_id, status)

    @app.post("/tasks/{task_id}/logs")
    def append_log(task_id: str, payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        content = payload.get("content", "")
        return api_stub.api_append_log(task_id, content)

    @app.get("/tasks/{task_id}/logs")
    def get_log(task_id: str) -> dict:
        return api_stub.api_get_log(task_id)

    @app.get("/tasks/{task_id}/results")
    def list_results(task_id: str) -> dict:
        return api_stub.api_list_results(task_id)

    def _get_task_pid(task_id: str) -> int | None:
        """获取任务 PID（兼容本次启动的 Popen 和恢复的孤儿进程）。"""
        return get_running_pid(task_id)

    @app.post("/tasks/{task_id}/pause")
    def pause_task(task_id: str) -> dict:
        """暂停任务 (SIGSTOP 进程组)。"""
        pid = _get_task_pid(task_id)
        if pid is None:
            return {"error": f"No running process for task {task_id}"}
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGSTOP)
            return {"ok": True, "task_id": task_id, "action": "paused"}
        except ProcessLookupError:
            return {"error": f"Process {pid} already exited"}
        except PermissionError as e:
            return {"error": f"Permission denied sending SIGSTOP: {e}"}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/tasks/{task_id}/resume")
    def resume_task(task_id: str) -> dict:
        """恢复任务 (SIGCONT 进程组)。"""
        pid = _get_task_pid(task_id)
        if pid is None:
            return {"error": f"No running process for task {task_id}"}
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGCONT)
            return {"ok": True, "task_id": task_id, "action": "resumed"}
        except ProcessLookupError:
            return {"error": f"Process {pid} already exited"}
        except PermissionError as e:
            return {"error": f"Permission denied sending SIGCONT: {e}"}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/tasks/{task_id}/cancel")
    def cancel_task(task_id: str) -> dict:
        """取消任务 (SIGTERM → SIGKILL)。"""
        from .task_service import update_task_status
        pid = _get_task_pid(task_id)
        if pid is None:
            update_task_status(task_id, "canceled")
            return {"ok": True, "task_id": task_id, "action": "canceled", "note": "process already exited"}
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            update_task_status(task_id, "canceled")
            return {"ok": True, "task_id": task_id, "action": "canceled", "note": "process exited before cancel"}
        try:
            os.killpg(pgid, signal.SIGTERM)
            # 等待进程退出（Popen 对象可能不存在，用 os.waitpid 兜底）
            proc = get_running_process(task_id)
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait(timeout=3)
            else:
                # 恢复的孤儿进程，无 Popen 对象，轮询 PID 存活状态
                import time
                for _ in range(10):
                    try:
                        os.kill(pid, 0)
                    except (ProcessLookupError, OSError):
                        break
                    time.sleep(0.5)
                else:
                    # 5 秒后仍存活，SIGKILL
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except ProcessLookupError:
            pass  # 进程已退出，达到预期效果
        update_task_status(task_id, "canceled")
        return {"ok": True, "task_id": task_id, "action": "canceled"}

    @app.get("/tasks/{task_id}/progress")
    def get_progress(task_id: str) -> dict:
        """获取任务结构化进度。"""
        return get_task_progress_info(task_id)

    @app.post("/tasks/discover")
    def discover_tasks() -> dict:
        """手动触发扫描 log 目录，发现并注册未被追踪的手动任务。

        发现和领养在后台线程执行，避免长时间阻塞 API。
        """
        import threading
        from .queue_stub import _discover_and_adopt
        from .task_service import discover_unregistered_tasks

        def _bg():
            try:
                n = discover_unregistered_tasks()
                if n > 0:
                    print(f"[discover] registered {n} new manual task(s)")
                _discover_and_adopt()
            except Exception as e:
                print(f"[discover] error: {e}")

        threading.Thread(target=_bg, daemon=True).start()
        return {"ok": True, "message": "discovery started in background"}

    # ANSI 转义码清理：保留 SGR 颜色序列（\x1b[...m），剥离其他控制序列
    _ANSI_NON_SGR = re.compile(r"\x1b\[[0-9;]*[A-HJKSTfhln]|\x1b\[\?[0-9;]*[hl]|\x1b\][^\x07]*\x07|\r")

    @app.get("/tasks/{task_id}/logs/stream")
    async def stream_logs(task_id: str) -> StreamingResponse:
        """SSE 实时日志流 — tail -f 方式读取日志文件。"""
        from .task_service import LOG_DIR
        log_path = LOG_DIR / f"{task_id}.log"

        def _clean(line: str) -> str:
            """剥离非 SGR 的 ANSI 转义码（光标移动、清屏等），保留颜色。"""
            return _ANSI_NON_SGR.sub("", line.rstrip())

        async def _sse():
            pos = 0
            if log_path.exists():
                # 先发送已有内容的最后 50 行
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    tail = lines[-50:] if len(lines) > 50 else lines
                    for line in tail:
                        yield f"data: {json.dumps({'line': _clean(line)})}\n\n"
                    pos = f.tell()

            # 持续 tail
            while True:
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        new_lines = f.readlines()
                        pos = f.tell()
                    for line in new_lines:
                        yield f"data: {json.dumps({'line': _clean(line)})}\n\n"

                # 检查任务是否已完成
                from .task_service import get_task
                task = get_task(task_id)
                if task and task.status in ("success", "fail", "canceled"):
                    yield f"data: {json.dumps({'event': 'task_ended', 'status': task.status})}\n\n"
                    break
                await asyncio.sleep(1)

        return StreamingResponse(_sse(), media_type="text/event-stream")


def _register_dataset_routes(app: FastAPI) -> None:
    @app.get("/datasets")
    def list_datasets() -> dict:
        return api_stub.api_list_datasets()

    @app.post("/datasets")
    def create_dataset(payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        return api_stub.api_create_dataset(payload)


def _register_template_routes(app: FastAPI) -> None:
    # --- existing ---
    @app.post("/templates/publish")
    def publish_templates(payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        return api_stub.api_publish_templates(payload)

    @app.post("/templates/history")
    def list_template_history(payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        return api_stub.api_list_template_history(payload.get("scenario"), payload.get("version"))

    @app.post("/templates/rollback")
    def rollback_template(payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        return api_stub.api_rollback_template(payload)

    # --- P2 new APIs ---
    @app.get("/templates")
    def list_templates(scenario: str | None = None) -> dict:
        return api_stub.api_list_templates(scenario)

    @app.get("/templates/{scenario}/{version}/files")
    def list_template_files(scenario: str, version: str) -> dict:
        return api_stub.api_list_template_files(scenario, version)

    @app.get("/templates/{scenario}/{version}/file")
    def get_template_file(scenario: str, version: str, path: str = "") -> dict:
        return api_stub.api_get_template_file(scenario, version, path)

    @app.post("/templates/{scenario}/{version}/file")
    def save_template_file(scenario: str, version: str, payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        return api_stub.api_save_template_file(
            scenario, version,
            payload.get("path", ""),
            payload.get("content", ""),
        )

    @app.delete("/templates/{scenario}/{version}")
    def delete_template(scenario: str, version: str) -> dict:
        return api_stub.api_delete_template(scenario, version)

    @app.post("/templates/{scenario}/{version}/apply")
    def apply_template(scenario: str, version: str, payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        return api_stub.api_apply_template(
            scenario, version,
            force=bool(payload.get("force", False)),
            backup=bool(payload.get("backup", True)),
        )

    @app.get("/templates/sync-status")
    def get_sync_status() -> dict:
        return api_stub.api_get_sync_status()

    @app.post("/templates/{scenario}/{version}/refresh-sha256")
    def refresh_template_sha256(scenario: str, version: str) -> dict:
        return api_stub.api_refresh_template_sha256(scenario, version)

    @app.get("/templates/backups")
    def list_backups() -> dict:
        return api_stub.api_list_backups()

    @app.post("/templates/backups/rollback")
    def rollback_from_backup(payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        backup_id = payload.get("backup_id", "")
        if not backup_id:
            return {"error": "backup_id is required"}
        return api_stub.api_rollback_from_backup(backup_id)

    # --- P3: env-based activation ---
    @app.post("/templates/{scenario}/{version}/activate-env")
    def activate_template_env(scenario: str, version: str) -> dict:
        return api_stub.api_activate_template_env(scenario, version)

    @app.get("/templates/active-env")
    def get_active_env_template() -> dict:
        return api_stub.api_get_active_env_template()


def _register_config_routes(app: FastAPI) -> None:
    @app.get("/config/env")
    def get_env() -> dict:
        return {"content": read_env()}

    @app.post("/config/env")
    def update_env(payload: Annotated[dict, DEFAULT_BODY]) -> dict:
        content = payload.get("content", "")
        write_env(content)
        return {"ok": True}


def _register_health_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})


def _register_routes(app: FastAPI) -> None:
    _register_task_routes(app)
    _register_dataset_routes(app)
    _register_template_routes(app)
    _register_config_routes(app)
    _register_health_routes(app)


def create_app() -> FastAPI:
    app = FastAPI(title="RD-Agent Scheduler")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _register_routes(app)

    @app.on_event("startup")
    def _on_startup():
        """API 启动时恢复之前存活的任务进程 + 启动定时 discover 线程。"""
        from .queue_stub import _ensure_queue, _discover_and_adopt
        from .task_service import discover_unregistered_tasks
        _ensure_queue()

        # 启动定时 discover 线程（每 30 秒扫描 log 目录发现手工任务）
        import threading
        import time

        def _periodic_discover():
            while True:
                try:
                    n = discover_unregistered_tasks()
                    if n > 0:
                        print(f"[periodic-discover] registered {n} new manual task(s)")
                    _discover_and_adopt()
                except Exception as e:
                    print(f"[periodic-discover] error: {e}")
                time.sleep(30)

        threading.Thread(target=_periodic_discover, daemon=True).start()

    return app


app = create_app()
