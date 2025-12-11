"""
FastAPI server wiring scheduler API stubs.
Single-process, single-worker mode (queue_stub WORKER_COUNT=1).
"""

from __future__ import annotations

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import api_stub
from .config_service import read_env, write_env


def create_app() -> FastAPI:
    app = FastAPI(title="RD-Agent Scheduler")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/tasks")
    def list_tasks():
        return api_stub.api_list_tasks()

    @app.post("/tasks")
    def create_task(payload: dict = Body(default_factory=dict)):
        return api_stub.api_create_task(payload)

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str):
        return api_stub.api_get_task(task_id)

    @app.post("/tasks/{task_id}/status")
    def update_task_status(task_id: str, payload: dict = Body(default_factory=dict)):
        status = payload.get("status", "")
        return api_stub.api_update_task_status(task_id, status)

    @app.post("/tasks/{task_id}/logs")
    def append_log(task_id: str, payload: dict = Body(default_factory=dict)):
        content = payload.get("content", "")
        return api_stub.api_append_log(task_id, content)

    @app.get("/tasks/{task_id}/logs")
    def get_log(task_id: str):
        return api_stub.api_get_log(task_id)

    @app.get("/tasks/{task_id}/results")
    def list_results(task_id: str):
        return api_stub.api_list_results(task_id)

    @app.get("/datasets")
    def list_datasets():
        return api_stub.api_list_datasets()

    @app.post("/datasets")
    def create_dataset(payload: dict = Body(default_factory=dict)):
        return api_stub.api_create_dataset(payload)

    @app.get("/config/env")
    def get_env():
        return {"content": read_env()}

    @app.post("/config/env")
    def update_env(payload: dict = Body(default_factory=dict)):
        content = payload.get("content", "")
        write_env(content)
        return {"ok": True}

    @app.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    return app


app = create_app()
