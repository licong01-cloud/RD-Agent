"""
FastAPI server wiring scheduler API stubs.
Single-process, single-worker mode (queue_stub WORKER_COUNT=1).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import api_stub
from .config_service import read_env, write_env

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
    return app


app = create_app()
