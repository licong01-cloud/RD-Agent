from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from types import ModuleType

HTTP_OK = 200
HTTP_FORBIDDEN = 403

_API_PATH = (
    Path(__file__).resolve().parents[2]
    / "rdagent"
    / "app"
    / "api_endpoints"
    / "qe_evolution_api.py"
)


def _load_api(workspace_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("qe_evolution_catalog_api_under_test", _API_PATH)
    if spec is None or spec.loader is None:
        message = f"failed to load QE workspace API from {_API_PATH}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORKSPACE_BASE = workspace_root
    return module


def _client(workspace_root: Path) -> TestClient:
    module = _load_api(workspace_root)
    app = FastAPI()
    app.include_router(module.router)
    return TestClient(app, follow_redirects=False)


def test_completed_loop_catalog_is_complete_sorted_and_relative(tmp_path: Path) -> None:
    loop_dir = tmp_path / "qe_task" / "Loop8"
    (loop_dir / "reports").mkdir(parents=True)
    (loop_dir / "status.txt").write_text("completed", encoding="utf-8")
    (loop_dir / "reports" / "result.json").write_text("{}", encoding="utf-8")
    (loop_dir / "pred.pkl").write_bytes(b"prediction")

    response = _client(tmp_path).get(
        "/api/v1/qe_workspace/tasks/qe_task/loops/Loop8/files",
    )

    assert response.status_code == HTTP_OK
    payload = response.json()
    assert payload["schema_version"] == "hmm_qe_asset_catalog_v1"
    assert payload["catalog_completeness"] == "complete"
    paths = [row["relative_path"] for row in payload["files"]]
    assert paths == sorted(paths)
    assert paths == ["pred.pkl", "reports/result.json", "status.txt"]
    assert all(not Path(path).is_absolute() for path in paths)
    assert all(row["sha256"] is None for row in payload["files"])
    assert response.history == []


def test_running_loop_catalog_is_explicitly_partial(tmp_path: Path) -> None:
    loop_dir = tmp_path / "qe_task" / "Loop1"
    loop_dir.mkdir(parents=True)
    (loop_dir / "status.txt").write_text("running", encoding="utf-8")

    response = _client(tmp_path).get(
        "/api/v1/qe_workspace/tasks/qe_task/loops/Loop1/files",
    )

    assert response.status_code == HTTP_OK
    payload = response.json()
    assert payload["catalog_completeness"] == "partial"
    assert "loop_not_terminal:running" in payload["warnings"]


def test_catalog_rejects_task_and_loop_path_traversal(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get(
        "/api/v1/qe_workspace/tasks/%2E%2E/loops/outside/files",
    )

    assert response.status_code == HTTP_FORBIDDEN


def test_catalog_lists_but_does_not_follow_symlink_directory(tmp_path: Path) -> None:
    loop_dir = tmp_path / "qe_task" / "Loop2"
    outside = tmp_path / "outside"
    loop_dir.mkdir(parents=True)
    outside.mkdir()
    (loop_dir / "status.txt").write_text("completed", encoding="utf-8")
    (outside / "secret.txt").write_text("outside", encoding="utf-8")
    link = loop_dir / "external"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    response = _client(tmp_path).get(
        "/api/v1/qe_workspace/tasks/qe_task/loops/Loop2/files",
    )

    assert response.status_code == HTTP_OK
    payload = response.json()
    paths = [row["relative_path"] for row in payload["files"]]
    assert "external" in paths
    assert "external/secret.txt" not in paths
    assert "symlink_not_followed:external" in payload["warnings"]
