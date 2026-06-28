from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    import pytest

HTTP_OK = 200
HTTP_BAD_REQUEST = 400

_ARTIFACT_API_PATH = (
    Path(__file__).resolve().parents[2]
    / "rdagent"
    / "app"
    / "api_endpoints"
    / "qe_workspace_artifacts_api.py"
)
_ARTIFACT_API_SPEC = importlib.util.spec_from_file_location("qe_workspace_artifacts_api_under_test", _ARTIFACT_API_PATH)
if _ARTIFACT_API_SPEC is None or _ARTIFACT_API_SPEC.loader is None:
    _message = f"failed to load qe_workspace_artifacts_api from {_ARTIFACT_API_PATH}"
    raise RuntimeError(_message)
_ARTIFACT_API_MODULE = importlib.util.module_from_spec(_ARTIFACT_API_SPEC)
_ARTIFACT_API_SPEC.loader.exec_module(_ARTIFACT_API_MODULE)
router = _ARTIFACT_API_MODULE.router


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("QE_WORKSPACE_ARTIFACT_STORE", str(tmp_path / "artifact_store"))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_workspace_artifact_upload_head_and_dedup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    payload = b"abc" * 1024
    sha = hashlib.sha256(payload).hexdigest()

    before = client.head(f"/api/v1/qe_workspace/artifacts/{sha}")
    assert before.status_code == HTTP_OK
    assert before.headers["X-Artifact-Exists"] == "0"
    assert before.headers["X-Artifact-Size"] == "0"
    before_json = client.get(f"/api/v1/qe_workspace/artifacts/{sha}")
    assert before_json.status_code == HTTP_OK
    assert before_json.json() == {
        "exists": False,
        "size": 0,
        "sha256": sha,
        "artifact_store_root": str(tmp_path / "artifact_store"),
    }

    uploaded = client.post(f"/api/v1/qe_workspace/artifacts/{sha}", content=payload)
    assert uploaded.status_code == HTTP_OK
    assert uploaded.json()["ok"] is True
    assert uploaded.json()["exists"] is True
    assert uploaded.json()["size"] == len(payload)
    assert (tmp_path / "artifact_store" / sha).read_bytes() == payload

    after = client.head(f"/api/v1/qe_workspace/artifacts/{sha}")
    assert after.status_code == HTTP_OK
    assert after.headers["X-Artifact-Exists"] == "1"
    assert int(after.headers["X-Artifact-Size"]) == len(payload)
    after_json = client.get(f"/api/v1/qe_workspace/artifacts/{sha}")
    assert after_json.json()["exists"] is True
    assert after_json.json()["size"] == len(payload)


def test_workspace_artifact_sha_mismatch_rejects_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(tmp_path, monkeypatch)
    payload = b"payload"
    wrong_sha = hashlib.sha256(b"different").hexdigest()

    response = client.post(f"/api/v1/qe_workspace/artifacts/{wrong_sha}", content=payload)

    assert response.status_code == HTTP_BAD_REQUEST
    assert "workspace artifact sha256 mismatch" in response.json()["detail"]
    store = tmp_path / "artifact_store"
    assert not (store / wrong_sha).exists()
    assert list(store.glob(".*.tmp")) == []


def test_workspace_artifact_rejects_invalid_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.head("/api/v1/qe_workspace/artifacts/not-a-sha")
    assert response.status_code == HTTP_BAD_REQUEST
    response_json = client.get("/api/v1/qe_workspace/artifacts/not-a-sha")
    assert response_json.status_code == HTTP_BAD_REQUEST
    assert "invalid workspace artifact sha256" in response_json.json()["detail"]
