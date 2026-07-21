from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi import HTTPException
from rdagent.app.api_endpoints import qe_environment_identity
from rdagent.app.api_endpoints.qe_dataset_identity import read_dataset_identity
from rdagent.app.api_endpoints.qe_submission_receipt import (
    get_submission_receipt,
    loop_lifecycle_lock,
    promote_submission_receipt_to_completed_from_verified_result_locked,
    reserve_submission,
    transition_submission_receipt,
)

HTTP_CONFLICT = 409


class _BackgroundTasks:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def add_task(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


def _load_api(workspace_root: Path, *, module_name: str) -> ModuleType:
    api_path = (
        Path(__file__).resolve().parents[2]
        / "rdagent"
        / "app"
        / "api_endpoints"
        / "qe_evolution_api.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, api_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORKSPACE_BASE = workspace_root
    return module


def _manifest_payload() -> dict[str, str]:
    return {
        "schema_version": "qe_dataset_manifest_v1",
        "deployment_snapshot_id": "qe_data_20260721",
        "cutoff_trade_date": "2026-06-30",
        "qlib_calendar_sha256": "a" * 64,
        "qlib_instruments_sha256": "b" * 64,
        "st_pit_snapshot_id": "qe_st_pit_20260630",
        "st_pit_manifest_sha256": "c" * 64,
    }


def test_environment_identity_is_deployment_cached_without_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def build_manifest() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"schema_version": "qe_execution_environment_manifest_v1", "value": "stable"}

    qe_environment_identity.reset_execution_environment_identity_cache_for_tests()
    monkeypatch.setattr(qe_environment_identity, "_build_manifest", build_manifest)
    first = qe_environment_identity.get_execution_environment_identity()
    second = qe_environment_identity.get_execution_environment_identity()

    assert calls == 1
    assert first == second
    assert first["execution_environment_snapshot_id"].startswith("qeenv_")
    assert "nvidia" not in json.dumps(first).lower()
    qe_environment_identity.reset_execution_environment_identity_cache_for_tests()


def test_dataset_identity_reads_published_manifest_and_missing_manifest_is_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "factor_data"
    root.mkdir()
    payload = _manifest_payload()
    payload["dataset_manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    (root / "qe_dataset_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setenv("QE_QLIB_DATA_PATH", str(root))
    monkeypatch.delenv("QE_DATASET_IDENTITY_ROOTS", raising=False)

    complete = read_dataset_identity(data_root_uri=str(root), node_id="wsl2-5080")
    assert complete["complete"] is True
    assert complete["dataset"]["dataset_manifest_sha256"] == payload["dataset_manifest_sha256"]

    (root / "qe_dataset_manifest.json").unlink()
    incomplete = read_dataset_identity(data_root_uri=str(root), node_id="wsl2-5080")
    assert incomplete["complete"] is False
    assert incomplete["reason_code"] == "qe_dataset_manifest_missing"
    assert incomplete["acquisition_suggestions"]


def test_submission_receipt_binds_current_environment_or_rejects_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, module_name="qe_environment_binding_api")
    environment = {
        "schema_version": "qe_execution_environment_manifest_v1",
        "execution_environment_snapshot_id": "qeenv_test_snapshot",
        "execution_environment_manifest_sha256": "d" * 64,
        "manifest": {"test": "environment"},
    }
    monkeypatch.setattr(module, "get_execution_environment_identity", lambda: dict(environment))
    request = module.LoopRunRequest(
        loop_index=1,
        config={"value": 1},
        submission_intent_hash="e" * 64,
        execution_identity_hash="f" * 64,
        execution_environment_snapshot_id=environment["execution_environment_snapshot_id"],
        execution_environment_manifest_sha256=environment["execution_environment_manifest_sha256"],
    )
    background = _BackgroundTasks()
    response = asyncio.run(module.create_and_run_loop("qe_task", request, background))

    assert response.execution_identity_hash == "f" * 64
    receipt = get_submission_receipt(
        tmp_path / "qe_task" / "Loop1",
        loop_id="Loop1",
        submission_intent_hash="e" * 64,
    )
    assert receipt is not None
    assert receipt["execution_environment_manifest_sha256"] == "d" * 64

    mismatch = request.model_copy(
        update={"execution_environment_manifest_sha256": "0" * 64},
    )
    with pytest.raises(HTTPException) as caught:
        asyncio.run(module.create_and_run_loop("other_task", mismatch, background))
    assert caught.value.status_code == HTTP_CONFLICT
    assert caught.value.detail["reason_code"] == "qe_execution_environment_identity_mismatch"


def test_verified_result_repairs_failed_receipt_under_shared_lock(tmp_path: Path) -> None:
    loop_dir = tmp_path / "qe_task" / "Loop1"
    intent_hash = "1" * 64
    reserve_submission(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        request_digest="2" * 64,
    )
    transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        status="failed",
    )
    loop_dir.mkdir(parents=True, exist_ok=True)
    (loop_dir / "qlib_results_enhanced.json").write_text("{}", encoding="utf-8")
    with loop_lifecycle_lock(loop_dir, "Loop1"):
        repaired = promote_submission_receipt_to_completed_from_verified_result_locked(
            loop_dir,
            loop_id="Loop1",
            submission_intent_hash=intent_hash,
        )

    assert repaired["status"] == "completed"
    persisted = get_submission_receipt(loop_dir, loop_id="Loop1", submission_intent_hash=intent_hash)
    assert persisted is not None
    assert persisted["status"] == "completed"
