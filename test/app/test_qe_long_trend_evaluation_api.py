from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import HTTPException
from rdagent.app.api_endpoints import qe_long_trend_evaluation as qelt
from rdagent.app.api_endpoints.qe_dataset_identity import read_dataset_identity
from rdagent.app.api_endpoints.qe_workspace_catalog import build_workspace_catalog


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _bundle(environment_sha: str) -> dict[str, object]:
    files = {path: f"# {path}\n" for path in qelt._ALLOWED_BUNDLE_PATHS if path != "bundle_manifest.json"}
    rows = [
        {
            "relative_path": path,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "size_bytes": len(source.encode()),
        }
        for path, source in sorted(files.items())
    ]
    manifest = {
        "schema_version": qelt.BUNDLE_SCHEMA,
        "evaluator_version": "qe_long_trend_evaluator_v1",
        "evaluator_source_sha256": "b" * 64,
        "execution_environment_snapshot_id": "qeenv-fixture",
        "execution_environment_manifest_sha256": environment_sha,
        "python_abi": {"implementation": "CPython", "version": "3.10", "cache_tag": "cpython-310"},
        "files": rows,
    }
    bundle_sha = _sha_json(manifest)
    manifest["bundle_sha256"] = bundle_sha
    files["bundle_manifest.json"] = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": qelt.BUNDLE_SCHEMA,
        "bundle_sha256": bundle_sha,
        "evaluator_source_sha256": "b" * 64,
        "execution_environment_snapshot_id": "qeenv-fixture",
        "execution_environment_manifest_sha256": environment_sha,
        "manifest": manifest,
        "files": files,
    }


def _request(environment_sha: str) -> qelt.QELongTrendJobRequest:
    bundle = _bundle(environment_sha)
    snapshot = {
        "snapshot_id": "qlib-st-pit-active-h5-daily-20180801-20260630",
        "manifest_sha256": "c" * 64,
        "start_date": "2018-08-01",
        "end_date": "2026-06-30",
        "lineage_parent_ids": [],
    }
    return qelt.QELongTrendJobRequest(
        evaluation_id="qelt_" + "a" * 64,
        run_id="qe_task_loop:task-1:Loop1",
        node_id="wsl2-5080",
        profile_id="qe_long_trend_v1",
        profile_sha256="d" * 64,
        evaluator_version="qe_long_trend_evaluator_v1",
        evaluator_source_sha256="b" * 64,
        execution_environment_snapshot_id="qeenv-fixture",
        execution_environment_manifest_sha256=environment_sha,
        bundle_sha256=str(bundle["bundle_sha256"]),
        qe_dataset_contract_id="qlib-st-pit-active-h5-daily-20180801-20260630",
        feature_snapshot=snapshot,
        outcome_snapshot=snapshot,
        feature_data_root_uri="/home/qe/factor_data",
        outcome_data_root_uri="/home/qe/factor_data",
        input_manifest_sha256="e" * 64,
        input_artifact_hashes={"prediction_sha256": "f" * 64},
        artifact_paths={"prediction": "mlruns/exp/rec/artifacts/pred.pkl"},
        artifact_hashes={"prediction": "f" * 64},
        recorder_ref={"experiment_id": "exp", "recorder_id": "rec"},
        catalog_digest="1" * 64,
        catalog_completeness="complete",
        backtest_freq="1day",
        evaluation_asof="2026-06-30",
        bundle=bundle,
        resource_session={"session_id": "qers-qelt", "source_run_key": "qelt:qelt_" + "a" * 64},
        resource_session_token="resource-secret",
        resource_callback_url="http://127.0.0.1:8001/api/v1/qe-resource",
    )


def test_job_reservation_is_secret_free_idempotent_and_conflict_detecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_sha = "2" * 64
    monkeypatch.setattr(
        qelt,
        "get_execution_environment_identity",
        lambda: {
            "execution_environment_snapshot_id": "qeenv-fixture",
            "execution_environment_manifest_sha256": environment_sha,
        },
    )
    loop_dir = tmp_path / "task-1" / "Loop1"
    loop_dir.mkdir(parents=True)
    request = _request(environment_sha)

    first = qelt.reserve_long_trend_job(
        loop_dir=loop_dir,
        task_id="task-1",
        loop_id="Loop1",
        request=request,
    )
    replay = qelt.reserve_long_trend_job(
        loop_dir=loop_dir,
        task_id="task-1",
        loop_id="Loop1",
        request=request,
    )
    assert first["duplicate_replay"] is False
    assert replay["duplicate_replay"] is True
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(
            pool.map(
                lambda _index: qelt.reserve_long_trend_job(
                    loop_dir=loop_dir,
                    task_id="task-1",
                    loop_id="Loop1",
                    request=request,
                ),
                range(4),
            ),
        )
    assert all(item["duplicate_replay"] is True for item in concurrent)
    job_dir = loop_dir / "long_trend_evaluations" / request.evaluation_id
    public_request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    assert "resource_session_token" not in public_request
    assert "resource-secret" not in json.dumps(public_request)
    secret = json.loads((job_dir / "secret.json").read_text(encoding="utf-8"))
    assert secret["resource_session_token"] == "resource-secret"

    changed = request.model_copy(update={"strategy_topk": 31})
    with pytest.raises(qelt.QELongTrendNodeError) as exc_info:
        qelt.reserve_long_trend_job(
            loop_dir=loop_dir,
            task_id="task-1",
            loop_id="Loop1",
            request=changed,
        )
    assert exc_info.value.reason_code == "QELT_NODE_JOB_IDENTITY_CONFLICT"


def test_incomplete_pre_job_directory_is_rebuilt_without_touching_durable_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_sha = "2" * 64
    monkeypatch.setattr(
        qelt,
        "get_execution_environment_identity",
        lambda: {
            "execution_environment_snapshot_id": "qeenv-fixture",
            "execution_environment_manifest_sha256": environment_sha,
        },
    )
    loop_dir = tmp_path / "task-1" / "Loop1"
    evaluation_id = "qelt_" + "a" * 64
    orphan = loop_dir / "long_trend_evaluations" / evaluation_id
    orphan.mkdir(parents=True)
    (orphan / "partial.tmp").write_text("crash", encoding="utf-8")

    receipt = qelt.reserve_long_trend_job(
        loop_dir=loop_dir,
        task_id="task-1",
        loop_id="Loop1",
        request=_request(environment_sha),
    )
    assert receipt["duplicate_replay"] is False
    assert (orphan / "job.json").is_file()
    assert not (orphan / "partial.tmp").exists()


def test_unconfigured_workspace_router_is_explicitly_unavailable() -> None:
    router = qelt.build_long_trend_router(None)
    create_route = next(route for route in router.routes if route.path.endswith("/long-trend-evaluations"))
    with pytest.raises(HTTPException) as exc_info:
        import asyncio

        asyncio.run(create_route.endpoint("task", "Loop1", _request("2" * 64), object()))
    assert getattr(exc_info.value, "status_code", None) == 503


def test_artifact_catalog_never_exposes_job_secret(tmp_path: Path) -> None:
    job_dir = tmp_path / "qelt_" / "job"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "evaluation_id": "qelt_" + "a" * 64,
                "status": "queued",
                "current_attempt_id": None,
            },
        ),
        encoding="utf-8",
    )
    (job_dir / "secret.json").write_text('{"resource_session_token":"secret"}', encoding="utf-8")
    catalog = qelt.build_long_trend_artifact_catalog(job_dir)
    assert catalog["artifacts"] == []
    assert "secret" not in json.dumps(catalog)
    with pytest.raises(HTTPException) as forbidden:
        qelt._artifact_download_path(job_dir, "request.json")
    assert forbidden.value.status_code == 403


def test_artifact_download_is_limited_to_current_catalog_members(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    artifacts = job_dir / "attempts" / "attempt-1" / "artifacts"
    artifacts.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "evaluation_id": "qelt_" + "a" * 64,
                "status": "succeeded",
                "current_attempt_id": "attempt-1",
            },
        ),
        encoding="utf-8",
    )
    receipt = artifacts / "worker_compact_receipt.json"
    receipt.write_text('{"schema_version":"qe_long_trend_worker_compact_v1"}', encoding="utf-8")
    assert qelt._artifact_download_path(
        job_dir,
        "attempts/attempt-1/artifacts/worker_compact_receipt.json",
    ) == receipt.resolve()


def test_general_workspace_catalog_excludes_qelt_secret(tmp_path: Path) -> None:
    loop_dir = tmp_path / "task-1" / "Loop1"
    job_dir = loop_dir / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    job_dir.mkdir(parents=True)
    (loop_dir / "status.txt").write_text("completed", encoding="utf-8")
    (job_dir / "job.json").write_text('{"status":"queued"}', encoding="utf-8")
    (job_dir / "secret.json").write_text('{"resource_session_token":"secret"}', encoding="utf-8")

    catalog = build_workspace_catalog(loop_dir, task_id="task-1", loop_id="Loop1")
    paths = {row["relative_path"] for row in catalog["files"]}
    assert any(path.endswith("/job.json") for path in paths)
    assert not any(path.endswith("/secret.json") for path in paths)
    assert "restricted_qelt_secret_not_catalogued" in catalog["warnings"]


def test_long_trend_snapshot_identity_survives_missing_legacy_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "factor_data"
    root.mkdir()
    (root / "meta.json").write_text(
        json.dumps(
            {
                "snapshot_id": "qlib-st-pit-active-h5-daily-20180801-20260630",
                "start": "2018-08-01",
                "end": "2026-06-30",
                "lineage_parent_ids": [],
            },
        ),
        encoding="utf-8",
    )
    (root / "daily_pv.h5").write_bytes(b"daily")
    (root / "sector_data.h5").write_bytes(b"sector")
    monkeypatch.setenv("RDAGENT_FACTOR_DATA_WSL", str(root))
    monkeypatch.delenv("QE_QLIB_DATA_PATH", raising=False)
    monkeypatch.delenv("QE_DATASET_IDENTITY_ROOTS", raising=False)
    monkeypatch.delenv("QE_REGISTERED_DATASET_ROOTS", raising=False)

    identity = read_dataset_identity(data_root_uri=str(root), node_id="wsl2-5080")
    assert identity["complete"] is False
    assert identity["reason_code"] == "qe_dataset_manifest_missing"
    assert identity["long_trend_snapshot"]["end_date"] == "2026-06-30"
    assert len(identity["long_trend_snapshot"]["manifest_sha256"]) == 64


def test_typed_cancel_only_signals_the_exact_current_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "job"
    attempt_dir = job_dir / "attempts" / "attempt-1"
    attempt_dir.mkdir(parents=True)
    process_identity = {"pid": 1234, "start_ticks": 99, "command_sha256": "a" * 64}
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "evaluation_id": "qelt_" + "a" * 64,
                "request_sha": "b" * 64,
                "status": "running",
                "current_attempt_id": "attempt-1",
            },
        ),
        encoding="utf-8",
    )
    (attempt_dir / "process_identity.json").write_text(json.dumps(process_identity), encoding="utf-8")
    monkeypatch.setattr(qelt, "_process_identity_alive", lambda identity: identity == process_identity)
    signals = []
    monkeypatch.setattr(qelt.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(qelt.os, "killpg", lambda pgid, signal: signals.append((pgid, signal)), raising=False)

    receipt = qelt.cancel_long_trend_attempt(
        job_dir,
        intent=qelt.QELongTrendCancelIntent(
            expected_attempt_id="attempt-1",
            expected_process_identity=process_identity,
            expected_request_sha="b" * 64,
        ),
    )
    assert receipt["status"] == "signal_sent"
    assert receipt["process_group_id"] == 1234
    assert signals == [(1234, qelt.signal.SIGTERM)]

    with pytest.raises(qelt.QELongTrendNodeError):
        qelt.cancel_long_trend_attempt(
            job_dir,
            intent=qelt.QELongTrendCancelIntent(
                expected_attempt_id="attempt-2",
                expected_process_identity=process_identity,
                expected_request_sha="b" * 64,
            ),
        )


def test_typed_cancel_rejects_worker_that_is_not_process_group_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "job"
    attempt_dir = job_dir / "attempts" / "attempt-1"
    attempt_dir.mkdir(parents=True)
    process_identity = {"pid": 1234, "start_ticks": 99, "command_sha256": "a" * 64}
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "evaluation_id": "qelt_" + "a" * 64,
                "request_sha": "b" * 64,
                "status": "running",
                "current_attempt_id": "attempt-1",
            },
        ),
        encoding="utf-8",
    )
    (attempt_dir / "process_identity.json").write_text(json.dumps(process_identity), encoding="utf-8")
    monkeypatch.setattr(qelt, "_process_identity_alive", lambda identity: identity == process_identity)
    monkeypatch.setattr(qelt.os, "getpgid", lambda _pid: 999, raising=False)
    monkeypatch.setattr(
        qelt.os,
        "killpg",
        lambda *_args: pytest.fail("must fail closed before signaling"),
        raising=False,
    )

    with pytest.raises(qelt.QELongTrendNodeError, match="process-group leader"):
        qelt.cancel_long_trend_attempt(
            job_dir,
            intent=qelt.QELongTrendCancelIntent(
                expected_attempt_id="attempt-1",
                expected_process_identity=process_identity,
                expected_request_sha="b" * 64,
            ),
        )


def test_dispatcher_spawn_is_singleton_by_durable_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls: list[dict[str, object]] = []
    identity = {"pid": 4321, "start_ticks": 101, "command_sha256": "c" * 64}

    class Process:
        pid = 4321

    def fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        popen_calls.append({"command": command, **kwargs})
        return Process()

    monkeypatch.setattr(qelt.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(qelt, "_capture_process_identity", lambda pid: identity if pid == 4321 else None)
    monkeypatch.setattr(qelt, "_process_identity_alive", lambda value: value == identity)

    assert qelt.spawn_long_trend_dispatcher(tmp_path) is True
    assert qelt.spawn_long_trend_dispatcher(tmp_path) is False
    assert len(popen_calls) == 1
    persisted = json.loads((tmp_path / qelt.DISPATCHER_IDENTITY_FILE).read_text(encoding="utf-8"))
    assert persisted == identity
