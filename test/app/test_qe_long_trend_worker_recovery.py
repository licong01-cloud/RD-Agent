from __future__ import annotations

# ruff: noqa: SLF001 - this unit module intentionally verifies private durable recovery primitives.
import hashlib
import io
import json
import socket
import urllib.error
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, Self

import pytest
from rdagent.app.api_endpoints import qe_long_trend_registration_replayer as replayer
from rdagent.app.api_endpoints import qe_long_trend_worker as worker
from rdagent.app.api_endpoints.qe_long_trend_worker import _queued_jobs


class _CallbackResponse:
    def __init__(self, status: int = 200, payload: dict | None = None) -> None:
        self.status = status
        self._body = json.dumps(payload or {"status": "accepted"}).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _resource_payload(*, sequence_no: int = 1, phase: str = "long_trend_eval") -> dict:
    evaluation_id = "qelt_" + "a" * 64
    return {
        "session_id": "qers_" + "b" * 32,
        "source_run_key": f"qelt:{evaluation_id}",
        "task_id": "task-1",
        "loop_id": "Loop3",
        "loop_index": 3,
        "node_id": "node-1",
        "sequence_no": sequence_no,
        "phase": phase,
        "phase_status": "running" if phase == "long_trend_eval" else phase,
        "metadata": {"evaluation_id": evaluation_id},
    }


def _write_outbox(job_dir: Path, payload: dict | None = None) -> Path:
    outbox = job_dir / "outbox" / "000001.json"
    outbox.parent.mkdir(parents=True)
    worker._atomic_json(outbox, worker._new_outbox_row(payload or _resource_payload()))
    worker._atomic_json(
        job_dir / "secret.json",
        {
            "resource_callback_url": "http://127.0.0.1:8001/api/v1/quantevolver/evolution/webhook/loop-resource-phase",
            "resource_session_token": "secret-token",
        },
    )
    return outbox


def _write_job(root: Path, task: str, loop: str, evaluation: str, created_at: str) -> Path:
    job_dir = root / task / loop / "long_trend_evaluations" / evaluation
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "evaluation_id": evaluation,
                "status": "queued",
                "created_at": created_at,
            },
        ),
        encoding="utf-8",
    )
    return job_dir


def test_cpu_slot_queue_is_fifo_by_created_at_then_identity(tmp_path: Path) -> None:
    later = _write_job(tmp_path, "task-b", "Loop2", "qelt_" + "b" * 64, "2026-07-22T02:00:00Z")
    first = _write_job(tmp_path, "task-a", "Loop1", "qelt_" + "a" * 64, "2026-07-22T01:00:00Z")
    assert _queued_jobs(tmp_path) == [first, later]


def test_pending_registration_replay_uses_fixed_hashed_loop_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_root = tmp_path / "task-1" / "Loop3"
    loop_root.mkdir(parents=True)
    adapter = loop_root / "long_trend_postprocess_adapter.py"
    descriptor = loop_root / "qe_long_trend_postprocess_descriptor.json"
    pending = loop_root / "postprocess_registration_pending.json"
    adapter.write_text("raise SystemExit(0)\n", encoding="utf-8")
    descriptor.write_text('{"schema_version":"qe_long_trend_postprocess_descriptor_v1"}', encoding="utf-8")
    pending.write_text('{"status":"pending"}', encoding="utf-8")
    pending_root = tmp_path / replayer.PENDING_DIR
    pending_root.mkdir()
    index_path = pending_root / "task-1__Loop3.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": replayer.PENDING_INDEX_SCHEMA,
                "task_id": "task-1",
                "loop_id": "Loop3",
                "loop_relative_path": "task-1/Loop3",
                "descriptor_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
                "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                "pending_receipt_sha256": hashlib.sha256(pending.read_bytes()).hexdigest(),
            },
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append({"command": command, **kwargs})
        index_path.unlink()
        pending.unlink()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(replayer.subprocess, "run", fake_run)
    replayer._replay_one(tmp_path, index_path)

    assert len(calls) == 1
    assert calls[0]["command"][1] == str(adapter)
    assert calls[0]["cwd"] == str(loop_root)
    assert "shell" not in calls[0]


def test_pending_registration_replay_rejects_path_escape(tmp_path: Path) -> None:
    pending_root = tmp_path / replayer.PENDING_DIR
    pending_root.mkdir()
    index_path = pending_root / "task-1__Loop3.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": replayer.PENDING_INDEX_SCHEMA,
                "task_id": "task-1",
                "loop_id": "Loop3",
                "loop_relative_path": "../outside",
                "descriptor_sha256": "a" * 64,
                "adapter_sha256": "c" * 64,
                "pending_receipt_sha256": "b" * 64,
            },
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="loop path is unsafe"):
        replayer._replay_one(tmp_path, index_path)


def test_pending_registration_replay_cleans_index_when_success_receipt_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_root = tmp_path / "task-1" / "Loop3"
    loop_root.mkdir(parents=True)
    adapter = loop_root / "long_trend_postprocess_adapter.py"
    descriptor = loop_root / "qe_long_trend_postprocess_descriptor.json"
    adapter.write_text("raise SystemExit(0)\n", encoding="utf-8")
    descriptor.write_text('{"schema_version":"qe_long_trend_postprocess_descriptor_v1"}', encoding="utf-8")
    (loop_root / "qe_long_trend_registration.json").write_text(
        json.dumps(
            {
                "schema_version": "qe_long_trend_registration_v1",
                "evaluation_id": "qelt_" + "a" * 64,
                "request_sha": "b" * 64,
            },
        ),
        encoding="utf-8",
    )
    pending_root = tmp_path / replayer.PENDING_DIR
    pending_root.mkdir()
    index_path = pending_root / "task-1__Loop3.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": replayer.PENDING_INDEX_SCHEMA,
                "task_id": "task-1",
                "loop_id": "Loop3",
                "loop_relative_path": "task-1/Loop3",
                "descriptor_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
                "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                "pending_receipt_sha256": "c" * 64,
            },
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        replayer.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("successful receipt must be reconciled without rerunning adapter"),
    )

    replayer._replay_one(tmp_path, index_path)

    assert not index_path.exists()


def test_registration_replayer_continues_polling_after_empty_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans: list[Path] = []
    sleeps: list[float] = []

    def record_scan(workspace: Path) -> None:
        scans.append(workspace)

    monkeypatch.setattr(replayer, "_replay_cycle", record_scan)

    def stop_after_two_scans(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:  # noqa: PLR2004 - terminate the synthetic loop after two scans.
            raise KeyboardInterrupt

    monkeypatch.setattr(replayer.time, "sleep", stop_after_two_scans)
    with pytest.raises(KeyboardInterrupt):
        replayer._run_replay_loop(tmp_path, poll_interval_seconds=7.5)

    assert scans == [tmp_path, tmp_path]
    assert sleeps == [7.5, 7.5]


def test_restart_monitor_uses_standard_failure_finalizer_when_terminal_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "task" / "Loop1" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    attempt_dir = job_dir / "attempts" / "attempt-1"
    (attempt_dir / "artifacts").mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "evaluation_id": "qelt_" + "a" * 64,
                "job_id": "job-1",
                "request_sha": "b" * 64,
                "status": "running",
                "current_attempt_id": "attempt-1",
            },
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "_process_alive", lambda _identity: False)
    events = []
    monkeypatch.setattr(worker, "_queue_resource_event", lambda *args, **kwargs: events.append((args, kwargs)))

    worker._monitor_existing(job_dir, attempt_dir, {"pid": 999})

    terminal = json.loads((attempt_dir / "artifacts" / "worker_terminal_receipt.json").read_text(encoding="utf-8"))
    compact = json.loads((attempt_dir / "artifacts" / "worker_compact_receipt.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert len(terminal["family_status"]) == 6  # noqa: PLR2004 - six approved model families.
    assert compact["family_status"] == terminal["family_status"]
    assert events[-1][1]["phase"] == "failed"


def test_cancel_delivery_failure_does_not_misclassify_worker_as_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "task" / "Loop1" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    attempt_dir = job_dir / "attempts" / "attempt-1"
    (attempt_dir / "artifacts").mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "evaluation_id": "qelt_" + "a" * 64,
                "job_id": "job-1",
                "request_sha": "b" * 64,
                "status": "running",
                "current_attempt_id": "attempt-1",
            },
        ),
        encoding="utf-8",
    )
    (attempt_dir / "cancel_intent.json").write_text(
        json.dumps({"status": "delivery_failed"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "_queue_resource_event", lambda *_args, **_kwargs: None)

    worker._finalize_attempt(job_dir, attempt_dir, returncode=2)

    terminal = json.loads((attempt_dir / "artifacts" / "worker_terminal_receipt.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["reason_code"] == "QELT_NODE_PROCESS_IDENTITY_CONFLICT"


def test_api_lifecycle_recovery_scan_replays_pending_outbox_without_new_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_calls: list[Path] = []
    registration_scans: list[Path] = []

    def record_registration_scan(root: Path) -> None:
        registration_scans.append(root)

    monkeypatch.setattr(replayer, "_replay_once", record_registration_scan)
    monkeypatch.setattr(worker, "_replay_outboxes", lambda root: replay_calls.append(root) or 1)

    replayer._run_recovery_scan(tmp_path.resolve())

    assert registration_scans == [tmp_path.resolve()]
    assert replay_calls == [tmp_path.resolve()]


def test_replay_orders_each_durable_sequence_before_later_outboxes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "task-1" / "Loop3" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    outbox_dir = job_dir / "outbox"
    outbox_dir.mkdir(parents=True)
    for sequence_no in (2, 1):
        (outbox_dir / f"{sequence_no:06d}.json").write_text("{}", encoding="utf-8")
    delivered_paths: list[Path] = []
    monkeypatch.setattr(
        worker,
        "_deliver_outbox",
        lambda _job_dir, path: delivered_paths.append(path) or True,
    )

    assert worker._replay_outboxes(tmp_path) == 0
    assert [path.name for path in delivered_paths] == ["000001.json", "000002.json"]


def test_http_409_is_structured_conflict_and_is_not_replayed_each_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "task-1" / "Loop3" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    outbox = _write_outbox(job_dir)
    calls = 0

    def raise_conflict(*_args: Any, **_kwargs: Any) -> NoReturn:
        nonlocal calls
        calls += 1
        body = io.BytesIO(
            json.dumps(
                {
                    "detail": {
                        "reason_code": "QE_RESOURCE_EVENT_PHASE_INVALID",
                        "message": "transition completed -> failed is not allowed",
                    },
                },
            ).encode("utf-8"),
        )
        callback_url = "http://callback"
        raise urllib.error.HTTPError(callback_url, worker.HTTP_CONFLICT_STATUS, "Conflict", {}, body)

    monkeypatch.setattr(worker.urllib.request, "urlopen", raise_conflict)
    assert worker._deliver_outbox(job_dir, outbox) is False
    row = json.loads(outbox.read_text(encoding="utf-8"))
    assert row["delivered"] is False
    assert row["delivery_state"] == "conflict_reconciliation_required"
    assert row["next_attempt_at"] is not None
    assert row["last_delivery_error"]["error_type"] == "http_conflict"
    assert row["last_delivery_error"]["http_status"] == worker.HTTP_CONFLICT_STATUS
    assert row["last_delivery_error"]["aistock_reason_code"] == "QE_RESOURCE_EVENT_PHASE_INVALID"
    assert row["last_delivery_error"]["payload_identity"] == row["payload_identity"]
    assert row["last_delivery_error"]["sequence_no"] == 1
    assert row["last_delivery_error"]["event_sha256"] == row["event_sha256"]

    assert worker._deliver_outbox(job_dir, outbox) is False
    assert calls == 1


def test_conflict_replays_after_persisted_backoff_and_can_be_delivered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "task-1" / "Loop3" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    outbox = _write_outbox(job_dir)
    row = json.loads(outbox.read_text(encoding="utf-8"))
    row.update(
        {
            "delivery_state": "conflict_reconciliation_required",
            "delivery_attempt_count": 1,
            "next_attempt_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            "last_delivery_error": {"reason_code": "QELT_RESOURCE_CALLBACK_HTTP_CONFLICT"},
        },
    )
    worker._atomic_json(outbox, row)
    monkeypatch.setattr(worker.urllib.request, "urlopen", lambda *_args, **_kwargs: _CallbackResponse())

    assert worker._deliver_outbox(job_dir, outbox) is True
    delivered = json.loads(outbox.read_text(encoding="utf-8"))
    assert delivered["delivered"] is True
    assert delivered["delivery_state"] == "delivered"
    assert delivered["delivery_attempt_count"] == 2  # noqa: PLR2004 - one conflict plus one successful retry.
    assert delivered["next_attempt_at"] is None
    assert delivered["last_delivery_error"] is None


@pytest.mark.parametrize(
    ("failure", "expected_state", "expected_reason", "expected_type"),
    [
        (
            urllib.error.HTTPError(
                "http://callback",
                500,
                "Unavailable",
                {},
                io.BytesIO(b'{"detail":{"reason_code":"BACKEND_UNAVAILABLE","message":"retry"}}'),
            ),
            "retryable_http",
            "QELT_RESOURCE_CALLBACK_HTTP_5XX",
            "http_5xx",
        ),
        (
            urllib.error.HTTPError(
                "http://callback",
                599,
                "Unavailable",
                {},
                io.BytesIO(b'{"detail":{"reason_code":"BACKEND_UNAVAILABLE","message":"retry"}}'),
            ),
            "retryable_http",
            "QELT_RESOURCE_CALLBACK_HTTP_5XX",
            "http_5xx",
        ),
        (
            urllib.error.HTTPError(
                "http://callback",
                600,
                "Rejected",
                {},
                io.BytesIO(b'{"detail":{"reason_code":"CALLBACK_REJECTED","message":"reject"}}'),
            ),
            "http_rejected",
            "QELT_RESOURCE_CALLBACK_HTTP_REJECTED",
            "http_rejected",
        ),
        (
            urllib.error.URLError(TimeoutError("wrapped timeout")),
            "retryable_timeout",
            "QELT_RESOURCE_CALLBACK_TIMEOUT",
            "timeout",
        ),
        (
            urllib.error.URLError(socket.timeout("socket timeout")),  # noqa: UP041 - explicit socket contract.
            "retryable_timeout",
            "QELT_RESOURCE_CALLBACK_TIMEOUT",
            "timeout",
        ),
        (
            urllib.error.URLError("connection refused"),
            "retryable_network",
            "QELT_RESOURCE_CALLBACK_NETWORK_FAILED",
            "network",
        ),
        (
            TimeoutError("timed out"),
            "retryable_timeout",
            "QELT_RESOURCE_CALLBACK_TIMEOUT",
            "timeout",
        ),
        (
            OSError("transport os failure"),
            "retryable_network",
            "QELT_RESOURCE_CALLBACK_NETWORK_FAILED",
            "os_error",
        ),
    ],
)
def test_retryable_callback_failures_keep_distinct_durable_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_state: str,
    expected_reason: str,
    expected_type: str,
) -> None:
    job_dir = tmp_path / "task-1" / "Loop3" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    outbox = _write_outbox(job_dir)

    def raise_failure(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise failure

    monkeypatch.setattr(worker.urllib.request, "urlopen", raise_failure)
    assert worker._deliver_outbox(job_dir, outbox) is False
    row = json.loads(outbox.read_text(encoding="utf-8"))
    assert row["delivery_state"] == expected_state
    assert row["last_delivery_error"]["reason_code"] == expected_reason
    assert row["last_delivery_error"]["error_type"] == expected_type
    assert row["next_attempt_at"] is not None


def test_invalid_retry_timestamp_fails_closed_before_network_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "task-1" / "Loop3" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    outbox = _write_outbox(job_dir)
    row = json.loads(outbox.read_text(encoding="utf-8"))
    row["next_attempt_at"] = "not-a-timestamp"
    worker._atomic_json(outbox, row)
    monkeypatch.setattr(
        worker.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("invalid retry timestamp must fail before network I/O"),
    )

    with pytest.raises(ValueError, match="Invalid isoformat string"):
        worker._deliver_outbox(job_dir, outbox)


def test_delivery_uses_exclusive_lock_and_delivered_outbox_is_not_resent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "task-1" / "Loop3" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    outbox = _write_outbox(job_dir)
    lock_paths: list[Path] = []
    calls = 0

    @contextmanager
    def record_lock(path: Path) -> Iterator[None]:
        lock_paths.append(path)
        yield

    def accepted(*_args: Any, **_kwargs: Any) -> _CallbackResponse:
        nonlocal calls
        calls += 1
        return _CallbackResponse()

    monkeypatch.setattr(worker, "_exclusive_file_lock", record_lock)
    monkeypatch.setattr(worker.urllib.request, "urlopen", accepted)

    assert worker._deliver_outbox(job_dir, outbox) is True
    assert worker._deliver_outbox(job_dir, outbox) is True
    assert calls == 1
    assert lock_paths == [outbox.with_suffix(".lock"), outbox.with_suffix(".lock")]


def _write_queue_job(job_dir: Path) -> dict:
    evaluation_id = job_dir.name
    job = {
        "evaluation_id": evaluation_id,
        "job_id": "qelt_job_1",
        "task_id": "task-1",
        "loop_id": "Loop3",
    }
    worker._atomic_json(
        job_dir / "request.json",
        {
            "node_id": "node-1",
            "resource_session": {
                "session_id": "qers_" + "b" * 32,
                "source_run_key": f"qelt:{evaluation_id}",
            },
        },
    )
    worker._atomic_json(
        job_dir / "secret.json",
        {
            "resource_callback_url": "http://127.0.0.1:8001/callback",
            "resource_session_token": "secret-token",
        },
    )
    return job


def test_exact_duplicate_is_delivered_once_and_different_hash_cannot_overwrite_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "task-1" / "Loop3" / "long_trend_evaluations" / ("qelt_" + "a" * 64)
    job_dir.mkdir(parents=True)
    job = _write_queue_job(job_dir)
    calls = 0

    def accepted(*_args: Any, **_kwargs: Any) -> _CallbackResponse:
        nonlocal calls
        calls += 1
        return _CallbackResponse(payload={"status": "idempotent"})

    monkeypatch.setattr(worker.urllib.request, "urlopen", accepted)
    worker._queue_resource_event(
        job_dir,
        job,
        "attempt-1",
        sequence_no=1,
        phase="long_trend_eval",
        phase_status="running",
    )
    outbox = job_dir / "outbox" / "000001.json"
    original = json.loads(outbox.read_text(encoding="utf-8"))
    assert original["payload_identity"] == worker._payload_identity(original["payload"])
    worker._queue_resource_event(
        job_dir,
        job,
        "attempt-1",
        sequence_no=1,
        phase="long_trend_eval",
        phase_status="running",
    )
    assert calls == 1
    assert json.loads(outbox.read_text(encoding="utf-8"))["event_sha256"] == original["event_sha256"]

    with pytest.raises(RuntimeError, match="QELT_RESOURCE_OUTBOX_IDENTITY_CONFLICT"):
        worker._queue_resource_event(
            job_dir,
            job,
            "attempt-2",
            sequence_no=1,
            phase="long_trend_eval",
            phase_status="running",
        )
    after_conflict = json.loads(outbox.read_text(encoding="utf-8"))
    assert after_conflict["event_sha256"] == original["event_sha256"]
    assert after_conflict["payload"]["metadata"]["attempt_id"] == "attempt-1"
