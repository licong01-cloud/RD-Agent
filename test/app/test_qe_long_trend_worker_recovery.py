from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
from rdagent.app.api_endpoints import qe_long_trend_registration_replayer as replayer
from rdagent.app.api_endpoints import qe_long_trend_worker as worker
from rdagent.app.api_endpoints.qe_long_trend_worker import _queued_jobs


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


def _write_outbox_job(root: Path, *, suffix: str = "a") -> tuple[Path, Path, dict[str, object]]:
    evaluation_id = "qelt_" + suffix * 64
    job_dir = root / "task-1" / "Loop1" / "long_trend_evaluations" / evaluation_id
    outbox = job_dir / "outbox" / "000001.json"
    outbox.parent.mkdir(parents=True)
    (job_dir / "secret.json").write_text(
        json.dumps(
            {
                "resource_callback_url": "http://127.0.0.1:8001/resource",
                "resource_session_token": "secret",
            },
        ),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "session_id": f"qers-{suffix}",
        "source_run_key": f"qelt:{evaluation_id}",
        "task_id": "task-1",
        "loop_id": "Loop1",
        "loop_index": 1,
        "node_id": "node-1",
        "sequence_no": 1,
        "phase": "long_trend_eval",
        "phase_status": "running",
        "metadata": {"evaluation_id": evaluation_id, "job_id": "job-1", "attempt_id": "attempt-1"},
    }
    worker._persist_outbox_event(outbox, payload)
    return job_dir, outbox, payload


class _Response:
    def __init__(self, status: int, payload: object) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self, _limit: int = -1) -> bytes:
        return self._body


def _http_error(status: int, payload: object) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1:8001/resource",
        status,
        "callback rejected",
        hdrs=None,
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


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

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
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

    monkeypatch.setattr(replayer, "_replay_cycle", lambda workspace: scans.append(workspace))

    def stop_after_two_scans(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
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
    assert len(terminal["family_status"]) == 6
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
    monkeypatch.setattr(replayer, "_replay_once", lambda root: registration_scans.append(root))
    monkeypatch.setattr(worker, "_replay_outboxes", lambda root: replay_calls.append(root) or 1)

    replayer._run_recovery_scan(tmp_path.resolve())

    assert registration_scans == [tmp_path.resolve()]
    assert replay_calls == [tmp_path.resolve()]


def test_same_sequence_different_event_hash_never_overwrites_durable_outbox(tmp_path: Path) -> None:
    _job_dir, outbox, payload = _write_outbox_job(tmp_path)
    original = outbox.read_bytes()

    assert worker._persist_outbox_event(outbox, payload)["event_sha256"] == worker._canonical_event_sha256(payload)
    conflicting = {**payload, "phase": "failed"}
    with pytest.raises(RuntimeError, match="QELT_RESOURCE_OUTBOX_EVENT_CONFLICT"):
        worker._persist_outbox_event(outbox, conflicting)

    assert outbox.read_bytes() == original


def test_http_409_is_structured_deferred_and_later_delivers_same_outbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _job_dir, outbox, payload = _write_outbox_job(tmp_path)
    now = datetime(2026, 7, 24, tzinfo=timezone.utc)
    clock = {"now": now}
    monkeypatch.setattr(worker, "_now_utc", lambda: clock["now"])

    conflict = _http_error(
        409,
        {
            "detail": {
                "reason_code": "QE_RESOURCE_EVENT_PHASE_INVALID",
                "message": "transition completed -> long_trend_eval is not allowed",
            },
        },
    )

    def raise_conflict(*_args: object, **_kwargs: object) -> NoReturn:
        raise conflict

    monkeypatch.setattr(worker.urllib.request, "urlopen", raise_conflict)
    assert worker._deliver_outbox(outbox.parents[1], outbox) is False
    persisted = json.loads(outbox.read_text(encoding="utf-8"))
    assert persisted["delivery_state"] == "reconciliation_required"
    assert persisted["delivered"] is False
    assert persisted["delivery_attempt_count"] == 1
    assert persisted["last_delivery_error"]["error_type"] == "http_conflict"
    assert persisted["last_delivery_error"]["http_status"] == worker.HTTP_CONFLICT_STATUS
    assert persisted["last_delivery_error"]["reason_code"] == "QE_RESOURCE_EVENT_PHASE_INVALID"
    assert persisted["payload_identity"]["sequence_no"] == 1
    assert persisted["event_sha256"] == worker._canonical_event_sha256(payload)
    next_attempt_at = datetime.fromisoformat(persisted["next_attempt_at"])

    monkeypatch.setattr(
        worker.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("deferred 409 outbox must not be sent every replay cycle"),
    )
    assert worker._replay_outboxes(tmp_path) == 1
    assert json.loads(outbox.read_text(encoding="utf-8"))["delivery_attempt_count"] == 1

    clock["now"] = next_attempt_at + timedelta(seconds=1)
    monkeypatch.setattr(
        worker.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(200, {"data": {"status": "idempotent"}}),
    )
    assert worker._replay_outboxes(tmp_path) == 0
    delivered = json.loads(outbox.read_text(encoding="utf-8"))
    assert delivered["delivery_state"] == "delivered"
    assert delivered["delivered"] is True
    assert delivered["event_sha256"] == persisted["event_sha256"]

    attempts = delivered["delivery_attempt_count"]
    monkeypatch.setattr(
        worker.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("delivered outbox must not be sent again"),
    )
    assert worker._deliver_outbox(outbox.parents[1], outbox) is True
    assert json.loads(outbox.read_text(encoding="utf-8"))["delivery_attempt_count"] == attempts


def test_api_restart_preserves_conflict_backoff_without_resending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _job_dir, outbox, _payload_value = _write_outbox_job(tmp_path, suffix="e")
    now = datetime(2026, 7, 24, tzinfo=timezone.utc)
    monkeypatch.setattr(worker, "_now_utc", lambda: now)
    conflict = _http_error(
        409,
        {"detail": {"reason_code": "QE_RESOURCE_EVENT_PHASE_INVALID", "message": "reconcile first"}},
    )

    def raise_conflict(*_args: object, **_kwargs: object) -> NoReturn:
        raise conflict

    monkeypatch.setattr(worker.urllib.request, "urlopen", raise_conflict)
    assert worker._deliver_outbox(outbox.parents[1], outbox) is False
    persisted = outbox.read_bytes()

    monkeypatch.setattr(replayer, "_replay_once", lambda _root: None)
    monkeypatch.setattr(
        worker.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("restart recovery must honor persisted next_attempt_at"),
    )
    replayer._run_recovery_scan(tmp_path.resolve())

    assert outbox.read_bytes() == persisted
    row = json.loads(outbox.read_text(encoding="utf-8"))
    assert row["delivery_state"] == "reconciliation_required"
    assert row["delivery_attempt_count"] == 1


def test_invalid_persisted_next_attempt_fails_closed_without_sending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _job_dir, outbox, _payload_value = _write_outbox_job(tmp_path, suffix="f")
    row = json.loads(outbox.read_text(encoding="utf-8"))
    row["delivery_state"] = "reconciliation_required"
    row["next_attempt_at"] = "not-a-timestamp"
    outbox.write_text(json.dumps(row), encoding="utf-8")
    monkeypatch.setattr(
        worker.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("invalid durable retry state must fail before network I/O"),
    )

    with pytest.raises(RuntimeError, match="QELT_RESOURCE_OUTBOX_NEXT_ATTEMPT_INVALID"):
        worker._deliver_outbox(outbox.parents[1], outbox)


def test_replay_delivers_same_evaluation_outboxes_in_sequence_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir, first_outbox, first_payload = _write_outbox_job(tmp_path, suffix="9")
    second_payload = {
        **first_payload,
        "sequence_no": 2,
        "phase": "completed",
        "phase_status": "succeeded",
    }
    second_outbox = first_outbox.with_name("000002.json")
    worker._persist_outbox_event(second_outbox, second_payload)
    delivered_sequences: list[int] = []

    def accept(request, **_kwargs):  # type: ignore[no-untyped-def]
        delivered_sequences.append(int(json.loads(request.data)["sequence_no"]))
        return _Response(200, {"data": {"status": "accepted"}})

    monkeypatch.setattr(worker.urllib.request, "urlopen", accept)

    assert worker._replay_outboxes(tmp_path) == 0
    assert delivered_sequences == [1, 2]
    assert json.loads((job_dir / "outbox" / "000001.json").read_text(encoding="utf-8"))["delivered"] is True
    assert json.loads((job_dir / "outbox" / "000002.json").read_text(encoding="utf-8"))["delivered"] is True


@pytest.mark.parametrize(
    ("suffix", "failure", "expected_state", "expected_error_type", "expected_reason"),
    [
        (
            "b",
            _http_error(503, {"detail": {"reason_code": "AIstock_UNAVAILABLE", "message": "retry later"}}),
            "retryable_http_error",
            "http_5xx",
            "AIstock_UNAVAILABLE",
        ),
        (
            "c",
            urllib.error.URLError(ConnectionRefusedError("connection refused")),
            "retryable_transport_error",
            "network_unreachable",
            "QELT_RESOURCE_CALLBACK_NETWORK_UNREACHABLE",
        ),
        (
            "d",
            TimeoutError("timed out"),
            "retryable_transport_error",
            "timeout",
            "QELT_RESOURCE_CALLBACK_TIMEOUT",
        ),
    ],
)
def test_http_5xx_network_and_timeout_keep_distinct_retryable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    failure: Exception,
    expected_state: str,
    expected_error_type: str,
    expected_reason: str,
) -> None:
    _job_dir, outbox, _payload_value = _write_outbox_job(tmp_path, suffix=suffix)
    monkeypatch.setattr(worker, "_now_utc", lambda: datetime(2026, 7, 24, tzinfo=timezone.utc))

    def raise_failure(*_args: object, **_kwargs: object) -> NoReturn:
        raise failure

    monkeypatch.setattr(worker.urllib.request, "urlopen", raise_failure)
    assert worker._deliver_outbox(outbox.parents[1], outbox) is False

    persisted = json.loads(outbox.read_text(encoding="utf-8"))
    assert persisted["delivery_state"] == expected_state
    assert persisted["last_delivery_error"]["error_type"] == expected_error_type
    assert persisted["last_delivery_error"]["reason_code"] == expected_reason
    assert persisted["next_attempt_at"] is not None
