from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_registration_replayer_continues_polling_after_empty_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans: list[Path] = []
    sleeps: list[float] = []

    monkeypatch.setattr(replayer, "_replay_once", lambda workspace: scans.append(workspace))

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
