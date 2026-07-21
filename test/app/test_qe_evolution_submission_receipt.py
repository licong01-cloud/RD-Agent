from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import multiprocessing
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from rdagent.app.api_endpoints.qe_kill_receipt import (
    KillReceiptConflictError,
    execute_typed_kill_intent,
)
from rdagent.app.api_endpoints.qe_submission_receipt import (
    SubmissionReceiptError,
    SubmissionReceiptTransitionError,
    canonical_request_digest,
    get_submission_receipt,
    reserve_submission,
    transition_submission_receipt,
)

if TYPE_CHECKING:
    from types import ModuleType

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_FORBIDDEN = 403
HTTP_CONFLICT = 409
HTTP_INTERNAL_SERVER_ERROR = 500
TEST_PROCESS_ID = 12345
EXPECTED_RETRY_RECEIPT_COUNT = 2
TYPED_KILL_COMMAND_ID = "macmd_2b84ea4e40d2d69ca8cc3c71d938ad30"
TYPED_KILL_INTENT_HASH = hashlib.sha256(b"typed-kill-intent").hexdigest()
TYPED_PROCESS_IDENTITY = {
    "pid": 43210,
    "pgid": 43210,
    "start_time_ticks": 987654321,
}

_API_PATH = (
    Path(__file__).resolve().parents[2]
    / "rdagent"
    / "app"
    / "api_endpoints"
    / "qe_evolution_api.py"
)


class CapturingBackgroundTasks:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def add_task(self, func: Any, *args: Any, **kwargs: Any) -> None:
        self.calls.append((func, args, kwargs))


def _load_api(workspace_root: Path, *, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, _API_PATH)
    if spec is None or spec.loader is None:
        message = f"failed to load QE workspace API from {_API_PATH}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORKSPACE_BASE = workspace_root
    return module


def _request(
    module: ModuleType,
    *,
    intent_seed: str = "intent-a",
    config_value: int = 1,
    callback_url: str = "http://callback-a",
) -> Any:
    return module.LoopRunRequest(
        loop_index=1,
        config={"value": config_value},
        experiment_files={"conf.yaml": "model: test"},
        wsl_command="python run.py",
        callback_url=callback_url,
        model_source={"source_task_id": "source", "source_loop": "Loop2"},
        submission_intent_hash=hashlib.sha256(intent_seed.encode("utf-8")).hexdigest(),
    )


def _create(module: ModuleType, request: Any) -> tuple[Any, CapturingBackgroundTasks]:
    background = CapturingBackgroundTasks()
    response = asyncio.run(module.create_and_run_loop("qe_task", request, background))
    return response, background


def _reserve_in_child(
    workspace_root: str,
    intent_hash: str,
    request_digest: str,
    result_queue: Any,
) -> None:
    receipt, created = reserve_submission(
        Path(workspace_root) / "qe_task" / "Loop1",
        task_id="qe_task",
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        request_digest=request_digest,
    )
    result_queue.put(("ok", created, receipt["status"]))


def _reserve_running_submission_with_identity(
    tmp_path: Path,
) -> tuple[Path, str]:
    loop_dir = tmp_path / "qe_task" / "Loop1"
    intent_hash = hashlib.sha256(b"typed-kill-submission").hexdigest()
    request_digest = hashlib.sha256(b"typed-kill-request").hexdigest()
    reserve_submission(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        request_digest=request_digest,
    )
    transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        status="started",
    )
    transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        status="running",
        process_identity=TYPED_PROCESS_IDENTITY,
    )
    return loop_dir, intent_hash


def test_concurrent_same_hash_post_registers_execution_once(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_concurrent_api")
    barrier = threading.Barrier(2)

    def submit() -> tuple[Any, CapturingBackgroundTasks]:
        barrier.wait(timeout=5)
        return _create(module, _request(module))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: submit(), range(2)))

    responses = [result[0] for result in results]
    backgrounds = [result[1] for result in results]
    assert sum(len(background.calls) for background in backgrounds) == 1
    assert sorted(response.duplicate_replay for response in responses) == [False, True]
    assert {response.loop_id for response in responses} == {"Loop1"}
    assert len({response.request_digest for response in responses}) == 1


def test_cross_process_same_hash_reservation_creates_one_receipt(tmp_path: Path) -> None:
    intent_hash = hashlib.sha256(b"cross-process-intent").hexdigest()
    request_digest = hashlib.sha256(b"cross-process-request").hexdigest()
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_reserve_in_child,
            args=(str(tmp_path), intent_hash, request_digest, result_queue),
        )
        for _index in range(2)
    ]

    timeout_seconds = 45 if os.name == "nt" else 15
    deadline = time.monotonic() + timeout_seconds
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))

        alive = [process.pid for process in processes if process.is_alive()]
        assert not alive, (
            "cross-process receipt reservation did not finish within the bounded "
            f"timeout: pids={alive} timeout_seconds={timeout_seconds}"
        )
        assert [process.exitcode for process in processes] == [0, 0]
        outcomes = [result_queue.get(timeout=5) for _index in processes]
        assert sorted(outcomes) == [("ok", False, "reserved"), ("ok", True, "reserved")]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()


def test_different_hash_returns_409(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_conflict_api")
    _create(module, _request(module, intent_seed="first"))

    with pytest.raises(HTTPException) as exc_info:
        _create(module, _request(module, intent_seed="different"))

    assert exc_info.value.status_code == HTTP_CONFLICT
    assert exc_info.value.detail["reason_code"] == "qe_workspace_submission_identity_conflict"


def test_same_intent_with_different_execution_payload_returns_409(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_digest_conflict_api")
    _create(module, _request(module, config_value=1))

    with pytest.raises(HTTPException) as exc_info:
        _create(module, _request(module, config_value=2))

    assert exc_info.value.status_code == HTTP_CONFLICT
    assert exc_info.value.detail["reason_code"] == "qe_workspace_submission_identity_conflict"


def test_callback_url_is_not_part_of_execution_digest(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_callback_api")
    first, first_background = _create(module, _request(module, callback_url="http://callback-a"))
    replay, replay_background = _create(module, _request(module, callback_url="http://callback-b"))

    assert first.duplicate_replay is False
    assert replay.duplicate_replay is True
    assert first.request_digest == replay.request_digest
    assert len(first_background.calls) == 1
    assert replay_background.calls == []


def test_receipt_survives_server_restart(tmp_path: Path) -> None:
    first_module = _load_api(tmp_path, module_name="qe_receipt_restart_api_first")
    created, _background = _create(first_module, _request(first_module))

    second_module = _load_api(tmp_path, module_name="qe_receipt_restart_api_second")
    receipt = asyncio.run(second_module.get_loop_submission("qe_task", "Loop1"))
    replay, replay_background = _create(second_module, _request(second_module))

    assert receipt["submission_intent_hash"] == created.submission_intent_hash
    assert receipt["request_digest"] == created.request_digest
    assert receipt["status"] == "reserved"
    assert replay.duplicate_replay is True
    assert replay_background.calls == []


def test_status_distinguishes_not_reserved_from_reserved_not_started(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_status_api")

    assert asyncio.run(module.get_loop_submission("qe_task", "Loop1"))["status"] == "not_reserved"
    assert asyncio.run(module.get_loop_status("qe_task", "Loop1"))["status"] == "not_found"

    created, _background = _create(module, _request(module))
    status = asyncio.run(module.get_loop_status("qe_task", "Loop1"))

    assert created.receipt_status == "reserved"
    assert status["status"] == "reserved_not_started"
    assert status["receipt_status"] == "reserved"
    assert status["submission_intent_hash"] == created.submission_intent_hash


def test_terminal_status_query_repairs_nonterminal_receipt_before_return(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_terminal_reconcile_api")
    created, _background = _create(module, _request(module))
    loop_dir = tmp_path / "qe_task" / "Loop1"
    transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=created.submission_intent_hash,
        status="started",
    )
    transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=created.submission_intent_hash,
        status="running",
        pid=TEST_PROCESS_ID,
    )
    loop_dir.mkdir(parents=True)
    (loop_dir / "status.txt").write_text("completed", encoding="utf-8")

    status = asyncio.run(module.get_loop_status("qe_task", "Loop1"))
    receipt = asyncio.run(module.get_loop_submission("qe_task", "Loop1"))

    assert status["status"] == "completed"
    assert status["receipt_status"] == "completed"
    assert receipt["status"] == "completed"


def test_reserved_submission_can_be_cancelled_before_process_start(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_cancel_api")
    _created, _background = _create(module, _request(module))

    result = asyncio.run(module.kill_loop("qe_task", "Loop1"))
    receipt = asyncio.run(module.get_loop_submission("qe_task", "Loop1"))

    assert result == {
        "killed": False,
        "pid": None,
        "status": "cancelled",
        "receipt_status": "cancelled",
    }
    assert receipt["status"] == "cancelled"
    assert (tmp_path / "qe_task" / "Loop1" / "status.txt").read_text() == "cancelled"


def test_typed_kill_pre_start_cancellation_persists_receipt_and_blocks_popen(
    tmp_path: Path,
) -> None:
    loop_dir = tmp_path / "qe_task" / "Loop1"
    intent_hash = hashlib.sha256(b"typed-pre-start-submission").hexdigest()
    reserve_submission(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        request_digest=hashlib.sha256(b"typed-pre-start-request").hexdigest(),
    )

    receipt = execute_typed_kill_intent(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        command_id=TYPED_KILL_COMMAND_ID,
        kill_intent_generation=1,
        kill_intent_hash=TYPED_KILL_INTENT_HASH,
        expected_submission_intent_hash=intent_hash,
        expected_process_identity=None,
        expected_phase="pre_process_start",
    )

    submission = get_submission_receipt(loop_dir, loop_id="Loop1")
    assert receipt["status"] == "cancelled"
    assert receipt["terminal_reason"] == "cancelled_before_process_start"
    assert receipt["signal_sent"] is False
    assert submission is not None
    assert submission["status"] == "cancelled"
    assert (loop_dir / "status.txt").read_text(encoding="utf-8") == "cancelled"


def test_typed_kill_uses_exact_process_incarnation_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_dir, submission_intent_hash = _reserve_running_submission_with_identity(tmp_path)
    captured_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt.capture_process_identity",
        lambda _pid: dict(TYPED_PROCESS_IDENTITY),
    )
    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt._signal_process_group",
        lambda pgid, sig: captured_signals.append((pgid, int(sig))),
    )

    first = execute_typed_kill_intent(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        command_id=TYPED_KILL_COMMAND_ID,
        kill_intent_generation=1,
        kill_intent_hash=TYPED_KILL_INTENT_HASH,
        expected_submission_intent_hash=submission_intent_hash,
        expected_process_identity=TYPED_PROCESS_IDENTITY,
        expected_phase=None,
    )
    replay = execute_typed_kill_intent(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        command_id=TYPED_KILL_COMMAND_ID,
        kill_intent_generation=1,
        kill_intent_hash=TYPED_KILL_INTENT_HASH,
        expected_submission_intent_hash=submission_intent_hash,
        expected_process_identity=TYPED_PROCESS_IDENTITY,
        expected_phase=None,
    )

    assert first["status"] == "reconciling"
    assert first["signal_sent"] is True
    assert first == replay
    assert captured_signals == [(TYPED_PROCESS_IDENTITY["pgid"], 15)]


def test_typed_kill_process_incarnation_mismatch_never_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_dir, submission_intent_hash = _reserve_running_submission_with_identity(tmp_path)
    captured_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt._signal_process_group",
        lambda pgid, sig: captured_signals.append((pgid, int(sig))),
    )
    mismatched_identity = {**TYPED_PROCESS_IDENTITY, "start_time_ticks": 987654322}

    receipt = execute_typed_kill_intent(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        command_id="macmd_6e18b58e89ba4f72bc4b6d1ed91a1b58",
        kill_intent_generation=1,
        kill_intent_hash=hashlib.sha256(b"typed-kill-mismatch").hexdigest(),
        expected_submission_intent_hash=submission_intent_hash,
        expected_process_identity=mismatched_identity,
        expected_phase=None,
    )

    assert receipt["status"] == "failed"
    assert receipt["terminal_reason"] == "kill_execution_incarnation_mismatch"
    assert captured_signals == []


def test_typed_kill_completed_result_wins_after_signal_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_dir, submission_intent_hash = _reserve_running_submission_with_identity(tmp_path)
    captured_signals: list[tuple[int, int]] = []
    observations = 0

    def capture(_pid: int) -> dict[str, int]:
        nonlocal observations
        observations += 1
        if observations == 1:
            return dict(TYPED_PROCESS_IDENTITY)
        message = "process exited after exact signal"
        raise SubmissionReceiptError(message)

    def signal_then_publish_result(pgid: int, sig: Any) -> None:
        captured_signals.append((pgid, int(sig)))
        loop_dir.mkdir(parents=True, exist_ok=True)
        (loop_dir / "qlib_results_enhanced.json").write_text('{"sharpe": 1.0}', encoding="utf-8")

    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt.capture_process_identity",
        capture,
    )
    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt._signal_process_group",
        signal_then_publish_result,
    )

    receipt = execute_typed_kill_intent(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        command_id=TYPED_KILL_COMMAND_ID,
        kill_intent_generation=1,
        kill_intent_hash=TYPED_KILL_INTENT_HASH,
        expected_submission_intent_hash=submission_intent_hash,
        expected_process_identity=TYPED_PROCESS_IDENTITY,
        expected_phase=None,
    )

    submission = get_submission_receipt(loop_dir, loop_id="Loop1")
    assert captured_signals == [(TYPED_PROCESS_IDENTITY["pgid"], 15)]
    assert receipt["status"] == "completed", receipt
    assert receipt["terminal_reason"] == "completed_result_wins_cancellation_race"
    assert receipt["signal_sent"] is True
    assert submission is not None
    assert submission["status"] == "completed"
    assert (loop_dir / "status.txt").read_text(encoding="utf-8") == "completed"


def test_typed_kill_allows_next_generation_only_after_no_signal_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_dir, submission_intent_hash = _reserve_running_submission_with_identity(tmp_path)
    first_hash = hashlib.sha256(b"typed-kill-generation-one").hexdigest()
    second_hash = hashlib.sha256(b"typed-kill-generation-two").hexdigest()
    mismatched_identity = {**TYPED_PROCESS_IDENTITY, "start_time_ticks": 987654322}
    first = execute_typed_kill_intent(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        command_id=TYPED_KILL_COMMAND_ID,
        kill_intent_generation=1,
        kill_intent_hash=first_hash,
        expected_submission_intent_hash=submission_intent_hash,
        expected_process_identity=mismatched_identity,
        expected_phase=None,
    )
    captured_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt.capture_process_identity",
        lambda _pid: dict(TYPED_PROCESS_IDENTITY),
    )
    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt._signal_process_group",
        lambda pgid, sig: captured_signals.append((pgid, int(sig))),
    )

    second = execute_typed_kill_intent(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        command_id=TYPED_KILL_COMMAND_ID,
        kill_intent_generation=2,
        kill_intent_hash=second_hash,
        expected_submission_intent_hash=submission_intent_hash,
        expected_process_identity=TYPED_PROCESS_IDENTITY,
        expected_phase=None,
    )

    assert first["status"] == "failed"
    assert first["signal_sent"] is False
    assert first["terminal_reason"] == "kill_execution_incarnation_mismatch"
    assert second["status"] == "reconciling"
    assert second["kill_intent_generation"] == first["kill_intent_generation"] + 1
    assert captured_signals == [(TYPED_PROCESS_IDENTITY["pgid"], 15)]


def test_typed_kill_rejects_conflicting_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_dir, submission_intent_hash = _reserve_running_submission_with_identity(tmp_path)
    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt.capture_process_identity",
        lambda _pid: dict(TYPED_PROCESS_IDENTITY),
    )
    monkeypatch.setattr(
        "rdagent.app.api_endpoints.qe_kill_receipt._signal_process_group",
        lambda _pgid, _sig: None,
    )
    execute_typed_kill_intent(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        command_id=TYPED_KILL_COMMAND_ID,
        kill_intent_generation=1,
        kill_intent_hash=TYPED_KILL_INTENT_HASH,
        expected_submission_intent_hash=submission_intent_hash,
        expected_process_identity=TYPED_PROCESS_IDENTITY,
        expected_phase=None,
    )

    with pytest.raises(KillReceiptConflictError):
        execute_typed_kill_intent(
            loop_dir,
            task_id="qe_task",
            loop_id="Loop1",
            command_id="macmd_6e18b58e89ba4f72bc4b6d1ed91a1b58",
            kill_intent_generation=1,
            kill_intent_hash=hashlib.sha256(b"conflicting-active-kill").hexdigest(),
            expected_submission_intent_hash=submission_intent_hash,
            expected_process_identity=TYPED_PROCESS_IDENTITY,
            expected_phase=None,
        )


def test_cancelled_reserved_background_call_does_not_start_process(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_cancel_race_api")
    _created, background = _create(module, _request(module))
    asyncio.run(module.kill_loop("qe_task", "Loop1"))

    assert len(background.calls) == 1
    func, args, kwargs = background.calls[0]
    asyncio.run(func(*args, **kwargs))

    receipt = asyncio.run(module.get_loop_submission("qe_task", "Loop1"))
    loop_dir = tmp_path / "qe_task" / "Loop1"
    assert receipt["status"] == "cancelled"
    assert (loop_dir / "status.txt").read_text() == "cancelled"
    assert not (loop_dir / "pid.txt").exists()
    assert not (loop_dir / "config.json").exists()


def test_loop_cleanup_preserves_task_level_submission_receipt(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_cleanup_api")
    _create(module, _request(module))
    loop_dir = tmp_path / "qe_task" / "Loop1"
    loop_dir.mkdir(parents=True)
    (loop_dir / "temporary.txt").write_text("temporary", encoding="utf-8")

    cleanup = asyncio.run(module.cleanup_loop_workspace("qe_task", "Loop1"))
    receipt = asyncio.run(module.get_loop_submission("qe_task", "Loop1"))

    assert cleanup["existed"] is True
    assert not loop_dir.exists()
    assert receipt["status"] == "reserved"


def test_openapi_requires_submission_intent_and_exposes_receipt_endpoint(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_openapi_api")
    app = FastAPI()
    app.include_router(module.router)
    client = TestClient(app)

    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["LoopRunRequest"]

    assert "submission_intent_hash" in request_schema["required"]
    assert (
        "/api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/submission"
        in schema["paths"]
    )
    assert "/api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/kill" in schema["paths"]
    assert "/api/v1/qe_workspace/tasks/{task_id}/loops/{loop_id}/kill-intents" in schema["paths"]
    typed_kill_schema = schema["components"]["schemas"]["TypedKillIntentRequest"]
    assert {
        "command_id",
        "kill_intent_generation",
        "kill_intent_hash",
        "expected_submission_intent_hash",
    }.issubset(typed_kill_schema["required"])


def test_invalid_submission_intent_is_explicit_and_does_not_create_receipt(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_invalid_hash_api")
    request = _request(module, callback_url="")
    request.submission_intent_hash = "not-a-sha"

    with pytest.raises(HTTPException) as exc_info:
        _create(module, request)

    assert exc_info.value.status_code == HTTP_BAD_REQUEST
    assert exc_info.value.detail["reason_code"] == "qe_workspace_submission_intent_invalid"
    receipt_dir = tmp_path / "qe_task" / ".submission_receipts"
    assert not list(receipt_dir.glob("*.json")) if receipt_dir.exists() else True


def test_corrupt_receipt_is_explicit_and_never_registers_background_task(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_corrupt_api")
    request = _request(module)
    receipt_dir = tmp_path / "qe_task" / ".submission_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / f"Loop1.{request.submission_intent_hash}.json").write_text(
        "{broken",
        encoding="utf-8",
    )
    background = CapturingBackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(module.create_and_run_loop("qe_task", request, background))

    assert exc_info.value.status_code == HTTP_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail["reason_code"] == "qe_workspace_submission_receipt_error"
    assert background.calls == []


def test_receipt_file_contains_no_callback_or_experiment_content(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_storage_api")
    request = _request(module, callback_url="http://sensitive-callback")
    _create(module, request)

    receipt_path = (
        tmp_path
        / "qe_task"
        / ".submission_receipts"
        / f"Loop1.{request.submission_intent_hash}.json"
    )
    raw = receipt_path.read_text(encoding="utf-8")

    assert "sensitive-callback" not in raw
    assert "model: test" not in raw
    assert "request_digest" in raw


def test_terminal_attempt_allows_new_retry_intent_and_cleans_loop_workspace(
    tmp_path: Path,
) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_retry_api")
    first_request = _request(module, intent_seed="first-attempt")
    first_response, _first_background = _create(module, first_request)
    loop_dir = tmp_path / "qe_task" / "Loop1"
    loop_dir.mkdir(parents=True)
    (loop_dir / "stale-artifact.txt").write_text("stale", encoding="utf-8")
    transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=first_request.submission_intent_hash,
        status="failed",
    )

    retry_request = _request(module, intent_seed="retry-attempt", config_value=2)
    retry_response, retry_background = _create(module, retry_request)

    assert first_response.duplicate_replay is False
    assert retry_response.duplicate_replay is False
    assert len(retry_background.calls) == 1
    assert not (loop_dir / "stale-artifact.txt").exists()
    receipt_files = sorted(
        (tmp_path / "qe_task" / ".submission_receipts").glob("Loop1.*.json"),
    )
    assert len(receipt_files) == EXPECTED_RETRY_RECEIPT_COUNT


def test_legacy_single_receipt_file_is_read_and_migrated_on_transition(
    tmp_path: Path,
) -> None:
    loop_dir = tmp_path / "qe_task" / "Loop1"
    intent_hash = hashlib.sha256(b"legacy-receipt").hexdigest()
    request_digest = canonical_request_digest(
        loop_index=1,
        config={"value": 1},
        experiment_files=None,
        wsl_command="python run.py",
        model_source=None,
    )
    reserve_submission(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        request_digest=request_digest,
    )
    receipt_root = tmp_path / "qe_task" / ".submission_receipts"
    hashed_path = receipt_root / f"Loop1.{intent_hash}.json"
    legacy_path = receipt_root / "Loop1.json"
    hashed_path.replace(legacy_path)

    legacy = get_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
    )
    migrated = transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        status="started",
    )

    assert legacy is not None
    assert legacy["status"] == "reserved"
    assert migrated["status"] == "started"
    assert hashed_path.exists()


def test_experiment_file_escape_fails_background_attempt_without_overwriting_receipt(
    tmp_path: Path,
) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_path_escape_api")
    request = _request(module, callback_url="")
    request.experiment_files = {
        "../.submission_receipts/Loop1.json": "malicious overwrite",
    }
    _created, background = _create(module, request)

    func, args, kwargs = background.calls[0]
    asyncio.run(func(*args, **kwargs))

    receipt = asyncio.run(
        module.get_loop_submission(
            "qe_task",
            "Loop1",
            submission_intent_hash=request.submission_intent_hash,
        ),
    )
    error_log = tmp_path / "qe_task" / "Loop1" / "error.log"
    assert receipt["status"] == "failed"
    assert "QE_WORKSPACE_PATH_ESCAPE" in error_log.read_text(encoding="utf-8")
    persisted = get_submission_receipt(
        tmp_path / "qe_task" / "Loop1",
        loop_id="Loop1",
        submission_intent_hash=request.submission_intent_hash,
    )
    assert persisted is not None
    assert persisted["submission_intent_hash"] == request.submission_intent_hash


def test_task_identity_escape_is_rejected_before_receipt_creation(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_task_escape_api")
    request = _request(module, callback_url="")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            module.create_and_run_loop(
                "..",
                request,
                CapturingBackgroundTasks(),
            ),
        )

    assert exc.value.status_code == HTTP_FORBIDDEN
    assert not (tmp_path / ".submission_receipts").exists()


def test_model_source_escape_fails_attempt_without_external_symlink(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_model_source_escape_api")
    request = _request(module, callback_url="")
    request.model_source = {
        "source_task_id": "..",
        "source_loop": "Loop2",
        "cross_node": False,
    }
    _created, background = _create(module, request)

    func, args, kwargs = background.calls[0]
    asyncio.run(func(*args, **kwargs))

    receipt = get_submission_receipt(
        tmp_path / "qe_task" / "Loop1",
        loop_id="Loop1",
        submission_intent_hash=request.submission_intent_hash,
    )
    error_log = tmp_path / "qe_task" / "Loop1" / "error.log"
    assert receipt is not None
    assert receipt["status"] == "failed"
    assert "invalid QE workspace task_id" in error_log.read_text(encoding="utf-8")
    assert not (tmp_path / "qe_task" / "Loop1" / "mlruns").exists()


@pytest.mark.parametrize("unsafe_kind", ["parent", "symlink"])
def test_tar_extraction_rejects_escape_and_link_members(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    module = _load_api(tmp_path, module_name=f"qe_receipt_tar_{unsafe_kind}_api")
    loop_dir = tmp_path / "qe_task" / "Loop1"
    loop_dir.mkdir(parents=True)
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        if unsafe_kind == "parent":
            member = tarfile.TarInfo("../escape.txt")
            data = b"escape"
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        else:
            member = tarfile.TarInfo("mlruns/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape.txt"
            archive.addfile(member)
    payload.seek(0)

    safe_extract = module._safe_extract_tar_to_loop  # noqa: SLF001
    with (
        tarfile.open(fileobj=payload, mode="r:gz") as archive,
        pytest.raises(RuntimeError, match=r"QE_WORKSPACE_(PATH_ESCAPE|TAR_UNSAFE)"),
    ):
        safe_extract(archive, loop_dir)

    assert not (tmp_path / "qe_task" / "escape.txt").exists()


def test_tar_extraction_preserves_valid_nested_mlruns_files(tmp_path: Path) -> None:
    module = _load_api(tmp_path, module_name="qe_receipt_tar_valid_api")
    loop_dir = tmp_path / "qe_task" / "Loop1"
    loop_dir.mkdir(parents=True)
    payload = io.BytesIO()
    expected = b"model-params"
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        member = tarfile.TarInfo("mlruns/1/run/artifacts/params.pkl")
        member.size = len(expected)
        archive.addfile(member, io.BytesIO(expected))
    payload.seek(0)

    safe_extract = module._safe_extract_tar_to_loop  # noqa: SLF001
    with tarfile.open(fileobj=payload, mode="r:gz") as archive:
        safe_extract(archive, loop_dir)

    assert (
        loop_dir / "mlruns" / "1" / "run" / "artifacts" / "params.pkl"
    ).read_bytes() == expected


def test_receipt_state_machine_is_strict_and_terminal_is_immutable(tmp_path: Path) -> None:
    loop_dir = tmp_path / "qe_task" / "Loop1"
    intent_hash = hashlib.sha256(b"state-machine").hexdigest()
    request_digest = canonical_request_digest(
        loop_index=1,
        config={"value": 1},
        experiment_files=None,
        wsl_command="python run.py",
        model_source=None,
    )
    reserve_submission(
        loop_dir,
        task_id="qe_task",
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        request_digest=request_digest,
    )

    transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        status="started",
    )
    transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        status="running",
        pid=TEST_PROCESS_ID,
    )
    completed = transition_submission_receipt(
        loop_dir,
        loop_id="Loop1",
        submission_intent_hash=intent_hash,
        status="completed",
    )

    assert completed["status"] == "completed"
    assert completed["pid"] == TEST_PROCESS_ID
    assert completed["started_at"]
    assert completed["running_at"]
    assert completed["finished_at"]
    assert get_submission_receipt(loop_dir, loop_id="Loop1") == completed

    with pytest.raises(SubmissionReceiptTransitionError):
        transition_submission_receipt(
            loop_dir,
            loop_id="Loop1",
            submission_intent_hash=intent_hash,
            status="running",
            pid=TEST_PROCESS_ID,
        )
    assert not list((tmp_path / "qe_task" / ".submission_receipts").glob("*.tmp"))
