from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdagent.app.scheduler import api_stub, worker_stub
from rdagent.app.scheduler.models import TaskRecord


def test_api_create_task_preserves_custom_type_payload_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _create(record: TaskRecord) -> TaskRecord:
        record.id = 7
        captured["record"] = record
        return record

    monkeypatch.setattr(api_stub, "create_task", _create)
    monkeypatch.setattr(
        api_stub,
        "submit_task",
        lambda *, task_id: captured.setdefault("submitted_task_id", task_id),
    )

    response = api_stub.api_create_task(
        {
            "name": "factor-refresh",
            "task_type": "official_evaluation",
            "payload": {
                "handler_task_type": "official_factor_full_compute",
                "factor_names": ["factor_a"],
            },
            "env_overrides": {"AISTOCK_DATASET_GENERATION": "r8"},
        },
    )

    record = captured["record"]
    assert isinstance(record, TaskRecord)
    assert record.task_type == "official_evaluation"
    assert record.payload["handler_task_type"] == "official_factor_full_compute"
    assert record.env_overrides == {"AISTOCK_DATASET_GENERATION": "r8"}
    assert response["task"]["task_type"] == "official_evaluation"
    assert captured["submitted_task_id"] == "7"


def test_legacy_task_record_defaults_to_rdagent() -> None:
    record = TaskRecord(name="legacy")
    assert record.task_type == "rdagent"
    assert record.payload == {}


def test_run_task_routes_custom_handler_without_rdagent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    task = SimpleNamespace(
        task_type="official_evaluation",
        payload={"factor_names": ["factor_a"]},
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(worker_stub, "get_task", lambda _task_id: task)
    monkeypatch.setattr(worker_stub, "_ensure_dirs", lambda: None)
    monkeypatch.setattr(
        worker_stub,
        "_run_custom_python_handler",
        lambda task_id, task_type, payload: captured.update(
            task_id=task_id, task_type=task_type, payload=payload,
        )
        or 0,
    )

    assert worker_stub.run_rdagent_task("7") == 0
    assert captured == {
        "task_id": "7",
        "task_type": "official_evaluation",
        "payload": {"factor_names": ["factor_a"]},
    }


def test_custom_handler_preserves_dataset_environment_and_result_envelope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aistock_root = tmp_path / "AIstock"
    script_dir = aistock_root / "backend" / "scripts"
    script_dir.mkdir(parents=True)
    script = script_dir / "run_official_evaluation_wsl.py"
    script.write_text(
        """
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(json.dumps({
    "type": "result",
    "data": {
        "success": payload.get("factor_names") == ["factor_a"],
        "generation": os.environ.get("AISTOCK_DATASET_GENERATION"),
    },
}))
""".strip(),
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs"
    states: list[str] = []
    results: list[dict] = []
    task = SimpleNamespace(
        env_overrides={
            "AISTOCK_REPO_ROOT": str(aistock_root),
            "AISTOCK_DATASET_GENERATION": "20260920-v14-unified",
            "AISTOCK_SCHEDULER_PYTHON": sys.executable,
        },
    )

    monkeypatch.setattr(worker_stub, "PROJECT_ROOT", tmp_path / "RD-Agent")
    monkeypatch.setattr(worker_stub, "LOG_DIR", log_dir)
    monkeypatch.setattr(worker_stub, "get_task", lambda _task_id: task)
    monkeypatch.setattr(worker_stub, "update_task_status", lambda _task_id, status: states.append(status))
    monkeypatch.setattr(worker_stub, "append_task_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_stub, "record_result", lambda _task_id, result: results.append(result))
    monkeypatch.setattr(worker_stub, "_save_pid_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_stub, "_remove_pid_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_stub, "_start_tail_thread", lambda *_args, **_kwargs: None)

    code = worker_stub._run_custom_python_handler(  # noqa: SLF001
        "7",
        "official_evaluation",
        {"factor_names": ["factor_a"]},
    )

    assert code == 0
    assert states == ["running", "success"]
    assert results[-1]["success"] is True
    assert results[-1]["generation"] == "20260920-v14-unified"
    assert results[-1]["task_type"] == "official_evaluation"


def test_unknown_custom_task_type_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    states: list[str] = []
    results: list[dict] = []
    task = SimpleNamespace(env_overrides={"AISTOCK_REPO_ROOT": str(tmp_path)})
    monkeypatch.setattr(worker_stub, "get_task", lambda _task_id: task)
    monkeypatch.setattr(worker_stub, "update_task_status", lambda _task_id, status: states.append(status))
    monkeypatch.setattr(worker_stub, "append_task_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_stub, "record_result", lambda _task_id, result: results.append(result))
    monkeypatch.setattr(worker_stub, "_remove_pid_file", lambda *_args, **_kwargs: None)

    code = worker_stub._run_custom_python_handler("8", "unknown_custom", {})  # noqa: SLF001

    assert code == 1
    assert states == ["fail"]
    assert results[-1]["success"] is False
    assert "unsupported custom scheduler task_type" in results[-1]["error"]


@pytest.mark.parametrize(
    ("task_type", "result", "expected"),
    [
        ("official_evaluation", {"success": True, "rows": 1}, "success"),
        ("official_evaluation", {}, "fail"),
        ("unknown_custom", {"success": True}, "fail"),
    ],
)
def test_recovery_requires_allowlisted_structured_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_type: str,
    result: dict[str, object],
    expected: str,
) -> None:
    recorded: list[dict[str, object]] = []
    log_path = tmp_path / "task.log"
    if result:
        log_path.write_text(f"{json.dumps(result)}\n", encoding="utf-8")
    monkeypatch.setattr(worker_stub, "record_result", lambda _task_id, item: recorded.append(item))

    status = worker_stub._finalize_recovered_task(  # noqa: SLF001
        "9",
        SimpleNamespace(task_type=task_type, rdagent_log_dir=None),
        log_path,
    )

    assert status == expected
    assert recorded[-1]["recovered_after_restart"] is True
