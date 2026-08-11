from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import HTTPException
from starlette.requests import Request

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CURSOR_PATH = _REPO_ROOT / "rdagent" / "app" / "api_endpoints" / "qe_log_cursor.py"
_API_PATH = _REPO_ROOT / "rdagent" / "app" / "api_endpoints" / "qe_evolution_api.py"
_TAIL_ASSERTION_LINES = 10
_HTTP_GONE = 410
_HTTP_BAD_REQUEST = 400
_LARGE_APPEND_BYTES = 2 * 1024 * 1024


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"failed to load module from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cursor_module = _load_module(_CURSOR_PATH, "qe_log_cursor_under_test")
CURSOR_CONFLICT_REASON_CODE = cursor_module.CURSOR_CONFLICT_REASON_CODE
CURSOR_REASON_CODE = cursor_module.CURSOR_REASON_CODE
QELogCursorError = cursor_module.QELogCursorError
QELogCursorState = cursor_module.QELogCursorState
consume_new_lines = cursor_module.consume_new_lines
decode_cursor = cursor_module.decode_cursor
encode_cursor = cursor_module.encode_cursor
initial_cursor_state = cursor_module.initial_cursor_state
resolve_resume_cursor = cursor_module.resolve_resume_cursor
stream_uuid_for_task = cursor_module.stream_uuid_for_task
validate_cursor_state = cursor_module.validate_cursor_state


def _task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task-1"
    loop_dir = task_dir / "Loop1"
    loop_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def _request(*, last_event_id: str | None = None) -> Request:
    headers = []
    if last_event_id is not None:
        headers.append((b"last-event-id", last_event_id.encode("ascii")))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


def test_cursor_round_trip_is_opaque_and_deterministic(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    state = QELogCursorState(
        stream_uuid=stream_uuid_for_task("task-1", task_dir),
        seq=7,
        offsets={"Loop1/run.log": 123},
    )

    token = encode_cursor(state)

    assert "Loop1" not in token
    assert decode_cursor(token) == state
    assert encode_cursor(state) == token

    terminal_state = QELogCursorState(
        stream_uuid=state.stream_uuid,
        seq=state.seq + 1,
        offsets=state.offsets,
        identities=state.identities,
        terminal=True,
    )
    assert decode_cursor(encode_cursor(terminal_state)).terminal is True


@pytest.mark.parametrize(
    "token",
    ["", "not-base64", "e30", pytest.param("A" * 40000, id="oversized")],
)
def test_malformed_cursor_fails_closed(token: str) -> None:
    with pytest.raises(QELogCursorError) as exc_info:
        decode_cursor(token)
    assert exc_info.value.reason_code == CURSOR_REASON_CODE


def test_initial_tail_is_bounded_and_reconnect_does_not_replay(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    log_file = task_dir / "Loop1" / "run.log"
    log_file.write_text("".join(f"line-{index}\n" for index in range(500)), encoding="utf-8")

    state, lines = initial_cursor_state(
        task_id="task-1",
        task_dir=task_dir,
        max_lines=_TAIL_ASSERTION_LINES,
    )

    assert len(lines) == _TAIL_ASSERTION_LINES
    assert lines[0] == "[Loop1] line-490"
    assert lines[-1] == "[Loop1] line-499"
    assert state.offsets == {"Loop1/run.log": log_file.stat().st_size}
    assert state.seq == 1

    resumed = decode_cursor(encode_cursor(state))
    same_state, replay = consume_new_lines(resumed, task_id="task-1", task_dir=task_dir)
    assert replay == []
    assert same_state.seq == state.seq

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("line-500\nline-501\n")
    advanced, new_lines = consume_new_lines(same_state, task_id="task-1", task_dir=task_dir)
    assert new_lines == ["[Loop1] line-500", "[Loop1] line-501"]
    assert advanced.seq == state.seq + 1


def test_new_loop_is_read_from_zero_after_existing_cursor(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    (task_dir / "Loop1" / "run.log").write_text("old\n", encoding="utf-8")
    state, _ = initial_cursor_state(task_id="task-1", task_dir=task_dir)
    loop_two = task_dir / "Loop2"
    loop_two.mkdir()
    (loop_two / "run.log").write_text("new-loop\n", encoding="utf-8")

    advanced, lines = consume_new_lines(state, task_id="task-1", task_dir=task_dir)

    assert lines == ["[Loop2] new-loop"]
    assert advanced.seq == state.seq + 1


def test_large_append_is_consumed_in_bounded_chunks(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    log_file = task_dir / "Loop1" / "run.log"
    log_file.write_text("baseline\n", encoding="utf-8")
    state, _ = initial_cursor_state(task_id="task-1", task_dir=task_dir)
    with log_file.open("ab") as handle:
        handle.write(b"x" * _LARGE_APPEND_BYTES)

    first, first_lines = consume_new_lines(state, task_id="task-1", task_dir=task_dir)
    second, second_lines = consume_new_lines(first, task_id="task-1", task_dir=task_dir)

    assert first.offsets["Loop1/run.log"] - state.offsets["Loop1/run.log"] < _LARGE_APPEND_BYTES
    assert second.offsets["Loop1/run.log"] == log_file.stat().st_size
    assert first_lines
    assert second_lines


def test_truncated_or_wrong_stream_cursor_returns_expired(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    log_file = task_dir / "Loop1" / "run.log"
    log_file.write_text("content\n", encoding="utf-8")
    state, _ = initial_cursor_state(task_id="task-1", task_dir=task_dir)
    log_file.write_text("", encoding="utf-8")

    with pytest.raises(QELogCursorError, match="truncated") as truncated:
        validate_cursor_state(state, task_id="task-1", task_dir=task_dir)
    assert truncated.value.reason_code == CURSOR_REASON_CODE

    wrong = QELogCursorState(stream_uuid="0" * 32, seq=0, offsets={})
    with pytest.raises(QELogCursorError, match="another task") as wrong_stream:
        validate_cursor_state(wrong, task_id="task-1", task_dir=task_dir)
    assert wrong_stream.value.reason_code == CURSOR_REASON_CODE


def test_replaced_source_with_same_or_larger_size_expires_cursor(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    log_file = task_dir / "Loop1" / "run.log"
    log_file.write_text("original\n", encoding="utf-8")
    state, _ = initial_cursor_state(task_id="task-1", task_dir=task_dir)
    replacement = task_dir / "Loop1" / "replacement.log"
    replacement.write_text("replacement-content\n", encoding="utf-8")
    replacement.replace(log_file)

    with pytest.raises(QELogCursorError, match="replaced") as replaced:
        validate_cursor_state(state, task_id="task-1", task_dir=task_dir)
    assert replaced.value.reason_code == CURSOR_REASON_CODE


def test_query_and_last_event_id_must_not_conflict() -> None:
    assert resolve_resume_cursor("same", "same") == "same"
    assert resolve_resume_cursor(None, "header") == "header"
    with pytest.raises(QELogCursorError) as exc_info:
        resolve_resume_cursor("query", "header")
    assert exc_info.value.reason_code == CURSOR_CONFLICT_REASON_CODE


def test_endpoint_rejects_invalid_cursor_before_streaming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RDAGENT_STATE_ROOT", str(tmp_path / "state"))
    qe_evolution_api = _load_module(_API_PATH, f"qe_evolution_api_cursor_{tmp_path.name}")
    monkeypatch.setattr(qe_evolution_api, "_get_task_dir", lambda _task_id: _task_dir(tmp_path))

    with pytest.raises(HTTPException) as expired:
        asyncio.run(qe_evolution_api.stream_task_logs("task-1", _request(), after_cursor="invalid"))
    assert expired.value.status_code == _HTTP_GONE
    assert expired.value.detail["reason_code"] == CURSOR_REASON_CODE

    with pytest.raises(HTTPException) as conflict:
        asyncio.run(
            qe_evolution_api.stream_task_logs(
                "task-1",
                _request(last_event_id="header"),
                after_cursor="query",
            ),
        )
    assert conflict.value.status_code == _HTTP_BAD_REQUEST
    assert conflict.value.detail["reason_code"] == CURSOR_CONFLICT_REASON_CODE

    missing_dir = tmp_path / "removed-terminal-task"
    monkeypatch.setattr(qe_evolution_api, "_get_task_dir", lambda _task_id: missing_dir)
    terminal = QELogCursorState(
        stream_uuid=stream_uuid_for_task("task-1", missing_dir),
        seq=9,
        offsets={"Loop1/run.log": 123},
        terminal=True,
    )
    response = asyncio.run(
        qe_evolution_api.stream_task_logs(
            "task-1",
            _request(last_event_id=encode_cursor(terminal)),
            after_cursor=None,
        ),
    )

    async def _collect_terminal_reconnect() -> list[bytes]:
        return [chunk async for chunk in response.body_iterator]

    assert asyncio.run(_collect_terminal_reconnect()) == []


def test_sse_contract_has_ids_and_non_persistent_heartbeats() -> None:
    source = _API_PATH.read_text(encoding="utf-8")
    assert 'parts = [f"id: {cursor}"]' in source
    assert 'yield ": heartbeat\\n\\n"' in source
    assert "cursor_state is not None and cursor_state.terminal" in source
    assert "await request.is_disconnected()" in source
    assert "seen_offsets" not in source
    assert "Task directory not found yet" not in source
    assert "SSE stream timeout" not in source
