from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import os
import shlex
import stat
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_API_PATH = (
    Path(__file__).resolve().parents[2]
    / "rdagent"
    / "app"
    / "api_endpoints"
    / "qe_evolution_api.py"
)
_SECRET_FILE_NAME = "qe_resource_session_secret.json"  # noqa: S105 - filename, not a credential.
_PRIVATE_FILE_MODE = 0o600
_TEST_STDOUT_FD = 17
_FORBIDDEN_CHILD_ENV_KEYS = (
    "TDX_DB_PASSWORD",
    "DATABASE_URL",
    "PGPASSWORD",
    "POSTGRES_PASSWORD",
    "OPENAI_API_KEY",
    "SERVICE_AUTH_TOKEN",
    "GITHUB_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_CLIENT_SECRET",
    "QE_RESOURCE_SESSION_TOKEN",
    "LD_PRELOAD",
    "LD_AUDIT",
    "tdx_db_password",
    "OpenAI_Api_Key",
    "ld_preload",
)
_PRESERVED_CHILD_ENV = {
    "AISTOCK_PREDICTION_STORE_BASE_URL": "http://prediction-store:9000",
    "FACTOR_CACHE_DIR": "/frozen/factor-cache",
    "QE_RESOURCE_SESSION_ID": "qers-test",
    "QLIB_DATA_PATH": "/frozen/qlib-bin",
}


def _load_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("RDAGENT_STATE_ROOT", str(tmp_path / "state"))
    spec = importlib.util.spec_from_file_location(
        f"qe_evolution_runtime_boundary_{tmp_path.name}",
        _API_PATH,
    )
    if spec is None or spec.loader is None:
        message = f"failed to load QE evolution API from {_API_PATH}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORKSPACE_BASE = tmp_path / "workspace"
    return module


def test_spawn_uses_explicit_noninteractive_bash_and_scrubbed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_popen(argv: list[str], **kwargs: Any) -> object:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    untrusted_bash_env = str(tmp_path / "untrusted-bash-env")
    untrusted_env = str(tmp_path / "untrusted-env")
    forbidden_parent_env = dict.fromkeys(_FORBIDDEN_CHILD_ENV_KEYS, "forbidden-marker")
    parent_env = {
        "PATH": "/usr/bin",
        "QE_ALLOWED": "yes",
        **_PRESERVED_CHILD_ENV,
        **forbidden_parent_env,
        "BASH_ENV": untrusted_bash_env,
        "ENV": untrusted_env,
        "BASHOPTS": "sourcepath",
        "SHELLOPTS": "braceexpand",
        "BASH_FUNC_injected%%": "() { printf injected; }",
    }

    result = module._spawn_qe_process(  # noqa: SLF001
        command="printf '%s\\n' ready",
        stdout_fd=_TEST_STDOUT_FD,
        env=parent_env,
        cwd=tmp_path,
    )

    assert result is sentinel
    assert captured["argv"] == [
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-o",
        "errexit",
        "-o",
        "pipefail",
        "-c",
        "printf '%s\\n' ready",
    ]
    kwargs = captured["kwargs"]
    assert kwargs["stdout"] == _TEST_STDOUT_FD
    assert kwargs["stderr"] is module.subprocess.STDOUT
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert "shell" not in kwargs
    assert kwargs["env"] == {
        "PATH": "/usr/bin",
        "QE_ALLOWED": "yes",
        **_PRESERVED_CHILD_ENV,
    }
    assert parent_env["BASH_ENV"] == untrusted_bash_env
    assert "BASH_FUNC_injected%%" in parent_env
    assert all(parent_env[key] == "forbidden-marker" for key in _FORBIDDEN_CHILD_ENV_KEYS)


@pytest.mark.skipif(os.name != "posix", reason="requires the Linux QE execution boundary")
def test_spawn_propagates_pipeline_failure_and_does_not_source_bash_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    bash_env = tmp_path / "bash_env.sh"
    sourced_marker = tmp_path / "bash_env_was_sourced"
    exported_function_marker = tmp_path / "exported_function_ran"
    profile_markers = {
        name: tmp_path / f"{name.removeprefix('.')}_was_sourced"
        for name in (".bash_profile", ".profile", ".bashrc")
    }
    bash_env.write_text(
        f"printf sourced > {shlex.quote(str(sourced_marker))}\n",
        encoding="utf-8",
    )
    for profile_name, marker in profile_markers.items():
        (tmp_path / profile_name).write_text(
            f"printf sourced > {shlex.quote(str(marker))}\n",
            encoding="utf-8",
        )
    log_path = tmp_path / "run.log"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _PRIVATE_FILE_MODE)
    try:
        process = module._spawn_qe_process(  # noqa: SLF001
            command=(
                "if type injected >/dev/null 2>&1; then injected; fi; "
                "printf 'before\\n'; false | true; printf 'after\\n'"
            ),
            stdout_fd=log_fd,
            env={
                **os.environ,
                "BASH_ENV": str(bash_env),
                "BASH_FUNC_injected%%": (
                    "() { printf function > "
                    f"{shlex.quote(str(exported_function_marker))}; }}"
                ),
                "HOME": str(tmp_path),
                "LD_AUDIT": str(tmp_path / "untrusted-audit.so"),
                "LD_PRELOAD": str(tmp_path / "untrusted-preload.so"),
            },
            cwd=tmp_path,
        )
    finally:
        os.close(log_fd)

    assert process.wait(timeout=10) != 0
    output = log_path.read_text(encoding="utf-8")
    assert "before" in output
    assert "after" not in output
    assert not sourced_marker.exists()
    assert not exported_function_marker.exists()
    assert not any(marker.exists() for marker in profile_markers.values())
    assert "cannot be preloaded" not in output


@pytest.mark.skipif(os.name != "posix", reason="requires the Linux QE execution boundary")
def test_legacy_fallback_qrun_receives_only_credential_safe_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    module.WORKSPACE_BASE = tmp_path / "workspace with spaces"
    for key in _FORBIDDEN_CHILD_ENV_KEYS:
        monkeypatch.setenv(key, "forbidden-marker")
    for key, value in _PRESERVED_CHILD_ENV.items():
        monkeypatch.setenv(key, value)

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    qrun = fake_bin / "qrun"
    qrun.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        f"forbidden = {_FORBIDDEN_CHILD_ENV_KEYS!r}\n"
        "time.sleep(0.25)\n"
        "present = sorted(key for key in forbidden if key in os.environ)\n"
        "preserved = '|'.join((\n"
        "    os.environ.get('AISTOCK_PREDICTION_STORE_BASE_URL', ''),\n"
        "    os.environ.get('QE_RESOURCE_SESSION_ID', ''),\n"
        "))\n"
        "Path('fallback_env_probe.txt').write_text(\n"
        "    ','.join(present) + '\\n' + preserved,\n"
        "    encoding='utf-8',\n"
        ")\n"
        "raise SystemExit(91 if present else 0)\n",
        encoding="utf-8",
    )
    qrun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    task_id = "fallback_task"
    loop_id = "Loop1"
    intent_hash = hashlib.sha256(b"credential-safe-fallback").hexdigest()
    experiment_files = {"conf.yaml": "model: test"}
    loop_dir = module.WORKSPACE_BASE / task_id / loop_id
    request_digest = module.canonical_request_digest(
        loop_index=1,
        config={},
        experiment_files=experiment_files,
        wsl_command=None,
        model_source=None,
    )
    module.reserve_submission(
        loop_dir,
        task_id=task_id,
        loop_id=loop_id,
        submission_intent_hash=intent_hash,
        request_digest=request_digest,
    )

    asyncio.run(
        module._run_qlib_backtest(  # noqa: SLF001
            task_id,
            loop_id,
            {},
            experiment_files,
            wsl_command=None,
            submission_intent_hash=intent_hash,
        ),
    )

    probe = (loop_dir / "fallback_env_probe.txt").read_text(encoding="utf-8")
    receipt = module.get_submission_receipt(
        loop_dir,
        loop_id=loop_id,
        submission_intent_hash=intent_hash,
    )
    assert probe == (
        "\n"
        f"{_PRESERVED_CHILD_ENV['AISTOCK_PREDICTION_STORE_BASE_URL']}|"
        f"{_PRESERVED_CHILD_ENV['QE_RESOURCE_SESSION_ID']}"
    )
    assert receipt is not None
    assert receipt["status"] == "completed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are the production contract")
def test_resource_session_secret_is_private_from_initial_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    target = tmp_path / _SECRET_FILE_NAME
    original_fdopen = module.os.fdopen
    observed_open_modes: list[int] = []

    def inspect_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        observed_open_modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return original_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(module.os, "fdopen", inspect_fdopen)

    decoded = module._write_experiment_file(  # noqa: SLF001
        target=target,
        relative_path=_SECRET_FILE_NAME,
        content='{"secret":"initial"}',
    )

    assert decoded is False
    assert observed_open_modes == [_PRIVATE_FILE_MODE]
    assert target.read_text(encoding="utf-8") == '{"secret":"initial"}'
    assert stat.S_IMODE(target.stat().st_mode) == _PRIVATE_FILE_MODE
    assert not list(tmp_path.glob(f".{_SECRET_FILE_NAME}.*.tmp"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are the production contract")
def test_resource_session_secret_overwrite_replaces_permissive_inode_with_private_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    target = tmp_path / _SECRET_FILE_NAME
    target.write_text("old", encoding="utf-8")
    target.chmod(0o644)

    module._write_experiment_file(  # noqa: SLF001
        target=target,
        relative_path=_SECRET_FILE_NAME,
        content="new",
    )

    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == _PRIVATE_FILE_MODE
    assert not list(tmp_path.glob(f".{_SECRET_FILE_NAME}.*.tmp"))


def test_resource_session_secret_replace_failure_cleans_temporary_and_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    target = tmp_path / _SECRET_FILE_NAME
    target.write_text("old", encoding="utf-8")
    replace_failure = "injected replace failure"

    def fail_replace(_source: Any, _target: Any) -> None:
        raise OSError(replace_failure)

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        module._write_experiment_file(  # noqa: SLF001
            target=target,
            relative_path=_SECRET_FILE_NAME,
            content="new",
        )

    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(f".{_SECRET_FILE_NAME}.*.tmp"))


def test_resource_session_secret_fsync_failure_cleans_temporary_and_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    target = tmp_path / _SECRET_FILE_NAME
    target.write_text("old", encoding="utf-8")
    fsync_failure = "injected fsync failure"

    def fail_fsync(_fd: int) -> None:
        raise OSError(fsync_failure)

    monkeypatch.setattr(module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match=fsync_failure):
        module._write_experiment_file(  # noqa: SLF001
            target=target,
            relative_path=_SECRET_FILE_NAME,
            content="new",
        )

    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(f".{_SECRET_FILE_NAME}.*.tmp"))


def test_ordinary_experiment_text_file_keeps_existing_write_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    target = tmp_path / "conf.yaml"
    target.write_text("old", encoding="utf-8")
    if os.name == "posix":
        target.chmod(0o640)
    original_mode = stat.S_IMODE(target.stat().st_mode)

    def reject_private_writer(_target: Path, _content: str) -> None:
        message = "ordinary experiment files must not use the secret writer"
        raise AssertionError(message)

    monkeypatch.setattr(module, "_atomic_write_private_text", reject_private_writer)
    decoded = module._write_experiment_file(  # noqa: SLF001
        target=target,
        relative_path="conf.yaml",
        content="new",
    )

    assert decoded is False
    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == original_mode


def test_ordinary_base64_experiment_file_keeps_binary_decode_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    target = tmp_path / "payload.bin"
    payload = b"\x00qe-binary\xff"

    decoded = module._write_experiment_file(  # noqa: SLF001
        target=target,
        relative_path="payload.bin.b64",
        content=base64.b64encode(payload).decode("ascii"),
    )

    assert decoded is True
    assert target.read_bytes() == payload


def test_resource_session_secret_rejects_base64_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    target = tmp_path / _SECRET_FILE_NAME

    with pytest.raises(RuntimeError, match="QE_RESOURCE_SESSION_SECRET_ENCODING_INVALID"):
        module._write_experiment_file(  # noqa: SLF001
            target=target,
            relative_path=f"{_SECRET_FILE_NAME}.b64",
            content="e30=",
        )

    assert not target.exists()


def test_background_writer_routes_resource_session_secret_through_private_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_api(tmp_path, monkeypatch)
    task_id = "qe_task"
    loop_id = "Loop1"
    intent_hash = hashlib.sha256(b"runtime-boundary-integration").hexdigest()
    experiment_files = {_SECRET_FILE_NAME: "session-envelope"}
    loop_dir = module.WORKSPACE_BASE / task_id / loop_id
    request_digest = module.canonical_request_digest(
        loop_index=1,
        config={},
        experiment_files=experiment_files,
        wsl_command="true",
        model_source=None,
    )
    module.reserve_submission(
        loop_dir,
        task_id=task_id,
        loop_id=loop_id,
        submission_intent_hash=intent_hash,
        request_digest=request_digest,
    )
    original_writer = module._write_experiment_file  # noqa: SLF001
    routed_paths: list[str] = []

    def capture_writer(*, target: Path, relative_path: str, content: str) -> bool:
        routed_paths.append(relative_path)
        return original_writer(
            target=target,
            relative_path=relative_path,
            content=content,
        )

    spawn_failure = "stop after experiment-file materialization"

    def reject_spawn(**_kwargs: Any) -> Any:
        raise RuntimeError(spawn_failure)

    monkeypatch.setattr(module, "_write_experiment_file", capture_writer)
    monkeypatch.setattr(module, "_spawn_qe_process", reject_spawn)

    asyncio.run(
        module._run_qlib_backtest(  # noqa: SLF001
            task_id,
            loop_id,
            {},
            experiment_files,
            wsl_command="true",
            submission_intent_hash=intent_hash,
        ),
    )

    target = loop_dir / _SECRET_FILE_NAME
    receipt = module.get_submission_receipt(
        loop_dir,
        loop_id=loop_id,
        submission_intent_hash=intent_hash,
    )
    assert routed_paths == [_SECRET_FILE_NAME]
    assert target.read_text(encoding="utf-8") == "session-envelope"
    assert receipt is not None
    assert receipt["status"] == "failed"
    assert spawn_failure in (loop_dir / "error.log").read_text(encoding="utf-8")
