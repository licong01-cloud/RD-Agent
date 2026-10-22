from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdagent.app.runtime_state import require_state_root
from tools.rdagent_release import _write_deployment_environment, build_release


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_state_root_is_required_and_external(monkeypatch, tmp_path):
    monkeypatch.delenv("RDAGENT_STATE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="RDAGENT_STATE_ROOT_REQUIRED"):
        require_state_root()
    monkeypatch.setenv("RDAGENT_STATE_ROOT", str(tmp_path / "state"))
    assert require_state_root() == (tmp_path / "state").resolve()


def test_scheduler_config_and_history_use_external_state_in_fresh_process(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    state = tmp_path / "state"
    config = state / "config" / "rdagent.env"
    config.parent.mkdir(parents=True)
    config.write_text("MODE=before\n", encoding="utf-8")
    code = """
from pathlib import Path
import os
from rdagent.app.scheduler import config_service, storage, template_service
root = Path(os.environ['RDAGENT_STATE_ROOT']).resolve()
assert config_service.ENV_PATH == root / 'config' / 'rdagent.env'
assert config_service.HISTORY_ROOT == storage.HISTORY_ROOT == template_service.HISTORY_ROOT == root / 'history'
assert storage.META_FILE == root / 'history' / '.meta_history.jsonl'
assert config_service.read_env() == 'MODE=before\\n'
config_service.write_env('MODE=after\\n', task_id='release-test')
assert config_service.read_env() == 'MODE=after\\n'
records = storage.list_history()
assert len(records) == 1 and records[0].task_id == 'release-test'
backup = Path(records[0].backup_path)
assert backup.is_relative_to(root / 'history')
assert backup.read_text() == 'MODE=before\\n'
"""
    environment = {**os.environ, "PYTHONPATH": str(repo), "RDAGENT_STATE_ROOT": str(state)}
    subprocess.run(  # noqa: S603 - fixed interpreter and test-owned code/state
        [sys.executable, "-B", "-c", code], cwd=tmp_path, env=environment, check=True,
    )


def test_scheduler_history_rejects_missing_state_root_in_fresh_process(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(repo)}
    environment.pop("RDAGENT_STATE_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-B", "-c", "from rdagent.app.scheduler import storage"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "RDAGENT_STATE_ROOT_REQUIRED" in result.stderr


def test_deployment_environment_retains_registry_and_sets_effective_workspace(tmp_path: Path) -> None:
    state = tmp_path / "state"
    path = tmp_path / "node.env"
    retained = b"# node configuration\r\nQE_DATASET_RELEASE_REGISTRY_ROOTS=/data/registry\r\nCUSTOM_SETTING=keep\r\n"
    path.write_bytes(retained + b"QE_WORKSPACE_WSL=/old/source/workspace\nQE_WORKSPACE_ROOT=/old/state\n")
    _write_deployment_environment(path, state)
    result = path.read_bytes()
    assert result.startswith(retained)
    assert result.count(b"QE_WORKSPACE_WSL=") == 1
    assert f"QE_WORKSPACE_WSL={state / 'qe_workspace'}\n".encode() in result
    assert b"/old/source/workspace" not in result
    assert list(tmp_path.glob(".*.tmp")) == []


def test_deployment_environment_does_not_delete_an_existing_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "node.env"
    path.write_bytes(b"CUSTOM_SETTING=original\n")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(b"another writer")
    with pytest.raises(FileExistsError):
        _write_deployment_environment(path, tmp_path / "state")
    assert temporary.read_bytes() == b"another writer"
    assert path.read_bytes() == b"CUSTOM_SETTING=original\n"


def test_deployment_environment_rejects_a_concurrent_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "node.env"
    path.write_bytes(b"CUSTOM_SETTING=original\n")
    read_bytes = Path.read_bytes
    reads = 0
    validation_read_number = 2

    def concurrent_read(target: Path) -> bytes:
        nonlocal reads
        if target == path:
            reads += 1
            if reads == validation_read_number:
                target.write_bytes(b"CUSTOM_SETTING=another_writer\n")
        return read_bytes(target)

    monkeypatch.setattr(Path, "read_bytes", concurrent_read)
    with pytest.raises(RuntimeError, match="RDAGENT_RELEASE_ENVIRONMENT_CHANGED"):
        _write_deployment_environment(path, tmp_path / "state")
    assert path.read_bytes() == b"CUSTOM_SETTING=another_writer\n"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="atomic release symlink is a POSIX deployment contract")
def test_release_is_exact_atomic_and_receipted(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "remote", "add", "origin", "https://example.invalid/RD-Agent.git"],
        check=True,
    )
    (source / "tracked.txt").write_text("exact\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-m", "exact"], check=True)
    sha = _git(source, "rev-parse", "HEAD")
    result = build_release(source, tmp_path / "releases", tmp_path / "state", sha, "main", "test")
    assert Path(result["current"]).resolve() == Path(result["release_path"])
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["merge_sha"] == sha
    assert receipt["tree_hash"] == _git(source, "rev-parse", "HEAD^{tree}")
    assert not (Path(result["release_path"]) / "git_ignore_folder").exists()
    environment = Path(result["environment_file"]).read_text()
    assert f"QE_WORKSPACE_WSL={tmp_path / 'state' / 'qe_workspace'}\n" in environment


@pytest.mark.skipif(os.name != "posix", reason="release builder is POSIX-only")
def test_release_rejects_dirty_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(source)], check=True)
    (source / "dirty").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="RDAGENT_RELEASE_SOURCE_DIRTY"):
        build_release(source, tmp_path / "releases", tmp_path / "state", "0" * 40, "main", "test")
