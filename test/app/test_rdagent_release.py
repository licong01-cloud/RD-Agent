from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from rdagent.app.runtime_state import require_state_root
from tools.rdagent_release import build_release


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_state_root_is_required_and_external(monkeypatch, tmp_path):
    monkeypatch.delenv("RDAGENT_STATE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="RDAGENT_STATE_ROOT_REQUIRED"):
        require_state_root()
    monkeypatch.setenv("RDAGENT_STATE_ROOT", str(tmp_path / "state"))
    assert require_state_root() == (tmp_path / "state").resolve()


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


@pytest.mark.skipif(os.name != "posix", reason="release builder is POSIX-only")
def test_release_rejects_dirty_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(source)], check=True)
    (source / "dirty").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="RDAGENT_RELEASE_SOURCE_DIRTY"):
        build_release(source, tmp_path / "releases", tmp_path / "state", "0" * 40, "main", "test")
