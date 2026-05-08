import asyncio
import base64
import io
import json
import tarfile

import pytest

from rdagent.app.api_endpoints import qe_evolution_api as api


def test_recorder_isolation_rejects_target_symlink(tmp_path):
    source = tmp_path / "source" / "mlruns"
    target = tmp_path / "target" / "mlruns"
    source.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    try:
        target.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")
    status_file = tmp_path / "target" / "status.txt"
    status_file.write_text("running")

    with pytest.raises(RuntimeError, match="QE_BACKTEST_TARGET_MLRUNS_IS_SYMLINK"):
        api._validate_recorder_isolation(tmp_path / "target", status_file, target, source)

    manifest = json.loads((tmp_path / "target" / api.RECORDER_ISOLATION_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["recorder_isolation_status"] == "failed"
    assert manifest["reason"] == "QE_BACKTEST_TARGET_MLRUNS_IS_SYMLINK"


def test_recorder_isolation_rejects_same_realpath(tmp_path):
    source = tmp_path / "shared" / "mlruns"
    source.mkdir(parents=True)
    status_file = tmp_path / "status.txt"
    status_file.write_text("running")

    with pytest.raises(RuntimeError, match="QE_BACKTEST_SOURCE_TARGET_REALPATH_COLLISION"):
        api._validate_recorder_isolation(tmp_path, status_file, source, source)


def test_recorder_isolation_passes_for_independent_target(tmp_path):
    source = tmp_path / "source" / "mlruns"
    target = tmp_path / "target" / "mlruns"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    status_file = tmp_path / "target" / "status.txt"
    status_file.write_text("running")

    api._validate_recorder_isolation(tmp_path / "target", status_file, target, source)

    manifest = json.loads((tmp_path / "target" / api.RECORDER_ISOLATION_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["recorder_isolation_status"] == "passed"
    assert manifest["source_mlruns_realpath"] != manifest["target_mlruns_realpath"]


def test_recorder_isolation_rejects_target_under_source(tmp_path):
    source = tmp_path / "source" / "mlruns"
    target = source / "nested_target"
    target.mkdir(parents=True)
    status_file = tmp_path / "status.txt"
    status_file.write_text("running")

    with pytest.raises(RuntimeError, match="QE_BACKTEST_TARGET_MLRUNS_UNDER_SOURCE"):
        api._validate_recorder_isolation(tmp_path, status_file, target, source)


def _tar_bytes_with_file(name: str, data: bytes = b"ok") -> io.BytesIO:
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    bio.seek(0)
    return bio


def test_safe_extract_rejects_path_traversal(tmp_path):
    with tarfile.open(fileobj=_tar_bytes_with_file("../escape.txt"), mode="r:gz") as tar:
        with pytest.raises(RuntimeError, match="QE_BACKTEST_UNSAFE_MLRUNS_TAR"):
            api._safe_extract_mlruns_tar(tar, tmp_path)


def test_safe_extract_rejects_link_members(tmp_path):
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tar:
        info = tarfile.TarInfo("mlruns/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/outside"
        tar.addfile(info)
    bio.seek(0)

    with tarfile.open(fileobj=bio, mode="r:gz") as tar:
        with pytest.raises(RuntimeError, match="QE_BACKTEST_UNSAFE_MLRUNS_TAR"):
            api._safe_extract_mlruns_tar(tar, tmp_path)


def test_same_node_model_source_fails_before_subprocess(tmp_path, monkeypatch):
    workspace = tmp_path / "qe_workspace"
    source = workspace / "source_task" / "Loop1" / "mlruns"
    source.mkdir(parents=True)
    monkeypatch.setattr(api, "WORKSPACE_BASE", workspace)

    asyncio.run(
        api._run_qlib_backtest(
            "target_task",
            "Loop2",
            config={},
            experiment_files=None,
            wsl_command="python should_not_run.py",
            model_source={"source_task_id": "source_task", "source_loop": "Loop1"},
        )
    )

    loop_dir = workspace / "target_task" / "Loop2"
    manifest = json.loads((loop_dir / api.RECORDER_ISOLATION_MANIFEST).read_text(encoding="utf-8"))
    assert (loop_dir / "status.txt").read_text(encoding="utf-8") == "failed"
    assert manifest["recorder_isolation_status"] == "failed"
    assert manifest["reason"] == "QE_BACKTEST_LEGACY_SYMLINK_MODEL_SOURCE_DISABLED"
    assert not (loop_dir / "pid.txt").exists()


def _tar_payload_b64(*members: tuple[str, bytes]) -> str:
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(bio.getvalue()).decode("ascii")


def test_safe_extract_rejects_control_file_outside_mlruns(tmp_path):
    with tarfile.open(fileobj=_tar_bytes_with_file("conf.yaml"), mode="r:gz") as tar:
        with pytest.raises(RuntimeError, match="member outside mlruns"):
            api._safe_extract_mlruns_tar(tar, tmp_path)


def test_safe_extract_rejects_parent_traversal_before_normalizing(tmp_path):
    with tarfile.open(fileobj=_tar_bytes_with_file("../conf.yaml"), mode="r:gz") as tar:
        with pytest.raises(RuntimeError, match="unsafe member path"):
            api._safe_extract_mlruns_tar(tar, tmp_path)


def test_cross_node_payload_success_writes_passed_manifest(tmp_path, monkeypatch):
    workspace = tmp_path / "qe_workspace"
    monkeypatch.setattr(api, "WORKSPACE_BASE", workspace)
    payload = _tar_payload_b64(("mlruns/0/run1/artifacts/params.pkl", b"params"))

    asyncio.run(
        api._run_qlib_backtest(
            "target_task",
            "Loop3",
            config={},
            experiment_files={"mlruns_params.tar.gz.b64": payload},
            wsl_command="python -V",
            model_source={"cross_node": True},
        )
    )

    loop_dir = workspace / "target_task" / "Loop3"
    manifest = json.loads((loop_dir / api.RECORDER_ISOLATION_MANIFEST).read_text(encoding="utf-8"))
    assert (loop_dir / "status.txt").read_text(encoding="utf-8") == "completed"
    assert manifest["recorder_isolation_status"] == "passed"
    assert (loop_dir / "mlruns" / "0" / "run1" / "artifacts" / "params.pkl").is_file()
    assert not (loop_dir / "mlruns_params.tar.gz").exists()
    assert (loop_dir / "pid.txt").is_file()


def test_cross_node_unsafe_tar_fails_with_manifest_and_no_subprocess(tmp_path, monkeypatch):
    workspace = tmp_path / "qe_workspace"
    monkeypatch.setattr(api, "WORKSPACE_BASE", workspace)
    payload = _tar_payload_b64(("conf.yaml", b"malicious"))

    asyncio.run(
        api._run_qlib_backtest(
            "target_task",
            "Loop4",
            config={},
            experiment_files={"mlruns_params.tar.gz.b64": payload},
            wsl_command="python should_not_run.py",
            model_source={"cross_node": True},
        )
    )

    loop_dir = workspace / "target_task" / "Loop4"
    manifest = json.loads((loop_dir / api.RECORDER_ISOLATION_MANIFEST).read_text(encoding="utf-8"))
    assert (loop_dir / "status.txt").read_text(encoding="utf-8") == "failed"
    assert manifest["recorder_isolation_status"] == "failed"
    assert manifest["reason"] == "QE_BACKTEST_UNSAFE_MLRUNS_TAR"
    assert "QE_BACKTEST_UNSAFE_MLRUNS_TAR" in (loop_dir / "error.log").read_text(encoding="utf-8")
    assert not (loop_dir / "pid.txt").exists()
