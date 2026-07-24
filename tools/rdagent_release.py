from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def build_release(
    source: Path, release_root: Path, state_root: Path, merge_sha: str, target_ref: str, actor: str
) -> dict:
    if os.name != "posix":
        raise RuntimeError("RDAGENT_RELEASE_POSIX_REQUIRED")
    source, release_root, state_root = source.resolve(), release_root.resolve(), state_root.resolve()
    if _git(source, "status", "--porcelain"):
        raise RuntimeError("RDAGENT_RELEASE_SOURCE_DIRTY")
    resolved_sha = _git(source, "rev-parse", f"{merge_sha}^{{commit}}")
    if resolved_sha != merge_sha or subprocess.run(
        ["git", "-C", str(source), "merge-base", "--is-ancestor", merge_sha, target_ref], check=False,
    ).returncode:
        raise RuntimeError("RDAGENT_RELEASE_MERGE_IDENTITY_INVALID")
    if (
        state_root == source
        or source in state_root.parents
        or state_root == release_root
        or release_root in state_root.parents
    ):
        raise RuntimeError("RDAGENT_RELEASE_STATE_ROOT_INVALID")
    tree = _git(source, "rev-parse", f"{merge_sha}^{{tree}}")
    release_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    release = release_root / merge_sha
    if release.exists():
        raise RuntimeError("RDAGENT_RELEASE_ALREADY_EXISTS")
    subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach", str(release), merge_sha], check=True)
    manifest = {"repository": _git(source, "remote", "get-url", "origin"), "merge_sha": merge_sha, "tree_hash": tree}
    manifest_hash = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest |= {"manifest_hash": manifest_hash}
    (release / ".rdagent-release-manifest.json").write_bytes(_canonical(manifest) + b"\n")
    current = release_root / "current"
    previous = str(current.resolve()) if current.exists() else None
    temp = release_root / f".current-{os.getpid()}"
    temp.symlink_to(release, target_is_directory=True)
    os.replace(temp, current)
    receipt = {
        **manifest,
        "release_path": str(release), "state_root": str(state_root), "node": socket.gethostname(),
        "actor": actor, "deployed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_path_before": previous, "runtime_path_after": str(release), "rollback_target": previous,
    }
    receipts = state_root / "deployments" / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts / f"{merge_sha}-{receipt['node']}.json"
    receipt_path.write_bytes(_canonical(receipt) + b"\n")
    env = state_root / "deployments" / f"{receipt['node']}.env"
    env.write_text(
        "\n".join([
            f"RDAGENT_STATE_ROOT={state_root}",
            f"QE_WORKSPACE_ROOT={state_root / 'qe_workspace'}",
            f"QE_WORKSPACE_ARTIFACT_STORE={state_root / 'artifact_cas'}",
            f"QE_FACTOR_DATA_DIR={state_root / 'factor_data'}",
            f"QE_FACTOR_DATA_DEBUG_DIR={state_root / 'factor_data_debug'}",
            f"RD_AGENT_REGISTRY_DB_PATH={state_root / 'registry' / 'registry.sqlite'}",
        ]) + "\n", encoding="utf-8",
    )
    return {**receipt, "receipt_path": str(receipt_path), "environment_file": str(env), "current": str(current)}


def main() -> int:
    p = argparse.ArgumentParser()
    for name in ("source", "release_root", "state_root", "merge_sha", "target_ref", "actor"):
        p.add_argument(f"--{name.replace('_', '-')}", required=True)
    a = p.parse_args()
    result = build_release(
        Path(a.source), Path(a.release_root), Path(a.state_root), a.merge_sha, a.target_ref, a.actor
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
