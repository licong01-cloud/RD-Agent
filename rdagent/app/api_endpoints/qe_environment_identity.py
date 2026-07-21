"""Cached, content-addressed identity for a QE execution deployment.

This is deliberately *not* a resource monitor.  It is captured once per
RD-Agent process/deployment, contains no GPU or VRAM telemetry, and is served
unchanged to every QE submission until the owning service restarts or its
explicit cache is reset by deployment code.  AIstock binds the returned
snapshot to a durable multi-alpha submission receipt so an exact recovery does
not mistake a same-named environment for the same installed runtime.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXECUTION_ENVIRONMENT_SCHEMA_VERSION = "qe_execution_environment_manifest_v1"

_CACHE_LOCK = threading.Lock()
_CACHE_STATE: dict[str, dict[str, Any] | None] = {"identity": None}


class ExecutionEnvironmentIdentityError(RuntimeError):
    """The owning QE service could not construct a trustworthy deployment identity."""


def get_execution_environment_identity() -> dict[str, Any]:
    """Return the immutable identity captured for this RD-Agent process.

    The first call is the only potentially expensive operation: it reads Python
    distribution metadata and hashes the small, explicit QE executor file set.
    There are intentionally no subprocess probes and no GPU/VRAM queries.
    """

    with _CACHE_LOCK:
        cached_identity = _CACHE_STATE["identity"]
        if cached_identity is None:
            manifest = _build_manifest()
            manifest_hash = _sha256_json(manifest)
            cached_identity = {
                "schema_version": EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
                "execution_environment_snapshot_id": f"qeenv_{manifest_hash[:24]}",
                "execution_environment_manifest_sha256": manifest_hash,
                "manifest": manifest,
            }
            _CACHE_STATE["identity"] = cached_identity
        return _copy_json(cached_identity)


def reset_execution_environment_identity_cache_for_tests() -> None:
    """Clear the process cache for deterministic owning-service tests only."""

    with _CACHE_LOCK:
        _CACHE_STATE["identity"] = None


def _build_manifest() -> dict[str, Any]:
    dependency_files = _dependency_file_manifest()
    package_rows = _installed_packages()
    qlib = _qlib_identity(package_rows)
    return {
        "schema_version": EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": str(getattr(sys.implementation, "cache_tag", "") or ""),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(aliased=False, terse=False),
            "container_identity": str(os.environ.get("CONTAINER_IMAGE_DIGEST") or ""),
        },
        "qlib": qlib,
        "declared_runtime_identity": {
            "qlib_runtime_template_sha256": str(os.environ.get("QE_RUNTIME_TEMPLATE_SHA256") or "").strip(),
            "conda_environment_lock_sha256": str(os.environ.get("QE_CONDA_ENVIRONMENT_LOCK_SHA256") or "").strip(),
            "executor_code_commit": str(os.environ.get("QE_EXECUTOR_CODE_COMMIT") or "").strip(),
        },
        "installed_packages": package_rows,
        "executor_dependencies": dependency_files,
        "executor_file_set_sha256": _sha256_json(dependency_files),
    }


def _dependency_file_manifest() -> list[dict[str, str]]:
    root = Path(__file__).resolve().parent
    names = (
        "qe_environment_identity.py",
        "qe_evolution_api.py",
        "qe_submission_receipt.py",
        "qe_kill_receipt.py",
    )
    rows: list[dict[str, str]] = []
    for name in names:
        path = root / name
        if not path.is_file():
            message = f"QE execution dependency file is missing: {path.name}"
            raise ExecutionEnvironmentIdentityError(message)
        rows.append({"path": name, "sha256": _sha256_file(path)})
    return rows


def _installed_packages() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        distributions = list(importlib.metadata.distributions())
    except Exception as exc:  # pragma: no cover - host Python metadata failure.
        message = f"cannot enumerate installed Python distributions: {type(exc).__name__}: {exc}"
        raise ExecutionEnvironmentIdentityError(message) from exc
    for dist in distributions:
        name = str(dist.metadata.get("Name") or "").strip()
        version = str(dist.version or "").strip()
        if not name or not version:
            continue
        direct_url = dist.read_text("direct_url.json")
        direct_url_sha256 = _sha256_text(direct_url) if direct_url else ""
        installer = str(dist.read_text("INSTALLER") or "").strip()
        rows.append(
            {
                "name": name.lower(),
                "version": version,
                "installer": installer,
                "direct_url_sha256": direct_url_sha256,
            },
        )
    return sorted(rows, key=lambda item: (item["name"], item["version"], item["direct_url_sha256"]))


def _qlib_identity(packages: list[dict[str, str]]) -> dict[str, str]:
    qlib_rows = [item for item in packages if item["name"] in {"pyqlib", "qlib"}]
    if not qlib_rows:
        return {
            "distribution": "",
            "version": "",
            "source_sha256": "",
            "commit": "",
            "available": "false",
        }
    selected = qlib_rows[0]
    # A direct-url digest is immutable evidence of the installed source metadata.
    # It is not guessed as a Git commit when the package manager did not record one.
    return {
        "distribution": selected["name"],
        "version": selected["version"],
        "source_sha256": selected["direct_url_sha256"],
        "commit": "",
        "available": "true",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        message = f"cannot hash QE execution dependency {path.name}: {exc}"
        raise ExecutionEnvironmentIdentityError(message) from exc
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Mapping[str, Any] | list[dict[str, str]]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
