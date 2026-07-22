"""Read an immutable QE dataset deployment manifest without scanning data files.

The data deployment pipeline owns creation of ``qe_dataset_manifest.json``.
RD-Agent only verifies and exposes that already-frozen manifest for a configured
QE data root.  When a deployment predates the manifest, this module returns
explicit incomplete evidence and acquisition guidance; it never hashes a
mutable directory opportunistically or blocks a research run.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

DATASET_IDENTITY_SCHEMA_VERSION = "qe_dataset_identity_v1"
DATASET_EVIDENCE_SCHEMA_VERSION = "qe_dataset_identity_evidence_v1"
DATASET_MANIFEST_FILENAME = "qe_dataset_manifest.json"
_SHA256_HEX_LENGTH = 64
_REQUIRED_FIELDS = (
    "deployment_snapshot_id",
    "dataset_manifest_sha256",
    "cutoff_trade_date",
    "qlib_calendar_sha256",
    "qlib_instruments_sha256",
    "st_pit_snapshot_id",
    "st_pit_manifest_sha256",
)


def read_dataset_identity(  # noqa: PLR0911 - each explicit evidence branch is part of the public contract.
    *,
    data_root_uri: str | None,
    node_id: str,
) -> dict[str, Any]:
    """Return verified identity or explicit evidence that the deployment lacks it."""

    root, root_reason = _resolve_allowed_data_root(data_root_uri)
    if root is None:
        return _incomplete(
            reason_code=root_reason or "qe_dataset_identity_root_unavailable",
            missing=["resolved_data_root_uri", "qe_dataset_manifest.json"],
            suggestions=[
                "configure QE_QLIB_DATA_PATH or QE_DATASET_IDENTITY_ROOTS for this QE deployment",
                "publish qe_dataset_manifest.json with the immutable Qlib/ST PIT snapshot",
            ],
        )
    manifest_path = root / DATASET_MANIFEST_FILENAME
    long_trend_snapshot, long_trend_reason = _read_long_trend_snapshot_identity(root)
    if not manifest_path.is_file():
        return _incomplete(
            reason_code="qe_dataset_manifest_missing",
            missing=[DATASET_MANIFEST_FILENAME],
            suggestions=[
                "publish qe_dataset_manifest.json from the data deployment pipeline",
                "include Qlib bin, calendar, instruments, and QE ST PIT content hashes in the deployment manifest",
            ],
            resolved_data_root_uri=str(root),
            long_trend_snapshot=long_trend_snapshot,
            long_trend_snapshot_reason=long_trend_reason,
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _incomplete(
            reason_code="qe_dataset_manifest_unreadable",
            missing=[DATASET_MANIFEST_FILENAME],
            suggestions=["repair or republish the immutable QE dataset manifest"],
            detail=f"{type(exc).__name__}: {exc}",
            resolved_data_root_uri=str(root),
        )
    if not isinstance(raw, Mapping):
        return _incomplete(
            reason_code="qe_dataset_manifest_invalid",
            missing=list(_REQUIRED_FIELDS),
            suggestions=["republish qe_dataset_manifest.json as a canonical JSON object"],
            resolved_data_root_uri=str(root),
        )
    manifest = dict(raw)
    missing = [
        field
        for field in _REQUIRED_FIELDS
        if not isinstance(manifest.get(field), str) or not str(manifest[field]).strip()
    ]
    if missing:
        return _incomplete(
            reason_code="qe_dataset_manifest_incomplete",
            missing=missing,
            suggestions=["republish qe_dataset_manifest.json with all required immutable content identifiers"],
            resolved_data_root_uri=str(root),
        )
    declared_hash = str(manifest["dataset_manifest_sha256"]).strip().lower()
    if not _is_sha256(declared_hash):
        return _incomplete(
            reason_code="qe_dataset_manifest_invalid",
            missing=["dataset_manifest_sha256"],
            suggestions=["republish qe_dataset_manifest.json with a lowercase SHA-256 digest"],
            resolved_data_root_uri=str(root),
        )
    canonical_payload = dict(manifest)
    canonical_payload.pop("dataset_manifest_sha256", None)
    actual_hash = _sha256_json(canonical_payload)
    if actual_hash != declared_hash:
        return _incomplete(
            reason_code="qe_dataset_manifest_hash_mismatch",
            missing=[],
            suggestions=["republish qe_dataset_manifest.json after recomputing its canonical content hash"],
            detail=f"expected={declared_hash} actual={actual_hash}",
            resolved_data_root_uri=str(root),
        )
    dataset = {field: str(manifest[field]).strip() for field in _REQUIRED_FIELDS}
    dataset["resolved_node_id"] = str(node_id).strip()
    dataset["resolved_data_root_uri"] = str(root)
    return {
        "schema_version": DATASET_IDENTITY_SCHEMA_VERSION,
        "complete": True,
        "reason_code": None,
        "missing": [],
        "acquisition_suggestions": [],
        "dataset": dataset,
        "long_trend_snapshot": long_trend_snapshot,
        "long_trend_snapshot_reason": long_trend_reason,
    }


def _read_long_trend_snapshot_identity(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    names = ("meta.json", "daily_pv.h5", "sector_data.h5")
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        return None, f"long_trend_snapshot_files_missing:{','.join(missing)}"
    try:
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"long_trend_snapshot_meta_unreadable:{type(exc).__name__}:{exc}"
    if not isinstance(meta, Mapping):
        return None, "long_trend_snapshot_meta_invalid"
    snapshot_id = str(meta.get("snapshot_id") or "").strip().lower()
    start_date = str(meta.get("start") or "").strip()
    end_date = str(meta.get("end") or "").strip()
    lineage = meta.get("lineage_parent_ids", [])
    if not snapshot_id or not start_date or not end_date or not isinstance(lineage, list):
        return None, "long_trend_snapshot_meta_incomplete"
    files: dict[str, dict[str, Any]] = {}
    try:
        for name in names:
            path = (root / name).resolve()
            stat = path.stat()
            files[name] = {
                "size": int(stat.st_size),
                "sha256": _cached_file_sha256(str(path), int(stat.st_size), int(stat.st_mtime_ns)),
            }
        manifest_sha = _sha256_json(
            {
                "snapshot_id": snapshot_id,
                "start_date": start_date,
                "end_date": end_date,
                "meta": dict(meta),
                "files": files,
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        return None, f"long_trend_snapshot_identity_failed:{type(exc).__name__}:{exc}"
    return (
        {
            "snapshot_id": snapshot_id,
            "manifest_sha256": manifest_sha,
            "start_date": start_date,
            "end_date": end_date,
            "lineage_parent_ids": [str(value).strip() for value in lineage],
            "files": files,
        },
        None,
    )


@lru_cache(maxsize=16)
def _cached_file_sha256(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_allowed_data_root(value: str | None) -> tuple[Path | None, str | None]:
    requested = str(value or os.environ.get("QE_QLIB_DATA_PATH") or "").strip()
    if not requested:
        return None, "qe_dataset_identity_root_unavailable"
    try:
        candidate = Path(requested).resolve(strict=True)
    except OSError:
        return None, "qe_dataset_identity_root_unavailable"
    if not candidate.is_dir():
        return None, "qe_dataset_identity_root_unavailable"
    allowed = _configured_roots()
    if not allowed:
        return None, "qe_dataset_identity_root_not_configured"
    for root in allowed:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return candidate, None
    return None, "qe_dataset_identity_root_not_configured"


def _configured_roots() -> tuple[Path, ...]:
    values: list[str] = []
    for key in ("QE_QLIB_DATA_PATH", "RDAGENT_FACTOR_DATA_WSL"):
        default = str(os.environ.get(key) or "").strip()
        if default:
            values.append(default)
    for key in ("QE_DATASET_IDENTITY_ROOTS", "QE_REGISTERED_DATASET_ROOTS"):
        configured = str(os.environ.get(key) or "").strip()
        if configured:
            values.extend(item.strip() for item in configured.split(os.pathsep) if item.strip())
    roots: list[Path] = []
    for raw in values:
        try:
            root = Path(raw).resolve(strict=True)
        except OSError:
            continue
        if root.is_dir() and root not in roots:
            roots.append(root)
    return tuple(roots)


def _incomplete(
    *,
    reason_code: str,
    missing: list[str],
    suggestions: list[str],
    detail: str | None = None,
    resolved_data_root_uri: str | None = None,
    long_trend_snapshot: Mapping[str, Any] | None = None,
    long_trend_snapshot_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": DATASET_EVIDENCE_SCHEMA_VERSION,
        "complete": False,
        "reason_code": reason_code,
        "missing": list(missing),
        "acquisition_suggestions": list(suggestions),
        "dataset": None,
        "long_trend_snapshot": dict(long_trend_snapshot) if long_trend_snapshot is not None else None,
        "long_trend_snapshot_reason": long_trend_snapshot_reason,
    }
    if detail:
        payload["detail"] = detail
    if resolved_data_root_uri:
        payload["resolved_data_root_uri"] = resolved_data_root_uri
    return payload


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(char in "0123456789abcdef" for char in value)


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
