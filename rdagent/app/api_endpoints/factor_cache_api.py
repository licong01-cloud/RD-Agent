"""Factor-value cache API for AIstock QE execution nodes.

The Windows AIstock service owns the authoritative factor cache metadata and
uploads cache bundles through this API. The node service writes only inside a
server-side cache directory; it does not expose arbitrary worker workspace
access.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile


router = APIRouter(prefix="/api/v1/qe_workspace/factor-cache", tags=["factor-cache"])


def _default_cache_dir() -> Path:
    for env_name in ("FACTOR_CACHE_DIR", "AISTOCK_FACTOR_CACHE_DIR", "QE_FACTOR_CACHE_DIR"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return Path(value).expanduser()

    # Local WSL node: match AIstock's default cache path used by generated QE commands.
    aistock_wsl_cache = Path("/mnt/f/Dev/AIstock/rdagent_assets/factor_values")
    if aistock_wsl_cache.exists() or aistock_wsl_cache.parent.exists():
        return aistock_wsl_cache

    # Remote Linux node: use a node-local cache outside RD-Agent worker workspaces.
    return Path.home() / "aistock_cache" / "factor_values"


def _resolve_cache_root(cache_dir: Optional[str]) -> Path:
    raw = str(cache_dir or "").strip()
    root = Path(raw).expanduser() if raw else _default_cache_dir()
    if not root.is_absolute():
        raise HTTPException(status_code=400, detail="factor cache dir must be absolute")
    if root.name != "factor_values":
        raise HTTPException(status_code=400, detail="factor cache dir must end with factor_values")
    return root


def _safe_factor_name(factor_name: str) -> str:
    name = str(factor_name or "").strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        raise HTTPException(status_code=400, detail=f"unsafe factor name: {factor_name!r}")
    return name


def _safe_factor_file_name(factor_name: str) -> str:
    return f"{_safe_factor_name(factor_name)}.parquet"


def _read_meta(root: Path) -> Dict[str, Any]:
    meta_path = root / "_meta.json"
    if not meta_path.exists():
        return {"factors": {}}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"factor cache meta is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="factor cache meta must be a JSON object")
    payload.setdefault("factors", {})
    if not isinstance(payload.get("factors"), dict):
        raise HTTPException(status_code=500, detail="factor cache meta factors must be a JSON object")
    return payload


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


@router.get("/meta")
def get_factor_cache_meta(cache_dir: Optional[str] = Query(None)) -> Dict[str, Any]:
    root = _resolve_cache_root(cache_dir)
    meta = _read_meta(root)
    return {
        "ok": True,
        "cache_dir": str(root),
        "meta": meta,
        "factors": meta.get("factors", {}),
    }


@router.get("/factors/{factor_name}/status")
def get_factor_cache_factor_status(
    factor_name: str,
    cache_dir: Optional[str] = Query(None),
) -> Dict[str, Any]:
    name = _safe_factor_name(factor_name)
    root = _resolve_cache_root(cache_dir)
    path = root / "single" / _safe_factor_file_name(name)
    meta = _read_meta(root)
    entry = (meta.get("factors") or {}).get(name)
    exists = path.exists() and path.is_file()
    return {
        "ok": True,
        "factor_name": name,
        "cache_dir": str(root),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "meta": entry if isinstance(entry, dict) else None,
    }


@router.post("/sync")
async def sync_factor_cache_bundle(
    meta_json: str = Form(...),
    factor_names_json: str = Form(...),
    cache_dir: Optional[str] = Form(None),
    parquet_files: Optional[List[UploadFile]] = File(None),
) -> Dict[str, Any]:
    root = _resolve_cache_root(cache_dir)
    try:
        merged_meta = json.loads(meta_json)
        factor_names_raw = json.loads(factor_names_json)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid sync JSON payload: {exc}") from exc

    if not isinstance(merged_meta, dict) or not isinstance(merged_meta.get("factors", {}), dict):
        raise HTTPException(status_code=400, detail="meta_json must be an object with factors object")
    if not isinstance(factor_names_raw, list):
        raise HTTPException(status_code=400, detail="factor_names_json must be a JSON list")

    factor_names = [_safe_factor_name(str(name)) for name in factor_names_raw]
    expected_files = {_safe_factor_file_name(name): name for name in factor_names}
    uploads = parquet_files or []
    if len(uploads) != len(expected_files):
        raise HTTPException(
            status_code=400,
            detail=f"uploaded file count mismatch: expected={len(expected_files)} actual={len(uploads)}",
        )

    uploaded: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for upload in uploads:
        filename = Path(str(upload.filename or "")).name
        if filename not in expected_files:
            raise HTTPException(status_code=400, detail=f"unexpected factor cache file: {upload.filename}")
        if filename in seen:
            raise HTTPException(status_code=400, detail=f"duplicate factor cache file: {filename}")
        seen.add(filename)
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"empty factor cache file: {filename}")
        target = root / "single" / filename
        _atomic_write_bytes(target, content)
        uploaded.append(
            {
                "factor_name": expected_files[filename],
                "filename": filename,
                "size_bytes": len(content),
            }
        )

    missing_files = sorted(set(expected_files) - seen)
    if missing_files:
        raise HTTPException(status_code=400, detail=f"missing factor cache files: {missing_files}")

    _atomic_write_text(
        root / "_meta.json",
        json.dumps(merged_meta, ensure_ascii=False, indent=2, sort_keys=True),
    )
    return {
        "ok": True,
        "cache_dir": str(root),
        "uploaded_count": len(uploaded),
        "uploaded": uploaded,
        "meta_factor_count": len(merged_meta.get("factors") or {}),
    }
