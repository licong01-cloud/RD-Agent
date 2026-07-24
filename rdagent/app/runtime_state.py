from __future__ import annotations

import os
from pathlib import Path

STATE_ROOT_ENV = "RDAGENT_STATE_ROOT"


def require_state_root() -> Path:
    raw = os.environ.get(STATE_ROOT_ENV, "").strip()
    if not raw:
        raise RuntimeError("RDAGENT_STATE_ROOT_REQUIRED: explicit external runtime state root is required")
    root = Path(raw).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if root == repo or repo in root.parents or root in repo.parents:
        raise RuntimeError("RDAGENT_STATE_ROOT_INVALID: state root must be outside source and release directories")
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_path(*parts: str) -> Path:
    return require_state_root().joinpath(*parts)
