from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.metadata

from rdagent.log import rdagent_logger as logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_repo_root_best_effort() -> Path | None:
    candidates: list[Path] = []
    try:
        candidates.append(Path.cwd().resolve())
    except Exception:
        pass
    try:
        candidates.append(Path(__file__).resolve())
    except Exception:
        pass

    seen: set[Path] = set()
    for base in candidates:
        for p in [base, *base.parents]:
            if p in seen:
                continue
            seen.add(p)

            try:
                if (p / ".git").exists() and (p / "rdagent").exists():
                    return p
                if (p / "requirements.txt").exists() and (p / "rdagent").exists():
                    return p
            except Exception:
                continue
    return None


def _default_registry_path() -> Path:
    """Default registry path: <repo_root>/RDagentDB/registry.sqlite (must travel with project)."""
    repo_root = _find_repo_root_best_effort()
    if repo_root is None:
        # Fallback to current working directory to keep DB colocated with execution context.
        try:
            repo_root = Path.cwd().resolve()
        except Exception:
            repo_root = Path(".").resolve()

    override = os.getenv("RD_AGENT_REGISTRY_DB_PATH", "").strip()
    if override:
        try:
            override_path = Path(override).expanduser().resolve()
            try:
                override_path.relative_to(repo_root)
                return override_path
            except Exception:
                logger.warning(
                    f"[SQLiteRegistry] Ignore RD_AGENT_REGISTRY_DB_PATH (must be under repo root {repo_root}): {override_path}"
                )
        except Exception:
            pass

    return repo_root / "RDagentDB" / "registry.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10, isolation_level=None)
    # WAL improves concurrent read/write. busy_timeout reduces spurious lock errors.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn


def _best_effort_rdagent_version() -> str | None:
    try:
        return importlib.metadata.version("rdagent")
    except Exception:
        return None


def _read_text_best_effort(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _best_effort_git_sha(repo_root: Path | None = None) -> str | None:
    """Try to resolve current git sha without invoking git binary."""
    try:
        root = repo_root
        if root is None:
            # Try to infer repo root from this file location.
            cur = Path(__file__).resolve()
            for parent in [cur, *cur.parents]:
                if (parent / ".git").exists():
                    root = parent
                    break
        if root is None:
            return None

        git_dir = root / ".git"
        head = _read_text_best_effort(git_dir / "HEAD")
        if not head:
            return None
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[-1].strip()
            sha = _read_text_best_effort(git_dir / ref)
            if sha:
                return sha
            packed = _read_text_best_effort(git_dir / "packed-refs")
            if not packed:
                return None
            for line in packed.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.split(" ")
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
            return None
        # Detached HEAD.
        return head
    except Exception:
        return None


SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS task_runs (
        task_run_id TEXT PRIMARY KEY,
        scenario TEXT,
        status TEXT,
        created_at_utc TEXT,
        updated_at_utc TEXT,
        git_sha TEXT,
        rdagent_version TEXT,
        log_trace_path TEXT,
        params_json TEXT
    );
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS loops (
        task_run_id TEXT NOT NULL,
        loop_id INTEGER NOT NULL,
        action TEXT,
        status TEXT,
        has_result INTEGER DEFAULT 0,
        best_workspace_id TEXT,
        started_at_utc TEXT,
        ended_at_utc TEXT,
        error_type TEXT,
        error_message TEXT,
        ic_mean REAL,
        rank_ic_mean REAL,
        ann_return REAL,
        mdd REAL,
        turnover REAL,
        multi_score REAL,
        metrics_json TEXT,
        PRIMARY KEY (task_run_id, loop_id)
    );
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS workspaces (
        workspace_id TEXT PRIMARY KEY,
        task_run_id TEXT NOT NULL,
        loop_id INTEGER,
        workspace_role TEXT,
        experiment_type TEXT,
        step_name TEXT,
        status TEXT,
        workspace_path TEXT NOT NULL,
        meta_path TEXT,
        summary_path TEXT,
        manifest_path TEXT,
        created_at_utc TEXT,
        updated_at_utc TEXT
    );
    """.strip(),
    """
    CREATE INDEX IF NOT EXISTS idx_workspaces_task_loop ON workspaces(task_run_id, loop_id);
    """.strip(),
    """
    CREATE INDEX IF NOT EXISTS idx_workspaces_role ON workspaces(workspace_role);
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        task_run_id TEXT NOT NULL,
        loop_id INTEGER,
        workspace_id TEXT NOT NULL,
        artifact_type TEXT,
        name TEXT,
        version TEXT,
        status TEXT,
        primary_flag INTEGER DEFAULT 0,
        summary_json TEXT,
        entry_path TEXT,
        created_at_utc TEXT,
        updated_at_utc TEXT
    );
    """.strip(),
    """
    CREATE INDEX IF NOT EXISTS idx_artifacts_task_loop ON artifacts(task_run_id, loop_id);
    """.strip(),
    """
    CREATE INDEX IF NOT EXISTS idx_artifacts_workspace ON artifacts(workspace_id);
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS artifact_files (
        file_id TEXT PRIMARY KEY,
        artifact_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        path TEXT NOT NULL,
        sha256 TEXT,
        size_bytes INTEGER,
        mtime_utc TEXT,
        kind TEXT
    );
    """.strip(),
    """
    CREATE INDEX IF NOT EXISTS idx_artifact_files_artifact ON artifact_files(artifact_id);
    """.strip(),
]


@dataclass
class RegistryConfig:
    db_path: Path


class SQLiteRegistry:
    """Best-effort SQLite registry.

    Design goals:
    - Must not break the main workflow if anything goes wrong.
    - All writes occur in the main process (caller responsibility).
    - Works under coroutine-level concurrency via process-local lock + WAL.
    """

    def __init__(self, config: RegistryConfig | None = None):
        self.config = config or RegistryConfig(db_path=_default_registry_path())
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        db_path = self.config.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._initialized:
                return
            conn = _connect(db_path)
            try:
                for stmt in SCHEMA_SQL:
                    if stmt:
                        conn.execute(stmt)
            finally:
                conn.close()
            self._initialized = True

    def _execute_with_retry(self, fn, *, max_attempts: int = 10) -> None:
        # Exponential backoff on lock errors.
        delay = 0.05
        for attempt in range(1, max_attempts + 1):
            try:
                fn()
                return
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "locked" not in msg and "busy" not in msg:
                    raise
                if attempt == max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 1.0)

    def _best_effort(self, op_name: str, fn) -> None:
        try:
            self._ensure_initialized()
            with self._lock:
                self._execute_with_retry(fn)
        except Exception as e:
            # Never raise to caller.
            logger.warning(f"[SQLiteRegistry] {op_name} failed: {type(e).__name__}: {e}")

    def upsert_task_run(
        self,
        *,
        task_run_id: str,
        scenario: str | None,
        status: str,
        log_trace_path: str | None = None,
        git_sha: str | None = None,
        rdagent_version: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:

        def _fn() -> None:
            now = _utc_now_iso()
            resolved_git_sha = git_sha or _best_effort_git_sha()
            resolved_version = rdagent_version or _best_effort_rdagent_version()
            conn = _connect(self.config.db_path)
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_runs(
                        task_run_id, scenario, status, created_at_utc, updated_at_utc,
                        git_sha, rdagent_version, log_trace_path, params_json
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(task_run_id) DO UPDATE SET
                        scenario=excluded.scenario,
                        status=excluded.status,
                        updated_at_utc=excluded.updated_at_utc,
                        git_sha=excluded.git_sha,
                        rdagent_version=excluded.rdagent_version,
                        log_trace_path=excluded.log_trace_path,
                        params_json=excluded.params_json
                    """,
                    (
                        task_run_id,
                        scenario,
                        status,
                        now,
                        now,
                        resolved_git_sha,
                        resolved_version,
                        log_trace_path,
                        json.dumps(params or {}, ensure_ascii=False),
                    ),
                )
                conn.execute("COMMIT;")
            finally:
                conn.close()

        self._best_effort("upsert_task_run", _fn)

    def upsert_loop(
        self,
        *,
        task_run_id: str,
        loop_id: int,
        action: str | None,
        status: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:

        def _fn() -> None:
            now = _utc_now_iso()
            conn = _connect(self.config.db_path)
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO loops(
                        task_run_id, loop_id, action, status, started_at_utc, ended_at_utc,
                        error_type, error_message
                    ) VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(task_run_id, loop_id) DO UPDATE SET
                        action=excluded.action,
                        status=CASE
                            WHEN loops.status IN ('success','failed','aborted','skip') AND excluded.status='running' THEN loops.status
                            ELSE excluded.status
                        END,
                        started_at_utc=COALESCE(loops.started_at_utc, excluded.started_at_utc),
                        ended_at_utc=excluded.ended_at_utc,
                        error_type=excluded.error_type,
                        error_message=excluded.error_message
                    """,
                    (
                        task_run_id,
                        int(loop_id),
                        action,
                        status,
                        now,
                        now,
                        error_type,
                        error_message,
                    ),
                )
                conn.execute("COMMIT;")
            finally:
                conn.close()

        self._best_effort("upsert_loop", _fn)

    def upsert_workspace(
        self,
        *,
        workspace_id: str,
        task_run_id: str,
        loop_id: int | None,
        workspace_role: str | None,
        experiment_type: str | None,
        step_name: str | None,
        status: str,
        workspace_path: str,
        meta_path: str | None = None,
        summary_path: str | None = None,
        manifest_path: str | None = None,
    ) -> None:

        def _fn() -> None:
            now = _utc_now_iso()
            conn = _connect(self.config.db_path)
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO workspaces(
                        workspace_id, task_run_id, loop_id, workspace_role, experiment_type,
                        step_name, status, workspace_path, meta_path, summary_path, manifest_path,
                        created_at_utc, updated_at_utc
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(workspace_id) DO UPDATE SET
                        task_run_id=excluded.task_run_id,
                        loop_id=excluded.loop_id,
                        workspace_role=excluded.workspace_role,
                        experiment_type=excluded.experiment_type,
                        step_name=excluded.step_name,
                        status=excluded.status,
                        workspace_path=excluded.workspace_path,
                        meta_path=excluded.meta_path,
                        summary_path=excluded.summary_path,
                        manifest_path=excluded.manifest_path,
                        updated_at_utc=excluded.updated_at_utc
                    """,
                    (
                        workspace_id,
                        task_run_id,
                        loop_id,
                        workspace_role,
                        experiment_type,
                        step_name,
                        status,
                        workspace_path,
                        meta_path,
                        summary_path,
                        manifest_path,
                        now,
                        now,
                    ),
                )
                conn.execute("COMMIT;")
            finally:
                conn.close()

        self._best_effort("upsert_workspace", _fn)

    def update_loop_metrics(
        self,
        *,
        task_run_id: str,
        loop_id: int,
        metrics: dict[str, Any] | None,
        best_workspace_id: str | None = None,
        has_result: bool | None = None,
    ) -> None:

        def _to_float(v: Any) -> float | None:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        def _fn() -> None:
            now = _utc_now_iso()
            m = metrics or {}
            ic_mean = _to_float(m.get("IC"))
            rank_ic_mean = _to_float(m.get("Rank IC"))
            ann_return = _to_float(m.get("1day.excess_return_with_cost.annualized_return "))
            mdd = _to_float(m.get("1day.excess_return_with_cost.max_drawdown"))
            turnover = _to_float(m.get("1day.excess_return_with_cost.turnover"))
            multi_score = _to_float(m.get("multi_score"))
            conn = _connect(self.config.db_path)
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    UPDATE loops
                    SET
                        has_result=COALESCE(?, has_result),
                        best_workspace_id=COALESCE(?, best_workspace_id),
                        ic_mean=COALESCE(?, ic_mean),
                        rank_ic_mean=COALESCE(?, rank_ic_mean),
                        ann_return=COALESCE(?, ann_return),
                        mdd=COALESCE(?, mdd),
                        turnover=COALESCE(?, turnover),
                        multi_score=COALESCE(?, multi_score),
                        metrics_json=COALESCE(?, metrics_json),
                        ended_at_utc=?
                    WHERE task_run_id=? AND loop_id=?
                    """,
                    (
                        (1 if has_result else 0) if has_result is not None else None,
                        best_workspace_id,
                        ic_mean,
                        rank_ic_mean,
                        ann_return,
                        mdd,
                        turnover,
                        multi_score,
                        json.dumps(m, ensure_ascii=False) if m else None,
                        now,
                        task_run_id,
                        int(loop_id),
                    ),
                )
                conn.execute("COMMIT;")
            finally:
                conn.close()

        self._best_effort("update_loop_metrics", _fn)

    def upsert_artifact(
        self,
        *,
        artifact_id: str,
        task_run_id: str,
        loop_id: int | None,
        workspace_id: str,
        artifact_type: str | None,
        name: str | None,
        version: str | None = None,
        status: str | None = None,
        primary_flag: bool = False,
        summary: dict[str, Any] | None = None,
        entry_path: str | None = None,
    ) -> None:

        def _fn() -> None:
            now = _utc_now_iso()
            conn = _connect(self.config.db_path)
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO artifacts(
                        artifact_id, task_run_id, loop_id, workspace_id,
                        artifact_type, name, version, status, primary_flag,
                        summary_json, entry_path, created_at_utc, updated_at_utc
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        task_run_id=excluded.task_run_id,
                        loop_id=excluded.loop_id,
                        workspace_id=excluded.workspace_id,
                        artifact_type=excluded.artifact_type,
                        name=excluded.name,
                        version=excluded.version,
                        status=excluded.status,
                        primary_flag=excluded.primary_flag,
                        summary_json=excluded.summary_json,
                        entry_path=excluded.entry_path,
                        updated_at_utc=excluded.updated_at_utc
                    """,
                    (
                        artifact_id,
                        task_run_id,
                        loop_id,
                        workspace_id,
                        artifact_type,
                        name,
                        version,
                        status,
                        1 if primary_flag else 0,
                        json.dumps(summary or {}, ensure_ascii=False),
                        entry_path,
                        now,
                        now,
                    ),
                )
                conn.execute("COMMIT;")
            finally:
                conn.close()

        self._best_effort("upsert_artifact", _fn)

    def upsert_artifact_file(
        self,
        *,
        file_id: str,
        artifact_id: str,
        workspace_id: str,
        path: str,
        sha256: str | None = None,
        size_bytes: int | None = None,
        mtime_utc: str | None = None,
        kind: str | None = None,
    ) -> None:

        def _fn() -> None:
            conn = _connect(self.config.db_path)
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO artifact_files(
                        file_id, artifact_id, workspace_id, path, sha256,
                        size_bytes, mtime_utc, kind
                    ) VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(file_id) DO UPDATE SET
                        artifact_id=excluded.artifact_id,
                        workspace_id=excluded.workspace_id,
                        path=excluded.path,
                        sha256=excluded.sha256,
                        size_bytes=excluded.size_bytes,
                        mtime_utc=excluded.mtime_utc,
                        kind=excluded.kind
                    """,
                    (
                        file_id,
                        artifact_id,
                        workspace_id,
                        path,
                        sha256,
                        size_bytes,
                        mtime_utc,
                        kind,
                    ),
                )
                conn.execute("COMMIT;")
            finally:
                conn.close()

        self._best_effort("upsert_artifact_file", _fn)

    def _best_effort_file_meta(self, p: Path) -> tuple[int | None, str | None]:
        try:
            st = p.stat()
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            return int(st.st_size), mtime
        except Exception:
            return None, None

    def _best_effort_sha256(self, p: Path, *, max_bytes: int = 20 * 1024 * 1024) -> str | None:
        try:
            st = p.stat()
            if st.st_size > max_bytes:
                return None
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None


_registry_singleton: SQLiteRegistry | None = None


def get_registry() -> SQLiteRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = SQLiteRegistry()
    return _registry_singleton


def new_task_run_id() -> str:
    return uuid.uuid4().hex


def should_enable_registry() -> bool:
    # default enabled; can be disabled by env for safety.
    return os.getenv("RD_AGENT_DISABLE_SQLITE_REGISTRY", "0") != "1"
