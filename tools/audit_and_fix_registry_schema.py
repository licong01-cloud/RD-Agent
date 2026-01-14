import os
import sqlite3
from pathlib import Path
import re
import sys
import importlib


def _get_registry_path() -> Path:
    # 优先使用 AIstock/.env 的 RDAGENT_REGISTRY_SQLITE_PATH（与现有链路保持一致）
    # 也支持直接用 RD_AGENT_REGISTRY_DB_PATH（RD-Agent 原生约定）
    p = os.getenv("RDAGENT_REGISTRY_SQLITE_PATH") or os.getenv("RD_AGENT_REGISTRY_DB_PATH")
    if p:
        return Path(p).expanduser().resolve()
    # fallback：按仓库标准路径
    return Path(__file__).resolve().parents[1] / "RDagentDB" / "registry.sqlite"


def _load_schema_sql() -> list[str]:
    """Load authoritative schema SQL from RD-Agent source code.

    We intentionally do not duplicate schema in this script.
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    mod = importlib.import_module("rdagent.utils.registry.sqlite_registry")
    schema_sql = getattr(mod, "SCHEMA_SQL", None)
    if not isinstance(schema_sql, list):
        msg = "failed to load SCHEMA_SQL from rdagent.utils.registry.sqlite_registry"
        raise TypeError(msg)
    return list(schema_sql)


def _strip_sql_comments(sql: str) -> str:
    # Remove "-- ..." comments line-wise.
    lines = []
    for raw_line in sql.splitlines():
        line = raw_line
        if "--" in line:
            line = line.split("--", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _parse_create_table_columns(sql: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Parse CREATE TABLE statement to (table_name, [(col, ddl), ...]).

    Only extracts plain column definitions. Table constraints like PRIMARY KEY(...)
    are ignored because they can't be safely altered in-place in SQLite.
    """
    cleaned = _strip_sql_comments(sql).strip().rstrip(";")
    m = re.search(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*\((.*)\)\s*$", cleaned, re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    table = m.group(1)
    body = m.group(2)

    # Split by commas at top-level.
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    cols: list[tuple[str, str]] = []
    for raw in parts:
        line = " ".join(raw.split())
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("PRIMARY KEY") or upper.startswith("UNIQUE") or upper.startswith("FOREIGN KEY") or upper.startswith("CHECK"):
            continue

        # Column definition: <name> <type/constraints...>
        pieces = line.split(" ", 1)
        if len(pieces) != 2:
            continue
        col = pieces[0].strip('"`[]')
        ddl = pieces[1].strip()

        # If column-level PRIMARY KEY appears, it cannot be safely added later.
        if "PRIMARY KEY" in ddl.upper():
            continue

        cols.append((col, ddl))

    return table, cols


def _table_columns(cur: sqlite3.Cursor, table: str) -> dict[str, str]:
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return {r[1]: (r[2] or "") for r in rows}


def _table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    row = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _add_missing_columns(cur: sqlite3.Cursor, table: str, missing: list[tuple[str, str]]) -> None:
    for col, ddl in missing:
        # sqlite 支持 ADD COLUMN，但不支持 IF NOT EXISTS，这里先检查后执行
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def audit_and_fix() -> None:
    schema_sql = _load_schema_sql()

    # Build authoritative expected columns from SCHEMA_SQL.
    expected: dict[str, list[tuple[str, str]]] = {}
    for stmt in schema_sql:
        parsed = _parse_create_table_columns(stmt)
        if parsed is None:
            continue
        table, cols = parsed
        expected[table] = cols

    reg_path = _get_registry_path()
    if not reg_path.exists():
        raise SystemExit(f"registry.sqlite not found: {reg_path}")

    conn = sqlite3.connect(str(reg_path))
    try:
        cur = conn.cursor()

        # 1) Ensure missing tables/indexes are created exactly as SCHEMA_SQL defines.
        #    This is idempotent because statements are CREATE ... IF NOT EXISTS.
        for stmt in schema_sql:
            s = stmt.strip()
            if not s:
                continue
            cur.execute(s)

        # 2) Add missing columns for existing tables to align with authoritative schema.
        for table, cols in expected.items():
            if not _table_exists(cur, table):
                # Should not happen because step(1) created tables; keep safe anyway.
                print(f"[WARN] table still missing after SCHEMA_SQL execution: {table}")
                continue
            existing = _table_columns(cur, table)
            missing = [(c, ddl) for c, ddl in cols if c not in existing]
            if missing:
                print(f"[FIX] {table}: add missing columns: {[c for c,_ in missing]}")
                _add_missing_columns(cur, table, missing)

        conn.commit()

        print("Registry schema audit done. Current columns snapshot:")
        for table in expected:
            if _table_exists(cur, table):
                col_map = _table_columns(cur, table)
                print(f"- {table}: {len(col_map)} columns")
            else:
                print(f"- {table}: <missing>")
    finally:
        conn.close()


if __name__ == "__main__":
    audit_and_fix()
