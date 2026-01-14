import sqlite3
import os
from pathlib import Path

def _to_native_path(p_str: str) -> Path:
    if not p_str:
        return Path()
    is_windows = os.name == "nt"
    if is_windows and p_str.startswith("/mnt/"):
        parts = p_str.split("/")
        if len(parts) < 3:
            return Path(p_str)
        drive = parts[2].upper()
        return Path(f"{drive}:\\") / Path(*parts[3:])
    elif not is_windows and len(p_str) > 1 and p_str[1] == ":" and p_str[2] == "\\":
        drive = p_str[0].lower()
        rel = p_str[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}") / rel
    return Path(p_str)

def debug_export():
    db_path = 'f:/Dev/RD-Agent-main/RDagentDB/registry.sqlite'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    sql = "SELECT workspace_id, workspace_path FROM workspaces WHERE workspace_path IS NOT NULL AND workspace_path != '' LIMIT 10"
    rows = conn.execute(sql).fetchall()
    
    print(f"Fetched {len(rows)} workspaces")
    for r in rows:
        ws_id = r["workspace_id"]
        ws_path_str = r["workspace_path"]
        native_path = _to_native_path(ws_path_str)
        exists = native_path.exists()
        meta_exists = (native_path / "factor_meta.json").exists()
        print(f"ID: {ws_id}")
        print(f"  DB Path: {ws_path_str}")
        print(f"  Native Path: {native_path}")
        print(f"  Exists: {exists}, Meta Exists: {meta_exists}")

    conn.close()

if __name__ == "__main__":
    debug_export()
