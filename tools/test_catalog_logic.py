import json
import sqlite3
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

@dataclass
class WorkspaceRow:
    workspace_id: str
    workspace_path: str

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
    return Path(p_str)

def debug_catalog():
    db_path = 'f:/Dev/RD-Agent-main/RDagentDB/registry.sqlite'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    sql = "SELECT workspace_id, workspace_path FROM workspaces WHERE workspace_path IS NOT NULL AND workspace_path != ''"
    rows = conn.execute(sql).fetchall()
    print(f"Total workspaces in DB: {len(rows)}")
    
    factors_by_name = {}
    ws_with_meta = 0
    total_factors_seen = 0
    
    for r in rows:
        ws_path_str = r["workspace_path"]
        ws_root = _to_native_path(ws_path_str)
        meta_path = ws_root / "factor_meta.json"
        
        if meta_path.exists():
            ws_with_meta += 1
            try:
                content = json.loads(meta_path.read_text(encoding='utf-8'))
                factors = content.get("factors") or []
                for f in factors:
                    name = f.get("name")
                    if name:
                        total_factors_seen += 1
                        factors_by_name[name] = f
            except Exception as e:
                print(f"Error reading {meta_path}: {e}")
                
    print(f"Workspaces with factor_meta.json: {ws_with_meta}")
    print(f"Total factors seen across all files: {total_factors_seen}")
    print(f"Distinct factor names: {len(factors_by_name)}")
    
    conn.close()

if __name__ == "__main__":
    debug_catalog()
