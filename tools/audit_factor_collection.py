import json
import sqlite3
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def _to_native_path(p_str: str) -> Path:
    """Convert path between WSL and Windows format based on current OS."""
    if not p_str:
        return Path()
    is_windows = os.name == "nt"
    if is_windows and p_str.startswith("/mnt/"):
        parts = p_str.split("/")
        if len(parts) < 3: return Path(p_str)
        drive = parts[2].upper()
        return Path(f"{drive}:\\") / Path(*parts[3:])
    elif not is_windows and len(p_str) > 1 and p_str[1] == ":" and p_str[2] == "\\":
        drive = p_str[0].lower()
        rel = p_str[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}") / rel
    return Path(p_str)

def audit_factors():
    db_path = 'f:/Dev/RD-Agent-main/RDagentDB/registry.sqlite'
    if not os.path.exists(db_path):
        db_path = '/mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite'
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    rows = cur.execute("SELECT workspace_id, artifact_id FROM artifacts WHERE artifact_type='factor_meta'").fetchall()
    
    total_factors = 0
    empty_files = 0
    files_missing = 0
    
    for ws_id, art_id in rows:
        ws_row = cur.execute("SELECT workspace_path FROM workspaces WHERE workspace_id=?", (ws_id,)).fetchone()
        if not ws_row:
            continue
        
        ws_path_str = ws_row[0]
        ws_path = _to_native_path(ws_path_str)
        json_path = ws_path / 'factor_meta.json'
        
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding='utf-8'))
                factors = data.get('factors', [])
                count = len(factors)
                total_factors += count
                if count == 0:
                    empty_files += 1
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
        else:
            files_missing += 1
            
    print(f"Audit Result:")
    print(f"  Total Factor Meta Records in DB: {len(rows)}")
    print(f"  Total Individual Factors Found: {total_factors}")
    print(f"  Empty Meta Files: {empty_files}")
    print(f"  Files Missing on Disk: {files_missing}")

    # Check loops with has_result=1 but no factor_meta
    loops_with_res = cur.execute("SELECT task_run_id, loop_id, action FROM loops WHERE has_result=1").fetchall()
    print(f"\nLoops with results (has_result=1): {len(loops_with_res)}")
    
    factor_loops = [r for r in loops_with_res if r[2] == 'factor']
    print(f"  - action='factor': {len(factor_loops)}")
    
    model_loops = [r for r in loops_with_res if r[2] == 'model']
    print(f"  - action='model': {len(model_loops)}")

    conn.close()

if __name__ == "__main__":
    audit_factors()
