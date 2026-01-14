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

def check_empty_factors():
    db_path = 'f:/Dev/RD-Agent-main/RDagentDB/registry.sqlite'
    if not os.path.exists(db_path):
        db_path = '/mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite'
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    rows = cur.execute("""
        SELECT a.workspace_id, w.workspace_path, w.experiment_type, l.action
        FROM artifacts a 
        JOIN workspaces w ON a.workspace_id = w.workspace_id 
        JOIN loops l ON w.task_run_id = l.task_run_id AND w.loop_id = l.loop_id
        WHERE a.artifact_type='factor_meta'
    """).fetchall()
    
    empty_count = 0
    for ws_id, ws_path_str, exp_type, action in rows:
        ws_path = _to_native_path(ws_path_str)
        json_path = ws_path / 'factor_meta.json'
        
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding='utf-8'))
            factors = data.get('factors', [])
            if not factors:
                empty_count += 1
                parquet_path = ws_path / 'combined_factors_df.parquet'
                has_parquet = parquet_path.exists()
                print(f"Empty factors in WS: {ws_id}")
                print(f"  Path: {ws_path}")
                print(f"  ExpType: {exp_type}, Action: {action}")
                print(f"  Has combined_factors_df.parquet: {has_parquet}")
                if has_parquet:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(parquet_path)
                        print(f"  Parquet columns: {list(df.columns)}")
                    except Exception as e:
                        print(f"  Error reading parquet: {e}")
                print("-" * 20)
    
    print(f"Total empty factor_meta.json: {empty_count}")
    conn.close()

if __name__ == "__main__":
    check_empty_factors()
