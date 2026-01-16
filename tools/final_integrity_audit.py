import sys
from pathlib import Path

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sqlite3
import os
from pathlib import Path

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

def final_audit():
    db_path = 'f:/Dev/RD-Agent-main/RDagentDB/registry.sqlite'
    if not os.path.exists(db_path):
        db_path = '/mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite'
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 1. Get all experiment workspaces from DB
    db_experiment_ws = set()
    rows = cur.execute("SELECT workspace_id, workspace_path FROM workspaces WHERE workspace_role='experiment_workspace'").fetchall()
    for ws_id, _ in rows:
        db_experiment_ws.add(ws_id)
    
    print(f"DB marked experiment workspaces: {len(db_experiment_ws)}")
    
    # 2. Scan disk for actual result files
    workspace_root = Path('f:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/')
    if not workspace_root.exists():
        workspace_root = Path('/mnt/f/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/')
        
    disk_with_results = []
    
    print(f"Scanning disk {workspace_root} for results...")
    for ws_dir in workspace_root.iterdir():
        if not ws_dir.is_dir():
            continue
        
        has_res = (ws_dir / "qlib_res.csv").exists() or \
                  (ws_dir / "combined_factors_df.parquet").exists() or \
                  (ws_dir / "ret.pkl").exists()
        
        if has_res:
            disk_with_results.append(ws_dir.name)
            
    print(f"Disk folders with results (qlib_res/factors/ret.pkl): {len(disk_with_results)}")
    
    # 3. Find missing ones
    missing_in_db = [ws_id for ws_id in disk_with_results if ws_id not in db_experiment_ws]
    print(f"Workspaces with results but NOT marked as experiment_workspace in DB: {len(missing_in_db)}")
    
    for ws_id in missing_in_db:
        ws_dir = workspace_root / ws_id
        res_types = []
        if (ws_dir / "qlib_res.csv").exists(): res_types.append("qlib_res.csv")
        if (ws_dir / "combined_factors_df.parquet").exists(): res_types.append("combined_factors_df.parquet")
        if (ws_dir / "ret.pkl").exists(): res_types.append("ret.pkl")
        print(f"  - {ws_id}: {res_types}")

    # 4. Check materialization status for all experiment workspaces
    not_done = cur.execute("SELECT COUNT(*) FROM loops l JOIN workspaces w ON l.task_run_id=w.task_run_id AND l.loop_id=w.loop_id WHERE w.workspace_role='experiment_workspace' AND (l.materialization_status != 'done' OR l.materialization_status IS NULL)").fetchone()[0]
    print(f"Loops linked to experiment workspaces NOT 'done': {not_done}")

    conn.close()

if __name__ == "__main__":
    final_audit()
