import os
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def register_workspaces(db_path, workspace_root):
    db_path = Path(db_path)
    workspace_root = Path(workspace_root)
    
    if not db_path.exists():
        print(f"Error: Database {db_path} not found.")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # 获取已注册的 workspace_id 集合
    existing_ws = set()
    rows = cur.execute("SELECT workspace_id FROM workspaces").fetchall()
    existing_ws.update(row[0] for row in rows)
    
    print(f"Scanning {workspace_root}...")
    count = 0
    registered = 0
    
    # 遍历所有子目录
    for ws_dir in workspace_root.iterdir():
        if not ws_dir.is_dir():
            continue
        
        count += 1
        ws_id = ws_dir.name
        
        if ws_id in existing_ws:
            continue
            
        # 尝试猜测 task_run_id 和 loop_id
        # 很多 workspace_id 就是 UUID，或者带有 synthetic_ 前缀
        # 我们优先看里面有没有 workspace_meta.json
        meta_path = ws_dir / "workspace_meta.json"
        task_run_id = "unknown_legacy"
        loop_id = 0
        experiment_type = "unknown"
        
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                task_run_id = meta.get("task_run_id", task_run_id)
                loop_id = meta.get("loop_id", loop_id)
                experiment_type = meta.get("experiment_type", experiment_type)
            except:
                pass
        
        # 检查有没有结果文件，决定角色
        role = "other"
        if (ws_dir / "qlib_res.csv").exists() or (ws_dir / "combined_factors_df.parquet").exists():
            role = "experiment_workspace"
            if (ws_dir / "qlib_res.csv").exists():
                experiment_type = "model"
            elif (ws_dir / "combined_factors_df.parquet").exists():
                experiment_type = "factor"
            
            # 对于 legacy 数据，如果没有 meta 信息，使用 workspace_id 作为唯一标识，避免 loop 冲突
            if task_run_id == "unknown_legacy":
                task_run_id = f"legacy_{ws_id}"
                loop_id = 0
        
        now = _utc_now_iso()
        
        # 插入 workspaces 表
        try:
            cur.execute("""
                INSERT INTO workspaces (
                    workspace_id, task_run_id, loop_id, workspace_role, 
                    experiment_type, status, workspace_path, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ws_id, task_run_id, int(loop_id), role, 
                experiment_type, "present", str(ws_dir.absolute()), now, now
            ))
            
            # 如果 task_run_id 不在 task_runs 表中，也补一条
            cur.execute("""
                INSERT OR IGNORE INTO task_runs (task_run_id, status, created_at_utc, updated_at_utc)
                VALUES (?, ?, ?, ?)
            """, (task_run_id, "legacy", now, now))
            
            # 如果 (task_run_id, loop_id) 不在 loops 表中，也补一条
            # 同时更新 has_result 标志
            has_result = 1 if role == "experiment_workspace" else 0
            cur.execute("""
                INSERT INTO loops (task_run_id, loop_id, action, status, has_result, started_at_utc, ended_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_run_id, loop_id) DO UPDATE SET
                    has_result = excluded.has_result,
                    action = COALESCE(loops.action, excluded.action)
            """, (task_run_id, int(loop_id), experiment_type, "legacy", has_result, now, now))
            
            registered += 1
        except Exception as e:
            print(f"Error registering {ws_id}: {e}")
            
        if (registered + 1) % 1000 == 0:
            conn.commit()
            print(f"Processed {count} directories, registered {registered} new workspaces...")

    conn.commit()
    conn.close()
    print(f"Finished. Scanned {count} directories, registered {registered} new workspaces.")

if __name__ == "__main__":
    register_workspaces(
        "f:/Dev/RD-Agent-main/RDagentDB/registry.sqlite", 
        "f:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/"
    )
