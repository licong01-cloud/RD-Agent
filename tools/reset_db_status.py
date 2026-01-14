
import sqlite3
from pathlib import Path
import os

def reset_db():
    db_path = Path('RDagentDB/registry.sqlite')
    if not db_path.exists():
        print(f"Database file not found at {db_path.absolute()}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_runs';")
        if cursor.fetchone():
            cursor.execute("UPDATE task_runs SET status='aborted' WHERE status='running';")
            print(f"Updated {cursor.rowcount} task_runs rows.")
            
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='loops';")
        if cursor.fetchone():
            cursor.execute("UPDATE loops SET status='aborted' WHERE status='running';")
            print(f"Updated {cursor.rowcount} loops rows.")
            
        conn.commit()
        conn.close()
        print("Database reset successful.")
    except Exception as e:
        print(f"Database reset failed: {e}")

if __name__ == "__main__":
    reset_db()
