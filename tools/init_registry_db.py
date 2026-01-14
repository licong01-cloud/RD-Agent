import sqlite3
import argparse
from pathlib import Path

def init_db(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 启用 WAL 模式以提高并发性能
    cur.execute("PRAGMA journal_mode=WAL;")
    
    # 定义表结构及字段注释 (通过 SQL 语句前的注释体现)
    schema_queries = [
        # task_runs: 记录 RD-Agent 任务运行的全局元数据
        """
        CREATE TABLE IF NOT EXISTS task_runs (
            task_run_id TEXT PRIMARY KEY,      -- 任务运行唯一标识
            scenario TEXT,                     -- 任务场景名称 (如 fin_quant)
            status TEXT,                       -- 任务状态 (running, success, failed)
            created_at_utc TEXT,               -- 创建时间 (UTC ISO)
            updated_at_utc TEXT,               -- 最近更新时间 (UTC ISO)
            git_sha TEXT,                      -- 代码 Git Commit SHA
            rdagent_version TEXT,              -- RD-Agent 软件版本
            log_trace_path TEXT,               -- 任务全局日志路径
            params_json TEXT                   -- 任务启动参数快照 (JSON)
        );
        """,
        # loops: 记录演进过程中每一轮 Loop 的执行状态与核心指标
        """
        CREATE TABLE IF NOT EXISTS loops (
            task_run_id TEXT NOT NULL,         -- 所属任务 ID
            loop_id INTEGER NOT NULL,          -- 循环轮次 (0, 1, 2...)
            action TEXT,                       -- 动作类型 (factor, model, strategy)
            status TEXT,                       -- 循环状态 (running, success, failed)
            has_result INTEGER DEFAULT 0,      -- 是否产生有效成果文件 (0/1)
            best_workspace_id TEXT,            -- 本轮最优工作区 ID
            started_at_utc TEXT,               -- Loop 开始时间
            ended_at_utc TEXT,                 -- Loop 结束时间
            error_type TEXT,                   -- 异常类型名称
            error_message TEXT,                -- 异常详细信息
            ic_mean REAL,                      -- IC 均值 (仅因子实验)
            rank_ic_mean REAL,                 -- Rank IC 均值 (仅因子实验)
            ann_return REAL,                   -- 年化收益率
            mdd REAL,                          -- 最大回撤
            turnover REAL,                     -- 换手率
            multi_score REAL,                  -- 综合评分 (如 Sharpe)
            metrics_json TEXT,                 -- 完整指标字典 (JSON)
            log_dir TEXT,                      -- 本轮循环关联的日志目录
            materialization_status TEXT,       -- Phase2 物料补齐状态 (pending, done, failed)
            materialization_error TEXT,        -- 物料补齐失败原因
            materialization_updated_at_utc TEXT, -- 物料状态最后更新时间
            
            -- Phase3 资产固化与增量同步扩展
            asset_bundle_id TEXT,              -- 资产包唯一标识 (UUID)
            is_solidified INTEGER DEFAULT 0,   -- 是否已完成核心资产固化 (0/1)
            sync_status TEXT DEFAULT 'pending', -- AIstock 同步状态 (pending, synced)
            
            PRIMARY KEY (task_run_id, loop_id)
        );
        """,
        # factor_registry: 因子核心成果表，支持离线持久化与增量同步
        """
        CREATE TABLE IF NOT EXISTS factor_registry (
            factor_name TEXT,                  -- 因子名称
            expression TEXT,                   -- 因子表达式 (与 formula_hint 对齐)
            performance_json TEXT,             -- 结构化绩效指标 (JSON)
            asset_bundle_id TEXT,              -- 关联的资产包 ID
            workspace_id TEXT,                 -- 来源工作区 ID
            task_run_id TEXT,                  -- 来源任务 ID
            loop_id INTEGER,                   -- 来源 Loop ID
            updated_at_utc TEXT,               -- 入库/更新时间
            PRIMARY KEY (factor_name, asset_bundle_id)
        );
        """,
        # workspaces: 记录实验产生的工作目录索引
        """
        CREATE TABLE IF NOT EXISTS workspaces (
            workspace_id TEXT PRIMARY KEY,     -- 工作区唯一 ID (通常为目录名)
            task_run_id TEXT NOT NULL,         -- 关联的任务 ID
            loop_id INTEGER,                   -- 关联的 Loop 轮次
            workspace_role TEXT,               -- 角色 (experiment_workspace 等)
            experiment_type TEXT,              -- 实验类型 (factor, model)
            step_name TEXT,                    -- 关联的 Step 名称
            status TEXT,                       -- 工作区状态
            workspace_path TEXT NOT NULL,      -- 文件系统绝对路径
            meta_path TEXT,                    -- workspace_meta.json 相对路径
            summary_path TEXT,                 -- experiment_summary.json 相对路径
            manifest_path TEXT,                -- manifest.json 相对路径
            created_at_utc TEXT,               -- 创建时间
            updated_at_utc TEXT                -- 更新时间
        );
        """,
        # artifacts: 记录结构化产出物 (JSON/模型等)
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,      -- 产出物唯一 ID
            task_run_id TEXT NOT NULL,         -- 关联任务 ID
            loop_id INTEGER,                   -- 关联 Loop 轮次
            workspace_id TEXT NOT NULL,        -- 关联工作区 ID
            artifact_type TEXT,                -- 类型 (report, factor_meta, factor_perf, feedback)
            name TEXT,                         -- 产出物显示名称
            version TEXT,                      -- Schema 版本 (如 v1)
            status TEXT,                       -- 状态 (present, missing)
            primary_flag INTEGER DEFAULT 0,    -- 是否为该类型的主要产出物 (0/1)
            summary_json TEXT,                 -- 产出物内容摘要 (JSON)
            entry_path TEXT,                   -- 入口文件相对路径
            model_type TEXT,                   -- 模型类型 (仅模型/策略)
            model_conf_json TEXT,              -- 模型配置 (JSON)
            dataset_conf_json TEXT,            -- 数据集配置 (JSON)
            feature_schema_json TEXT,          -- 特征定义 (JSON)
            created_at_utc TEXT,               -- 创建时间
            updated_at_utc TEXT                -- 更新时间
        );
        """,
        # artifact_files: 记录产出物包含的具体文件列表
        """
        CREATE TABLE IF NOT EXISTS artifact_files (
            file_id TEXT PRIMARY KEY,          -- 文件记录 ID
            artifact_id TEXT NOT NULL,         -- 关联产出物 ID
            workspace_id TEXT NOT NULL,        -- 关联工作区 ID
            path TEXT NOT NULL,                -- 相对工作区的路径
            sha256 TEXT,                       -- 文件哈希
            size_bytes INTEGER,                -- 文件大小 (字节)
            mtime_utc TEXT,                    -- 文件最后修改时间
            kind TEXT                          -- 文件种类
        );
        """,
        # 索引定义
        "CREATE INDEX IF NOT EXISTS idx_workspaces_task_loop ON workspaces(task_run_id, loop_id);",
        "CREATE INDEX IF NOT EXISTS idx_workspaces_role ON workspaces(workspace_role);",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_task_loop ON artifacts(task_run_id, loop_id);",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_workspace ON artifacts(workspace_id);",
        "CREATE INDEX IF NOT EXISTS idx_artifact_files_artifact ON artifact_files(artifact_id);",
    ]
    
    print(f"Initializing database at: {db_path}")
    for sql in schema_queries:
        try:
            cur.execute(sql.strip())
        except Exception as e:
            print(f"Error executing statement: {e}")
    
    # 增量升级现有表结构 (针对旧版 DB)
    try:
        # loops 表增加字段
        existing_cols = [row[1] for row in cur.execute("PRAGMA table_info(loops)").fetchall()]
        if "asset_bundle_id" not in existing_cols:
            cur.execute("ALTER TABLE loops ADD COLUMN asset_bundle_id TEXT;")
        if "is_solidified" not in existing_cols:
            cur.execute("ALTER TABLE loops ADD COLUMN is_solidified INTEGER DEFAULT 0;")
        if "sync_status" not in existing_cols:
            cur.execute("ALTER TABLE loops ADD COLUMN sync_status TEXT DEFAULT 'pending';")
        if "updated_at_utc" not in existing_cols:
            cur.execute("ALTER TABLE loops ADD COLUMN updated_at_utc TEXT;")
    except Exception as e:
        print(f"Error altering table loops: {e}")
            
    conn.commit()
    conn.close()
    print("Database initialization completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize RD-Agent Registry SQLite Database.")
    parser.add_argument("--db", default="RDagentDB/registry.sqlite", help="Path to the registry sqlite file.")
    args = parser.parse_args()
    init_db(args.db)
