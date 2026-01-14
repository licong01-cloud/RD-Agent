# RD-Agent → AIstock：SQLite Registry + Workspace Meta/Artifacts 设计方案（v1）

> **重要说明（2025-12-29）：**
> 本文档的设计内容已完整并入 Phase 2 最终版设计文档：
> - `2025-12-29_Phase2_Detail_Design_RD-Agent_AIstock_Final.md` 的 **附录 A：SQLite Registry 设计**
>
> 自本日期起，本文件仅作为历史溯源材料保留，不再单独维护或作为设计入口。所有与 Registry 相关的最新需求与字段合同，以 Phase2 最终版文档为准。

## 0. 背景与目标
RD-Agent 负责策略研发与演进，AIstock 负责回测/模拟盘/实盘。为了让 AIstock **无需遍历扫描全部 workspace**，且能按 **任务（task）→ loop → workspace → artifact** 追溯与拉取成果，需要 RD-Agent 输出：

- Workspace 内：结构化元信息与成果清单（可搬运、可审计）。
- Workspace 外：一个可并发写入、可 SQL 查询的 Registry（SQLite）。

本方案目标：

- AIstock 只读 SQLite，就能：
  - 列出任务列表、任务状态
  - 明确每个任务有哪些 loop、有无成果
  - 对“有成果”的 loop 定位到对应 workspace
  - 获取该 workspace 下的 artifacts（模型/特征/配置/报告）与文件清单（hash/size/path）
  - 可按常用指标（<=10列）进行筛选，其余指标在 `metrics_json`
- RD-Agent 在并行机制下稳定写入（主进程写，WAL，busy_timeout，重试退避，进程内 lock 串行事务）。
- 对 RD-Agent 主流程影响最小：写失败可降级，不影响任务推进。

## 1. 已确认的设计选项（来自用户确认）

- **DB 放置路径（按最终实现）**：`<repo_root>/RDagentDB/registry.sqlite`（随项目目录迁移，且不提交仓库）
- **task_run_id**：每次启动生成 UUID；session resume 继续写同一 task_run_id
- **并行写入策略**：只在主进程写 SQLite + WAL + busy_timeout + 重试退避 + 进程内 lock 串行事务
- **artifact 粒度**：两级结构 `artifacts（组件）+ artifact_files（文件清单）`
- **artifact 关联**：同时保存 `workspace_id` 与 `(task_run_id, loop_id)`
- **成果判定（qlib）**：
  - model loop：`ret.pkl` 与 `qlib_res.csv` 均存在
  - factor loop：`combined_factors_df.parquet` 存在
- **Parquet 引擎一致性**：
  - RD-Agent 与 AIstock 推荐统一使用 `pyarrow` 读取/写入 Parquet，以减少类型兼容性差异
- **失败/跳过/异常**：写入 loop 记录（status=failed/aborted/skip）
- **指标来源**：从 workspace 内已有文件读取（`qlib_res.csv`、`ret.pkl` 等）提取
- **AIstock 消费方式**：现阶段 AIstock 只读 SQLite
- **上游同步策略**：短中期以 cherry-pick 为主
- **侵入程度**：短中期选 A（通用层 hook + 新模块），后续再评估是否需要更深入（B）。

## 2. RD-Agent 并行机制与写入约束

### 2.1 并行发生在哪里
- 并行主要体现在：**多个 loop 并发推进**。
  - `LoopBase.run()` 会启动：
    - 1 个 `kickoff_loop()` 负责不断生成新的 loop index（li=0,1,2...）
    - N 个 `execute_loop()` worker（N= `RD_AGENT_SETTINGS.get_max_parallel()`）从队列取 loop 执行 step
- 同一 loop 的 step 仍是串行（direct_exp_gen→coding→running→feedback）。

### 2.2 subprocess 约束
- 当 `force_subproc=True` 时，`LoopBase._run_step()` 会使用 `ProcessPoolExecutor` 在子进程执行 step。
- **强约束**：Registry 写入必须发生在主进程（step 返回后），不得在子进程写 DB。

### 2.3 写入位置（推荐）
- 在 `LoopBase._run_step()` 中：
  - step 开始：写入/更新 task、loop、workspace 的“运行中”状态
  - step 结束：写入/更新 step 结果；若为 `running` 且成功则写入成果摘要与 artifacts

## 3. 数据对象定义

### 3.1 Task（任务）
- 一次 `rdagent.app.cli fin_quant ...` 启动对应一个 `task_run_id`。
- session resume（从 session folder load）应复用同一个 `task_run_id`。

### 3.2 Loop（迭代轮次）
- 以 `loop_id`（整数，从 0 递增）标识。
- 每个 loop 的 `action` 为 factor 或 model（qlib quant 场景）。

### 3.3 Workspace
- RD-Agent 当前 workspace 目录名为 UUID（`FBWorkspace.workspace_path`）。
- 每个 experiment 通常包含：
  - `experiment_workspace`：实际运行与产物落盘位置
  - `sub_workspace_list`：候选代码注入/开发 workspace

### 3.4 Artifact（组件级）
- Artifact 是 AIstock 消费的“组件”，而不是单个文件：
  - `model`：模型产物（mlflow 或 pickle 等）
  - `feature_set`：特征/因子集合（定义/快照/版本）
  - `config_snapshot`：训练/实验配置快照
  - `report`：研究证据（指标汇总、曲线、图表等）

Artifact 再关联文件清单（artifact_files），用于校验/复制/同步。

## 4. SQLite Schema（v1）

### 4.1 通用约定
- 所有时间统一存 `*_at_utc`（ISO 8601 字符串或 unix timestamp，建议 ISO 8601）。
- 主键：
  - `task_runs.task_run_id`（TEXT）
  - `loops(task_run_id, loop_id)`（复合）
  - `workspaces.workspace_id`（TEXT）
  - `artifacts.artifact_id`（TEXT，uuid）
  - `artifact_files(file_id)`（TEXT，uuid）

### 4.2 表：task_runs
- 目的：任务级索引与审计

字段建议：
- `task_run_id` TEXT PRIMARY KEY
- `scenario` TEXT
- `status` TEXT  -- running/success/failed/aborted
- `created_at_utc` TEXT
- `updated_at_utc` TEXT
- `git_sha` TEXT
- `rdagent_version` TEXT
- `log_trace_path` TEXT  -- LOG_SETTINGS.trace_path
- `params_json` TEXT  -- 市场、数据源、segments 等

### 4.3 表：loops
- 目的：AIstock 直接判断“哪个 loop 有成果”

字段建议：
- `task_run_id` TEXT NOT NULL
- `loop_id` INTEGER NOT NULL
- `action` TEXT  -- factor/model
- `status` TEXT  -- running/success/failed/aborted/skip
- `has_result` INTEGER DEFAULT 0  -- 0/1
- `best_workspace_id` TEXT  -- 指向最关键的 experiment_workspace
- `started_at_utc` TEXT
- `ended_at_utc` TEXT
- `error_type` TEXT
- `error_message` TEXT

指标列（<=10，推荐 v1 先放这些）：
- `ic_mean` REAL
- `rank_ic_mean` REAL
- `ann_return` REAL
- `mdd` REAL
- `turnover` REAL
- `multi_score` REAL

扩展：
- `metrics_json` TEXT  -- 全量指标 JSON（可空）

约束：
- PRIMARY KEY (`task_run_id`, `loop_id`)

### 4.4 表：workspaces
- 目的：明确每个 workspace 的出处与入口指针

字段建议：
- `workspace_id` TEXT PRIMARY KEY
- `task_run_id` TEXT NOT NULL
- `loop_id` INTEGER
- `workspace_role` TEXT  -- experiment_workspace/sub_workspace
- `experiment_type` TEXT  -- qlib_factor/qlib_model（或 factor/model）
- `step_name` TEXT  -- 最后一次更新来自哪个 step（coding/running/feedback）
- `status` TEXT  -- running/success/failed/aborted
- `workspace_path` TEXT NOT NULL
- `meta_path` TEXT  -- workspace 内 workspace_meta.json（相对或绝对）
- `summary_path` TEXT  -- experiment_summary.json
- `manifest_path` TEXT  -- manifest.json
- `created_at_utc` TEXT
- `updated_at_utc` TEXT

索引建议：
- `CREATE INDEX idx_workspaces_task_loop ON workspaces(task_run_id, loop_id);`
- `CREATE INDEX idx_workspaces_role ON workspaces(workspace_role);`

### 4.5 表：artifacts
- 目的：组件级产物索引（模型/特征/配置/报告），可供 AIstock 拉取

字段建议：
- `artifact_id` TEXT PRIMARY KEY
- `task_run_id` TEXT NOT NULL
- `loop_id` INTEGER
- `workspace_id` TEXT NOT NULL
- `artifact_type` TEXT  -- model/feature_set/config_snapshot/report
- `name` TEXT  -- 可读名称
- `version` TEXT  -- 可选
- `status` TEXT  -- present/missing
- `primary` INTEGER DEFAULT 0  -- 该 loop 的主 artifact（可选）

成果摘要（可选，便于 AIstock 直接查看）：
- `summary_json` TEXT  -- 例如特征数、模型类型、Alpha158/SOTA/new 计数等

指针：
- `entry_path` TEXT  -- 组件入口（例如 model 目录、features 定义文件等）

时间：
- `created_at_utc` TEXT
- `updated_at_utc` TEXT

索引：
- `CREATE INDEX idx_artifacts_task_loop ON artifacts(task_run_id, loop_id);`
- `CREATE INDEX idx_artifacts_workspace ON artifacts(workspace_id);`

### 4.6 表：artifact_files
- 目的：文件级指纹（校验/同步/复制）

字段建议：
- `file_id` TEXT PRIMARY KEY
- `artifact_id` TEXT NOT NULL
- `workspace_id` TEXT NOT NULL
- `path` TEXT NOT NULL  -- 相对 workspace 的路径（推荐）
- `sha256` TEXT
- `size_bytes` INTEGER
- `mtime_utc` TEXT
- `kind` TEXT  -- model/config/data/report

索引：
- `CREATE INDEX idx_artifact_files_artifact ON artifact_files(artifact_id);`

## 5. Workspace 内文件（可搬运真相源）

即便 AIstock 只读 DB，仍建议在 workspace 内落文件，作为“可搬运/可审计”的真相源：

- `workspace_meta.json`：小而稳定（归属关系、状态、指针）
- `experiment_summary.json`：详细成果（指标、关键产物路径、因子清单摘要）
- `manifest.json`：对 AIstock 的 release 契约入口（后续逐步完善）

v1 最小要求：
- 只要 DB 可用，AIstock 可只读 DB。
- 当 DB 不可用/迁移时，可通过 workspace 内文件重建。

## 6. 写入时机与算法（主进程 hook）

### 6.1 写入点
- 在 `LoopBase._run_step()` 的主进程逻辑中加入 hook：
  - step 开始（进入执行前）
  - step 结束（拿到 result/exception 后）

### 6.2 需要从上下文中提取的关键字段
- `task_run_id`：本次任务 id（持久化到 session，便于 resume）
- `loop_id`：`li`
- `step_name`：`name`（direct_exp_gen/coding/running/feedback）
- `action`：优先从 `prev_out.get('direct_exp_gen',{}).get('propose').action` 获取
- `workspace_id/workspace_path`：
  - 如果 step result 是 Experiment：
    - 取 `exp.experiment_workspace.workspace_path` 作为 experiment_workspace
    - 取 `exp.sub_workspace_list[*].workspace_path` 作为 sub_workspace

### 6.3 成果判定与 loop.has_result 更新
- 若 `action=model` 且 `experiment_workspace/ret.pkl` 与 `experiment_workspace/qlib_res.csv` 均存在：
  - `loops.has_result=1`
  - `loops.best_workspace_id=<experiment_workspace_id>`
  - `loops.status=success`
  - 从 `qlib_res.csv`、`ret.pkl` 抽取指标：
    - 写入 `ic_mean/rank_ic_mean/ann_return/mdd/turnover/multi_score`
    - 其余写入 `metrics_json`

补充（稳定契约输出，推荐）：

- 对 model loop，RD-Agent 可额外输出：
  - `ret_schema.parquet` / `ret_schema.json`：稳定承载回测结果对象（index reset 成列）
  - `signals.parquet` / `signals.json`：强固定 schema 的可执行信号表（AIstock 主入口）
- 以上文件不参与 v1 的 `has_result` 判定（避免引入额外依赖导致误判），但应纳入 artifacts/files 记录，供 AIstock 自动发现。

- 若 `action=factor` 且 `experiment_workspace/combined_factors_df.parquet` 存在：
  - 同上更新 loop 记录（指标可为空或仅写 `summary_json`）

### 6.4 artifacts 写入（v1）
对 qlib quant：

- **model loop**（running 成功后）：
  - artifact: `model`，entry_path = `mlruns/` 或模型导出目录（若可定位）
  - artifact: `config_snapshot`，entry_path = `conf*.yaml`（已在 workspace）
  - artifact: `report`，entry_path = `qlib_res.csv`、`ret.pkl`
    - 建议同时纳入（若存在）：
      - `ret_schema.parquet`
      - `ret_schema.json`
      - `signals.parquet`
      - `signals.json`

- **factor loop**（running 成功后）：
  - artifact: `feature_set`，entry_path = `combined_factors_df.parquet`
  - artifact: `config_snapshot`，entry_path = `conf*.yaml`

对每个 artifact：
- 生成 `artifact_id=uuid4`
- 填充 `artifact_files`：至少把关键文件纳入（路径+sha256+size+mtime）

### 6.5 并发写入实现建议
- 使用 SQLite：
  - `PRAGMA journal_mode=WAL;`
  - `PRAGMA busy_timeout=10000;`
- 写入事务：
  - 进程内单例 `Lock`，串行执行
  - 每次写入短事务：`BEGIN IMMEDIATE` + upsert + commit
- 锁冲突处理：
  - 捕获 `database is locked`，指数退避重试（例如 50ms 起，最多 2-5s）

### 6.6 降级策略（必须）
- Registry 写入失败不得中断主流程：
  - 记录 warning 日志
  - 可选写一个 `registry_write_failures.log`
  - 下一次成功写入时继续

## 7. AIstock 消费 SQL（示例）

### 7.1 列出任务
```sql
SELECT task_run_id, scenario, status, created_at_utc, updated_at_utc
FROM task_runs
ORDER BY created_at_utc DESC;
```

### 7.2 查看某任务所有 loop，并标识哪些有成果
```sql
SELECT loop_id, action, status, has_result, best_workspace_id,
       ic_mean, ann_return, mdd, turnover, multi_score
FROM loops
WHERE task_run_id = ?
ORDER BY loop_id ASC;
```

### 7.3 找某任务“有成果”的 loop
```sql
SELECT loop_id, action, best_workspace_id
FROM loops
WHERE task_run_id = ? AND has_result = 1
ORDER BY loop_id ASC;
```

### 7.4 定位成果 workspace 入口（manifest/summary）
```sql
SELECT workspace_path, meta_path, summary_path, manifest_path
FROM workspaces
WHERE workspace_id = ?;
```

### 7.5 拉取某个成果 workspace 的 artifacts
```sql
SELECT artifact_id, artifact_type, name, entry_path, summary_json
FROM artifacts
WHERE workspace_id = ?
ORDER BY artifact_type;
```

### 7.6 拉取 artifact 的文件清单（用于复制/校验）
```sql
SELECT path, sha256, size_bytes, mtime_utc, kind
FROM artifact_files
WHERE artifact_id = ?
ORDER BY path;
```

### 7.7 按指标门槛筛选（示例：mdd<0.4）
```sql
SELECT task_run_id, loop_id, best_workspace_id, ic_mean, ann_return, mdd
FROM loops
WHERE has_result = 1 AND ABS(mdd) < 0.4
ORDER BY ic_mean DESC;
```

## 8. 对主程序的影响评估

### 8.1 性能开销
- 每个 step 增加一次/少量 SQLite upsert：I/O 级开销，通常远小于 LLM/qlib 执行。
- 只要事务短小，且使用 WAL，整体开销可控。

### 8.2 稳定性影响
- 采用降级策略：DB 不可写只告警，不影响 workflow。
- 关键风险点在于路径/权限：`<repo_root>/RDagentDB/` 需要可写。

### 8.4 DB 目录随项目迁移（实现说明）

- DB 默认路径：`<repo_root>/RDagentDB/registry.sqlite`
- 该目录在本仓库已加入 `.gitignore`，用于“随项目目录一起走，但不提交到 git”。
- 可选覆盖环境变量：
  - `RD_AGENT_REGISTRY_DB_PATH`：允许指定 DB 文件路径
  - 强约束：该路径必须位于 `<repo_root>` 之下，否则会被忽略并回退到默认路径。

### 8.3 代码侵入范围（选 A：最小侵入）
- 新增一个 registry 模块（建议 `rdagent/utils/registry/` 或 `rdagent/utils/artifact_registry/`）
- 修改 `LoopBase._run_step()`：加入 hook 调用（少量代码）
- 可选：在 session dump/load 时持久化 `task_run_id`

不改 workspace 命名规则，不改 runner 主逻辑。

## 9. 风险清单与缓解

- **[SQLite 锁冲突]**：WAL + busy_timeout + 重试退避 + 进程内 lock
- **[子进程写 DB]**：严格禁止；写入仅发生在 `_run_step` 主进程返回后
- **[DB 损坏/迁移]**：workspace 内 meta/summary/manifest 作为可重建真相源
- **[路径跨平台]**：workspace_path 记录时建议同时存：
  - 原始路径（WSL/Linux path）
  - 可选的 Windows path 映射（若有稳定映射规则）
- **[指标抽取失败]**：不阻塞；仅置 `metrics_json` 为空并告警

## 10. 上游同步工作量评估（cherry-pick 优先）

### 10.1 冲突热点
- `rdagent/utils/workflow/loop.py`（`LoopBase._run_step`）：上游也可能改这里

### 10.2 最小化冲突策略
- 将新增逻辑尽量收敛为：
  - `_run_step` 中调用一个独立函数（例如 `registry_hook.on_step_finished(...)`）
  - 具体 schema/SQL/文件 IO 全部在新模块中

### 10.3 cherry-pick vs merge/rebase（简述）
- cherry-pick：每次只挑必要修复，冲突面小，适合你们当前平台化改造阶段。
- merge/rebase：长期更接近上游，但冲突与回归成本更高，需要更强 CI。

## 11. 开发执行清单（可直接落地）

### Phase 1：基础 Registry（tasks/loops/workspaces）
- [x] 创建 `RDagentDB/` 目录（运行时自动创建）
- [x] 初始化 SQLite（建表 + WAL + 索引）
- [x] 生成与持久化 `task_run_id`（支持 resume）
- [x] 在 `LoopBase._run_step()` hook：写 task/loop/workspace 基本状态
- [x] 按成果判定规则更新 `loops.has_result/best_workspace_id`

### Phase 2：Artifacts（artifacts + artifact_files）
- [x] 在 running 成功后写入 artifacts（model/feature_set/config_snapshot/report，包含 missing 状态）
- [x] 计算关键文件 sha256/size/mtime 写入 artifact_files（best-effort）

### Phase 3：workspace 内 meta/summary（可搬运）
- [x] 写 `workspace_meta.json`（归属关系、状态、指针、registry/task_run/loop 快照）
- [x] 写 `experiment_summary.json`（指标摘要 + 关键产物路径 + artifacts 摘要）
- [x] 写 `manifest.json`（对 AIstock 的 workspace 入口契约：pointers + artifacts + 快照）

增强建议（signals 可执行信号表）：

- 对 model loop 的 `manifest.json/experiment_summary.json`：
  - `files` 字段应包含（如存在）：
    - `ret_schema.parquet` / `ret_schema.json`
    - `signals.parquet` / `signals.json`
- DB 的 `artifact_files` 应记录上述文件的 `path/sha256/size/mtime`，供 AIstock 做文件级校验与同步。

补充（Phase 3 实现增强）：

- 三文件均包含：
  - `pointers`（meta/summary/manifest 相对路径）
  - `result_criteria`（required_files + has_result 的文件级判定口径）
  - `key_metrics`（扁平关键指标摘要）
  - `registry.db_path`（best-effort 写入，用于排障；AIstock 正常消费以 workspaces 表为准）
  - `workspace_row`（对应 workspaces 表的关键字段快照）
  - `task_run` / `loop`（任务与 loop 的最小快照，便于“只搬运一个 workspace”时仍可审计）

### Phase 4：AIstock 对接与验收
- [ ] AIstock 通过 SQL：列出 task/loop、有成果 loop、定位 artifacts
- [ ] 验收：无需扫描 workspace 即可找到最新成果并拉取

建议验收步骤（已在 WSL conda 环境验证 DB 初始化与 schema）：

1) 初始化 DB（在项目根目录执行）：

```bash
cd <repo_root>
python -c "from rdagent.utils.registry.sqlite_registry import get_registry; r=get_registry(); r._ensure_initialized(); print(r.config.db_path)"
ls -lh RDagentDB/registry.sqlite
```

2) 校验表结构（示例）：

```sql
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;
PRAGMA table_info(task_runs);
PRAGMA table_info(loops);
PRAGMA table_info(workspaces);
PRAGMA table_info(artifacts);
PRAGMA table_info(artifact_files);
```

3) 跑一次最小任务后校验写入闭环：

- `task_runs` 是否写入
- `loops.has_result=1` 的 loop 能否找到 `best_workspace_id`
- `workspaces` 能否定位 `manifest_path/summary_path`
- `artifacts` 是否包含 `present/missing`
- `artifact_files.path` 是否为相对 workspace 的路径

---

## 附录 A：v1 指标列（<=10）建议
- `ic_mean`
- `rank_ic_mean`
- `ann_return`
- `mdd`
- `turnover`
- `multi_score`

其余全部放 `metrics_json`。
