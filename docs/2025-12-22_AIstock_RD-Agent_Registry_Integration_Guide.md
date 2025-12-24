# AIstock 对接 RD-Agent 成果 Registry（SQLite + Workspace Manifest）说明

## 0. 目的

AIstock 侧**无需扫描 RD-Agent workspace 目录**，仅通过读取 RD-Agent 输出的：

- 项目内 SQLite：`RDagentDB/registry.sqlite`
- 每个 workspace 内的 `manifest.json` / `experiment_summary.json`

即可实现：

- 按 task_run → loop → workspace → artifacts 定位成果
- 按关键指标（例如 `mdd`）筛选候选
- 获取可复制的文件清单（相对路径 + hash/size/mtime）

> 约束：
> - 本方案中 AIstock 对接以**只读**方式使用 SQLite 与 workspace 文件。
> - `RDagentDB/` 目录随项目目录迁移，但不提交到 git（已加入 `.gitignore`）。

---

## 1. 文件位置与运行环境

### 1.1 DB 文件位置（必须跟项目走）

- DB 默认路径（Linux/WSL）：
  - `<repo_root>/RDagentDB/registry.sqlite`

例如你的项目根目录为：`/mnt/f/Dev/RD-Agent-main`，则：

- `/mnt/f/Dev/RD-Agent-main/RDagentDB/registry.sqlite`

### 1.2 可选环境变量

- `RD_AGENT_DISABLE_SQLITE_REGISTRY=1`
  - 禁用 SQLite registry（不建议在对接阶段启用）

- `RD_AGENT_REGISTRY_DB_PATH`
  - 指定 DB 文件路径
  - 强约束：该路径必须位于 `<repo_root>` 之下，否则 RD-Agent 会忽略并回退到默认路径

### 1.3 Parquet 读取依赖（AIstock 推荐）

- AIstock 侧推荐统一使用 **`pyarrow`** 读取 Parquet（与 RD-Agent 运行依赖一致）：
  - `pandas.read_parquet(..., engine="pyarrow")`
  - `duckdb` / `polars` / `spark` 等也均可直接读取 Parquet
- 不推荐在 AIstock 侧引入 `fastparquet` 作为主引擎，以减少类型兼容性差异。

---

## 2. SQLite Schema（AIstock 消费关注点）

SQLite 内包含 5 张表：

- `task_runs`：一次 RD-Agent CLI 启动
- `loops`：每轮迭代（factor/model）与关键指标
- `workspaces`：workspace 归属与入口指针（meta/summary/manifest）
- `artifacts`：组件级成果（model/feature_set/config_snapshot/report）
- `artifact_files`：文件级指纹（相对路径 + hash/size/mtime）

AIstock 侧消费的最小闭环链路是：

`task_runs` → `loops.best_workspace_id` → `workspaces.manifest_path` → 读取 workspace 内 `manifest.json`（作为可搬运真相源）

---

## 3. AIstock 推荐消费流程（闭环）

### 3.1 选择 task_run

```sql
SELECT task_run_id, scenario, status, created_at_utc, updated_at_utc
FROM task_runs
ORDER BY created_at_utc DESC
LIMIT 20;
```

### 3.2 列出该 task_run 的 loops（并筛选有成果的 loop）

```sql
SELECT loop_id, action, status, has_result, best_workspace_id,
       ic_mean, ann_return, mdd, turnover, multi_score
FROM loops
WHERE task_run_id = ?
ORDER BY loop_id ASC;
```

示例筛选（mdd 门槛）：

```sql
SELECT loop_id, action, best_workspace_id, ic_mean, ann_return, mdd
FROM loops
WHERE task_run_id = ?
  AND has_result = 1
  AND mdd IS NOT NULL
  AND ABS(mdd) < 0.4
ORDER BY ic_mean DESC;
```

### 3.3 定位成果 workspace 的入口文件

```sql
SELECT workspace_id, workspace_role, experiment_type,
       workspace_path, meta_path, summary_path, manifest_path
FROM workspaces
WHERE workspace_id = ?;
```

说明：

- `workspace_path` 是绝对路径（WSL/Linux），指向 workspace 根目录
- `meta_path/summary_path/manifest_path` 是**相对 workspace 的路径**

AIstock 侧应优先拼接：

- `manifest_abs_path = workspace_path + "/" + manifest_path`

### 3.4 读取 workspace 的 manifest（推荐作为对接入口）

- 文件：`<workspace_path>/<manifest_path>`，默认为 `manifest.json`
- 内容包含：
  - `task_run` / `loop` / `workspace_row` 快照（用于“只搬运一个 workspace 目录”时仍可审计）
  - `key_metrics`（扁平关键指标摘要）
  - `result_criteria`（required_files + has_result 的判定口径）
  - `artifacts[]`（组件级产物摘要，含 `artifact_id/artifact_type/status/entry_path/files`）

> 注意：
> - `manifest.json` 的 `artifacts[]` 仅用于快速查看/搬运；
> - AIstock 若需要严格/完整文件清单，应以 DB 的 `artifact_files` 为准。

### 3.5 从 DB 拉取 artifacts / artifact_files（用于复制/同步）

按 workspace 拉 artifacts：

```sql
SELECT artifact_id, artifact_type, name, status, entry_path, summary_json
FROM artifacts
WHERE workspace_id = ?
ORDER BY artifact_type;
```

按 artifact 拉文件清单：

```sql
SELECT path, sha256, size_bytes, mtime_utc, kind
FROM artifact_files
WHERE artifact_id = ?
ORDER BY path;
```

说明：

- `artifact_files.path` 为**相对 workspace 的路径**
- 对目录型 entry（如 `mlruns/`），`artifact_files` 会包含：
  - 根目录的一个记录（sha256 为空）
  - 以及部分关键文件（数量受限，best-effort）

---

## 4. Artifact 类型与约定

### 4.1 qlib `model` loop

- 成果判定：`ret.pkl` 与 `qlib_res.csv` 同时存在
- 典型 artifacts：
  - `report`：入口 `qlib_res.csv`（files 里包含 `qlib_res.csv`/`ret.pkl`）
  - `model`：入口 `mlruns/`
  - `config_snapshot`：入口 workspace 根目录（files 里包含最多 50 个 yaml）

### 4.2 qlib `factor` loop

- 成果判定：`combined_factors_df.parquet` 存在
- 典型 artifacts：
  - `feature_set`：入口 `combined_factors_df.parquet`
  - `config_snapshot`：入口 workspace 根目录（files 里包含最多 50 个 yaml）

### 4.3 `status=missing`

即使产物缺失，也会写入 `artifacts.status="missing"`，用于 AIstock 诊断：

- 该 loop 产生了 workspace，但成果文件未生成/被清理
- 可结合 `workspaces.summary_path` 与 `log_trace_path` 排查

---

## 5. Phase 4 验收（AIstock 自测清单）

### 5.1 初始化 DB

在项目根目录执行：

```bash
python -c "from rdagent.utils.registry.sqlite_registry import get_registry; r=get_registry(); r._ensure_initialized(); print(r.config.db_path)"
ls -lh RDagentDB/registry.sqlite
```

### 5.2 结构校验

```sql
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;
PRAGMA table_info(task_runs);
PRAGMA table_info(loops);
PRAGMA table_info(workspaces);
PRAGMA table_info(artifacts);
PRAGMA table_info(artifact_files);
```

### 5.3 写入闭环校验（跑一次最小任务后）

- `task_runs` 行数 > 0
- `loops` 行数 > 0
- 至少 1 个 `loops.has_result = 1`
- 对应 `best_workspace_id` 在 `workspaces` 中可查
- `workspaces.manifest_path` 指向的文件存在且可 JSON 解析
- `artifacts` 至少包含 `report/model/config_snapshot` 或 `feature_set/config_snapshot`

补充（signals 可执行信号表）：

- 对 `action=model` 且 `has_result=1` 的 best workspace：
  - workspace 根目录应存在 `signals.parquet` 或 `signals.json`（至少一个）
  - `workspaces.manifest_path` 指向的 `manifest.json` 的 `files` 中应包含 `signals.parquet/signals.json` 的相对路径（若存在）
  - DB `artifact_files` 中应能查到 `path IN ('signals.parquet','signals.json')` 的记录（如存在）

---

## 6. 最小实现建议（AIstock 侧代码结构）

建议 AIstock 实现一个只读服务层：

- `RegistryReader(db_path)`
  - `list_task_runs()`
  - `list_loops(task_run_id)`
  - `get_best_workspaces(task_run_id)`
  - `list_artifacts(workspace_id)`
  - `list_artifact_files(artifact_id)`

- `WorkspaceManifestReader(workspace_path, manifest_path)`
  - `load_manifest()`
  - `validate_manifest_schema_best_effort()`

其中：

- DB 是权威索引（可筛选/聚合）
- workspace `manifest.json` 是可搬运真相源与 debug 入口

---

## 7. 成果内容清单与格式契约（B：严格解析）

本章节定义：AIstock 在拿到 workspace 后，如何**逐个文件**做解析与校验。

### 7.1 `qlib_res.csv`（关键指标表，CSV）

来源（权威生成方式）：workspace 内 `read_exp_res.py`。

- 生成方式：
  - `metrics = pd.Series(latest_recorder.list_metrics())`
  - `metrics.to_csv("qlib_res.csv")`

因此 `qlib_res.csv` 的契约为：

- **文件类型**：CSV
- **行语义**：每一行是一条指标
- **列结构**：两列
  - 第 1 列：指标名（作为 index 列；AIstock 读取时建议 `index_col=0`）
  - 第 2 列：指标值（数值或可转数值的字符串）

AIstock 推荐解析方式：

```python
import pandas as pd

metrics_series = pd.read_csv("qlib_res.csv", index_col=0).iloc[:, 0]
# metrics_series 是一个 pandas.Series: index=metric_name, value=metric_value
```

指标 key → Registry `loops` 列映射（qlib quant 场景）：

- `loops.ic_mean` ← `metrics_series["IC"]`
- `loops.rank_ic_mean` ← `metrics_series["Rank IC"]`
- `loops.ann_return` ← `metrics_series["1day.excess_return_with_cost.annualized_return"]`
- `loops.mdd` ← `metrics_series["1day.excess_return_with_cost.max_drawdown"]`
- `loops.turnover` ← `metrics_series["1day.excess_return_with_cost.turnover"]`（如存在）
- `loops.multi_score` ← 若 RD-Agent 写入则以 DB 为准（AIstock 不建议自行复算）

兼容性注意事项（必须）：

- 少数历史实现里，`annualized_return` 的 key 可能存在尾部空格（例如 `"1day.excess_return_with_cost.annualized_return "`）。
- AIstock 解析时建议对 index 做一次 `strip()` 的容错：

```python
metrics_series.index = metrics_series.index.map(lambda s: str(s).strip())
```

### 7.2 `ret.pkl`（回测曲线/报告对象，Pickle）

来源（权威生成方式）：workspace 内 `read_exp_res.py`：

- `latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")`
- 然后 `ret_data_frame.to_pickle("ret.pkl")`

因此 `ret.pkl` 的契约为：

- **文件类型**：pickle（Python pickle）
- **对象类型**：`pandas.DataFrame`（或 pandas 可兼容读取的对象）
- **读取方式**：

```python
import pandas as pd

ret_obj = pd.read_pickle("ret.pkl")
```

字段/列约定（严格程度说明）：

- RD-Agent 当前实现只保证：`pd.read_pickle` 能成功读出对象，并且对象可用于“回测曲线/回测报告”的可视化/分析。
- 该对象的列名/具体内容由 qlib 的 `portfolio_analysis/report_normal_1day.pkl` 决定，可能随 qlib 版本变化。

AIstock 消费建议（稳定口径）：

- 将 `ret.pkl` 作为“结果证据文件”进行**存档与可视化**，而不是把它作为唯一的指标计算来源。
- 指标筛选以 DB 的 `loops` 列 + `metrics_json` 为准。

### 7.2.1 `ret_schema.parquet` / `ret_schema.json`（稳定 schema 的回测结果表，推荐）

来源（权威生成方式）：workspace 内 `read_exp_res.py`。

- `ret_schema.parquet`
  - **文件类型**：Parquet（建议 `pyarrow` 读取）
  - **对象类型**：`pandas.DataFrame`
  - **写出约束**：`index=False`（index 已被 reset 成列，避免依赖 pickle/index 语义）
- `ret_schema.json`
  - **文件类型**：JSON（`pandas.DataFrame.to_json(orient="table")`）
  - **特点**：包含 `schema` + `data`，适合作为“零依赖/跨语言”的备选解析入口

AIstock 推荐解析方式（Python）：

```python
import pandas as pd

df = pd.read_parquet("ret_schema.parquet", engine="pyarrow")
```

兼容性说明（必须）：

- 该表用于稳定承载回测结果对象，但其列集合仍可能受 qlib 报告结构影响。
- 若 AIstock 需要“直接变成可执行策略”，应优先消费下文的 `signals.*`（可执行信号表）。

### 7.3 `combined_factors_df.parquet`（因子矩阵，Parquet）

来源（权威生成方式）：`rdagent/scenarios/qlib/developer/factor_runner.py` 与 `model_runner.py`。

写入方式要点：

- `combined_factors` 为 `pandas.DataFrame`
- `combined_factors.index`：来自各子实现 factor DataFrame 的 index 拼接；至少包含 `datetime`
- `combined_factors.columns`：会被设置为二级 MultiIndex：
  - level-0 固定为 `"feature"`
  - level-1 为因子名（字符串）
- 使用 `engine="pyarrow"` 写出：`combined_factors.to_parquet("combined_factors_df.parquet", engine="pyarrow")`

因此 `combined_factors_df.parquet` 的契约为：

- **文件类型**：Parquet（pyarrow 兼容）
- **对象类型**：`pandas.DataFrame`
- **index**：至少包含 `datetime`（通常为 MultiIndex：`instrument` + `datetime`，但 AIstock 以实际文件为准）
- **columns**：二级 MultiIndex，第一层恒为 `feature`

AIstock 推荐解析方式：

```python
import pandas as pd

df = pd.read_parquet("combined_factors_df.parquet")

# 合同约束：必须存在 datetime
if isinstance(df.index, pd.MultiIndex):
    assert "datetime" in df.index.names
else:
    # 兼容：如果不是 MultiIndex，也应能通过列或 index 表达 datetime（不推荐）
    raise ValueError("combined_factors_df.parquet index is expected to include 'datetime'")

# 合同约束：columns 第一层为 feature
if isinstance(df.columns, pd.MultiIndex):
    assert len(df.columns.levels) >= 2
    assert set(df.columns.get_level_values(0)) == {"feature"}
```

### 7.4 `mlruns/`（MLflow Tracking 目录）

来源（权威生成方式）：`QlibFBWorkspace.execute()` 会在 workspace 下创建 `mlruns/` 并设置 `MLFLOW_TRACKING_URI` 指向该目录，然后 qlib 训练/回测会写入 MLflow。

因此 `mlruns/` 的契约为：

- **类型**：目录
- **位置**：`<workspace_root>/mlruns/`
- **用途**：保存 qlib 运行时的 MLflow experiments / runs / artifacts
- **典型结构**（MLflow 标准布局，AIstock 以“目录整体复制/归档”为主）：
  - `mlruns/<experiment_id>/<run_id>/`
    - `meta.yaml`
    - `metrics/`（逐指标文件）
    - `params/`（参数文件）
    - `artifacts/`（产物，如 code_diff.txt、模型文件等）

AIstock 消费建议（稳定口径）：

- 若需要完整复现实验/追溯：**整体复制 `mlruns/`**（不要只挑某几个文件）。
- 若只需要指标：以 `qlib_res.csv` + DB 的 `loops/metrics_json` 为准。

### 7.5 `conf*.yaml`（实验配置快照）

来源（权威使用方式）：

- qlib 执行入口：`qrun <config.yaml>`
- config 文件存在于 workspace 根目录（模板注入 + runner 选择）

因此 `conf*.yaml` 的契约为：

- **类型**：YAML 文本
- **位置**：workspace 根目录下的若干 `conf*.yaml`
- **用途**：描述 qlib 的 dataset/model/backtest/handler 等配置

AIstock 消费建议（稳定口径）：

- 将 workspace 根目录下 `conf*.yaml` 全量归档为 `config_snapshot` 的文件集合。
- 如果 AIstock 需要读取关键配置（例如数据源、region、provider_uri），可解析 YAML，并优先读取：
  - `qlib_init.provider_uri`
  - `qlib_init.region`

### 7.6 `signals.parquet` / `signals.json`（强固定 schema 的可执行信号表，AIstock 主入口）

目标：

- AIstock 从 RD-Agent workspace **直接得到可执行的交易信号/目标仓位表**，并可无歧义映射到 miniQMT 的调仓/下单逻辑。
- 同时保留充分的策略治理/审计字段，支持后续“策略上线评审/策略管理”。

文件形态：

- `signals.parquet`
  - **文件类型**：Parquet（建议 `pyarrow` 读取）
  - **对象类型**：`pandas.DataFrame`
  - **写出约束**：`index=False`
- `signals.json`
  - **文件类型**：JSON（建议 `orient="table"`，包含 schema+data）

强固定字段契约（AIstock 必须按此解析；RD-Agent 侧必须按此导出）：

- **标识与时间**
  - `trade_date`：交易日期（ISO8601 日期字符串，或可解析为日期的字符串）
  - `instrument`：标的代码（必须能映射为 miniQMT 可识别 code）
- **执行目标（核心）**
  - `signal`：信号强度（float，正/负均允许；纯多头策略可约束为非负）
  - `target_weight`：目标权重（float，可为负表示做空；若不做空则应 >=0）
  - `target_position`：目标仓位数量（float/int，允许为空；若提供则以该列为准）
  - `price_ref`：参考价格（float，允许为空；用于从 weight 换算 position）
  - `universe_flag`：是否在当日可交易池（bool/int）
- **排序与解释（策略治理/审计）**
  - `score`：模型分数/打分（float，允许为空）
  - `rank`：当日 rank（int，允许为空）
  - `pred_return`：预测收益（float，允许为空）
  - `confidence`：置信度（float，允许为空）
  - `volatility_est`：波动率估计（float，允许为空）
  - `max_weight`：单标的最大权重约束（float，允许为空）
  - `min_weight`：单标的最小权重约束（float，允许为空）
  - `sector`：行业/板块（str，允许为空）
  - `industry`：细分行业（str，允许为空）
- **溯源（强烈建议 AIstock 入库）**
  - `generated_at_utc`：生成时间戳（ISO8601）
  - `task_run_id`：本次任务 id（UUID）
  - `loop_id`：loop id（int）
  - `workspace_id`：workspace id（UUID）
  - `model_version`：模型/策略版本（str，允许为空）

权重策略元信息（推荐；AIstock 可按需消费）：

- `weight_method`：权重策略名称（例如 `topk_equal_weight`）
- `topk`：TopK 参数（int）
- `n_drop`：dropout 参数（int；如未启用可为 0）
- `rebalance_freq`：调仓频率（例如 `1d`）

兼容性注意事项（必须）：

- v1 的 `signals.*` 可能由 workspace 内的结果提取脚本生成，部分溯源字段可能暂时为空（例如 `task_run_id/loop_id/workspace_id/model_version`）。
- AIstock 侧解析时应允许上述字段为 null/空字符串，并优先以 SQLite/manifest 的 workspace 上下文做关联。

AIstock 推荐解析方式（Python）：

```python
import pandas as pd

signals = pd.read_parquet("signals.parquet", engine="pyarrow")

required = [
    "trade_date",
    "instrument",
    "signal",
    "target_weight",
    "target_position",
    "price_ref",
    "universe_flag",
    "score",
    "rank",
    "pred_return",
    "confidence",
    "volatility_est",
    "max_weight",
    "min_weight",
    "sector",
    "industry",
    "generated_at_utc",
    "task_run_id",
    "loop_id",
    "workspace_id",
    "model_version",
]

missing = [c for c in required if c not in signals.columns]
if missing:
    raise ValueError(f"signals schema mismatch, missing columns: {missing}")

# 允许 signals 存在额外列（例如 weight_method/topk/n_drop/rebalance_freq），不应当作为错误
```

miniQMT 执行建议（策略生成/调仓口径）：

- 若 `target_position` 不为空：优先按 `target_position` 调仓
- 否则使用 `target_weight` + 账户权益 + `price_ref` 换算目标仓位

> 注意：`signals.*` 是“可执行层主入口”，`ret_schema.*` 与 `ret.pkl` 作为“研究证据/回测报告”存档。

---

## 8. “成果包含内容列表”（按 artifact_type 汇总）

本节给出 AIstock 拉取时的最小“文件集合”建议（以 DB 为准，manifest 仅辅助展示）。

### 8.1 `report`

- 必备：
  - `qlib_res.csv`
- 推荐：
  - `ret.pkl`
  - `ret_schema.parquet`
  - `ret_schema.json`
  - `signals.parquet`
  - `signals.json`

### 8.2 `feature_set`

- 必备：
  - `combined_factors_df.parquet`

### 8.3 `model`

- 推荐：
  - `mlruns/`（整体目录）

### 8.4 `config_snapshot`

- 推荐：
  - `conf*.yaml`（最多 50 个文件记录写入 DB 的 `artifact_files`，AIstock 如需全量可直接扫描 workspace 根目录的 yaml）
