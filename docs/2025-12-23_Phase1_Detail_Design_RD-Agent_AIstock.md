# Phase 1 详细设计：单策略、多因子，打通 registry 与 AIstock

> 目标：在不大改现有 RD-Agent/Qlib 结构的前提下，稳定打通“回测结果 → registry.sqlite → AIstock 消费”的闭环，为后续各阶段提供可靠基线。

---

## 1. 范围与现状基线

### 1.1 范围

- 本阶段只覆盖：
  - RD-Agent 通过 Qlib 回测生成结果；
  - 在 workspace 下生成必要文件（signals/ret/qlib_res 等）；
  - 将这些文件结构化写入 `registry.sqlite`；
  - AIstock 通过 registry 发现并消费这些成果。
- 不涉及：
  - 多策略、多窗口、多数据源路由；
  - 因子元数据/组合指标（在 Phase 2 实现）；
  - 实时/在线链路（由数据服务层与后续 Phase 处理）。

### 1.2 现状基线（简述）

- Qlib 回测：
  - 配置文件：`conf_baseline.yaml`、`conf_combined_factors_dynamic.yaml` 等；
  - 回测产物：`ret.pkl`、`qlib_res.csv`、`combined_factors_df.parquet`、`mlruns/`。
- RD-Agent workflow：
  - `QlibFBWorkspace.execute()` 负责调用 `qrun`，并读取 `ret.pkl`/`qlib_res.csv`；
  - `rdagent/utils/workflow/loop.py` 在 `action == "model"` 且关键文件存在时，写入：
    - `workspace_meta.json`、`experiment_summary.json`、`manifest.json`；
    - `workspaces`、`artifacts`、`artifact_files` 三张表；
    - `loops.has_result` 根据文件存在性设置。
- AIstock 现有能力：
  - 可通过工具/脚本连接到 `registry.sqlite`；
  - 但尚未有规范化的“如何发现回测成果并消费 signals/ret_schema”的约定文档。

---

## 2. 实现逻辑设计

### 2.1 RD-Agent 侧：has_result 与 artifacts 生成逻辑

#### 2.1.1 has_result 判定

- 对于 `action == "model"` 的 loop：
  - 必要条件（全部满足）：
    - `ret.pkl` 存在；
    - `qlib_res.csv` 存在；
  - 充分条件（Phase 1）：
    - 满足上述必要条件即可将 `has_result=1`；
    - `signals.*`、`ret_schema.*` 如存在则写入 artifacts，但不作为 Phase 1 判定的硬条件（实际实现中已基本满足）。

- 对于 `action == "factor"` 的 loop：
  - 当前不作为 AIstock 可直接消费的最终成果；
  - 即便 `result.h5` 存在，也优先考虑 `model` loop 对外暴露。

#### 2.1.2 workspace_meta / summary / manifest

- 生成位置：
  - `workspace_meta.json`：记录 workspace 的基础信息（路径、创建时间、task_run_id、loop_id 等）。
  - `experiment_summary.json`：记录关键指标（如 IC、收益、回撤等），Phase 1 可仅包含基础字段。
  - `manifest.json`：列出 workspace 下关键文件及其类型，以便 registry 与外部系统引用。

- 写入时机：
  - 在 Qlib 回测结束且 `ret.pkl`/`qlib_res.csv` 检查通过后；
  - 在设置 `has_result` 之前，确保 JSON 均已落盘。

#### 2.1.3 artifacts 与 artifact_files

- `artifacts`：一行代表一种逻辑产物（如“回测指标表”、“回测曲线”、“signals 文件”）。
- `artifact_files`：一行代表一个具体文件路径。

Phase 1 要求至少登记：

- 回测指标：
  - `artifact_type = 'backtest_metrics'`（建议命名），文件：`qlib_res.csv`；
- 回测曲线数据：
  - `artifact_type = 'backtest_curve'`，文件：`ret.pkl`；
- 策略信号（如存在）：
  - `artifact_type = 'signals'`，文件：`signals.parquet`/`signals.json`；
- ret_schema（如存在）：
  - `artifact_type = 'ret_schema'`，文件：`ret_schema.parquet`/`ret_schema.json`；
- qlib 运行配置与日志（可选）：
  - `artifact_type = 'qlib_config'`、`'qlib_log'` 等，对 AIstock 暂不强依赖。

### 2.2 AIstock 侧：发现与消费逻辑

#### 2.2.1 发现“可用成果”

- SQL 示例：

```sql
SELECT
  tr.id AS task_run_id,
  l.id AS loop_id,
  l.action,
  l.has_result,
  w.id AS workspace_id,
  w.workspace_path
FROM task_runs tr
JOIN loops l ON l.task_run_id = tr.id
JOIN workspaces w ON w.loop_id = l.id
WHERE l.action = 'model' AND l.has_result = 1;
```

- AIstock 通过上述查询获得一组 `(task_run, loop, workspace)` 作为候选结果集。

#### 2.2.2 获取 artifacts 列表

- 通过 `artifacts` / `artifact_files`：

```sql
SELECT
  a.id AS artifact_id,
  a.artifact_type,
  f.file_path
FROM artifacts a
JOIN artifact_files f ON f.artifact_id = a.id
WHERE a.workspace_id = :workspace_id;
```

- AIstock 对 `artifact_type` 做规范化解析：
  - `'backtest_metrics'` → `qlib_res.csv`；
  - `'backtest_curve'` → `ret.pkl`；
  - `'signals'` → `signals.parquet`/`signals.json`；
  - `'ret_schema'` → `ret_schema.parquet`/`ret_schema.json`。

#### 2.2.3 数据消费

- AIstock 推荐：
  - 通过相对路径 + 仓库根/工作目录，组装为绝对路径；
  - 使用 pandas/pyarrow 等读入 DataFrame / Series：
    - `qlib_res.csv`：回测指标表；
    - `ret.pkl`：时间序列回测曲线（可绘制净值/回撤曲线）；
    - `signals.*`：策略信号（用于后续模拟盘/人工交易参考）；
    - `ret_schema.*`：信号含义与字段规范（用于下游系统消费）。

#### 2.2.4 最小结果展示（AIstock 侧）

- **后端接口建议**：
  - `GET /api/rdagent/strategies`：
    - 功能：列出“已导入 RD-Agent 策略”（你现有的列表接口，可复用）；
    - 字段：策略 ID、名称、形态、输出模式、来源 key（含 task_run/loop/workspace）、状态、创建时间、版本等。
  - `GET /api/rdagent/strategies/{strategy_id}/result`：
    - 功能：根据策略 ID 查找对应的 workspace 与 artifacts，返回回测指标和曲线数据；
    - 处理流程：
      1. 根据策略表中的 `source_key` 解析出 `task_run_id` / `loop_id` / `workspace_id`，或直接存储 `workspace_id`；
      2. 通过 `artifacts` / `artifact_files` 查找：
         - `artifact_type='backtest_metrics'` → `qlib_res.csv`；
         - `artifact_type='backtest_curve'` → `ret.pkl`；
      3. 用 pandas 读取文件，提取若干核心指标和净值曲线；
      4. 将结果以 JSON 形式返回给前端，例如：

```json
{
  "metrics": {
    "annual_return": 0.18,
    "max_drawdown": 0.35,
    "sharpe": 1.20
  },
  "equity_curve": [
    {"date": "2025-01-02", "nav": 1.01},
    {"date": "2025-01-03", "nav": 1.02}
  ]
}
```

- **前端展示（最小版本）**：
  - 在现有“已导入 RD-Agent 策略”列表中：
    - 每行增加一个“查看结果”按钮，跳转到策略详情页；
  - 策略详情页最小布局：
    - 顶部：展示关键回测指标卡片（年化收益、最大回撤、Sharpe 等）；
    - 中部：一张净值/收益曲线图（使用 `equity_curve` 数据绘制）；
    - 底部：保留来源信息（task_run/loop/workspace、创建时间等），便于调试。

- **实现约束**：
  - 当前阶段不强制展示 signals/ret_schema 细节，仅做“基础可视化”；
  - 所有数据均通过 registry.sqlite + artifacts 发现与读取，不解析 RD-Agent 日志；
  - 若某个 loop 缺失 `backtest_metrics` 或 `backtest_curve`，后端返回友好错误信息，前端提示“该实验结果不完整”。

---

## 3. 规范 / 规则 / 限制

### 3.1 命名与路径规范

- `workspace_path`：统一使用相对于 RD-Agent 仓库根的路径或绝对路径，但需在文档中明确约定；
- `file_path`：在 `artifact_files` 中记录相对于 `workspace_path` 的相对路径，便于跨环境移动；
- 不在 DB 中记录 Windows/WSL 混排路径，保持单一风格（建议统一 WSL 风格 `/mnt/f/...`，Windows 侧由 AIstock 做一次转换）。

### 3.2 artifacts 类型规范（Phase 1）

- 必须类型：
  - `backtest_metrics`、`backtest_curve`；
- 推荐类型：
  - `signals`、`ret_schema`；
- 可选类型：
  - `qlib_config`、`qlib_log`、`mlruns` 等。

### 3.3 限制

- Phase 1 不引入因子层 artifacts（factor_meta/factor_perf），避免一次性改动过多；
- 不改变 Qlib 回测内部逻辑，仅在回测完成后读取其产物；
- 不改变 RD-Agent 现有因子/策略演进流程，只在结果写入与回填层加固。

---

## 4. 开发计划与任务拆分

### 4.1 RD-Agent 侧

1. **梳理并锁定现有 loop 写入逻辑**
   - 确认 `rdagent/utils/workflow/loop.py` 中 model 分支的行为；
   - 确定 `has_result` 判定条件与 meta/summary/manifest 写入顺序。

2. **补充/修正 artifacts 类型与路径**
   - 确保：
     - `qlib_res.csv` → `backtest_metrics`；
     - `ret.pkl` → `backtest_curve`；
     - `signals.*` → `signals`；
     - `ret_schema.*` → `ret_schema`；
   - 补充缺失的 `artifact_files` 行，保证 `artifact_id` 与 `workspace_id` 关联正确。

3. **完善 backfill 工具**
   - 确保 `tools/backfill_registry_artifacts.py`：
     - 能扫描历史 workspace，检测已存在与缺失的 JSON/artifacts；
     - 支持幂等重跑（通过 stable UUID 或确定性键值）；
     - 提供 `--cleanup-existing` 选项用于清理重复/错误记录。

### 4.2 AIstock 侧

1. **编写 registry 访问封装**
   - 提供一个内部模块：
     - `list_model_loops_with_result()` → 返回 task_run/loop/workspace 列表；
     - `list_artifacts(workspace_id)` → 返回 artifacts 与文件路径；
   - 内部使用 SQLAlchemy/原生 sqlite3 均可，注意连接池与并发访问。

2. **构建基础浏览/调试界面（可选但推荐）**
   - 在 AIstock UI 中增加“RD-Agent 实验浏览”页面：
     - 列出最近的 model loops；
     - 可点击查看回测指标与简要结果。

---

## 5. Phase 1 验收标准（落地版）

1. **技术验收**：
   - 任选一个或多个 task_run：
     - 至少有 1 个 `action='model' AND has_result=1` 的 loop；
   - 对每个这样的 loop：
     - `workspace_meta.json`、`experiment_summary.json`、`manifest.json` 存在且 JSON 可解析；
     - `artifacts` / `artifact_files` 中至少存在：
       - 一条 `artifact_type='backtest_metrics'` 指向 `qlib_res.csv`；
       - 一条 `artifact_type='backtest_curve'` 指向 `ret.pkl`；
     - backfill 工具重复执行不会引入重复记录或破坏已有记录。

2. **AIstock 功能验收**：
   - 通过 AIstock 内部脚本或 UI：
     - 能列出最近 N 个 “有结果的回测 loop”；
     - 对任一 loop：
       - 成功读取 `qlib_res.csv` 并展示主要指标；
       - 成功读取 `ret.pkl` 并绘制一张简单的净值/回撤曲线；
   - 所有这些操作不依赖 RD-Agent 内部日志或硬编码 workspace 目录结构，只通过 registry 与 manifest/artifacts 完成。

3. **文档验收**：
   - `2025-12-22_AIstock_RD-Agent_Registry_Integration_Guide.md` 中已：
     - 更新 Phase 1 验收步骤与 SQL/Python 示例；
     - 明确 `artifact_type` 约定与 AIstock 消费方式；
   - 本 Phase 1 详细设计文档与顶层设计保持一致，无冲突。
