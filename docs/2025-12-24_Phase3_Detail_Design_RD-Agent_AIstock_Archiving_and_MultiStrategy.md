# Phase 3 详细设计：成果归档 & Workspace 清理策略 + 多策略/多因子扩展

> 前提：Phase 1/2 已完成，实现了：RD-Agent → registry.sqlite → AIstock 的结构化成果暴露（含因子/策略 artifacts）以及 AIstock 侧最小成果归档（因子库/策略库雏形）。
> 同时，AIstock 侧已在《DataServiceLayer 详细设计》约定的数据视图基础上，完成数据服务层的基础框架搭建（snapshot/history/streaming 等），但尚未与 RD-Agent 产出的策略/因子做在线绑定。
> 本阶段在此基础上，完成成果资产化闭环与多策略扩展，为后续在线路径与数据服务层接入打下“资产与管理”层面的基础。

---

## 1. 目标与范围

### 1.1 目标

1. **成果完整归档与安全清理**：
   - 在 AIstock 侧实现一套清晰的“归档等级”与对应的 DB/文件存储结构；
   - 对于达到“完整归档”条件的实验/因子/策略，允许通过工具安全清理对应的 RD-Agent workspace/log，而不丢失有价值的信息。

2. **多策略/多因子实验管理**：
   - 在 StrategyConfig/ExperimentRequest 的基础上，支持：
     - 多策略、多参数版本并行实验；
     - 多因子组合与对照组实验；
     - 与外部因子/外部策略的联合回测和对比。

3. **AIstock 侧集中资产池增强**：
   - 把 Phase 2 中的“最小因子库/策略库”扩展为可支持筛选、组合、复用的资产层；
   - 为后续“从因子库/策略库中选择组合 → 下发给 RD-Agent 或 miniQMT”的工作流奠定数据基础。

### 1.2 范围

- **包含**：
  - 成果归档等级定义与 schema 设计；
  - AIstock 侧归档流水线与历史/增量导入逻辑；
  - Workspace/log 清理策略与工具设计；
  - 多策略/多因子实验配置与在 registry 中的表示方式；
  - 对 Phase 2 中 importer/因子库/策略库结构的扩展。

- **不包含（留给后续 Phase 4/5）**：
  - 在线策略引擎 + 数据服务层的**实时集成细节**（即“从因子库/策略库中选择策略 → 绑定到数据服务层视图 → 实时/准实时执行”的整条链路）；
  - 模拟盘/实盘执行逻辑与接口；
  - 自动化的“基于 LiveFeedback 的演进实验调度”。

### 1.3 与数据服务层及在线策略的关系（与 Phase 2 对齐）

- Phase 2 中已经明确：
  - **RD-Agent 侧职责**：
    - 通过统一 `write_loop_artifacts(...)` 产出稳定的因子/策略 artifacts（`factor_meta`/`factor_perf`/`feedback`/回测图表等）；
    - 确保这些 artifacts 足以支撑 AIstock 因子库/策略库与回测视图展示；
  - **AIstock 侧职责**：
    - 实现从 registry.sqlite + workspace artifacts 导入到本地 DB 的 importer；
    - 在内部搭建数据服务层基础设施（snapshot/history/streaming 等），可供未来在线策略复用；
  - **暂不要求** RD-Agent 在 Phase 2 就切换为通过数据服务层获取行情，离线回测仍可延续现有 h5/bin 流程。

- 本 Phase 3 的角色是：
  - 在 Phase 2 最小归档与数据服务层“基础设施就绪”的前提下：
    - 完成成果的 **Level 2 完整归档** 与 workspace 清理策略；
    - 将因子/策略/实验抽象为可在 AIstock 内集中管理、筛选与组合的“资产池”；
    - 为后续在线策略接入数据服务层提供**清晰的资产边界与标识**（例如通过 `aistock_strategies`/`aistock_experiments` 的主键与档案信息）。

- 真正的“在线策略接入数据服务层”工作流（示意）：
  - 用户在 AIstock 因子库/策略库中选择一组策略/因子 →
  - 为其绑定数据服务层视图（如某个账户、市场、频率、实时行情流）→
  - 将该组合策略下发到在线策略引擎/miniQMT，执行模拟盘或实盘；
- 以上工作流 **不在本 Phase 3 的开发范围内**，而是明确规划为 Phase 4/5 的内容；本阶段仅保证：
  - 当 Phase 4/5 启动时，AIstock 侧已经拥有：
    - 完整可依赖的因子/策略/实验资产库（含归档级别与文件/指标信息）；
    - 可复用的数据服务层基础设施；
  - 使后续在线集成工作可以仅在 AIstock 侧完成，**无需再对 RD-Agent 的 artifacts 结构做破坏性调整**。

---

## 2. 成果归档等级与数据结构

### 2.1 归档等级定义

为实验/策略/因子组合定义 3 个归档等级：

- **Level 0（未归档）**：
  - 仅 registry.sqlite + RD-Agent workspace 中的原始 artifacts 存在；
  - AIstock 只能临时读取结果，不保证长期可回溯；
  - 不允许清理 workspace。

- **Level 1（最小归档，Phase 2 已实现的部分）**：
  - AIstock DB 中已有：
    - 因子库最小字段（name/source/description_cn/formula_hint/tags）与主要表现指标（ic_mean/ic_ir/coverage+主窗口收益/回撤/Sharpe）；
    - 策略库最小字段（策略 ID/source/关联因子集/关键回测指标/推荐状态等）；
  - 但**文件级 artifacts 仍主要依赖 RD-Agent workspace**；
  - 清理 workspace 仍然会导致部分信息丢失，仅允许清理 log 目录。

- **Level 2（完整归档，Phase 3 目标）**：
  - 在 Level 1 基础上，AIstock 侧还具备：
    - 关键文件的本地化副本：
      - 回测指标表（`qlib_res.csv` 等）已导入 DB 或存储在 AIstock 管理的对象存储，并与 DB 记录关联；
      - 回测曲线（`ret.pkl`）已转换为时序表或标准化的压缩存储格式；
      - 因子元数据、表现、反馈 JSON 已复制到 AIstock 存储（或完全吸收入库）；
    - DB 中有明确字段标记该实验/策略/因子组合已达到 Level 2；
  - 对于 Level 2 记录，可通过专用脚本清理对应 RD-Agent workspace，以释放磁盘空间，而不影响 AIstock 的展示、分析与筛选能力。

### 2.2 AIstock DB 结构扩展（示意）

在 Phase 2 的基础上，进一步扩展以下表结构（字段示意）：

- `aistock_factors`：
  - `id` (PK)
  - `name`
  - `version`
  - `source` (rdagent_generated/alpha/external)
  - `description_cn`
  - `formula_hint`
  - `tags` (JSON/text)
  - `first_experiment_id`
  - `created_at`

- `aistock_factor_performance`：
  - `id` (PK)
  - `factor_id` (FK → aistock_factors)
  - `window_name`（如 `train_2018_2020`/`test_2021_2025`）
  - `ic_mean`
  - `ic_ir`
  - `coverage`
  - `annual_return`
  - `max_drawdown`
  - `sharpe`
  - `experiment_id`

- `aistock_factor_combinations`：
  - `id` (PK)
  - `name`
  - `factor_ids` (JSON 数组)
  - `experiment_id`

- `aistock_strategies`：
  - `id` (PK)
  - `strategy_key`（如 task_run/loop/workspace 三元组）
  - `name`
  - `shape`（`portfolio`/`signal` 等）
  - `output_mode`（`target_weight`/`long_short` 等）
  - `source` (rdagent/handcrafted/external)
  - `factor_combination_id` (FK)
  - `created_at`
  - `recommendation_status`（rdagent_accept/rdagent_reject/manual_review 等）

- `aistock_experiments`：
  - `id` (PK)
  - `task_run_id`
  - `loop_id`
  - `workspace_id`
  - `strategy_id` (FK)
  - `archiving_level` (0/1/2)
  - `metrics_json`（整合 qlib_res/experiment_summary 中的关键指标）
  - `feedback_json`（整合 feedback.json 中的决策与限制）
  - `file_refs_json`（记录已复制到 AIstock 存储的文件路径/对象键）
  - `created_at`
  - `imported_at`

具体字段可以在实施时细化，但本设计明确：**Phase 3 要把这些表结构固化到可实现程度**。

---

## 3. AIstock 归档流水线设计

### 3.1 总体流程

归档流水线由三层组成：

1. **扫描层**（Scan）：
   - 基于 registry.sqlite：
     - 枚举所有 `action='model' AND has_result=1` 的 loops；
     - 读取对应的 workspaces 与 artifacts 列表；
   - 对于每个 loop：读取 `archiving_level`（如果已存在）决定需要执行的动作。

2. **抽取层**（Extract）：
   - 利用 Phase 2 已有 importer 逻辑，从：
     - `factor_meta.json` / `factor_perf.json` / `feedback.json`；
     - `experiment_summary.json` / `qlib_res.csv` / `ret.pkl`；
   - 抽取并组装成 DB 行（因子/策略/实验等）。

3. **持久化层**（Persist）：
   - 写入/更新 AIstock DB 中的相关表；
   - 复制/迁移必要文件到 AIstock 管理的存储位置（如 `aistock-archives/exp_{id}/...`）；
   - 更新 `aistock_experiments.archiving_level`。

### 3.2 历史全量归档任务

- 设计一个批处理任务（可为管理命令或独立脚本）：

  - `aistock_rdagent_archive --mode full --since <date>`：
    - **输入**：
      - `RD-Agent` 仓库根路径；
      - 对应的 `registry.sqlite` 路径；
      - 归档目标存储路径（DB 连接、文件存储根目录）。
    - **行为**：
      1. 扫描所有 loops/experiments；
      2. 对 `archiving_level=0` 或空的记录，执行 Level 1 + Level 2 的归档动作；
      3. 对 `archiving_level=1` 的记录，仅执行文件级归档（补齐 Level 2 差异）；
      4. 将处理结果写入日志，并更新 DB 中的归档状态。

- 该任务需要具备：
  - **幂等性**：多次执行不会产生重复记录或冲突；
  - **断点续跑能力**：可按时间/ID 分批归档，避免一次性处理量过大。

### 3.3 增量归档任务

- 在新实验完成后（Phase 1/2 既有流程）：
  - AIstock 或 CI/CD 管道可以触发一个轻量级增量任务：
    - 只处理最新的若干 `task_run/loop`；
    - 将其从 Level 0/1 快速提升到 Level 1/2。

- 也可以使用定时任务（如每小时/每天）扫描：
  - 近 N 小时内新增的 loops；
  - 尚未达到目标 `archiving_level` 的记录。

---

## 4. Workspace/log 清理策略与工具

### 4.1 清理前置条件

- 仅允许对满足以下条件的 workspace 执行清理：
  - 对应 `aistock_experiments.archiving_level >= 2`；
  - 最近一段时间（如 30 天）内未被标记为“需要重新回溯/调试”；
  - 不在当前的“活跃调试列表”中（例如 RD-Agent 正在调试的 runs）。

### 4.2 清理粒度

- **log 目录**：
  - 在 Phase 1/2 已经不被 AIstock 消费的前提下，可在较低门槛（Level 1 即可）下进行定期清理；
  - 清理方式：按日期/大小/数量进行滚动删除，保留最近 N 天。

- **workspace**：
  - 建议分阶段清理：
    - 阶段一（保守）：只删除最重的中间产物（如中间缓存、临时训练文件），保留 JSON/配置与关键 CSV/PKL；
    - 阶段二（完全）：在 Level 2 完整归档后，可删除整个 workspace 目录，仅保留在 AIstock 存储中的归档副本。

### 4.3 清理工具

- 在 RD-Agent 仓库中新增一个脚本（示意）：`tools/cleanup_archived_workspaces.py`：
  - 输入：
    - `--min-archiving-level 2`；
    - `--dry-run`（仅打印将要删除的目录）；
    - `--confirm`（实际执行删除）。
  - 行为：
    1. 连接 AIstock DB 或导入导出的归档状态表；
    2. 过滤出符合清理条件的 workspace 路径列表；
    3. 在 dry-run 模式下打印列表，供人工审核；
    4. 在 confirm 模式下实际删除对应目录并记录日志。

---

## 5. 多策略/多因子实验设计扩展

### 5.1 StrategyConfig 与 ExperimentRequest 扩展

- 在 Phase 2 的基础上，扩展 StrategyConfig/ExperimentRequest：
  - 支持在一个 Experiment 中包含多条策略配置：
    - 不同参数版本（如不同风控参数、不同持仓上限）；
    - 不同因子组合（在同一数据集上多组合对比）；
  - 支持明确标记“对照组”（如现有 SOTA 策略）与“新试验组”（RD-Agent 新演进策略）。

- 在 registry 中：
  - 为多策略实验建立一对多关系：
    - 一个 `task_run` 下多个 `loop`，每个 loop 对应一个策略版本；
  - 在 AIstock DB 中的 `aistock_experiments` / `aistock_strategies` 表体现这种结构。

### 5.2 外部因子/策略接入

- Phase 3 要求：
  - 在因子库结构中，`source` 字段必须支持：
    - `rdagent_generated` / `alpha_library` / `external_manual` 等；
  - AIstock importer 能将：
    - 外部因子表（parquet）与外部策略配置（YAML/JSON）按照统一 schema 记录到因子库/策略库中；
  - RD-Agent 在 ExperimentRequest 中可以引用这些外部因子/策略，形成“混合实验”。

---

## 6. Phase 3 开发计划与验收标准

### 6.1 开发计划（概要）

- **RD-Agent 侧**：
  1. 与 AIstock 侧确认归档状态获取接口（或导出归档状态表的方式）；
  2. 实现 `tools/cleanup_archived_workspaces.py`，支持 dry-run 和 confirm；
  3. 对 workspace 结构进行必要的轻量调整，以便未来清理粒度更精细（例如分离中间缓存与关键 artifacts）。

- **AIstock 侧**：
  1. 完善 DB 表结构（因子库/策略库/实验表/组合表）；
  2. 在 Phase 2 importer 基础上，实现 Level 2 归档逻辑：
     - 文件复制/迁移到 AIstock 存储；
     - `archiving_level` 字段更新；
  3. 实现历史全量归档任务与定期增量归档任务；
  4. 在 UI 中增加归档状态展示与筛选（可选但推荐）。

### 6.2 验收标准

1. **归档功能验收**：
   - 任选若干历史和新产生的 experiments：
     - 在运行归档任务后，AIstock DB 中有完整的因子/策略/实验记录；
     - AIstock 存储中存在对应的归档文件；
     - `archiving_level` 正确从 0/1 提升到 2。

2. **清理功能验收**：
   - 在 dry-run 模式下运行清理脚本：
     - 输出将被删除的 workspace 列表，与预期一致；
   - 在 confirm 模式下执行一次有限范围的清理：
     - 被清理的 workspace 在文件系统中确实消失；
     - AIstock 仍能基于自身 DB/存储完整展示这些 experiments 的因子/策略信息与回测结果（不依赖 RD-Agent workspace）。

3. **多策略/多因子扩展验收**：
   - 至少有一组包含多策略/多因子组合的 Experiment 被成功归档：
     - 在因子库/策略库中可看到每个因子/策略的独立记录与效果；
     - 在实验详情中可看到不同策略/组合之间的对比信息。

---

> 本 Phase 3 详细设计建立在 Phase 2 的成果和接口约定之上，重点解决“成果资产化闭环”和“安全清理 workspace/log”的问题，同时为后续在线路径与数据服务层集成提供多策略/多因子管理能力。后续 Phase 可在此基础上进一步引入 LiveFeedback、实时数据服务与模拟盘/实盘闭环。
