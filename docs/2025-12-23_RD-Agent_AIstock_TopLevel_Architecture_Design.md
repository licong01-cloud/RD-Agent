# RD-Agent × AIstock 顶层架构设计草案（v1）

> 说明：本方案基于《2025-12-23_RD-Agent_AIstock_Ultimate_Vision_and_Phased_Goals.md》作为前置约束，目标是给出整体架构图、分阶段实施路线与验收标准，并评估各阶段对 RD-Agent / Qlib 的侵入性与工作量，避免后续重构与无效投入。

---

## 1. 整体架构概览（目标态）

### 1.1 逻辑分层

自下而上，将整体系统划分为五层：

1. **数据源层（Data Sources）**：
   - miniQMT（实时行情、委托、成交、账户/持仓）
   - TDX / tushare（历史行情与部分衍生数据）
   - AIstock 自有 timescaledb（本地化时序数据）
   - 其他外部数据源（经济指标、行业分类、新闻情绪等，预留扩展）

2. **数据服务层（Data Service Layer, AIstock 中心）**：
   - 统一封装上述多数据源，提供标准化的数据接口：
     - 面向 Qlib/回测的标准数据集（qlib bin、Alpha158 所需字段）；
     - 面向 RD-Agent/因子研发的 pandas/HDF5/Parquet 视图（daily_pv.h5、moneyflow.h5、静态因子表等）；
     - 面向模拟盘/实盘的实时行情与交易视图（账户、持仓、订单、成交、盘口等）。
   - 屏蔽底层格式、字段、频率差异，数据消费方不感知具体数据源；
   - 实时场景默认优先使用 miniQMT，其他源用于补充与回溯。
   - **优先级说明**：数据服务层与 RD-Agent/回测主线属于同等级基础能力，应尽早建设；尤其是基于 miniQMT 的实时数据链路，是后续模拟盘与实盘自动化的前提之一，可与 Phase 2/3 的 artifacts 与配置演进并行开发。

3. **实验执行层（Experiment Engine, RD-Agent + Qlib）**：
   - RD-Agent：
     - 因子生成与修复（factor.py + result.h5）；
     - 因子组合与 SOTA 管理（combined_factors_df.parquet）；
     - 模型训练与回测（通过 Qlib YAML / StrategyConfig 驱动）；
     - 统一 artifacts 生成（write_loop_artifacts）。
   - Qlib：
     - DataHandler / DatasetH / TSDatasetH；
     - 模型训练与验证；
     - 回测与指标计算（ret.pkl / qlib_res.csv 等）。

4. **实验管理与策略库（AIstock Research Console）**：
   - 策略库：StrategyConfig 列表（含外部与内部策略，含仓位与风控逻辑）；
   - 因子库：来自 RD-Agent 的 factor_meta.json / factor_perf.json；
   - 实验库：registry.sqlite + artifacts 的可视化浏览与检索；
   - 实验编排：AIstock 根据用户输入/预设模版生成 ExperimentRequest，下发给 RD-Agent。

5. **模拟盘 / 实盘执行层（Execution & Risk Layer, 以 AIstock 为主）**：
   - 模拟盘：基于统一数据服务层 + 策略信号执行模拟撮合；
   - 实盘：调用券商/交易通道（miniQMT 等）下单与风控；
   - 风控与监控：持仓/风险限额/回撤控制/熔断与降级；
   - 将真实执行数据（成交、滑点、风控事件）结构化为 LiveFeedback 反馈给 RD-Agent。

### 1.2 核心契约对象

- **ExperimentRequest**：AIstock → RD-Agent
  - 含 StrategyConfig / FactorSetConfig / TrainTestConfig / 目标与约束。
- **ExperimentResult**：RD-Agent → AIstock
  - 含 registry.sqlite 中的 task_run / loops / workspaces + 全量 artifacts（signals、回测指标、因子元数据、反馈、图表）。
- **LiveFeedback**：AIstock → RD-Agent
  - 含模拟盘/实盘中的真实表现指标与改进目标。

这些对象的具体 schema 在后续详细设计中细化，本顶层设计只定义其角色与流向。

### 1.3 策略接口与职责边界原则

- **AIstock 侧职责**：
  - 接收用户在前端的自然语言策略描述与配置选择（如任务类型、优化目标、风险偏好、市场与时间段、模型/特征偏好等）；
  - 通过 LLM 与规则解析，将自然语言与前端勾选组合成结构化的 `StrategyConfig` / `FactorSetConfig`；
  - 结合训练/验证/回测数据集的选择，组装出统一的 `ExperimentRequest` JSON（包含 StrategyConfig / FactorSetConfig / TrainTestConfig / 目标与约束）；
  - 通过 API 或任务队列将 `ExperimentRequest` 发送给 RD-Agent。

- **RD-Agent 侧职责**：
  - 只接受结构化的 `ExperimentRequest`（JSON/对象），**不直接接收自然语言策略描述**；
  - 在入口处解析 `ExperimentRequest` 为内部对象，限制候选模型、设置 primary metric，将配置映射到 Qlib 模板的 market/segments/provider_uri 等参数；
  - 执行实验并在 workspace 下输出统一的 JSON：如 `workspace_meta.json`、`data_profile.json`、`experiment_summary.json` 等，以及规范化的 artifacts（signals、ret_schema、factor_meta、factor_perf、feedback、图表等）。

- **AIstock 对结果的消费方式**：
  - 只依赖上述 JSON 与 registry/artifacts 作为“官方出口”来做结果入库、展示与监控；
  - 不再解析 RD-Agent 内部日志或依赖 workspace 目录结构细节；
  - 任何未来的 UI/工作流/分析模块均基于这些结构化结果构建，保证 RD-Agent 内部实现可自由演进而不影响 AIstock 侧集成。

### 1.4 在线消费原则：RD-Agent 因子/策略在交易端必须使用实时数据

- **离线场景（训练/回测/研发）**：
  - RD-Agent/Qlib 在训练与历史回测中，可以继续使用 AIstock 侧导出的 `daily_pv.h5` / `moneyflow.h5` / `daily_basic.h5` / `static_factors.parquet` / `qlib_bin` 等文件型数据；
  - 这些文件由 AIstock 从自有数据库导出，是“research/backtest-ready”的标准视图。

- **在线场景（模拟盘/实盘交易）**：
  - 无论策略和因子是否由 RD-Agent 演进生成，只要在 AIstock 侧用于模拟盘或实盘交易，**一律通过数据服务层消费实时行情/账户数据，禁止再绕一圈转换成 h5 或 bin 文件再使用**；
  - 数据服务层负责将 miniQMT 等实时源标准化为内存视图（DataFrame/对象流），供在线策略引擎与因子计算函数直接调用；
  - 这样可以避免额外的文件 I/O 与链路延迟，确保真实交易与模拟盘的时效性与稳定性。

- **统一约束**：
  - RD-Agent 演进出来的因子/策略应尽量以“对 DataFrame/标准视图的纯函数形式”表达，使同一套逻辑既可被离线 wrapper 包装为文件型流程（训练/回测），也可被 AIstock 在线策略引擎包装为实时流程（直接消费数据服务层视图）。

---

## 2. 分阶段实施设计（含验收标准与改动评估）

> 各阶段均在前一阶段基础上增量演进，且每阶段产物均可在 AIstock 上实际使用（哪怕只是作为人工/半人工参考），避免一次性大改。

### Phase 1（现状整理）：单策略、多因子，打通 registry 与 AIstock

**目标（回顾 & 固化）：**

- RD-Agent 能稳定写入 registry.sqlite 与 workspace manifest：
  - `signals.*`, `ret_schema.*`, `qlib_res.csv`, `ret.pkl` 等 artifacts；
  - loops / workspaces / artifacts / artifact_files 关联关系完备。
- AIstock 能基于 registry：
  - 发现 `action='model' AND has_result=1` 的回测；
  - 拉取 `signals.*` / `ret_schema.*` / 回测指标，在自身系统内消费。

**验收标准（已基本满足）：**

- 多个 task_run / loop 被验证存在完整 artifacts（signals + ret + schema）；
- backfill 脚本幂等，可为历史 workspace 补齐 artifacts；
- AIstock 能通过 SQL/Python 脚本在 registry 上找到指定 loop 的成果，并在自身侧加载 `signals.parquet` / `ret_schema.parquet` 做基本分析。

**对 RD-Agent / Qlib 的改动评估：**

- RD-Agent：
  - 已做适度改动（loop.py hook 修复、backfill 脚本）——侵入性中等偏低；
- Qlib：
  - 无需修改，仅作为黑盒回测引擎使用；
- 后续：Phase 1 视为“稳态基线”，不建议再对这一层做结构性修改。

---

### Phase 2：统一 artifacts 生成函数 + 因子/策略成果结构化暴露

**核心思路：**

- 抽象统一的 `write_loop_artifacts(...)`：
  - 由 RD-Agent 实现，负责：
    - 写入 workspace_meta / experiment_summary / manifest；
    - 登记 `signals.*` / `ret_schema.*` / `qlib_res.csv` / `ret.pkl`；
    - 生成并登记：
      - `factor_meta.json`（因子元数据），
      - `factor_perf.json`（单因子与组合表现），
      - `feedback.json`（决策与反馈摘要），
      - 回测图表文件（`ret_curve.png`, `dd_curve.png`）。

**阶段性验收标准：**

1. **技术侧：**
   - 所有 `action='model' AND has_result=1` 的 loop，对应 workspace 下均存在：
     - factor_meta.json / factor_perf.json / feedback.json / 回测图表；
   - registry 的 artifacts / artifact_files 中可查询到这些文件的路径与元信息；
   - backfill 工具可回填 factor_meta / factor_perf 等新 artifacts（必要时）。

2. **AIstock 功能侧：**
   - 能基于 registry 构建：
     - 因子库视图（按因子名/来源/指标筛选）；
     - 策略视图（按 StrategyConfig/指标/时间段筛选）；
     - 回测曲线与指标面板（不再依赖 RD-Agent 内部 UI）。
   - 在 UI 上可对单个 loop 做：
     - 查看因子清单与说明；
     - 查看关键指标与回测曲线；
     - 以人工/半人工方式将信号用于模拟盘或实盘试验（作为研究参考）。

**改动量与侵入性评估：**

- RD-Agent：
  - 需要新增 `write_loop_artifacts` 函数，并将原先分散的写入逻辑迁移进该函数；
  - 在 loop 执行完成后增加一次统一调用；
  - 新增 JSON/图表生成逻辑；
  - 侵入性：**中等**（集中在结果写入与 backfill，控制好接口即可）。
- Qlib：
  - 仍作为黑盒，只需从其输出中提取指标与时间序列生成 JSON/图表；
  - 侵入性：**极低**。

---

### Phase 3：StrategyConfig / FactorSetConfig 与外部资产接入

**核心思路：**

- 在 AIstock 侧引入：
  - **StrategyConfig**：
    - 包含：策略类型、仓位与风险管理逻辑、参数、回测窗口、benchmark、目标指标、绑定的提示词模板 ID 等；
    - 明确不再长期依赖简单 TopkDropout，而是以具备仓位/风险管理能力的策略框架为基线。
  - **FactorSetConfig**：
    - 包含：所用因子列表（Alpha158 / RD-Agent 因子 / 外部因子）、来源标签、版本信息等。

- 支持外部因子与策略接入：
  - 外部因子以静态因子表（parquet）或 factor.py 形式被纳入 factor_meta/factor_perf 管线；
  - 外部策略以 Qlib YAML + StrategyConfig 的方式纳入策略库。

- 实验请求（ExperimentRequest）中显式携带 StrategyConfig / FactorSetConfig，由 RD-Agent 按此执行实验。

**阶段性验收标准：**

1. **配置与契约：**
   - 有一个明确、版本化的 StrategyConfig / FactorSetConfig JSON schema；
   - AIstock 能创建/编辑/存储这些配置，并通过 API 发送给 RD-Agent；
   - RD-Agent 能解析并将其映射到当前的 Qlib YAML 与因子/特征管线。

2. **外部资产接入：**
   - 至少有 1–2 组外部因子表（parquet）成功接入；
   - 至少有 1–2 条外部策略（YAML）在 RD-Agent 实验中被调用并写入 registry。

**改动量与侵入性评估：**

- RD-Agent：
  - 需要在 runner/experiment 架构中加入对 StrategyConfig / FactorSetConfig 的解释与注入；
  - 对 Qlib workspace 的 YAML 模板增加占位符（由配置填充）；
  - 侵入性：**中等偏上**（集中在配置与模板注入层）。
- Qlib：
  - 仍不直接修改，只是多了一些 YAML 模板；
  - 侵入性：**低**。

- AIstock：
  - 工作量较大：策略库/因子库/ExperimentRequest 管理界面与后端 API；
  - 但 RD-Agent 侧改动相对有边界，整体可控。

---

### Phase 4：多策略、多窗口与模拟盘反馈闭环

**核心思路：**

- 在 StrategyConfig 的基础上支持：
  - 多策略、多参数版本并行实验（同一因子集下不同策略）；
  - 多时间窗口（regime）评估。

- AIstock 将模拟盘/实盘中的 **LiveFeedback** 结构化发送给 RD-Agent：
  - 包含真实收益/回撤/滑点/风控事件等指标；
  - 指明希望改进的方向（降低回撤、降低滑点、保持收益等）。

- RD-Agent 以 LiveFeedback 为约束和目标，执行“定向演进实验”（而非完全探索式）。

**阶段性验收标准：**

1. **多策略/多窗口能力：**
   - 针对同一因子集，能配置并跑通至少 2–3 个不同策略配置 + 不同回测窗口；
   - registry 中能区分这些实验，并在 AIstock UI 中并排对比。

2. **LiveFeedback 闭环：**
   - 模拟盘/实盘模块能针对某个 strategy_instance_id 输出结构化 LiveFeedback；
   - RD-Agent 能基于该反馈启动新一轮 ExperimentRequest，并在 experiment_summary / feedback.json 中体现“本轮改进目标与结果”。

**改动量与侵入性评估：**

- RD-Agent：
  - 需要在 ExperimentRequest 解析层与 runner 层增加对多策略/多窗口组合的支持；
  - 需要在 feedback/summary 中记录“上一轮 LiveFeedback 与本轮对齐关系”；
  - 侵入性：**中等**，但集中在 orchestrator 层，可控。
- Qlib：
  - 依旧通过多 YAML / 多 run 的方式承载不同策略与窗口；
  - 侵入性：**极低**。
- AIstock：
  - 需要实现模拟盘/实盘的指标聚合与 LiveFeedback 输出，工作量较大，但与 RD-Agent 的接口清晰。

---

### Phase 5：工作流、多 agent 与 RAG 驱动的智能实验室

**核心思路：**

- 在前述数据契约与配置体系稳定后，引入：
  - 工作流引擎：将复杂实验拆分为显式步骤，可以重试、回滚与监控；
  - 多 agent：
    - 因子 agent：在约束下搜索/修正规则；
    - 策略 agent：在策略空间搜索与微调；
    - 分析 agent：基于历史 ExperimentResult + LiveFeedback 生成“下一步实验建议”；
  - RAG：
    - 从 registry + artifacts 构成的实验库中检索类似任务与成功经验，用于提示词增强与参数建议。

**阶段性验收标准：**

- 至少有一条“从策略/因子设想 → 批量实验 → 人工筛选候选 → 模拟盘 → LiveFeedback → 再实验”的完整工作流可在 AIstock UI 中被一键触发与监控；
- RD-Agent 日志与 feedback.json 能支持 RAG 检索（例如，根据错误模式、指标模式检索类似历史案例）。

**改动量与侵入性评估：**

- RD-Agent / Qlib：
  - 若前几 phase 的契约设计良好，本阶段对 RD-Agent/Qlib 的改动主要是“编排与调用方式”，核心引擎基本不需要重构；
  - 侵入性：**低–中**，取决于工作流系统与现有代码的集成方式。
- AIstock：
  - 需要较多前端/后端工作支持复杂工作流与多 agent 协作，但这是在稳定数据与接口之上的“上层增量”。

---

## 3. 总体改动风险与控制策略

1. **先定契约，后定实现**：
   - 先冻结/版本化 ExperimentRequest / ExperimentResult / LiveFeedback / StrategyConfig / FactorSetConfig / artifacts JSON schema；
   - 所有实现围绕这些契约增量演进，避免中途换协议导致大面积重构。

2. **统一出口函数，降低侵入性**：
   - `write_loop_artifacts(...)` 作为结果写入的“单一出口”，保证新增 artifacts 时只改一处；
   - backfill 工具只依赖该函数，实现逻辑可共享。

3. **尽量保持 Qlib 黑盒化**：
   - 所有策略/时间窗/数据的控制尽量通过 YAML/配置完成，不直接改 Qlib 源码；
   - 若确需修改 Qlib（如 bugfix），应通过 fork/patch 的方式最小化变更面。

4. **每阶段都有可用成果，并可向后兼容**：
   - Phase 2 完成后，AIstock 就已经可以作为“完整的研究控制台”使用；
   - Phase 3 引入外部资产与结构化配置，不影响 Phase 2 产物；
   - Phase 4/5 主要增加智能性与自动化，不破坏前期存量实验数据与接口。

本顶层设计文件作为后续各阶段详细设计文档的“框架与约束”，后续任何阶段性方案（包括具体 schema、字段与实现细节）都应在其指导下展开，以最大程度避免重构与无效开发工作。
