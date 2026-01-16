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

### 1.5 成果资产化与集中因子/策略库原则

- **资产范围**：
  - RD-Agent 侧的“成功成果”不仅包括策略（可执行组合/模型），还包括所有被接受的因子及其组合；
  - 这些成果必须在 AIstock 侧形成**集中资产池**：
    - 因子库（Factor Registry）：因子名称、中文描述、公式提示、来源（RD-Agent 演进 / Alpha 库 / 外部）、标签等；
    - 策略库（Strategy Registry）：策略配置、输出形态、关联因子集合、来源、回测与反馈信息等。

- **成果入库与归档**：
  - RD-Agent 通过统一的 artifacts 体系（Phase 1/2）将回测/因子成果结构化为 JSON/CSV/PKL 等标准文件（如 `factor_meta.json`、`factor_perf.json`、`feedback.json`、`qlib_res.csv` 等）；
  - AIstock 必须提供 importer/任务，将这些结构化成果：
    - 写入自有数据库（因子库/策略库/实验表）；
    - 按需复制关键文件到 AIstock 管理的文件存储或对象存储；
  - 对每个实验/策略/因子组合维护“归档状态”，标记是否已完成最小归档或完整归档。

- **历史成果补录**：
  - 需要通过脚本或工具对现有所有 RD-Agent workspace/log 进行**一次性全量扫描与补录**：
    - 先用 RD-Agent 侧 backfill 工具补齐 registry/artifacts；
    - 再用 AIstock 侧 importer 将历史中有价值的因子与策略成果导入本地数据库与文件存档；
  - 保证过去的演进成果不会因为后续的 workspace/log 清理而丢失。

- **清理策略约束**：
  - 在成果未完成“完整归档”前，RD-Agent 的 workspace 目录视为**成果唯一物理载体**，清理将导致信息不可恢复；
  - 只有当某个实验/策略/因子组合在 AIstock 侧达到预设归档等级（例如：关键指标、因子元数据、反馈、必要文件均已本地化）后，才允许通过策略性脚本清理对应 workspace/log 目录；
  - 顶层设计的后续 Phase（尤其 Phase 2/3）需围绕“从结构化 artifacts 到 AIstock 资产池”的归档闭环展开。

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

#### Phase 2 最终协议补充（2025-12-26）

> 本小节是 RD-Agent × AIstock 在 Phase 2 阶段的**最终实现协议**。RD-Agent 侧代码已经按本方案落地并通过真实任务验证，AIstock 侧可以仅依据本节内容完成后续阶段二的全部开发。

- **RD-Agent 侧保证：**
  - 对所有 `has_result=1` 的 loop×workspace：
    - workspace 下存在完整的 Phase2 artifacts：`factor_meta.json` / `factor_perf.json` / `feedback.json` / `ret_curve.png` / `dd_curve.png`；
    - 在 `registry.sqlite` 中有对应的 `artifacts` / `artifact_files` 记录；
  - 通过 backfill（log 模式）可以对历史任务补齐 Phase2，并将符合条件的 loop 的 `has_result` 从 0/NULL 更新为 1；
  - 通过三大导出脚本可以随时生成：
    - `tmp/factor_catalog.json`（因子全集：Alpha158 + RD-Agent 因子）、
    - `tmp/strategy_catalog.json`（策略配置全集）、
    - `tmp/loop_catalog.json`（所有 `has_result=1` 的 loop×workspace）。

- **AIstock 侧职责（Phase 2）：**
  - **数据获取与导入流程**：
    - 在 Windows 侧通过 WSL 命令触发 RD-Agent 在 WSL/Linux 中执行：
      - `backfill_registry_artifacts.py`（按 log 目录补齐 Phase2 与 `has_result`）；
      - 三个 `export_aistock_*.py` 脚本生成最新 Catalog JSON；
    - 从共享路径（例如 `F:\Dev\RD-Agent-main\tmp\`）读取：
      - `factor_catalog.json`
      - `strategy_catalog.json`
      - `loop_catalog.json`
    - 将上述 JSON 按约定 schema 导入自有数据库，采用 **全量覆盖 + upsert** 策略，保证幂等：
      - 因子表（如 `factor_registry`）：`(name, source)` 为主键；
      - 策略表（如 `strategy_registry`）：`strategy_id` 为主键；
      - 实验表（如 `loop_result`）：`(task_run_id, loop_id, workspace_id)` 为主键，并关联 `strategy_id`。

  - **只依赖三大 Catalog 与文件路径**：
    - 所有前端/后端功能只使用导入后的三张表及共享文件路径，不直接访问 RD-Agent 的 `registry.sqlite`、日志或 workspace 目录结构；
    - 若未来 Catalog schema 升级，将通过 `version` 字段版本化，保持向后兼容旧字段。

  - **功能目标（AIstock 侧）：**
    - 基于 `factor_catalog` + Phase2 的 `factor_meta`，构建因子库：支持按标签/来源/性能筛选并勾选因子集合；
    - 基于 `strategy_catalog` 构建策略库：展示策略配置（data/model/backtest），允许用户选择某个 `strategy_id` 生成 ExperimentRequest；
    - 基于 `loop_catalog` 构建实验库：以 loop 维度展示关键指标、收益/回撤曲线（根据 `paths` 拼接静态文件 URL）、以及 LLM 反馈文本，用于研究与人工决策。

> 达成以上约定后，Phase 2 可视为双方协议已冻结，AIstock 侧可以在不依赖 RD-Agent 内部实现细节的前提下，独立完成阶段二剩余所有开发工作。

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

### Phase 3 补充：AIstock 执行层 × miniQMT × Qlib 分工与模块结构（2025-12-28 更新）

> 本小节在 1.5「成果资产化」和 Phase 3 核心思路的基础上，明确 **AIstock 执行层** 在短中期（仅 A 股、只做多头、不考虑融资融券/期货/期权/ETF 套利）的具体分工与模块结构。
> 目的是让 AIstock 侧可以在不依赖 RD-Agent 内部实现的前提下，独立完成执行层编码与演进。

#### 3.x.1 约束前提与目标

- 交易标的：仅 A 股现货；
- 方向约束：只做多头，不涉及融券/卖空，不涉及融资融券业务；
- 衍生品：暂不考虑期货、期权、ETF 套利等场景；
- 执行通道：统一通过 miniQMT（XtQuantTrader）与券商侧交互；
- 研究/回测：继续由 RD-Agent + Qlib 承担，AIstock 仅在需要时本地复用 Qlib 的组合/风控/评估逻辑；
- 阶段目标：
  - Phase 3 期间，**模拟盘优先**，实盘在同一执行架构下自然扩展；
  - 初期模拟盘风控可以大量复用 Qlib 组合风险与评估逻辑。

#### 3.x.2 执行层逻辑分层与模块图（文字版）

从下到上将 AIstock 执行层拆成五层（与 1.1 的整体分层对齐）：

1. **执行网关层（Execution Gateway Layer，miniQMT 为主）**
   - 责任：
     - 管理与 miniQMT 的连接、会话、账号订阅；
     - 提供统一的报单/撤单接口：`place_order` / `cancel_order`；
     - 消费 XtQuant 回调：委托、成交、持仓、资金变化等。
   - 实现：
     - AIstock 内部实现一个 `MiniQMTExecutionGateway` 适配器，
       对外暴露领域友好的接口，对内调用：
       - `XtQuantTrader.order_stock(...)` / `cancel_order_stock(...)`；
       - `XtQuantTrader.subscribe(...)` / `run_forever()` 等；
     - **miniQMT 提供全部下单、撤单、账户与成交信息能力**；
     - AIstock 不直接操作柜台/交易所，只与 miniQMT 交互。

2. **账户与市场视图层（Account & Market View Layer，miniQMT + Data Service）**
   - 责任：
     - 将 miniQMT 的 XtAsset / XtPosition / XtOrder / XtTrade 结构化存入 AIstock DB；
     - 将 xtdata 的实时/历史行情经数据服务层标准化，统一对上提供：
       - `get_realtime_quotes` / `get_kline` / `get_trading_calendar` 等接口；
     - 为上层组合决策和风控提供“当前账户状态 + 市场状态”视图。
   - 实现：
     - **完全复用 miniQMT/xtdata 的数据接口**；
     - AIstock 负责：
       - 设计 DB 表结构（账户、持仓、订单、成交、资金流水）；
       - 编写 importer/同步服务，将回调/查询结果写入 DB；
       - 对上暴露统一的数据服务 API（与顶层 1.2 的 Data Service Layer 一致）。

3. **组合构建与仓位决策层（Portfolio & Positioning Layer，Qlib 逻辑可复用）**
   - 责任：
     - 将策略信号（预测得分、打分因子等）转化为**目标组合权重/目标持仓**；
     - 应用组合约束：
       - 单票权重上限、行业权重上限；
       - 换手限制、benchmark 约束、tracking error 等。
   - 与 Qlib 的关系：
     - 在研究/模拟环境中，可以直接使用 Qlib 提供的：
       - `WeightStrategyBase` 及相关策略（如 Topk/Enhanced Indexing）；
       - 风险模型（因子协方差、特异风险）与优化器；
     - 在执行路径上，推荐：
       - 将 Qlib 的组合优化算法/公式以“纯 Python 数学逻辑”的方式移植到 AIstock 服务；
       - 替换数据源为 AIstock 数据服务/xtdata，而非 Qlib provider；
       - 保证 **模拟盘与实盘共用同一套组合决策逻辑**，只是数据源不同。
   - 所属：
     - **AIstock 自研服务（组合决策服务）**，
     - 但可以大量复用 Qlib 的思路与代码片段（保持与 RD-Agent 研究侧一致）。

4. **订单生成与执行策略层（Order Generation & Execution Policy Layer，AIstock 自研）**
   - 责任：
     - 从「当前实盘持仓 + 目标持仓」推导出**调仓指令集**：
       - 需要买入/卖出的标的与数量；
     - 将调仓指令拆分为具体订单：
       - 下单时机（例如开盘/收盘/分时段）；
       - 价格类型（限价、对手方最优、本方最优、智能算法等）；
       - 是否使用 miniQMT 的 `smart_algo_order_async` 实现 TWAP/VWAP 等执行策略。
   - 实现：
     - **必须由 AIstock 自研**，因为这是策略风格与风险偏好的核心体现；
     - Qlib 可作为“在回测中模拟调仓结果”的参考，但实盘中：
       - 真实成交由 miniQMT/交易所决定；
       - 执行策略需要显式利用 XtQuant 的价格类型和智能算法能力。

5. **风控与监控层（Risk & Monitoring Layer，规则借鉴 Qlib，执行由 AIstock 自研）**
   - 责任：
     - 在线风控：
       - 单票持仓/下单金额上限；
       - 组合总仓位/行业敞口限制；
       - 最大日内/累计回撤阈值（结合实时净值/资产曲线）；
       - 黑名单/停牌/高风险证券过滤；
       - 全局 kill-switch（异常时一键止损、停策略、停止报单）。
     - 事后评估：
       - 利用 Qlib 的风险分析与成本/滑点模型，对模拟盘和实盘路径进行统一评估；
   - 与 Qlib 的关系：
     - **规则与指标体系可以完全借鉴 Qlib**（如 IR、max drawdown、tracking error 等）；
     - 在线执行逻辑（是否放行某个订单）必须在 AIstock 自己的风控服务中实现；
     - Qlib 更适合作为：
       - 模拟盘/回测的评估工具；
       - 实盘后的回放与对账工具。

#### 3.x.3 miniQMT / Qlib / AIstock 在执行层的职责边界

- **miniQMT / XtQuant：**
  - 提供：
    - 真实的券商交易通道：报单、撤单、异步回报；
    - 账户、持仓、资金与信用数据视图（本项目暂不使用信用部分）；
    - 基于交易所规则的价格类型与智能算法执行能力；
  - 不负责：
    - 量化策略逻辑、组合优化、因子计算；
    - 高层风控规则编排。

- **Qlib（在本项目中的定位）：**
  - 保持 RD-Agent 侧 **完全不改动**，只用于：
    - 研究/回测阶段的组合构建与回测；
    - 风险分析与成本/滑点模型评估；
  - 在 AIstock 侧：
    - 仅借鉴/复用其**组合优化、风险评估的数学与实现**；
    - 不直接作为实盘撮合引擎；
    - 不直接访问 miniQMT，而是通过 AIstock 数据服务消费数据。

- **AIstock 执行层：**
  - 负责：
    - 从策略信号到目标组合的决策（可使用移植后的 Qlib 逻辑）；
    - 从目标组合到具体订单的拆分与执行策略；
    - 在线风控与监控；
    - 与 miniQMT 的交易网关集成；
  - 不负责：
    - 低层撮合与柜台风控（由券商/miniQMT 负责）；
    - Qlib 内部模型训练与回测细节（由 RD-Agent 负责）。

> 按照上述分工，Phase 3 之后 AIstock 可以在不修改 RD-Agent/Qlib 源码的前提下，
> 独立完成模拟盘与实盘执行层的绝大部分开发工作，仅通过：
> - Phase 2/3 已定义的 JSON/Schema 与因子/策略/loop Catalog 同步研究成果；
> - miniQMT/xtdata 提供的行情与交易接口连接真实市场；
> - 自研的执行与风控服务承载 AIstock 策略逻辑与风险约束。

- **建议的 LiveFeedback 结构要素（示意）：**
  - 元信息：`strategy_instance_id`、`experiment_id`、`env`（preview/paper/live）、统计区间 `period`（start_date/end_date）；
  - 绩效指标汇总：`total_return`、`annual_return`、`max_drawdown`、`volatility`、`sharpe`、`win_rate`、`turnover` 等；
  - 执行与成本：`avg_slippage_bps`、`fill_ratio`、`order_reject_count`、`risk_block_count` 等；
  - 风险与事件：一组带时间戳与类型的 `events`（如 `risk_limit_hit`、`drawdown_alert`、`kill_switch`、`qmt_error` 等）；
  - 配置摘要（可选）：`strategy_config_hash`、`factor_set_hash` 等，便于 RD-Agent 将实盘/模拟盘结果对齐到具体实验版本。

- **数据来源与边界：**
  - AIstock：
    - 从预览/模拟盘/实盘执行路径中采集成交、持仓、账户、风控拦截和异常事件；
    - 基于本地 DB 中的账户/预览数据（如 `preview_account`/实盘账户表）按周期聚合上述指标；
    - 生成符合约定 schema 的 LiveFeedback JSON，并写入约定路径或通过 API 推送。
  - RD-Agent：
    - 定期扫描 LiveFeedback 输出目录或接收推送；
    - 将 `strategy_instance_id` / `experiment_id` 与内部 registry 关联；
    - 以 LiveFeedback 为约束与目标，发起“定向演进实验”（例如：降低回撤、改善风险收益比），并将该约束记录在新的 ExperimentRequest 中。

- **阶段划分建议：**
  - Phase 3：
    - 确定 LiveFeedback schema 并在文档中冻结；
    - 在 AIstock 执行路径中补齐必要的指标采集与落库（成交/持仓/风险事件等）；
    - 可先在模拟盘路径输出最小版本的 LiveFeedback JSON 作为联调验证，不要求全量覆盖。
  - Phase 4：
    - 在模拟盘与实盘路径上全面启用 LiveFeedback 生成与回传机制；
    - RD-Agent 侧实现基于 LiveFeedback 的实验触发与对齐逻辑，形成“模拟盘/实盘反馈闭环”。

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

---

## 附录：AIstock Phase 2 协议与数据结构（实现说明）

本附录是面向 AIstock 团队的 **Phase 2 实现说明**，在不深入 RD-Agent 内部代码的前提下，AIstock 可以完全据此完成后续开发与联调。

### A. RD-Agent 输出的三大 Catalog（JSON）

#### 1. `factor_catalog.json`

- 顶层结构：

  - `version: "v1"`
  - `generated_at_utc: str`（UTC ISO8601 时间）
  - `source: "rdagent_tools"`
  - `factors: list`

- 单条因子结构：

  - `name: str`
  - `expression: str`
    - Alpha158 因子：一定存在（Qlib 表达式）；
    - RD-Agent 因子：若暂缺，可由 AIstock 后续补充；
  - `source: "qlib_alpha158" | "rd_agent"`
  - `region: "cn"`
  - `tags: list[str]`
    - 示例：`["alpha158"]`, `["rdagent_generated"]`, `["risk"]` 等。

#### 2. `strategy_catalog.json`

- 顶层结构：

  - `version: str`
  - `generated_at_utc: str`
  - `source: str`
  - `strategies: list`

- 单条策略结构：

  - `strategy_id: str`
    - UUIDv5，基于 `step_name + action + template_files` 生成；
    - RD-Agent / AIstock 如需重建，必须遵守相同规则以保证一致性。
  - `scenario: null`（预留）
  - `step_name: str`（如 `"feedback"`）
  - `action: "factor" | "model"`
  - `workspace_example: {task_run_id, loop_id, workspace_id, workspace_path}`
  - `template_files: list[str]`
    - YAML 与 mlflow meta 等相对路径（相对 workspace 根目录）；
  - `data_config / dataset_config / portfolio_config / backtest_config / model_config`
    - 均为从 Qlib YAML 透传的配置子树，类型为 JSON 对象。

#### 3. `loop_catalog.json`

- 顶层结构：

  - `version: str`
  - `generated_at_utc: str`
  - `source: str`
  - `loops: list`

- 单条记录（一个 `(task_run_id, loop_id, workspace_id)`）：

  - 标识字段：
    - `task_run_id: str`
    - `loop_id: int`
    - `workspace_id: str`
    - `step_name: str`
    - `action: str`
    - `status: str`
    - `has_result: bool`（RD-Agent + backfill 维护，AIstock 只需信任该值）
  - 关联：
    - `strategy_id: str | null`（与 Strategy Catalog 中的 `strategy_id` 对齐）
  - 因子与指标：
    - `factor_names: list[str]`（来自 `factor_perf.combinations[*].factor_names`）
    - `metrics: dict`
      - 主 window（如 `main_window`）的关键指标：
        - `annualized_return`
        - `max_drawdown`
        - `ic_mean`
        - `rank_ic_mean`
        - `multi_score`
        - 其他补充指标按原样透传。
  - 决策与反馈：
    - `decision: bool | null`（来自 `feedback.json.decision`）
    - `summary_texts: {execution, value_feedback, shape_feedback}`
      - 三段中文文本摘要，分别描述执行情况、假设/价值评价、曲线形态与风险。
  - 关键文件路径：
    - `paths: {factor_meta, factor_perf, feedback, ret_curve, dd_curve}`
      - 值为文件名（如 `"factor_meta.json"`），不含绝对路径。

### B. RD-Agent 导出命令（固定协议）

在 WSL 中，RD-Agent 提供统一导出方式（路径可按实际安装位置调整）：

```bash
cd /mnt/f/Dev/RD-Agent-main

# 1) Alpha158 元信息（只需偶尔刷新）
python tools/export_alpha158_meta.py \
  --conf-yaml rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_dynamic.yaml \
  --output tmp/alpha158_meta.json

# 2) Factor Catalog
python tools/export_aistock_factor_catalog.py \
  --registry-sqlite RDagentDB/registry.sqlite \
  --alpha-meta tmp/alpha158_meta.json \
  --output tmp/factor_catalog.json

# 3) Strategy Catalog
python tools/export_aistock_strategy_catalog.py \
  --registry-sqlite RDagentDB/registry.sqlite \
  --output tmp/strategy_catalog.json

# 4) Loop Catalog（全量，has_result=1 的 loop×workspace）
python tools/export_aistock_loop_catalog.py \
  --registry-sqlite RDagentDB/registry.sqlite \
  --output tmp/loop_catalog.json \
  --limit 0
```

- 每次执行会生成 **全量快照**，覆盖写 `tmp/*.json`。
- AIstock 侧导入时，应采用“全量读取 + upsert”策略，而不是做增量 diff。

> 若 AIstock 运行在 Windows，可通过 `wsl -d <发行版> -- bash -lc "..."` 调用上述命令，并从 `F:\\Dev\\RD-Agent-main\\tmp\\` 读取生成的 JSON 文件。

### C. AIstock 侧建议的数据库结构设计

#### 1. `factor_registry` 表

- 主键：`(name, source)`
- 字段建议：
  - `expression: text`
  - `region: text`
  - `tags: jsonb`
  - `description_cn: text`（可从 Phase2 的 `factor_meta.json` 中补充）
  - `formula_hint: text`
  - `variables: jsonb`

#### 2. `strategy_registry` 表

- 主键：`strategy_id`
- 字段建议：
  - `step_name: text`
  - `action: text`
  - `template_files: jsonb`
  - `data_config: jsonb`
  - `dataset_config: jsonb`
  - `portfolio_config: jsonb`
  - `backtest_config: jsonb`
  - `model_config: jsonb`

#### 3. `loop_result` 表

- 主键：`(task_run_id, loop_id, workspace_id)`
- 外键：`strategy_id → strategy_registry(strategy_id)`
- 字段建议：
  - `status: text`
  - `has_result: boolean`
  - `step_name: text`
  - `action: text`
  - `metrics: jsonb`
  - `decision: boolean`
  - `summary_execution: text`
  - `summary_value_feedback: text`
  - `summary_shape_feedback: text`
  - `paths: jsonb`

### D. 分工与协议重申

- **RD-Agent 保证：**
  - 通过在线写入 + backfill，为所有“有结果”的 loop 生成完整的 Phase2 artifacts，并在 registry 中登记；
  - 通过上述脚本，随时导出三大 Catalog 的最新全量视图；
  - 不要求 AIstock 直接操作 `registry.sqlite` 或解析工作空间目录结构。

- **AIstock 需完成：**
  - 定时或按需触发 RD-Agent 在 WSL 中执行 backfill + 三大 Catalog 导出；
  - 从共享目录（例如 `F:\\Dev\\RD-Agent-main\\tmp\\`）读取三大 JSON；
  - 将 JSON 内容导入本地数据库对应表，使用 upsert 保证幂等；
  - 所有 UI 和后续逻辑只依赖这三张表，不再直接解析 RD-Agent 的 `registry.sqlite` 或 workspace 目录。

本附录配合正文 Phase 2 小节，可视为 AIstock 阶段二研发的“接口文档”，在协议不变的前提下，AIstock 侧可以独立完成后续开发与演进.
