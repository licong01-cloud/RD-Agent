# RD-Agent × AIstock Phase 3 详细设计最终版（2025-12-29）

> 本文件为 RD-Agent × AIstock Phase 3 设计的**最终整合版**，在内容上整合并去重自：
> - `2025-12-24_Phase3_Detail_Design_RD-Agent_AIstock_Archiving_and_MultiStrategy.md`
> - `2025-12-26_Phase3_Detail_Design_RD-Agent_AIstock_v1.md`
> - `2025-12-26_Phase3_Detail_Design_Supp_Execution_Migration_and_Retrain.md`
> - `2025-12-26_Phase3_Execution_and_Strategy_Preview_Design_for_AIstock.md`
>
> 所有 Phase 3 相关需求、接口与执行侧设计均以本文件为**唯一入口文档**；旧版文件保留用于追溯，不再单独作为需求来源。若与旧文存在表述不一致，以本文件为准，再回溯更新历史文档。

---

## 1. Phase 3 范围与总体目标

### 1.1 范围

- **从“研究端独立回测”升级为“AIstock 内部可复现并持续运维的执行/选股体系”**：
  - 在 AIstock 内构建因子执行引擎与策略执行引擎；
  - 迁移并对齐 RD-Agent 回测结果与 AIstock 内部回测/模拟执行结果；
  - 在 AIstock 内部实现模型重训与策略调整流程；
  - 提供策略预览（Strategy Preview）与多策略对比视图；
  - 打通 LiveFeedback 的数据通路，为 Phase 4 闭环奠定基础。

- **成果沉淀与多策略管理**：
  - 完成 AIstock 侧的结果归档（Archiving）与历史迁移；
  - 支持多策略/多因子同时管理与比较；
  - 定义策略生命周期管理与版本演进的基础机制。

### 1.2 总体目标

1. **执行迁移与对齐**
   - 对于选定的代表性策略/因子组合，AIstock 内部执行引擎跑出的回测/模拟结果，与 RD-Agent/qlib 回测在统计指标与曲线形态上尽可能对齐；
   - 对齐结果达到预设阈值（如收益曲线相关性、主要指标偏差范围等），视为执行迁移成功。

2. **AIstock 自主可用**
   - AIstock 在没有 RD-Agent 参与的情况下，可以：
     - 基于导入的因子共享包与模型，在内部重新训练、回测与调参；
     - 对策略进行启停管理、风险控制配置与资金规模管理；
     - 提供面向业务/运营侧可直接使用的策略视图与选股服务。

3. **多策略与归档能力**
   - 支持多策略同时在线/准在线评估；
   - 定义清晰的结果归档流程与数据结构，避免历史结果散落在多个系统；
   - 为 Phase 4/5 的全链路闭环与更复杂的执行场景提供扎实基础。

---

## 2. AIstock 执行层总览与五层架构

> 参考《Execution and Strategy Preview 设计》文档内容，这里做统一整合与固化。

### 2.1 执行层责任边界

- **RD-Agent**：
  - 专注于研究与策略生成：
    - 因子设计与演进；
    - 策略模板与实验配置；
    - 回测与结果分析；
  - 通过 Phase 2 的 Catalog + artifacts + 因子共享包，将成果“资产化”并暴露给外部。

- **AIstock**：
  - 专注于执行与运营：
    - 在本地数据服务层之上执行因子与策略；
    - 进行性能评估、风险控制、资金管理；
    - 对接 miniQMT/xtquant 等下单通道；
    - 与运营侧系统（账号、持仓、风控）融合。

### 2.2 AIstock 执行五层架构

1. **数据获取层（Data Access Layer）**
   - 与数据服务层对接，提供标准化的行情与因子视图（如 `MultiIndex(datetime, instrument)` 的 DataFrame）；
   - 支持历史窗口、实时 snapshot 与 streaming；
   - 保证与 RD-Agent/qlib 因子计算时的数据定义一致。

2. **因子执行层（Factor Execution Layer）**
   - 负责在 AIstock 环境中执行因子计算：
     - 基于 `rd_factors_lib` 与 Alpha158 表达式，生成因子矩阵；
     - 保证结果与 RD-Agent 离线结果对齐（在指定范围内）。

3. **策略执行层（Strategy Execution Layer）**
   - 负责将因子与模型组合为具体策略执行：
     - 接受策略配置（与 RD-Agent 策略模板相对应）；
     - 基于因子与行情生成信号（signal）与目标持仓；
     - 将输出传递给选股服务或下单通道。

4. **订单生成与交易适配层（Order & Trading Adapter Layer）**
   - 将策略输出转换为可执行订单；
   - 与 miniQMT/xtquant 等交易系统对接；
   - 接入风控与资金管理规则。

5. **监控与反馈层（Monitoring & LiveFeedback Layer）**
   - 负责收集执行结果与实时反馈：
     - 策略表现监控、回撤报警、成交统计等；
     - 为 LiveFeedback 输出结构化反馈数据；
   - 与 Phase 4 的闭环设计衔接。

> Phase 3 的实现重点集中在第 2~3 层（因子执行 + 策略执行）和第 5 层的“准备工作”（监控与 LiveFeedback 通路打通）。

### 2.3 核心演进策略：Alpha 大脑与执行插座解耦 (2026-01-04 更新)

根据最新的生产实践讨论，Phase 3 的架构重心进行了战略性调整，确立了“RD-Agent 作为 Alpha 大脑，AIstock/VN.py 作为执行四肢”的解耦模式：

1.  **Alpha 大脑 (Research & Signal)**：
    *   **RD-Agent** 承担所有“重型”研究任务：因子演进、模型重训、策略回测、参数优化。
    *   **产出物**：标准化的策略资产包（`.py` 因子文件 + `.pkl` 模型文件 + 实验元数据）。
2.  **执行插座 (Execution Socket)**：
    *   **AIstock** 转型为轻量级的“Alpha 信号消费端”与“执行中转站”。
    *   **InferenceEngine**：不再试图在 AIstock 内部复现复杂的训练过程，而是专注于高效、安全地执行 RD-Agent 导出的资产。
    *   **自动化基础**：`ENABLE_INGESTION_SCHEDULER` 默认设为启用，确保本地数据中心（Tushare daily_basic, moneyflow_ts, adj_factor）始终维持最新的“热数据”状态，无需人工干预。
3.  **VN.py 深度集成预留**：
    *   **标准化接口**：DSL 层（Data Service Layer）接口保持与 VN.py 数据契约兼容。
    *   **信号与执行分离**：AIstock 仅生成信号（Signal），由独立的执行层（未来接入 VN.py 的 OMS/RiskManager）负责具体下单。这确保了未来集成 VN.py 时可以实现“零重构”切换。

### 2.4 Phase 3 核心实现策略：纯离线补齐与非侵入式标准化 (2026-01-04 更新)

为了确保 RD-Agent 核心 Loop 的稳定性和轻量化，Phase 3 的三大核心需求（因子、策略、模型）采取**“纯离线实现、历史直接补齐”**的执行策略：

1.  **非侵入式标准化**：不在 RD-Agent 运行期间增加任何数据写入操作，也不修改核心 Prompt。
2.  **离线脚本驱动**：所有 Phase 3 的资产标准化（Wrapper 生成、策略 Python 化、模型元数据加固）均由 `tools/backfill_registry_artifacts.py` 离线脚本完成。
3.  **全量历史兼容**：通过离线补录脚本，可以对 Phase 2 积累的所有历史实验数据进行一键补齐，确保资产在 AIstock 侧的完整性，不会造成成果丢失。
4.  **LiveFeedback 演进驱动器 (D)**：本阶段暂不执行，待后续详细讨论方案后再行实现。

---

## 3. 执行迁移：因子执行引擎

### 3.1 目标

- 在 AIstock 内部，实现一套基于**因子共享包 + 数据服务层**的因子执行引擎：
  - 使用与 RD-Agent 一致的因子定义与数据视图；
  - 支持以近实时方式计算因子（至少日频，后续可扩展到更高频）；
  - 为策略执行与策略预览提供统一的因子输入。

### 3.2 输入与输出

- **输入**：
  - 因子定义：
    - 来自 `factor_catalog.json` 与因子共享包 `rd_factors_lib`；
  - 数据视图：
    - 来自数据服务层的标准化行情与基础因子视图；
  - 调度配置：
    - 计算频率（如日/分钟）、计算窗口、标的范围等。

- **输出**：
  - 因子值矩阵：
    - 形态为 `DataFrame(MultiIndex(datetime, instrument))`，列为一个或多个因子；
    - 可写入 AIstock 本地 DB 的因子表，或缓存于内存/中间件中供策略层读取。

### 3.3 关键设计要素

1. **因子实现加载**
   - 利用 Phase 2 已定义的 `impl_module` / `impl_func` / `impl_version`：
     - 在 AIstock 的 Python 运行环境中 `import rd_factors_lib.generated`；
     - 按 `impl_func` 名取到对应函数对象；
   - 针对 Alpha158：
     - 根据 `alpha158_meta` 中的表达式与字段名，通过统一的 ExpressionEngine 调用 qlib 或自研表达式引擎执行。

2. **数据接口与缓存策略**
   - 从数据服务层一口气拉取所需窗口（如过去 250 个交易日），减少反复 IO；
   - 对相同窗口与标的范围做缓存，避免多策略重复拉取。

3. **执行对齐与验证**
   - 选取若干典型任务（如代表性因子/策略），在以下维度对齐 RD-Agent：
     - 单因子横截面分布；
     - 时间序列统计（均值/方差/IC）；
     - 与 RD-Agent 导出的 `combined_factors_df.parquet` 对比误差；
   - 误差在预设阈值内视为对齐成功（具体容忍范围可根据实践调整）。

---

## 4. 执行迁移：策略执行引擎

### 4.1 目标

- 将 RD-Agent 中策略配置（基于 qlib 的 YAML 模板和 ExperimentConfig）迁移为 AIstock 内部可执行的策略配置：
  - 在不依赖 RD-Agent 的情况下，在 AIstock 内重现主要策略；
  - 支持与因子执行引擎、数据服务层协同工作；
  - 形成 AIstock 自主的策略运行与评估框架。

### 4.2 策略配置与 ExperimentRequest

- Phase 3 中，引入结构化的 `ExperimentRequest`（详见顶层架构与 Phase 3 v1 文档）：
  - 描述一次实验/回测/模拟任务的全量配置：
    - 任务类型（回测/模拟/重训）；
    - 使用的因子与模型；
    - 数据范围与分段（train/valid/test）；
    - 风控规则与交易约束；
  - 在 RD-Agent 与 AIstock 之间，`ExperimentRequest` 主要用于说明“对应关系”与对齐范围：
    - AIstock 端可将其转换为内部的策略配置对象。

### 4.3 策略执行引擎结构

- **输入**：
  - 因子流：来自因子执行引擎的因子矩阵；
  - 策略配置：来自策略库（Phase 2 的 Strategy Catalog）与 AIstock 内部配置；
  - 行情与账号信息：通过数据服务层与账户模块获取。

- **核心模块**：
  1. **信号生成模块**
     - 根据因子值与模型输出生成目标权重或打分；
  2. **组合构建模块**
     - 将信号转换为目标持仓：
       - TopK/EqualWeight 等常见逻辑；
       - 考虑基础交易成本与约束；
  3. **订单生成模块**
     - 将持仓变化转换为买卖订单；
       - 对接交易适配层；

- **输出**：
  - 回测/模拟的收益曲线与指标；
  - 每日或每个时间粒度的持仓与成交信息；
  - 可用于策略预览与 LiveFeedback 的中间数据。

### 4.4 对齐流程

1. 从 RD-Agent 已验收的 Loop 中选取代表性策略：
   - 通过 `loop_catalog` 与 `strategy_catalog` 获取策略配置；
2. 在 AIstock 内构造对应的策略配置对象；
3. 使用相同数据区间与数据服务层接口，运行 AIstock 策略执行引擎；
4. 将输出结果与 RD-Agent 回测结果进行对比：
   - 收益曲线重合度；
   - 关键指标（年化收益、回撤、Sharpe、胜率等）偏差；
5. 若在预设阈值内，视为执行迁移成功，并记录为标准策略模板。

---

## 5. 策略预览、选股中心与自选增强

### 5.1 策略选股中心 (Multi-Strategy Selection)

为支持从海量 RD-Agent 实验中筛选实战策略，AIstock 引入“选股中心”机制：

1.  **勾选机制**：在 `Strategies Catalog` 页面，每个策略条目后增加“添加到选股中心”按钮。只有被显式添加的策略才会出现在对比视野中。
2.  **动态视图**：`/rdagent/multi-selection` 页面汇聚选中的策略卡片，支持查看其来自 RD-Agent 的历史评估指标。
3.  **一键推理**：提供“执行选股”全局按钮，并发触发各策略的 `InferenceEngine` 推理。
4.  **选股清单与交互**：
    *   展示每个策略选出的股票列表。
    *   列表支持**单选**与**全选**复选框。
    *   提供“一键加入自选股池”功能。

### 5.2 自选股票池增强 (Watchlist 2.0)

实现从选股到跟踪的闭环：

1.  **分类管理**：
    *   在加入自选时，弹出分类选择器。
    *   支持从下拉框选择已有分类。
    *   支持直接输入新分类名称进行实时创建并归类。
2.  **绩效追踪**：
    *   **加入价格 (Entry Price)**：数据库记录股票加入自选时的即时市场价格。
    *   **视图排列**：在自选股列表中，`加入价格` 字段放置在 `最新价格` 之前。
    *   **实时涨幅**：新增 `加入以来涨幅` 字段，实时计算 `(最新价格 - 加入价格) / 加入价格`，动态评估策略选股后的实际表现。

---

## 6. 结果归档（Archiving）与多策略/多因子扩展

### 6.1 归档层级

> 参考《Archiving and MultiStrategy》文档中的分级概念，这里统一整理：

- **Level 0（基础归档）**：
  - 归档核心回测/模拟结果与主要指标；
  - 保存净值曲线、基准曲线、持仓与成交的基础信息；

- **Level 1（增强归档）**：
  - 在 Level 0 基础上，增加更丰富的中间数据：
    - 因子暴露、因子贡献；
    - 行业/风格暴露与贡献；

- **Level 2（深度归档）**：
  - 在 Level 1 基础上，记录更详细的执行与风控信息：
    - 细粒度的风险事件、限仓/限价限制触发情况；
    - 更高频率（如 Tick 级）的成交轨迹（如有需要）。

### 6.2 AIstock 归档流水线

1. **输入来源**：
   - 来自 AIstock 内部执行引擎（回测/模拟/真实执行）的输出；
   - 来自 RD-Agent 的历史回测结果（通过 Phase 2 的导入与迁移工具）。

2. **归档处理流程**：
   - 标准化输出格式：
     - 将各种来源的结果统一映射到归档数据模型；
   - 写入归档库：
     - 对应 Level 0/1/2 不同粒度的数据表；
   - 生成归档标识：
     - 唯一 ID（如策略 ID + 时间戳 + 来源等）。

3. **历史全量迁移与增量迁移**：
   - 全量迁移：
     - 利用 Phase 2 中的 Loop Catalog 与 artifacts，将 RD-Agent 历史回测结果转换为 AIstock 归档；
   - 增量迁移：
     - 对于新增策略或新增回测任务，周期性执行归档同步。

### 6.3 多策略与多因子扩展

- 在归档与策略管理层面，需要支持：
  - 同一策略在不同参数配置、资金规模或风险约束下的多个版本；
  - 不同因子集合的对比与组合；

- 数据结构：
  - 在归档记录中明确：
    - 因子组合 ID 或列表；
    - 策略配置版本号；
    - 资金规模与风控参数；

- 应用场景：
  - 在前端提供“多策略对比”、“多因子组合对比”等功能；
  - 为策略组合优化（Portfolio of Strategies）提供基础数据。

---

## 7. LiveFeedback 与 Phase 4 的衔接

### 7.1 Phase 3 的 LiveFeedback 准备

- 在 Phase 3 中，LiveFeedback 主要目标是：
  - 打通数据通路与接口规范；
  - 确保 AIstock 执行结果可以以结构化形式反馈给 RD-Agent；

- 典型反馈内容包括：
  - 实际执行表现（收益、回撤、成交质量等）；
  - 与回测/模拟的偏差；
  - 风控触发情况与异常事件；

### 7.2 接口与数据形态

- 在 AIstock 内部，将 LiveFeedback 数据整理为结构化对象：
  - 对应到策略 ID、归档 ID、时间范围等；
  - 以 JSON 或表格形式存储；

- 在与 RD-Agent 的交互层面：
  - Phase 3 以“文件/数据库导出 + 人工或工具辅助导入”为主；
  - Phase 4 才会引入更自动化、低延迟的闭环反馈机制。

---

## 8. 开发任务拆分与 Phase 3 验收要点

### 8.1 AIstock 侧主要任务

1. **因子执行引擎实现与对齐**  
   - 基于 `rd_factors_lib` + 数据服务层，实现可重用因子执行模块；
   - 与 RD-Agent 的因子结果（特别是代表性因子）进行对齐测试。

2. **策略执行引擎实现与对齐**  
   - 实现策略执行管线（信号→组合→订单）；
   - 对典型策略进行回测/模拟并与 RD-Agent 结果对齐。

3. **策略预览与多策略对比**  
   - 实现 `preview_*` 表与刷新任务；
   - 完成多策略净值对比与基准曲线展示；

4. **结果归档流水线与多策略扩展**  
   - 设计并实现归档数据模型；
   - 打通从执行引擎与 RD-Agent 历史结果到归档库的迁移过程。

5. **LiveFeedback 数据通路准备**  
   - 在归档与监控层面整理出结构化反馈；
   - 输出供 RD-Agent 消费的统一格式（即便初期只是文件导出）。

### 8.2 Phase 3 技术与功能验收

- **执行迁移对齐度**：
  - 对选定策略：
    - 收益曲线差异在可接受范围内；
    - 主要指标（年化收益、回撤、Sharpe 等）偏差在预设阈值内；

- **因子执行一致性**：
  - 若以同一时间窗口与标的范围计算因子：
    - AIstock 与 RD-Agent 因子矩阵差异在设定容忍区间内；

- **策略预览可用性**：
  - 至少支持若干策略的实时/准实时预览；
  - 支持与基准指数曲线的对比；

- **归档与多策略管理**：
  - 历史结果可以在 AIstock 内部统一查看、搜索与对比；
  - 策略版本与因子组合信息在归档中可追溯；

- **LiveFeedback 准备完成**：
  - AIstock 能输出结构化的执行结果与反馈；
  - RD-Agent 至少可以以离线方式消费这些反馈，为 Phase 4 做准备。

---

## 9. Phase 2 → 数据服务层 → Phase 3 的时序关系

- **推荐开发顺序**（面向 AIstock）：
  1. **完成 Phase 2 接入与验收**：
     - Catalog 导入 + artifacts 消费 + 因子/策略/实验库视图；
  2. **按数据服务层设计文档完成生产级实现（离线/研究场景）**：
     - 至少支撑研究/离线场景的数据访问，满足 REQ-DATASVC-P2-001，
       且不得以“最小可用版本”或 PoC 为由做字段或功能精简；
  3. **在此基础上推进 Phase 3**：
     - 因子执行引擎、策略执行引擎、策略预览、归档与 LiveFeedback 通路。

- Phase 3 的所有执行功能都建立在：
  - Phase 2 成果资产化与接入已经稳定；
  - 数据服务层接口稳定可用；
 这两点之上。

---

## 10. 总结

- 本文件给出了 Phase 3 在 RD-Agent × AIstock 体系下的统一设计：
  - AIstock 执行层五层架构与责任划分；
  - 执行迁移（因子执行 + 策略执行）与对齐路径；
  - 策略预览、多策略对比、结果归档与多因子扩展；
  - LiveFeedback 的 Phase 3 准备工作与 Phase 4 衔接；
  - 与 Phase 2、数据服务层的时序与依赖关系。
- 完成 Phase 3 后，AIstock 将具备在自身体系内闭环运营策略与执行的基础能力，RD-Agent 则更多聚焦于研究与创新，两者通过标准化的资产与接口实现解耦与协同。

---

## 11. 硬性要求（REQ Checklist，按 2025-12-30 项目规范对齐）

> 本节列出与 Phase 3 执行迁移与在线复用相关的关键 REQ，完整说明见：
> `docs/2025-12-30_Project_Development_Spec_RD-Agent_AIstock.md` 与
> `docs/2025-12-30_Phase1-3_Design_Update_RD-Agent_AIstock.md`。

- **REQ-FACTOR-P3-001：因子函数标准形态 (纯离线实现)**  
  RD-Agent 演进因子需提供形如 `factor_xxx(df: pd.DataFrame) -> pd.Series | pd.DataFrame` 的统一接口。实现方案如下：
  - **Wrapper 自动生成**：由离线脚本在同步代码至共享库时，自动识别类实现并包裹为标准函数。
  - **无痛迁移**：不修改 Loop 核心逻辑与提示词，AIstock 侧通过调用 Wrapper 即可获得标准化输出。

- **REQ-STRATEGY-P3-001：策略函数标准形态 (纯离线实现)**  
  策略函数需消费 `factors`、`prices`、`portfolio` 等结构化输入。实现方案如下：
  - **模板转译**：离线脚本读取 workspace 中的策略 YAML 模板，自动生成对应的 Python 接口函数 `get_strategy_config()` 或标准策略类。
  - **逻辑解耦**：策略逻辑在离线导出时完成从“模板描述”到“Python 对象”的转换。

- **REQ-DATASVC-P3-001：仅通过数据服务层获取执行数据**  
  所有进入模拟盘/实盘执行栈的 Phase 3 策略/模型，必须通过 DataService 提供的标准接口
  获取行情与因子数据，禁止直接访问底层数据源或临时文件。

- **REQ-MODEL-P3-010：qlib runtime 集成与模型复用**  
  AIstock 必须集成固定版本 qlib runtime，能够根据 `model_conf` / `dataset_conf` / 特征列表加载
  RD-Agent 导出的模型，并在自身执行栈中运行，实现与 RD-Agent 回测结果的数值对齐。对于来自 RD-Agent 的模型 loop，AIstock 应优先消费 workspace 中的 `model_meta.json`（包含 `model_type` / `model_conf` / `dataset_conf` / `feature_schema` 等字段）作为模型重建与在线推理的权威元数据来源。

- **REQ-LOOP-P3-001：基于 loop 的一键重放能力**  
  AIstock 需支持从选定 loop 的因子/模型/策略配置出发，通过 DataService 与执行引擎在本地
  重放该实验（至少在模拟盘环境），用于执行迁移验证与策略预览.

### 11.2 Phase 3 补充硬性要求 (2026-01-04)

- **REQ-SEC-P3-005：因子代码静态审计 (Sandbox)**  
  AIstock 必须对 RD-Agent 生成的动态因子代码进行静态安全扫描，拦截 `os`, `subprocess`, `shutil` 等高危库的调用，确保执行环境安全。

- **REQ-DATASVC-P3-005：异构数据实时 Join (Market Schema Only)**  
  `get_history_window` 接口必须支持在返回 xtquant 行情数据的同时，自动按 `(datetime, instrument)` 维度关联本地 PostgreSQL (`market` schema) 中的 Tushare 基础数据（daily_basic, moneyflow_ts, adj_factor）。禁止引入冗余的 schema 或表。

- **REQ-UI-P3-020：多策略选股对比与自选联动**  
  实现 `/rdagent/multi-selection` 页面，支持策略从 Catalog 动态添加、批量推理、结果分类加入自选股池。

- **REQ-WATCHLIST-P3-010：自选股池绩效追踪**  
  自选股表结构需补齐 `entry_price` 与 `category` 字段，前端实时计算并展示“加入以来涨幅”。

- **REQ-SCHEDULER-P3-001：数据调度默认启用**  
  AIstock 后端启动时，`ENABLE_INGESTION_SCHEDULER` 默认行为应为 True，确保本地数据地基的实时性。

---

## 12. Alpha 大脑与执行插座解耦

- **Alpha 大脑**：RD-Agent 负责研究与创新，提供策略与模型。
- **执行插座**：AIstock 负责执行与监控，提供策略执行引擎与 LiveFeedback 通路。

---

## 14. Phase 3 进度与交付数据结构 (2026-01-04 更新)

### 14.1 核心功能进度
目前 RD-Agent 侧 Phase 3 核心功能已全部完成开发并通过离线补录脚本验证：

| 核心需求 | 实现状态 | 交付物 |
| :--- | :--- | :--- |
| **因子函数标准化 (REQ-FACTOR-P3-001)** | 已完成 (100% 覆盖) | `rd-factors-lib` 中的 Wrapper 函数 + `factor_catalog.json` 中的 `interface_info` |
| **策略 Python 化 (REQ-STRATEGY-P3-001)** | 已完成 (100% 覆盖) | `rd-strategies-lib` 中的配置函数 + `strategy_catalog.json` 中的 `python_implementation` |
| **模型元数据权威加固 (REQ-MODEL-P3-010)** | 已完成 (100% 覆盖) | `model_catalog.json` 中的 `model_config` 与 `dataset_config` |
| **全量历史数据补齐** | 已完成 | 全量 `registry.sqlite` 补录及 AIstock Catalog 导出 |

### 14.2 详细数据结构 (AIstock 消费指引)

#### 14.2.1 因子目录 (factor_catalog.json)
每个因子条目新增 `interface_info` 与性能关联字段：
```json
{
  "name": "feature_PriceStrength_10D",
  "source": "rdagent_generated",
  "region": "cn",
  "expression": "...",
  "formula_hint": "...",
  "tags": ["rdagent"],
  "best_performance": "Sharpe: 2.10, Ann.Ret: 15.00%",
  "best_performance_sharpe": 2.10,
  "best_performance_ann_ret": 0.15,
  "impl_module": "rd_factors_lib.generated",
  "impl_func": "feature_PriceStrength_10D",
  "interface_info": {
    "type": "class",
    "standard_wrapper": "factor_feature_PriceStrength_10D"
  }
}
```
- **字段同步 (REQ-FACTOR-SYNC)**：确保 `expression` 与 `formula_hint` 始终一致；`region` 默认为 `"cn"`；`rdagent_generated` 因子若无标签则补全 `"rdagent"`。
- **表现关联 (REQ-FACTOR-ASSOC)**：`best_performance` 为汇总字符串，直接对接 AIstock UI 列表显示，解决“无回测记录”问题。顶级字段 `best_performance_*` 供数值筛选。
- **消费方式**：AIstock 执行引擎若发现 `type == 'class'`，应调用 `standard_wrapper` 指定的函数，其签名统一为 `def func(df: pd.DataFrame) -> pd.Series`。

#### 14.2.2 策略目录 (strategy_catalog.json)
每个策略条目新增 `python_implementation` 字段：
```json
{
  "strategy_name": "strategy_f9c2cf9fc7b84dc4a973f800e6bc791a",
  "python_implementation": {
    "module": "rd_strategies_lib.generated",
    "func": "get_strategy_f9c2cf9fc7b84dc4a973f800e6bc791a_config"
  }
}
```
- **消费方式**：调用该函数即可获得该策略对应的完整 qlib 策略配置字典，无需再手动解析 Workspace 里的 YAML。

#### 14.2.3 模型目录 (model_catalog.json)
模型条目现在包含权威的配置字段：
```json
{
  "model_type": "LGBModel",
  "model_config": {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": { "learning_rate": 0.2, ... }
  },
  "dataset_config": {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": { "handler": { ... }, "segments": { ... } }
  }
}
```
- **消费方式**：AIstock 侧 `InferenceEngine` 应优先使用这两个字段来重建模型和数据集实例，确保与 RD-Agent 研究端完全一致。

### 14.4 性能优化与资源管控策略 (2026-01-05 更新)

为解决高并发与大规模数据场景下的性能瓶颈，RD-Agent 侧实现了以下优化，AIstock 侧在调用相关 API 或同步数据时应注意：

1.  **零重计算 (Zero Re-computation)**：
    - `artifacts_writer.py` 与 `backfill` 脚本已移除所有对 `combined_factors_df.parquet` 的全量加载及描述统计计算。
    - 因子元数据提取现在仅依赖 Parquet Metadata (Schema) 读取，耗时从分钟级优化至毫秒级。
2.  **轻量级归档 (Lightweight Archiving)**：
    - 移除所有图形渲染（Matplotlib/Plotly）逻辑，不再在同步过程中生成收益曲线图片。
    - 收益曲线、回撤曲线等长序列数据仅在 `factor_perf.json` 中保留结构化数值，由前端实时渲染。
3.  **IO 吞吐优化**：
    - 替换 `pandas.read_csv` 为 Python 原生 `csv` 模块处理 `qlib_res.csv`，彻底规避大文件处理时的 OOM 风险。
4.  **API 响应加速 (Results API Server)**：
    - `/ops/materialize-and-refresh` 现在支持通过参数传递多个外部元数据（如 Alpha158 + Alpha360），通过离线脚本并行化处理，显著缩短 Catalog 刷新时间。

---

## 15. RD-Agent × AIstock 资产固化与增量同步规范 (2026-01-05 联合研发规范)

本章节定义了 RD-Agent 与 AIstock 之间关于“资产固化”与“增量同步”的统一技术标准。

### 15.1 核心资产固化机制 (Solidification)

为了支持在删除原始日志与 Workspace 目录后 AIstock 仍能独立运行，系统引入“固化”流程。

1.  **资产包自闭环 (Self-Contained Bundle)**：
    每个有成果的 Loop 对应一个全局唯一标识 `asset_bundle_id` (UUID)。资产包内路径结构已优化为**扁平化结构**，确保 AIstock 侧能直接发现核心文件：
    ```text
    /RDagentDB/production_bundles/{asset_bundle_id}/
    ├── config.yaml          # 核心配置文件（因子流水线、模型参数等）
    ├── factor.py           # 因子实现源代码
    ├── {workspace_id}_model.pkl  # 模型权重（带工作区前缀以防冲突）
    └── {workspace_id}_mlruns/    # MLflow 跟踪数据
    ```
2.  **路径转换与兼容性 (Path Compatibility)**：
    引入 `_to_native_path` 转换逻辑。当 RD-Agent 运行在 WSL 而同步脚本运行在 Windows 时，自动将 `/mnt/f/...` 转换为 `F:\...`。这允许 AIstock 直接在 Windows 环境下运行全量补齐脚本。
3.  **Alpha158 全量提取 (Full Factor Extraction)**：
    同步脚本不再仅依赖结果 Parquet 的列名，而是优先解析 Workspace 中的 Qlib YAML 配置文件，提取完整的 158 个因子定义。
4.  **因子指纹去重 (Expression Fingerprinting)**：
    导出 Catalog 时，基于因子表达式的指纹进行全局去重。对于相同表达式的因子，仅保留 Sharpe 值最高的最佳版本，解决历史数据中 3000+ 因子重复的问题。
5.  **剔除回测冗余 (Data Slimming)**：
    固化过程中强制执行物理文件清洗，**禁止**同步以下数据：
    - `*.parquet` / `*.csv`：历史因子数据与标签。
    - `*.png` / `*.html`：可视化图表。
    - `*.log`：实验调试日志。

### 15.2 数据库增量状态机设计

在 `registry.sqlite` 中确立以下关键状态字段：

*   **`loops` 表**：
    - `has_result` (int): 标记该 Loop 是否有可交付成果（1: 是, 0: 否）。
    - `is_solidified` (bool): 标记核心资产是否已提取至 `production_bundles` 且元数据已入库。
    - `sync_status` (string): `pending` (待 AIstock 同步), `synced` (已同步至 AIstock)。
*   **`factor_registry` 表**：
    - 存储因子的全量结构化元数据（表达式、Sharpe、Ann.Ret 等），作为 Catalog 导出的唯一权威源。

### 15.3 增量同步与文件传输协议

1.  **增量元数据 API (`GET /catalog/incremental`)**：
    - AIstock 侧通过 `last_sync_id` 请求自上次同步以来的新成果。
    - 接口仅返回 `is_solidified = 1` 且 `sync_status = 'pending'` 的记录。
2.  **文件分发 API (`GET /artifacts/bundle/{asset_bundle_id}`)**：
    - AIstock 侧通过 asset ID 拉取清洗后的资产压缩包，用于实盘选股复原。

### 15.4 两侧协同执行步骤

1.  **[RD 侧] 任务结束**：Loop 结束后仅做极简的 `upsert_loop` 记录，最大限度屏蔽读写干扰。
2.  **[API 侧] 固化刷新 (Ops)**：AIstock 触发 API，RD-Agent 异步执行“扫描新 Workspace -> 资产固化入库 -> 标记生产就绪”。
3.  **[AIstock 侧] 增量同步**：拉取增量 Catalog，下载资产包，并将同步状态回传至 RD-Agent。
4.  **[运维侧] 日志清理**：确认 `sync_status = 'synced'` 后，可安全删除原始 Workspace。

**[开发状态]：✅ 已完成 (2026-01-05)**
- **核心逻辑**：实现在 `rdagent/utils/solidification.py` 中的 `solidify_loop_assets` 函数。
- **状态机**：通过 `is_solidified` 和 `sync_status` 字段在 `loops` 表中追踪。
- **解耦优化**：已在 `rdagent/utils/workflow/loop.py` 中移除重型 `write_loop_artifacts` 调用，运行期仅执行极简 DB 注册。
- **隔离机制**：采用 `production_bundles/{asset_bundle_id}/{workspace_id}/` 结构，确保多 Workspace 因子不冲突。
- **字段定义**：
    - `asset_bundle_id` (UUID): 资产包唯一标识。
    - `is_solidified` (int 0/1): 是否已完成资产提取。
    - `sync_status` (string): `pending` | `synced` 同步状态。
    - `updated_at_utc` (ISO-8601): 最近一次状态更新时间，作为增量同步基准。

---

## 16. 全量历史数据补录与 API 详细功能设计

本章节为实现上述规范提供具体的脚本设计与 API 接口定义，确保历史成果与未来新任务在同一套标准下运行。

### 16.1 全量 Backfill 脚本设计 (`tools/backfill_registry_artifacts.py`)

为了将历史存量数据（包含 Alpha 因子）迁移至第 15 章定义的“资产固化”规范，Backfill 脚本需具备以下核心功能：

1.  **全量扫描与补全**：
    - 遍历 `log/` 目录下所有历史任务，解析 `combined_factors_df.parquet` 和 `qlib_res.csv`。
    - 在数据库中补全缺失的 `Loop` 和 `Workspace` 记录，标记 `has_result = 1`。
2.  **历史成果固化 (Migration to Solidified)**：
    - 支持 `--mode solidify-all` 模式：扫描所有有成果但未固化的历史 Workspace。
    - 执行“资产打包”：按照 `{asset_bundle_id}/{workspace_id}/` 结构移动代码、配置与权重。
    - **Alpha 因子底稿集成**：集成自 `alpha_all_meta.json`，自动补全因子表达式。
3.  **物理与逻辑双重校验**：
    - 校验物理文件是否存在，若 Workspace 已删除则在数据库标记为失效。
    - 确保生成的 `factor_meta.json` 符合最新 Schema (v1)。

**[开发状态]：✅ 已完成 (2026-01-05)**
- **新增模式**：`python tools/backfill_registry_artifacts.py --mode solidify-all --all-task-runs`。
- **实施成果**：已成功完成 RD-Agent 侧 230 个实验循环的全量固化迁移。
- **数据库升级**：`tools/init_registry_db.py` 已支持 `loops` 表扩容及 `factor_registry` 表创建。

### 16.2 增量同步与资产传输 API 设计 (`results_api_server.py`)

API 层负责屏蔽底层复杂的目录扫描，直接面向 AIstock 提供结构化增量数据。

#### 1. 增量 Catalog 接口
- **Endpoint**: `GET /catalog/incremental`
- **Query Params**:
    - `last_sync_time` (ISO-8601): 上次同步时间戳（对比 `updated_at_utc`）。
    - `limit`: 分页限制（默认 100）。
- **功能**: 查询数据库中 `is_solidified = 1` 且 `updated_at_utc > last_sync_time` 的记录。
- **返回**: 包含增量 Loop 及其嵌套的 `factors` 列表（从 `factor_registry` 表实时聚合）。

#### 2. 资产包下载接口
- **Endpoint**: `GET /artifacts/bundle/{asset_bundle_id}`
- **功能**: 将对应资产目录实时打包为 ZIP 流并返回。
- **规范**: AIstock 获取后解压，通过 `config.yaml` 引导加载本地执行环境。

#### 3. 补录状态回传接口
- **Endpoint**: `POST /ops/sync-confirm`
- **Payload**: `{"asset_bundle_id": "xxx", "status": "synced"}`
- **功能**: AIstock 入库后确认，RD-Agent 更新 `loops.sync_status = 'synced'`。

**[开发状态]：✅ 已完成 (2026-01-05)**
- **核心接口**：
    - `GET /catalog/incremental` (已上线)
    - `GET /artifacts/bundle/{id}` (已上线，流式压缩)
    - `POST /ops/sync-confirm` (已上线)
    - `GET /alpha360/meta` (补全对称性)
- **Ops 联动**：`POST /ops/materialize-and-refresh` 已自动集成 `solidify-all` 逻辑。

### 16.3 实施路径总结

RD-Agent 侧开发已全部结束，AIstock 侧可立即开始以下对接：
1.  **获取底稿**：通过 `/alpha158/meta` 和 `/alpha360/meta` 获取基础因子库。
2.  **增量拉取**：首次请求 `/catalog/incremental` (不带时间戳) 即可获取全量已固化的历史成果。
3.  **资产本地化**：针对感兴趣的 `asset_bundle_id`，通过接口拉取代码权重包，实现 AIstock 侧的逻辑闭环。

### 16.3 实施路径总结

开发完成后，通过执行以下“一次性”操作完成历史数据的大一统：
1.  **执行 `python tools/export_qlib_alpha_meta.py`**：建立 Alpha 因子底稿。
2.  **执行 `python tools/backfill_registry_artifacts.py --mode solidify-all`**：将所有历史成果（含 Alpha 指标）搬迁至资产包并写入数据库详情表。
3.  **AIstock 侧执行首次全量同步**：通过增量接口（初始时间设为 0）拉取所有已固化的历史资产。

---

## 17. 数据预处理与验证模块设计 (2026-01-06 补充)

### 17.1 设计目标

为确保 AIstock 侧实时选股功能的数据一致性与推理准确性，本章节定义了数据预处理与验证模块的详细设计规范：

1. **数据预处理一致性**：确保 AIstock 推理时应用的数据预处理逻辑与 RD-Agent 模型训练时完全一致
2. **输入数据完整性验证**：验证因子计算所需的数据字段完整性，避免运行时错误
3. **推理结果可靠性**：通过标准化预处理流程，保证推理结果的数值稳定性

### 17.2 RD-Agent 侧：元数据增强设计

#### 17.2.1 模型预处理配置导出 (REQ-MODEL-P3-020)

**目标**：在 `model_meta.json` 中补充模型训练时的数据预处理配置

**Schema 定义**：
```json
{
  "version": "v1",
  "task_run_id": "...",
  "loop_id": 123,
  "workspace_id": "...",
  "model_type": "LGBModel",
  "model_conf": { ... },
  "dataset_conf": { ... },
  "feature_schema": [ ... ],
  "preprocess_config": {
    "normalize": "zscore",        // 归一化方式: zscore | minmax | none
    "fillna": "forward_fill",     // 缺失值填充: forward_fill | backward_fill | mean | zero
    "clip": [-3, 3],              // 异常值裁剪范围，null 表示不裁剪
    "standardize_features": true, // 是否对特征进行标准化
    "description": "模型训练时的数据预处理配置，用于在线推理时复现相同的数据变换"
  }
}
```

**实现位置**：
- 文件：`rdagent/utils/artifacts_writer.py`
- 函数：`_extract_model_metadata_from_workspace()`
- 补充逻辑：从 YAML 配置中提取 `dataset.kwargs.handler.kwargs.preprocess` 字段

**提取优先级**：
1. 优先从 `dataset_conf.kwargs.handler.kwargs.preprocess` 提取
2. 若未找到，则使用默认配置：`{"normalize": "zscore", "fillna": "forward_fill", "clip": null}`

#### 17.2.2 因子输入 Schema 导出 (REQ-FACTOR-P3-020)

**目标**：在 `factor_meta.json` 中补充因子计算所需的数据 Schema

**Schema 定义**：
```json
{
  "version": "v1",
  "task_run_id": "...",
  "loop_id": 123,
  "factors": [
    {
      "name": "feature_PriceStrength_10D",
      "source": "rdagent_generated",
      "region": "cn",
      "description_cn": "价格强度因子",
      "formula_hint": "...",
      "expression": "...",
      "tags": ["rdagent"],
      "variables": { ... },
      "freq": "1d",
      "align": "close",
      "nan_policy": "dataservice_default",
      "input_schema": {
        "required_fields": ["open", "high", "low", "close", "volume"],
        "optional_fields": ["amount", "pct_chg", "turnover_rate"],
        "lookback_days": 10,
        "index_type": "MultiIndex(datetime, instrument)",
        "description": "因子计算所需的输入数据结构定义"
      }
    }
  ]
}
```

**实现位置**：
- 文件：`rdagent/utils/artifacts_writer.py`
- 函数：`_build_factor_meta_dict()`
- 补充逻辑：根据因子类型（技术因子、基本面因子等）生成默认的输入 Schema

**默认 Schema 规则**：
- **技术因子**：`required_fields = ["open", "high", "low", "close", "volume"]`
- **基本面因子**：`required_fields = ["pe", "pb", "market_cap", "turnover_rate"]`
- **资金流因子**：`required_fields = ["net_inflow", "net_outflow", "money_flow"]`
- **lookback_days**：根据因子名称推断（如 `10D` -> 10，`20D` -> 20）

#### 17.2.3 策略配置 Schema 增强 (REQ-STRATEGY-P3-020)

**目标**：在 `strategy_meta.json` 中补充完整的策略配置 Schema

**Schema 定义**：
```json
{
  "version": "v1",
  "strategy_name": "...",
  "impl_module": "rd_strategies_lib.generated",
  "impl_func": "get_strategy_xxx_config",
  "portfolio_config": {
    "signal_config": {
      "top_k": 50,              // 选股数量
      "min_score": 0.5,         // 最低分数阈值
      "max_positions": 50,      // 最大持仓数
      "score_field": "score"    // 分数字段名
    },
    "weight_config": {
      "method": "equal_weight", // 权重分配方式: equal_weight | market_cap_weight | score_weight
      "max_single_weight": 0.05 // 单只股票最大权重
    },
    "rebalance_config": {
      "freq": "1d",             // 调仓频率
      "rebalance_threshold": 0.1 // 调仓阈值
    },
    "risk_config": {
      "max_drawdown": 0.2,      // 最大回撤限制
      "max_single_loss": 0.05   // 单只股票最大亏损
    }
  }
}
```

**实现位置**：
- 文件：`rdagent/utils/artifacts_writer.py`
- 函数：`_sync_strategy_impl_to_shared_lib()`
- 补充逻辑：从 YAML 配置中提取 `portfolio` 相关配置

### 17.3 AIstock 侧：数据预处理模块设计

#### 17.3.1 DataPreprocessor 模块架构

**文件位置**：`backend/data_service/preprocessor.py`

**核心类设计**：
```python
class DataPreprocessor:
    """数据预处理模块"""

    def apply_model_preprocess(
        self,
        df: pd.DataFrame,
        preprocess_config: dict
    ) -> pd.DataFrame:
        """应用模型训练时的预处理配置"""

    def validate_factor_input(
        self,
        df: pd.DataFrame,
        input_schema: dict
    ) -> bool:
        """验证因子输入数据完整性"""

    def apply_feature_standardization(
        self,
        df: pd.DataFrame,
        feature_schema: list
    ) -> pd.DataFrame:
        """应用特征标准化"""

    def detect_and_handle_outliers(
        self,
        df: pd.DataFrame,
        clip_range: tuple | None
    ) -> pd.DataFrame:
        """检测并处理异常值"""
```

#### 17.3.2 预处理配置应用逻辑

**归一化处理**：
```python
def _apply_normalize(
    df: pd.DataFrame,
    method: str
) -> pd.DataFrame:
    if method == "zscore":
        return (df - df.mean()) / df.std()
    elif method == "minmax":
        return (df - df.min()) / (df.max() - df.min())
    else:
        return df
```

**缺失值处理**：
```python
def _apply_fillna(
    df: pd.DataFrame,
    method: str
) -> pd.DataFrame:
    if method == "forward_fill":
        return df.fillna(method="ffill")
    elif method == "backward_fill":
        return df.fillna(method="bfill")
    elif method == "mean":
        return df.fillna(df.mean())
    elif method == "zero":
        return df.fillna(0)
    else:
        return df
```

**异常值裁剪**：
```python
def _apply_clip(
    df: pd.DataFrame,
    clip_range: tuple | None
) -> pd.DataFrame:
    if clip_range:
        return df.clip(*clip_range)
    return df
```

#### 17.3.3 数据验证逻辑

**输入字段验证**：
```python
def validate_factor_input(
    self,
    df: pd.DataFrame,
    input_schema: dict
) -> bool:
    required_fields = input_schema.get("required_fields", [])
    missing = [f for f in required_fields if f not in df.columns]

    if missing:
        raise ValueError(
            f"因子输入数据缺少必需字段: {missing}. "
            f"当前可用字段: {df.columns.tolist()}"
        )

    return True
```

**数据质量检查**：
```python
def check_data_quality(
    self,
    df: pd.DataFrame
) -> dict:
    """检查数据质量并返回报告"""
    return {
        "total_rows": len(df),
        "missing_ratio": df.isnull().sum().to_dict(),
        "outlier_count": self._detect_outliers(df),
        "data_type_consistency": self._check_dtypes(df)
    }
```

### 17.4 AIstock 推理引擎集成

#### 17.4.1 InferenceEngine 预处理流程

**文件位置**：`backend/inference_engine.py`

**集成点**：在 `run_inference()` 方法中，因子计算之后、模型推理之前

**流程设计**：
```python
def run_inference(self, strategy_id, version_tag, trade_date, ...):
    # 1. 加载资产
    assets = self._load_strategy_assets(strategy_id, version_tag)

    # 2. 数据准备
    df_history = get_history_window(...)

    # 3. 因子计算
    df_factors = compute_func(df_history)

    # 4. 【新增】数据预处理
    preprocess_config = assets.get("preprocess_config")
    if preprocess_config:
        df_factors = self.preprocessor.apply_model_preprocess(
            df_factors,
            preprocess_config
        )

    # 5. 【新增】特征列表过滤
    feature_list = assets.get("flattened_feature_list")
    if feature_list:
        df_factors = df_factors[feature_list]

    # 6. 模型推理
    pred_scores = model.predict(df_factors)

    # 7. 信号持久化
    self._persist_signals(...)
```

#### 17.4.2 数据验证集成

**集成点**：在因子计算之前

**流程设计**：
```python
def run_inference(self, strategy_id, version_tag, trade_date, ...):
    # 1. 加载资产
    assets = self._load_strategy_assets(strategy_id, version_tag)

    # 2. 【新增】验证输入数据
    input_schema = assets.get("input_schema")
    if input_schema:
        self.preprocessor.validate_factor_input(
            df_history,
            input_schema
        )

    # 3. 数据准备
    df_history = get_history_window(...)

    # 4. 因子计算
    df_factors = compute_func(df_history)

    # ... 后续流程
```

### 17.5 开发任务拆分

#### 17.5.1 RD-Agent 侧任务

1. **补充模型预处理配置导出**
   - 文件：`rdagent/utils/artifacts_writer.py`
   - 函数：`_extract_model_metadata_from_workspace()`
   - 优先级：P0

2. **补充因子输入 Schema 导出**
   - 文件：`rdagent/utils/artifacts_writer.py`
   - 函数：`_build_factor_meta_dict()`
   - 优先级：P1

3. **补充策略配置 Schema 导出**
   - 文件：`rdagent/utils/artifacts_writer.py`
   - 函数：`_sync_strategy_impl_to_shared_lib()`
   - 优先级：P2

4. **更新 backfill 脚本**
   - 文件：`tools/backfill_registry_artifacts.py`
   - 优先级：P0

#### 17.5.2 AIstock 侧任务

1. **实现 DataPreprocessor 模块**
   - 文件：`backend/data_service/preprocessor.py`
   - 优先级：P0

2. **更新 InferenceEngine 集成预处理**
   - 文件：`backend/inference_engine.py`
   - 优先级：P0

3. **更新 InferenceEngine 集成验证**
   - 文件：`backend/inference_engine.py`
   - 优先级：P1

4. **测试完整选股流程**
   - 优先级：P0

### 17.6 验收标准

#### 17.6.1 RD-Agent 侧验收

- [ ] `model_meta.json` 包含 `preprocess_config` 字段
- [ ] `factor_meta.json` 包含 `input_schema` 字段
- [ ] `strategy_meta.json` 包含 `portfolio_config` 字段
- [ ] backfill 脚本能够正确生成增强的元数据文件

#### 17.6.2 AIstock 侧验收

- [ ] DataPreprocessor 模块能够正确应用预处理配置
- [ ] InferenceEngine 能够正确调用预处理模块
- [ ] 数据验证逻辑能够正确检测缺失字段
- [ ] 完整选股流程能够成功执行并产生正确结果

#### 17.6.3 集成验收

- [ ] AIstock 推理结果与 RD-Agent 回测结果在数值上对齐
- [ ] 预处理配置缺失时能够使用默认配置
- [ ] 数据验证失败时能够给出明确的错误提示

---
