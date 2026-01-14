# RD-Agent × AIstock Phase 3 详细设计补充：执行迁移与重训能力

> 本补充文档基于顶层架构与 Phase 3 原有设计，聚焦回答：
> - AIstock 侧如何在自身数据服务与执行引擎中，运行与 RD-Agent 等价的因子/策略/模型；
> - 如何基于 Phase 2 导入的成果，在 AIstock 环境中对不同因子组合使用相同模型结构重新训练；
> - RD-Agent 在 Phase 3 中继续扮演“研发工厂”角色的同时，如何为 AIstock 执行迁移提供支持。

---

## 1. Phase 3 总体目标（补充版）

在原有 Phase 3 “StrategyConfig / FactorSetConfig 与外部资产接入”的基础上，补充以下目标：

1. **执行迁移目标**：
   - 对于在 RD-Agent loop 中验证表现良好的“因子组合 + 策略 + 模型配置”，
   - AIstock 能在自身的数据服务与执行引擎中，运行**等价**的因子计算与策略执行逻辑，
   - 作为模拟盘与实盘信号的唯一来源，不再依赖 RD-Agent 进程或其运行时。

2. **重训能力目标**：
   - AIstock 能基于从 RD-Agent 导入的：
     - `FactorSetConfig`（因子集合定义）；
     - `StrategyConfig`（策略配置、仓位/风险逻辑）；
     - `model_config`（模型结构和超参数）；
   - 在本地环境中对这些配置进行**重新训练与评估**，
   - 并可作为模拟盘/实盘的在线推理模型。

3. **分工原则保持不变**：
   - RD-Agent 继续专注于：因子/策略演进与离线回测、反馈生成；
   - AIstock 专注于：在线执行、订单路由、风险控制，以及 LiveFeedback 的汇总与下发。

---

## 2. 因子执行迁移：从因子共享包到 AIstock 执行引擎

### 2.1 输入前提（来自 Phase 2）

- AIstock 已经具备：
  - 完整的因子库：
    - 每个因子具备：名称、来源、描述、表达式（Alpha158）、表现指标；
    - `impl_module` / `impl_func` / `impl_version` 等实现指针；
  - 本地安装的因子共享包（如 `rd_factors_lib`），版本号与元数据中的 `impl_version` 对齐；
  - 完整的因子表现与组合信息（`factor_perf` / `loop_catalog`）。

### 2.2 AIstock 因子执行引擎设计

- 在 AIstock 仓库中实现一套“因子执行引擎”模块，假定接口形态为：

  ```python
  class FactorEngine:
      def __init__(self, data_service):
          self.data_service = data_service  # 统一访问行情/财务/账户等数据

      def compute_factor(self, factor_name: str, universe: list[str], start: str, end: str) -> pd.DataFrame:
          """在给定时间区间和股票池上计算指定因子，返回 MultiIndex DataFrame。"""
          ...

      def compute_factor_set(self, factor_names: list[str], ...) -> pd.DataFrame:
          ...
  ```

- 因子引擎只依赖 AIstock 的 Data Service：
  - Data Service 提供标准化的行情/特征视图（如 `get_ohlcv`, `get_fundamental` 等）；
  - 因子实现以这些视图为输入，不直接访问文件系统或 RD-Agent 的中间产物。

### 2.3 参考实现与迁移策略

- 对于每个待迁移因子：

  1. 从因子元数据中读取 `impl_module` / `impl_func` / `impl_version`；
  2. 在 AIstock 研究环境中，通过 import 使用共享包中的实现函数作为**参考实现**：

     ```python
     from rd_factors_lib.generated import factor_xxx as factor_xxx_ref
     ```

  3. 在 AIstock 因子引擎中编写对应实现：

     ```python
     def factor_xxx(data_service, universe, start, end):
         # 使用 AIstock Data Service 获取数据
         df = data_service.get_feature_view(...)
         # 在本地实现逻辑，力求与 factor_xxx_ref 一致
         ...
     ```

  4. 使用标准测试集对齐：
     - RD-Agent 提供或 AIstock 构造一份统一的测试 DataFrame / 时间区间；
     - 在两种实现上分别计算因子值；
     - 检查差异是否在预设阈值内（例如绝对误差 < 1e-8，或统计显著性一致）。

- 文档中明确约束：
  - AIstock 可对实现做代码重构/优化，
  - **但必须在标准测试集上与 RD-Agent 参考实现保持数值一致，以保证因子逻辑含义不变。**

---

## 3. 策略与模型执行迁移

### 3.1 策略执行引擎迁移

- 在 AIstock 仓库实现一套策略执行引擎：

  ```python
  class StrategyEngine:
      def __init__(self, data_service, factor_engine, risk_engine):
          ...

      def run_backtest(self, strategy_config, model, factor_set, ...):
          """在 AIstock 环境中复现 RD-Agent 的回测流程，用于对账与回归测试。"""
          ...

      def run_realtime(self, strategy_config, model, factor_set, ...):
          """生成模拟盘/实盘信号。"""
          ...
  ```

- `strategy_config` 对应于 Phase 3 中定义的 `StrategyConfig` 对象/JSON：
  - 包含策略类型（如 Topk / 多空等）、仓位与风控逻辑、持仓周期、调仓频率等；
  - 由 AIstock 在前端/配置中维护，并可通过 API 下发给 RD-Agent 用于离线实验。

- 策略执行迁移目标：
  - 对于从 RD-Agent 导入的策略（`strategy_catalog` 中的某条记录），
  - AIstock 能构建一个等价的 `StrategyConfig`，并在自己的执行引擎中：
    - 复现 RD-Agent 回测路径（用于对账）；
    - 在实时行情驱动下生成交易信号（用于模拟盘/实盘）。

### 3.2 模型结构与训练迁移

- 基于 Phase 2 中导入的 `model_config` / 数据区间等元信息，AIstock 在本地实现：

  1. **模型结构构建**：
     - 将 `model_config` 映射为 AIstock 本地的模型定义（可使用 PyTorch/TF 等）；
     - 保持关键结构与超参数（层数、维度、loss、优化器等）一致。

  2. **训练与验证流程**：
     - 使用 AIstock 的 Data Service 构造训练/验证/测试集；
     - 在本地训练模型，并计算一组与 RD-Agent 接近的评估指标；

  3. **对账与验收**：
     - 在同一历史数据集上，对比：
       - RD-Agent 回测结果（从 `qlib_res.csv` / `loop_catalog.metrics`）；
       - AIstock 本地回测/模拟的结果；
     - 若偏差在可接受范围内，则认为迁移成功，可以纳入模拟盘/实盘候选模型池。

- 可选：模型产物导出/导入
  - 后续可定义一套标准模型文件格式（如 ONNX / TorchScript / 自定义 JSON+权重文件），
  - 允许 RD-Agent 导出训练好的模型权重，AIstock 直接加载并在本地执行推理；
  - 该部分可作为 Phase 3 中后段或 Phase 4 的增强内容。

---

## 4. AIstock 侧执行迁移的典型工作流

以下工作流展示了从 RD-Agent 成果到 AIstock 模拟盘/实盘执行的完整闭环，便于实现时对照：

1. **在 RD-Agent 中完成一批实验**：
   - RD-Agent 侧运行因子/策略演进任务，生成多个 `task_run` / `loop`；
   - 针对每个有效 loop，产出 Phase 2 artifacts：
     - `factor_meta.json`, `factor_perf.json`, `feedback.json`, 回测图表等；
     - 在因子元数据中写入 `impl_module` / `impl_func` / `impl_version`。

2. **AIstock 同步成果并挑选候选组合**：
   - 通过 RD-Agent 只读 API 同步：
     - 因子库、策略库、实验库、Alpha158 元信息；
   - 在 AIstock UI 中：
     - 按回测指标、反馈、因子来源等条件筛选候选因子组合与策略；
     - 人工或半自动标记“拟上线组合”。

3. **因子执行迁移与对齐测试**：
   - 对候选组合涉及的所有因子：
     - 在 AIstock 因子引擎中实现等价逻辑；
     - 使用共享包中的参考实现和标准测试集做数值对齐测试；
   - 记录迁移状态（如 `migrated`, `verified` 标志）。

4. **策略与模型执行迁移与重训**：
   - 基于 `strategy_catalog` 与 `model_config`：
     - 在 AIstock 策略引擎中构造等价 `StrategyConfig`；
     - 在 AIstock 本地数据上重训模型，并跑回测对账；
   - 对通过验收的组合，标记为“可用于模拟盘/实盘”。

5. **上线到模拟盘/实盘**：
   - 在 AIstock 的执行系统中：
     - 按选定的 `StrategyConfig` + `FactorSetConfig` + 本地训练好的模型，
     - 运行模拟盘或对接实盘交易通道（miniQMT 等），生成并执行买卖信号；
   - 将模拟盘/实盘的真实表现封装为 LiveFeedback，回传给 RD-Agent 用于后续定向演进。

---

## 5. 与原有 Phase 3 设计的关系

- 本补充设计与原有 Phase 3 “StrategyConfig / FactorSetConfig 与外部资产接入”目标相互补充：
  - StrategyConfig / FactorSetConfig 仍然是 AIstock → RD-Agent 的实验配置契约；
  - 因子共享包与只读成果 API 则是 RD-Agent → AIstock 的成果交付契约；
  - Phase 3 的新增责任在于：
    - AIstock 侧基于这两个方向的契约，在自身执行栈中完成策略/因子/模型的迁移与重训。

- 对后续 Phase 4/5：
  - 在执行迁移与重训能力稳定后，可以进一步：
    - 叠加多策略/多窗口组合试验；
    - 基于模拟盘/实盘 LiveFeedback 做定向演进；
    - 在 AIstock UI 中以工作流/多 Agent 方式自动化这些步骤。

本补充文档到此给出了 Phase 3 在“执行迁移”和“重训能力”方面的可直接落地设计，开发人员可据此实现具体模块与接口，无需额外猜测。

---

## 6. Phase 3 模块级分工与调用关系（RD-Agent × AIstock）

为方便工程实现，下面以模块和调用方向的角度，总结 Phase 3 时两个程序的职责边界：

- **RD-Agent 模块（主要在研发域）**：
  - `rdagent.core` / `rdagent.loop` 等：
    - 负责执行 ExperimentRequest / FactorSetConfig / StrategyConfig 对应的实验；
    - 产出 workspace、回测结果与 Phase 2/3 约定的 artifacts。
  - `rd_factors_lib`（因子共享包）：
    - 作为因子参考实现库，被 RD-Agent 演进流程写入；
    - 供 AIstock 在研究环境中 import，用于对齐测试。
  - `rdagent-results-api`（只读服务）：
    - 对外暴露 registry / Catalog / alpha158_meta / artifacts 只读视图；
    - 不提供任何“执行交易”或“实时信号”接口。

- **AIstock 模块（研究域 + 执行域）**：
  - 研究域：
    - `aistock.sync.rdagent_client`：
      - 调用 `rdagent-results-api` 同步因子/策略/loop/Alpha158/训练元数据；
    - `aistock.db`：
      - 存储上述成果，用于构建因子库/策略库/实验库视图；
    - `aistock.ui.research`：
      - 提供成果浏览、筛选和“拟上线组合”标记功能。
  - 执行域：
    - `aistock.data.service`：
      - 提供统一的数据访问接口（行情、财务、账户等）；
    - `aistock.factors.engine`：
      - 基于 Data Service 和从 RD-Agent 导入的因子定义，在本地实现等价因子计算；
    - `aistock.strategies.engine`：
      - 运行与 RD-Agent 等价的策略逻辑，调用因子引擎与模型；
    - `aistock.models`：
      - 根据 model_config 构建模型结构、训练与推理；
    - `aistock.execution`：
      - 模拟盘/实盘撮合与订单路由；
    - `aistock.feedback`：
      - 从执行结果构造 LiveFeedback，并通过合适渠道回传 RD-Agent。

- **调用方向约束：**
  - 研究域：
  246→    - AIstock 调用 RD-Agent（只读 API + 演进控制 API），RD-Agent 不反向调用 AIstock；
  247→  - 执行域：
  248→    - AIstock 内部各模块之间调用（data → factor → strategy → model → execution）；
  249→    - **执行域不调用 RD-Agent 的任何模块或 API**，确保实盘/模拟盘对 RD-Agent 运行时零依赖。
  250→
   251→工程实现时，按上述模块和调用方向拆分开发任务，既能保证 Phase 3 目标达成，又能为后续阶段的扩展预留清晰边界。

---

## 7. AIstock 实盘执行层实现进度 Checklist（内部跟踪用，2025-12-28 新增）

> 本节仅作为 **AIstock 内部执行层开发进度跟踪** 使用，不涉及 RD-Agent 代码改动。
> 路径命名基于当前 AIstock 仓库实际结构（`backend/...`），便于直接对上项目代码。

### 7.1 现有执行相关模块小结（基于 F:\Dev\AIstock）

- **数据服务与 xtquant 接入**（已存在骨架）：
  - `backend/data_service/api.py`
    - 已定义统一数据服务接口：`get_realtime_snapshot` / `get_history_window` / `get_intraday_window` 等；
    - 已通过 `xtquant_adapter` / `tdx_adapter` / `timescaledb_adapter` 实现行情来源选择与降级；
  - `backend/data_service/xtquant_adapter.py`
    - 已封装 `xtquant.xtdata.get_market_data` 等接口，返回标准化的 `MultiIndex(datetime, instrument)` K 线 DataFrame；
  - `backend/data_service/miniqmt_adapter.py`
    - 已定义 `Position` / `PortfolioState` / `Order` / `Trade` 四个 dataclass；
    - `load_portfolio_state_qmt` / `load_open_orders_qmt` / `load_trades_qmt` 目前为 `NotImplementedError` 占位。

- **QMT 客户端与下单封装**（已存在较完整实现）：
  - `backend/infra/qmt_client.py`
    - 管理与 miniQMT/xtquant 的连接与会话；
    - 封装账户、持仓、委托、成交查询与 `place_order` / `cancel_order` 等操作；
    - 内部直接使用 `XtQuantTrader` 与 `xtconstant`，是当前事实上的 **MiniQMT 执行网关实现**。

- **策略执行与风控封装**（已存在轻量执行器）：
  - `backend/infra/strategy_executor.py`
    - `SimpleStrategyExecutor`：
      - 通过 `qmt_client` 和 `RiskControlService` 承载交易信号执行逻辑；
      - 封装：幂等性控制（`trade_intent` 表）、风控检查、调用 QMT 下单、更新意图状态；
      - 目前以单标的信号执行为主（`execute_signal(strategy_id, symbol, side, quantity, price_type, price, ...)`）。

- **策略调度与运行入口**：
  - `backend/schedulers/strategy_scheduler.py`
    - 管理策略注册与调度循环，周期性调用各策略实例的 `run(symbol)` 方法；
  - `backend/main.py`
    - FastAPI 主入口，注册了 `qmt`、`portfolio`、`strategies` 等路由，并初始化 `strategy_scheduler` 与 `get_qmt_client_singleton`。

### 7.2 Phase 3 执行层实现任务 Checklist（建议以看板/表格跟踪）

以下 checklist 基于上述现有模块，结合 Phase 3 设计中“执行层五层架构”的目标，供 AIstock 团队按任务拆分与跟踪。

#### A. 数据服务与 MiniQMT 账户视图补齐

- [ ] **实现 miniQMT 账户视图适配器**（优先级：高）
  - 文件：`backend/data_service/miniqmt_adapter.py`
  - 目标：
    - 补全 `load_portfolio_state_qmt` / `load_open_orders_qmt` / `load_trades_qmt` 的实现；
    - 以 `backend/infra/qmt_client.py` 为依托，从 QMT 拉取账户、持仓、委托、成交信息并转换为 dataclass：
      - `PortfolioState`（含 `cash`、`equity`、`positions`、`timestamp`）；
      - `Order`（含 `status`、`created_at`）；
      - `Trade`（含 `traded_at`）。
  - 依赖：
    - `backend/infra/qmt_client.get_qmt_client_singleton()`；
    - QMT 连接与账号配置已在 `.env` 与 `backend/infra/qmt_client.py` 中就绪。

- [ ] **将 MiniQMT 账户视图接入公共数据服务 API**（优先级：中）
  - 文件：`backend/data_service/api.py`
  - 目标：
    - 在已有的 `get_portfolio_state` / `get_open_orders` / `get_trades` 基础上，确认并完善：
      - 调用 `miniqmt_adapter` 的实现是否正确；
      - 出错路径（QMT 不可用时）是否记录告警并向上抛出明确异常。

#### B. 执行网关与账户服务抽象对齐 Phase 3 设计

- [ ] **将 qmt_client 视为统一执行网关（文档与代码对齐）**（优先级：中）
  - 文件：`backend/infra/qmt_client.py`
  - 目标：
    - 在 Phase 3 文档中，将“MiniQMTExecutionGateway”的示例路径与命名，对齐到现有 `backend/infra/qmt_client.py`；
    - 如有必要，可在代码中增加轻量封装类 `QMTExecutionGateway`，但不改变现有 `place_order` / `cancel_order` 的行为。

- [ ] **抽象 AccountService（可选，视复杂度决定）**（优先级：中）
  - 推荐新增文件：`backend/services/account_service.py`
  - 目标：
    - 将“从 QMT/数据服务拉取账户视图 + 落库/缓存 + 提供统一查询接口”的逻辑集中到一个服务中；
    - 对上游执行与风控模块暴露：
      - `get_current_positions(account_id)`；
      - `get_cash_and_asset(account_id)`；
      - `get_open_orders(account_id)`。

#### C. 策略执行与风控扩展

- [ ] **基于 SimpleStrategyExecutor 完善执行与风控路径**（优先级：高）
  - 文件：`backend/infra/strategy_executor.py`
  - 目标：
    - 检查 `RiskControlService` 的实现与规则是否满足 Phase 3 初期“只做多头 A 股”的约束；
    - 在保持当前幂等与意图记录设计的前提下，
      将 `execute_signal` 扩展为可接受来自“组合层”的批量调仓请求（例如支持一次性调多个标的或带有目标仓位信息）。

- [ ] **将策略调度与执行器集成到 Phase 3 策略引擎设计**（优先级：中）
  - 文件：
    - `backend/schedulers/strategy_scheduler.py`
    - `backend/routers/strategies.py`（如存在）
  - 目标：
    - 在当前 `strategy.run(symbol)` 调用链上，引入或对齐 “因子/组合 → 信号 → SimpleStrategyExecutor.execute_signal” 的流程；
    - 为后续引入基于 `StrategyConfig` / `FactorSetConfig` 的统一策略执行入口预留钩子。

#### D. 与 Qlib 组合构建/风控评估的集成（研究与模拟盘）

- [ ] **在 AIstock 研究环境中集成 Qlib 组合构建与风险评估**（优先级：中）
  - 建议新增：`backend/research/portfolio_engine_qlib.py`（仅用于研究/模拟盘环境）；
  - 目标：
    - 使用 Qlib 的 `WeightStrategyBase` 等组合策略，在 AIstock 数据服务提供的数据上构建组合；
    - 将组合结果与 RD-Agent 回测结果对齐，验证配置与指标的一致性。

- [ ] **评估是否需要将 Qlib 组合逻辑移植到执行路径**（优先级：中-低）
  - 如决定在实盘/模拟盘中严格对齐 Qlib 组合逻辑，可按照 Phase 3 主文档第 7 章所述，将关键优化逻辑移植到 AIstock 独立的组合引擎模块中（命名可参考 `backend/execution/portfolio_engine.py`）。

#### E. 监控与 LiveFeedback 对接（与 RD-Agent 的后续闭环）

- [ ] **在执行路径中补充指标采集与日志结构化**（优先级：中）
  - 目标：
    - 在策略执行与 QMT 成交回报路径中，采集：成交价格、滑点、成交比率、风控拦截次数等；
    - 将关键事件以结构化日志或数据库记录形式落地，为后续构造 LiveFeedback 做准备。

- [ ] **设计 AIstock → RD-Agent 的 LiveFeedback 组装脚本**（优先级：中-低）
  - 目标：
    - 按顶层与 Phase 3 文档中 LiveFeedback 的 schema，将模拟盘/实盘结果封装；
    - 通过专门脚本或服务，将这些 JSON 输出到 RD-Agent 可消费的位置（文件或 API）。

> 建议 AIstock 团队在自己的仓库（例如 `docs/` 或 `backend/docs/`）中复制本 checklist，
> 并按内部规范增加负责人、预估工期与状态字段，用作实盘执行层的迭代看板.

#### F. 策略预览模式与多策略对比（Phase 3 范围）

- [ ] **设计并建表：策略预览核心表结构**（优先级：中）
  - 范围：**仅 Phase 3 新能力，不影响 Phase 2 设计与实现**。
  - 建议在 AIstock 后台数据库中新增以下几类表：
    - `preview_strategy`：记录预览实例的元信息（关联策略、初始资金、状态、创建时间等）；
    - `preview_trade`：记录虚拟成交明细（`preview_id`、交易日/时间、标的、方向、价格、数量、金额、原因等）；
    - `preview_position`：维护预览持仓视图（`preview_id`、交易日、标的、数量、持仓成本、最新价、市值等）；
    - `preview_account`：记录预览账户资金维度（`preview_id`、交易日、现金、总资产、当日/累计收益等）。

- [ ] **实现盘中预览计算与 1 分钟批量刷新机制**（优先级：中）
  - 目标：
    - 盘中 UI 展示使用数据库中最近一次计算结果，不做 tick 级推送；
    - 通过定时任务（建议每 1 分钟）统一为所有活跃预览实例刷新：
      - 最新行情价（通过 `backend/data_service/api.py` 或 xtquant/miniQMT 行情接口获取）；
      - `preview_position.last_price` / `market_value` 与 `preview_account.total_value`。
  - 要求：
    - 预览执行路径 **不调用 miniQMT 报单接口**，仅使用行情数据和策略输出信号进行“虚拟成交”计算；
    - 至少保证同时活跃预览实例数 ≥ 10，具体上限可结合资源评估配置。

- [ ] **实现收盘结算与历史净值曲线数据沉淀**（优先级：中）
  - 目标：
    - 每个交易日收盘后，对所有活跃或近期活跃的 `preview_id` 执行 EOD 结算：
      - 使用收盘价/日线行情计算当日收盘总市值与总资产；
      - 将结果写入当日 `preview_account.total_value` / `daily_pnl` / `cum_pnl` 等字段；
    - 为后续绘制净值曲线、计算最大回撤等绩效指标提供稳定数据源。

- [ ] **实现多策略预览页面与基准指数曲线展示**（优先级：中）
  - 目标：
    - 前端页面支持用户勾选一个或多个 `preview_strategy` 实例，
      在同一图表中展示其净值曲线（基于 `preview_account` 数据）；
    - 叠加至少一条指数基准曲线（推荐上证指数 `SH000001`）：
      - 后端通过数据服务获取指数日 K 线；
      - 将收盘价归一化为净值曲线，在前端与策略曲线同时展示；
    - 时间尺度与交易日历对齐，支持区间筛选与缩放。
  - 说明：
    - 该页面仅用于 **预览与对比评估**，不触发任何真实或模拟盘交易；
    - 费用与滑点可暂不纳入计算，后续如有需要可在 Phase 4+ 扩展。
