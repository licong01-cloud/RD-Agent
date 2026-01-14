# RD-Agent × AIstock Phase 3 详细设计（v1，2025-12-26）

> 本文件在顶层架构与 Phase 2 设计的基础上，给出 Phase 3 的**可直接执行的详细设计**，
> 重点聚焦：
> - 在 AIstock 侧实现与 RD-Agent 等价的因子/策略/模型执行能力；
> - 基于 RD-Agent 成果，在 AIstock 环境中对不同因子组合使用相同模型结构重新训练；
> - 继续保持 RD-Agent 作为“研发工厂”，AIstock 作为“执行与交易平台”的分工。
>
> 相关前置文档：
> - 顶层：`2025-12-23_RD-Agent_AIstock_TopLevel_Architecture_Design.md`
> - 顶层补充：`2025-12-26_RD-Agent_AIstock_TopLevel_Supp_FactorPackage_and_API.md`
> - Phase 2：`2025-12-23_Phase2_Detail_Design_RD-Agent_AIstock.md`
> - Phase 2 补充：`2025-12-26_Phase2_Detail_Design_Supp_FactorPackage_and_API.md`
> - Phase 3 补充：`2025-12-26_Phase3_Detail_Design_Supp_Execution_Migration_and_Retrain.md`

---

## 1. Phase 3 目标与范围

### 1.1 目标（与顶层设计对齐）

1. **执行迁移**：
   - 对于在 RD-Agent 中验证表现良好的“因子组合 + 策略 + 模型配置”，
   - AIstock 能在自身的数据服务与执行引擎中，运行等价逻辑并生成模拟盘/实盘信号。

2. **重训能力**：
   - AIstock 能基于从 RD-Agent 导入的 `FactorSetConfig` / `StrategyConfig` / `model_config`，
   - 在本地对不同因子组合使用相同模型结构进行重新训练与评估。

3. **研发与执行解耦**：
   - RD-Agent 继续只负责离线演进、回测与成果输出；
   - AIstock 负责所有在线执行、订单路由与风险控制。

### 1.2 与 Phase 2 的衔接

- Phase 2 已经提供：
  - 统一 artifacts 与 registry；
  - 三大 Catalog（因子/策略/loop）；
  - 因子共享包与只读成果 API；
  - AIstock 侧的成果接入与可视化能力。

- Phase 3 在此基础上增加：
  - AIstock 因子执行引擎；
  - AIstock 策略执行引擎；
  - AIstock 本地模型训练与推理能力；
  - 与 RD-Agent 回测结果的一致性验证流程；
  - 将等价实现纳入模拟盘/实盘的标准化工作流。

---

## 2. 因子执行迁移设计

> 详见：`2025-12-26_Phase3_Detail_Design_Supp_Execution_Migration_and_Retrain.md` 第 2 章，本节为实现概览。

### 2.1 AIstock 因子执行引擎接口

- 在 AIstock 仓库中实现模块（示例）：`aistock/factors/engine.py`：

  ```python
  class FactorEngine:
      def __init__(self, data_service):
          self.data_service = data_service

      def compute_factor(self, factor_name: str, universe: list[str], start: str, end: str) -> pd.DataFrame:
          """在给定时间区间和股票池上计算单个因子。"""
          ...

      def compute_factor_set(self, factor_names: list[str], universe: list[str], start: str, end: str) -> pd.DataFrame:
          """批量计算因子集合，返回 MultiIndex DataFrame。"""
          ...
  ```

- `data_service` 提供统一的数据访问接口（行情、财务、账户等），因子实现只依赖该接口。

### 2.2 利用因子共享包进行参考实现与对齐

- 从 Phase 2 导入的因子元数据中读取：
  - `impl_module`, `impl_func`, `impl_version`；
- 在 AIstock 研究环境中：
  - 通过 import 方式加载参考实现（来自 `rd_factors_lib`）；
  - 在 AIstock 因子引擎中编写对应实现，使其基于 AIstock Data Service；
  - 使用标准测试集对比两者输出，确保数值在可接受误差内一致；
- 对通过对齐测试的因子，在因子库中标记状态（如 `migrated=true, verified=true`）。

---

## 3. 策略与模型执行迁移设计

> 详见：`2025-12-26_Phase3_Detail_Design_Supp_Execution_Migration_and_Retrain.md` 第 3 章，本节为实现概览。

### 3.1 策略执行引擎

- 在 AIstock 仓库实现策略引擎模块（示例）：`aistock/strategies/engine.py`：

  ```python
  class StrategyEngine:
      def __init__(self, data_service, factor_engine, risk_engine):
          ...

      def run_backtest(self, strategy_config, model, factor_set, ...):
          ...

      def run_realtime(self, strategy_config, model, factor_set, ...):
          ...
  ```

- `strategy_config` 基于顶层与 Phase 3 中定义的 `StrategyConfig`：
  - 包含策略类型、仓位/风控逻辑、调仓频率、持仓周期等；
  - 兼容 RD-Agent 在 ExperimentRequest 中使用的配置结构。

### 3.2 模型训练与对账

- 基于从 RD-Agent 导入的：
  - `model_config`（模型类型与超参）；
  - 训练/验证/测试时间区间；
  - 因子集合定义；
- AIstock 在本地：
  - 使用自有 Data Service 构建数据集；
  - 训练模型并计算评估指标；
  - 在同一数据集上，与 RD-Agent 回测结果（来自 `qlib_res.csv` / `loop_catalog.metrics`）对比；
  - 对偏差在允许范围内的组合，标记为“可上线候选”。

---

## 4. 整体工作流与职责划分

> 完整流程图与文字说明参见 Phase 3 补充文档第 4 章，这里给出关键步骤索引：

1. RD-Agent 完成一批实验并通过 Phase 2 产物暴露结果；
2. AIstock 通过只读 API 同步并在 UI 中选择候选因子/策略/模型组合；
3. AIstock 基于因子共享包和元数据，在本地实现并对齐因子逻辑；
4. AIstock 构建等价策略配置和模型结构，并在本地训练与回测；
5. 对通过验收的组合，上线到模拟盘/实盘执行；
6. 将模拟盘/实盘表现封装为 LiveFeedback，回传 RD-Agent。

---

## 5. 与后续阶段的关系

- Phase 4：在本设计之上增加多策略/多窗口能力与 LiveFeedback 闭环；
- Phase 5：在稳定的数据与执行契约基础上，引入工作流、多 Agent 与 RAG 驱动的“智能实验室”。

本文件与 `2025-12-26_Phase3_Detail_Design_Supp_Execution_Migration_and_Retrain.md` 共同构成 Phase 3 的完整详细设计，开发实现时应两者结合阅读。

---

## 6. RD-Agent 侧 ExperimentRequest schema 与内部工具（2025-12-27 更新）

为支撑 Phase 3 中“AIstock 1>> RD-Agent”的结构化配置契约，本仓库已在 RD-Agent 侧预先落地了一套最小可用的 ExperimentRequest schema 及配套工具，用于本地实验与后续对接。

### 6.1 ExperimentRequest / StrategyConfig / FactorSetConfig 内部 schema

- 模块位置：`rdagent/core/experiment_request_schema.py`
- 已定义的主要 dataclass（仅在 RD-Agent 内部使用，尚未对外冻结为正式 JSON 协议）：
  - **`FactorRef`**：
    - 字段：`name`, `source`, `version`
  - **`FactorSetConfig`**：
    - 字段：`factors: list[FactorRef]`, `tags: list[str]`, `extra: dict[str, Any]`
  - **`StrategyConfig`**：
    - 字段：
      - `strategy_type`（如 `topk`, `long_short` 等）
      - `name`, `description`
      - `risk_model: dict`（仓位/风控相关参数）
      - `trading_rule: dict`（调仓频率、持仓周期、TopK 参数等）
      - `benchmark`
      - `prompt_template_id`
      - `extra: dict`
  - **`TrainTestConfig`**：
    - 字段：`train_start/train_end`, `valid_start/valid_end`, `test_start/test_end`, `dataset_extra`
  - **`ObjectiveConfig`**：
    - 字段：`primary_metric`, `direction`, `constraints`
  - **`ExperimentRequest`**：
    - 字段：
      - `kind`（`research` / `benchmark` / `ab_test`）
      - `strategy`
      - `factor_set`
      - `train_test`
      - `objective`
      - `name`, `description`, `user_tags`, `extra`

> 说明：当前这些 schema 仅在 RD-Agent 内部使用，用于统一组织配置与便于测试，不改变现有 Phase 2 入口行为。Phase 3 对外协议冻结时，将基于这些结构定义正式 JSON Schema。

### 6.2 从 ExperimentRequest JSON 生成 Qlib YAML 的内部工具

- 工具脚本：`tools/run_experiment_from_request.py`
- 功能：
  - 从本地 JSON 文件读取 ExperimentRequest；
  - 解析为上述 dataclass 结构；
  - 生成一份最小可用的 Qlib 配置 YAML，用于挂接到现有 RD-Agent/Qlib 流程。
- 使用示例（WSL）：

  ```bash
  cd /mnt/f/Dev/RD-Agent-main

  # 1）准备一个实验请求 JSON
  cat > experiment_request_demo.json << 'EOF'
  {
    "kind": "research",
    "name": "demo_experiment_from_request",
    "description": "Minimal demo ExperimentRequest to generate a Qlib YAML.",
    "strategy": {
      "strategy_type": "topk",
      "benchmark": "SH000300",
      "trading_rule": {
        "topk": 50,
        "n_drop": 10
      }
    },
    "factor_set": {
      "factors": [
        { "name": "F1", "source": "rdagent_generated" },
        { "name": "RESI5", "source": "qlib_alpha158" }
      ],
      "tags": ["demo"]
    },
    "train_test": {
      "train_start": "2015-01-01",
      "train_end": "2018-12-31",
      "valid_start": "2019-01-01",
      "valid_end": "2019-12-31",
      "test_start": "2020-01-01",
      "test_end": "2020-12-31",
      "dataset_extra": {
        "market": "csi300",
        "provider_uri": "your_qlib_provider_uri_here"
      }
    },
    "objective": {
      "primary_metric": "ic",
      "direction": "maximize",
      "constraints": {}
    },
    "user_tags": ["demo", "from_request"]
  }
  EOF

  # 2）生成规范化请求与 Qlib YAML
  python tools/run_experiment_from_request.py \
    --input experiment_request_demo.json \
    --output-dir tmp/exp_from_request_demo

  # 3）查看输出
  ls tmp/exp_from_request_demo
  # 预期：
  #   experiment_qlib_config.yaml
  #   experiment_request.normalized.json
  ```

- 实际生成的 YAML 示例（关键字段）：

  ```yaml
  data_handler_config:
    market: csi300
    provider_uri: your_qlib_provider_uri_here
    train:
      start_time: '2015-01-01'
      end_time: '2018-12-31'
    valid:
      start_time: '2019-01-01'
      end_time: '2019-12-31'
    test:
      start_time: '2020-01-01'
      end_time: '2020-12-31'
  port_analysis_config:
    strategy:
      topk: 50
      n_drop: 10
    backtest:
      benchmark: SH000300
  task:
    model: {}
    dataset:
      market: csi300
      provider_uri: your_qlib_provider_uri_here
  ```

- 当前验证结论：
  - ExperimentRequest JSON → dataclass 解析已通过上述 demo 用例验证；
  - YAML 生成逻辑可将时间区间与 market/provider 配置映射到 `data_handler_config` / `task.dataset`；
  - 后续可以在不破坏现有入口的前提下，将该 YAML 串接到指定 Qlib 场景或新的 runner 中，用于 Phase 3 的进一步开发与联调。


---

## 7. AIstock 执行层详细设计与开发指引（2025-12-28 新增）

> 本节面向 **AIstock 执行研发团队**，在顶层架构与前述 Phase 3 目标的基础上，
> 给出一套可以直接落地编码的执行层设计方案，明确：
> - 哪些模块直接使用 miniQMT / XtQuant；
> - 哪些模块可以复用或移植 Qlib 的组合/风险/评估逻辑；
> - 哪些模块需要 AIstock 完全自研。

> 约束前提：短中期仅考虑 **A 股现货、只做多头、不涉及融资融券/期货/期权/ETF 套利**。

### 7.1 执行层逻辑分层与总体结构

AIstock 执行层建议拆成五个逻辑层，每层都有清晰的外部依赖和实现方：

1. **执行网关层（Execution Gateway Layer，依赖 miniQMT）**
2. **账户与市场视图层（Account & Market View Layer，依赖 miniQMT + xtdata + AIstock Data Service）**
3. **组合构建与仓位决策层（Portfolio & Positioning Layer，可复用 Qlib 逻辑）**
4. **订单生成与执行策略层（Order Generation & Execution Policy Layer，AIstock 自研）**
5. **风控与监控层（Risk & Monitoring Layer，规则借鉴 Qlib，执行由 AIstock 自研）**

下面按层给出接口与实现建议，AIstock 可按模块拆分成独立 Python 包或微服务实现。

### 7.2 执行网关层：MiniQMTExecutionGateway

- **目标**：对上屏蔽 miniQMT/XtQuant 细节，对下完全依赖其提供的交易能力。
- **推荐模块位置（示例）**：`aistock/execution/gateway_miniqmt.py`

```python
class MiniQMTExecutionGateway:
    """AIstock 与 miniQMT 之间的统一执行网关。

    仅负责连接、报单、撤单与基础回调转发，不承担策略逻辑和风控.
    """

    def __init__(self, userdata_mini_path: str, session_id: int, logger):
        ...  # 内部创建 XtQuantTrader, 注册回调

    def connect(self) -> None:
        """启动交易线程并连接 miniQMT，异常时抛出自定义错误."""
        ...

    def subscribe_account(self, account_id: str) -> None:
        """订阅指定资金账号的交易回报."""
        ...

    def place_order(self, order_req: "OrderRequest") -> str:
        """统一下单接口，返回 AIstock 内部的 order_id.

        - 内部映射到 XtQuantTrader.order_stock(...)
        - 对价格类型、买卖方向做从领域模型到 xtconstant 的转换
        """
        ...

    def cancel_order(self, account_id: str, order_id: str) -> None:
        """撤单接口，内部调用 cancel_order_stock."""
        ...
```

- **必须直接使用的 XtQuant 能力**：
  - `XtQuantTrader` 及其回调类；
  - `order_stock` / `cancel_order_stock`；
  - `subscribe` / `run_forever`；
  - 交易市场与价格类型常量（`xtconstant`）。
- **网关不做的事情**：
  - 不做组合决策、不做风控规则判断；
  - 不直接访问 Qlib，只与 miniQMT/xtdata 通信。

### 7.3 账户与市场视图层：AccountService / MarketDataService

- **目标**：
  - 为上层提供统一的“账户状态 + 市场行情”视图；
  - 将 miniQMT / xtdata 的结构映射到 AIstock 自己的 DB 与领域模型中。

- **推荐模块位置（示例）**：
  - `aistock/execution/account_service.py`
  - `aistock/data/market_data_service.py`

- **AccountService 关键职责**：
  - 从 MiniQMTExecutionGateway 的回调中消费：
    - `XtAsset`（资金）、`XtPosition`（持仓）、`XtOrder`（委托）、`XtTrade`（成交）；
  - 按顶层文档建议，落库到类似：`account_asset` / `account_position` / `trade_order` / `trade_fill` 表；
  - 对外提供：
    - `get_current_positions(account_id)`：返回当前持仓列表；
    - `get_cash_and_asset(account_id)`：返回可用资金、总资产等；
    - `get_open_orders(account_id)`：返回当日未完全成交的订单。

- **MarketDataService 关键职责**：
  - 包装 xtdata，提供：
    - 历史 K 线：`get_history_kline(codes, period, start, end)`；
    - 实时行情订阅：`subscribe_quote(codes, callback)`；
    - 交易日历/交易时段：`get_trading_calendar`, `get_trading_period`。
  - 与顶层数据服务层设计一致，作为 Qlib/研究/执行的统一行情入口。

> 这一层完全依赖 miniQMT/xtdata 的接口能力，AIstock 只负责 **映射 + 落库 + 再暴露 API**。

### 7.4 组合构建与仓位决策层：PortfolioEngine（可复用 Qlib 逻辑）

- **目标**：
  - 从策略信号 + 因子 + 风险模型出发，计算 **目标权重/目标持仓**；
  - 这部分逻辑在模拟盘和实盘应尽量完全一致，只是输入数据源不同。

- **建议接口（示例）**：`aistock/execution/portfolio_engine.py`

```python
class PortfolioEngine:
    """组合构建与仓位优化引擎.

    可以在实现时直接移植/复用 Qlib 的组合优化逻辑，
    但数据源来自 AIstock Data Service，而非 Qlib provider.
    """

    def __init__(self, data_service, risk_model_repo, logger):
        ...

    def compute_target_portfolio(
        self,
        strategy_config: dict,
        signal_df: "pd.Series | pd.DataFrame",  # index: 股票代码
        current_positions: list["Position"],
    ) -> dict[str, float]:
        """根据策略配置和信号，输出目标权重字典 {stock_code: weight}."""
        ...
```

- **与 Qlib 的集成方式**：
  - 在研究/验证阶段：
    - 直接使用 Qlib 的 `WeightStrategyBase` / Enhanced Indexing 策略；
    - 利用 Qlib 的风险模型与优化器评估组合效果；
  - 在线上执行阶段：
    - 建议将上述策略逻辑“摘抄/移植”到 AIstock 的 `PortfolioEngine` 中：
      - 保持相同的目标函数与约束（如跟踪误差、行业/单票上限）；
      - 将 Qlib 的数据调用替换为 `data_service` 调用；
    - 这样可以保证 **模拟盘与实盘都使用相同的组合构建逻辑**。

### 7.5 订单生成与执行策略层：OrderPlanner / ExecutionPolicy

- **目标**：
  - 将「当前持仓 + 目标持仓」差分成一组具体订单；
  - 结合市场流动性与策略风格，决定下单时点、价格类型与是否启用智能算法。

- **建议接口（示例）**：`aistock/execution/order_planner.py`

```python
class OrderPlanner:
    def __init__(self, account_service, market_data_service, logger):
        ...

    def plan_orders(
        self,
        account_id: str,
        target_portfolio: dict[str, float],
    ) -> list["OrderRequest"]:
        """根据目标组合与当前账户状态，生成一组订单请求."""
        ...
```

- **ExecutionPolicy 建议**：
  - AIstock 自定义一组策略执行规则，例如：
    - 日内低频：开盘后 N 分钟内按限价单分批买入；
    - 对成交量较大的标的，可以使用 `LATEST_PRICE` 或对手方最优价；
    - 对流动性较差的标的或大额交易，优先使用 `smart_algo_order_async` 调用 miniQMT 的智能算法；
  - 这些规则应配置化写入 `StrategyConfig` 或独立的 `ExecutionPolicyConfig`，方便后续扩展。

> 本层完全由 AIstock 自研实现，miniQMT 仅作为执行通道，Qlib 只在回测/模拟中提供参考调仓逻辑。

### 7.6 风控与监控层：RiskEngine / MonitoringService

- **目标**：
  - 在线拦截/限制不合规或超风险的订单；
  - 统一监控模拟盘/实盘执行质量与风险事件。

- **RiskEngine 建议职责**：
  - 在线风控：
    - 单票/行业仓位上限检查；
    - 总仓位/现金占比约束；
    - 日内/累计回撤阈值（结合账户净值曲线）；
    - 黑名单股票/停牌/高风险股票过滤；
    - 全局 kill-switch（紧急情况下拒绝所有新单并尝试降仓）；
  - 与 OrderPlanner 集成方式：
    - `OrderPlanner` 在下单前调用 `RiskEngine.validate(order_batch)`；
    - 对不通过的订单返回拒绝原因或自动削减规模。

- **借鉴 Qlib 的部分**：
  - 风险指标体系：年化收益、波动率、最大回撤、信息比率、tracking error 等；
  - 成本/滑点模型：作为模拟盘与实盘后评估的一致工具；
  - 实现方式：
    - 实盘结束后，将成交与持仓路径重构为 Qlib 可识别的格式；
    - 使用 Qlib 的风险分析模块计算指标并写入 AIstock 的绩效数据库。

> 在线风控的“规则引擎 + 决策链路”必须由 AIstock 自研，Qlib 仅作为 **指标与离线评估工具**。

### 7.7 面向 AIstock 的开发 checklist

下面是 AIstock 执行侧按本设计可直接落地的开发任务清单（不涉及 RD-Agent 代码改动）：

1. **搭建 MiniQMTExecutionGateway**（依赖 XtQuantTrader）：
   - 完成连接、订阅、报单、撤单、回调转发；
   - 封装为内部统一接口，便于未来支持其他券商通道。

2. **实现 AccountService / MarketDataService**：
   - 设计并创建账户/持仓/订单/成交/资金等表结构；
   - 编写同步逻辑与查询 API，为上层组合与风控提供统一视图。

3. **实现 PortfolioEngine 并在模拟盘验证**：
   - 首先在研究环境直接调用 Qlib 的组合策略，验证配置与效果；
   - 再逐步移植关键优化逻辑到 AIstock 的 PortfolioEngine，使其完全独立于 Qlib provider；
   - 确保在相同输入条件下，RD-Agent 与 AIstock 组合结果在误差范围内一致。

4. **实现 OrderPlanner 与 ExecutionPolicy**：
   - 定义 OrderRequest/OrderResponse 领域模型；
   - 实现从目标组合到具体订单的拆分与调度；
   - 支持配置化的价格类型/智能算法选择。

5. **实现 RiskEngine 与 MonitoringService**：
   - 落地一批基础风控规则（单票上限、总仓位上限、黑名单等）；
   - 接入监控系统，对关键风险事件和执行异常报警；
   - 使用 Qlib 风险分析对模拟盘/实盘结果做周期性评估，写入绩效报表。

> 完成以上步骤后，AIstock 将具备一套与 RD-Agent 研究侧高度对齐，
> 但在实现上完全独立的执行层架构：
> - 研究与回测逻辑主要在 RD-Agent + Qlib；
> - 执行与风控逻辑完全在 AIstock + miniQMT；
> - 中间通过 Phase 2/3 约定的 JSON/Schema 与 Catalog 对接成果与配置.

### 7.8 策略预览模式与多策略对比评估（Phase 3 范围，独立于 Phase 2）

- **目标与场景：**
  - 在不占用 miniQMT 模拟盘/实盘账户、完全不下单的前提下，
    提供一个“策略/因子/模型输出的买卖信号与持仓表现”的 **预览环境**；
  - 支持在同一页面上并行展示多条策略的净值曲线与持仓结构，作为“进入模拟盘前”的预评估工具；
  - 本小节所述能力全部归入 Phase 3 范围，不要求对 Phase 2 的因子包/策略 API 设计作任何修改。

- **执行模式区分：**
  - `preview`（策略预览）：仅计算信号与虚拟持仓，不下单、不写入真实/模拟盘账户；
  - `paper`（模拟盘）：通过 miniQMT 模拟账户真实报单，由现有 `SimpleStrategyExecutor + qmt_client` 路径承载；
  - `live`（实盘）：未来阶段在审批后切换到 miniQMT 真实账户，仍复用同一套执行路径。
  - 本节聚焦 `preview` 模式的后端与数据设计，其余两种模式在 7.1–7.6 已覆盖。

- **预览数据持久化设计（AIstock DB 内部表，示意）：**
  - `preview_strategy`：
    - 记录每一个预览实例：
      - `id`、`strategy_id`、展示名称、初始资金 `initial_capital`、状态（running/stopped/archived）、创建/更新时间等；
    - 支持前端创建/停止/归档预览实例，为“预览列表”提供基础数据.
  - `preview_trade`：
    - 记录预览模式下的虚拟成交明细，用作实时/回放时更新持仓与成本：
      - `preview_id`、交易日 `trading_day`、时间 `datetime`、标的 `symbol`、方向 `side`（buy/sell）、价格 `price`、数量 `volume`、金额 `amount`、原因 `reason` 等；
    - 预览引擎每次根据策略信号“假想成交”时写入一条记录.
  - `preview_position`：
    - 维护每个预览实例的当前持仓视图：
      - `preview_id`、`trading_day`、`symbol`、`volume`、`avg_cost`、`last_price`、`market_value` 等；
    - 可选择：
      - 仅保留最新快照；或
      - 每个交易日收盘时落库一份“日终快照”，便于后续调试回放.
  - `preview_account`：
    - 记录预览虚拟账户的资金状态：
      - `preview_id`、`trading_day`、`cash`、`total_value`、`daily_pnl`、`cum_pnl` 等；
    - 作为后续净值曲线与绩效计算的主数据源.

- **盘中计算与 1 分钟批量刷新策略：**
  - 预览引擎与策略执行：
    - 使用与模拟盘/实盘一致的策略实现/因子/模型，只是将“执行后端”路由为 **SignalPreviewEngine**（概念），不调用 miniQMT；
    - 对每个预览实例：
      - 接收策略输出的买卖信号或目标持仓；
      - 使用当前行情价（来自 `backend/data_service/api.py`）作为虚拟成交价；
      - 在内存中更新虚拟账户状态，同时写入 `preview_trade`，更新 `preview_position` 与 `preview_account` 的最新状态.
  - UI 展示刷新：
    - 盘中页面优先使用“最近一次计算结果”，而不是 tick 级实时推送；
    - 建议通过 **定时任务（如每 1 分钟）**：
      - 统一为所有活跃预览实例拉取最新行情；
      - 批量更新各实例的 `preview_position.last_price` / `market_value` 与 `preview_account.total_value`；
      - 为前端提供一个“每分钟级别”的收益与持仓刷新频率，在资源占用与实时性之间取得平衡.

- **收盘结算与历史净值曲线：**
  - 每个交易日收盘后，运行 EOD 结算任务：
    - 使用 AIstock 数据服务提供的指数/个股收盘价，为每个 `preview_id` 计算当日收盘总市值与总资产；
    - 写入/更新当日 `preview_account.total_value`、`daily_pnl`、`cum_pnl` 等字段；
    - 同步更新历史峰值，用于最大回撤的后续统计.
  - 有了 `preview_account` 的逐日记录后：
    - 可在预览页面或离线工具中绘制任一预览实例的净值曲线（`total_value / initial_capital`）；
    - 支持计算年化收益、波动率、最大回撤等粗略绩效指标.

- **多策略对比与上证指数基准曲线：**
  - 预览页面需要支持：
    - 用户自定义勾选一个或多个 `preview_strategy` 实例，
      在同一图表中展示多条净值曲线，便于横向比较；
    - 叠加至少一条指数基准曲线（建议默认使用上证指数 `SH000001`）：
      - 从 AIstock 数据服务获取指数日 K 线；
      - 将收盘价序列归一化（如首日为 1.0），作为“指数净值曲线”绘制在同一坐标系中；
    - 时间坐标与交易日历对齐，支持拖动/缩放与区间筛选.
  - 为避免资源占用过高：
    - 后端可对“同时活跃的预览实例数”设置上限（例如至少支持 10 个，可配置）；
    - 提供 API 或管理界面停止/清理不活跃实例，释放计算与存储资源.

> 以上策略预览模式的设计仅依赖 AIstock 自有数据库与 Phase 2/3 已有的数据服务/策略接口，
> 是 Phase 3 的增量能力，不要求修改 Phase 2 已经开始实现的 FactorPackage/StrategyConfig/ExperimentRequest 等设计.

### 7.9 LiveFeedback 规划：Phase 3 准备与 Phase 4 闭环

- **设计引用：**
  - LiveFeedback 的统一 schema 与阶段划分，见顶层文档《2025-12-23_RD-Agent_AIstock_TopLevel_Architecture_Design.md》中 “Phase 4.x LiveFeedback 统一设计与 Phase 3 准备工作”。
  - 本小节只强调 Phase 3 期间 AIstock 需要完成的“埋点与数据准备”工作，以及 RD-Agent 对输入格式的期望.

- **Phase 3：AIstock 侧需要完成的准备工作（不要求形成完整闭环）：**
  - 在模拟盘/实盘执行路径中，确保以下信息可从本地 DB 中还原：
    - 账户与持仓路径（净值曲线、资金曲线）；
    - 成交与订单路径（价格、数量、成交比率、撤单/拒单情况）；
    - 风控与异常事件（被 RiskEngine 拒绝的订单、触发 kill-switch、QMT 连接异常等）；
  - 根据顶层文档的 LiveFeedback schema，为后续 JSON 生成预留好：
    - `strategy_instance_id` / `experiment_id` 的映射关系；
    - 统计周期（例如按自然日/交易日）聚合指标的 SQL/服务层封装；
  - 可以在模拟盘路径上实现一个最小版的 LiveFeedback 导出（例如仅包含收益、回撤、成交比率），用于与 RD-Agent 侧联调.

- **Phase 4：在此基础上完成的工作（本文件只做前瞻，不在 Phase 3 内实现）：**
  - AIstock：
    - 按统一 schema 周期性生成完整的 LiveFeedback JSON（包含绩效、执行、成本、事件等）；
    - 将文件写入约定目录或通过 API 推送到 RD-Agent 可访问的端点.
  - RD-Agent：
    - 在 orchestrator 层增加 LiveFeedback 消费逻辑：
      - 解析 LiveFeedback，识别对应的策略实例与实验版本；
      - 将“希望改进的目标”（例如控制回撤、提升 Sharpe）编码进新的 ExperimentRequest 中；
    - 将这类基于 LiveFeedback 触发的实验记录为“定向演进实验”，并在 summary/feedback.json 中挂接来源.

> 综上，Phase 3 主要任务是“让 AIstock 执行路径具备生成 LiveFeedback 所需的全部原始数据与聚合能力”，
> 真正的“模拟盘/实盘反馈闭环”则按顶层架构文档规划在 Phase 4 落地.

### 7.10 基于 loop 的选股与策略预览：AIstock 侧在 Phase 3 的硬性目标

> 本小节在顶层架构附录与 Phase 2 v2 设计的基础上，明确 Phase 3 中 AIstock 必须完成的、可直接用于研发与上线的“选股 + 策略预览”能力，不允许只做 PoC/精简版.

#### 7.10.1 能力目标与约束

- **目标 1：基于任意合格 loop 的“选股服务”**
  - 对于任意在 RD-Agent 侧通过回测与验证、并已写入 registry（`has_result=1`）的 loop，
    AIstock 需要在 Phase 3 内提供一个可在后端/前端直接调用的选股接口：
    - 输入：`loop_id`（含 `task_run_id`、`workspace_id` 等标识）、`trade_date` 或日期区间、股票池（如不指定则使用策略配置中的默认 universe）；
    - 输出：当期的候选股票列表及其 `score` / `rank` /（可选）`target_weight` 等字段.

- **目标 2：策略预览 UI 与后台流程**
  - 在 AIstock 前端提供策略预览页：
    - 用户可选择一个或多个 loop（或其对应的策略/配置）；
    - 为给定的回放区间触发选股与组合构建，并展示收益/回撤曲线、持仓变化等；
    - 不触达真实资金账户，仅在预览数据表（如 `preview_strategy`、`preview_trade`、`preview_position`、`preview_account`）中落库.

- **约束：不得做“简化版”实现**
  - 以上能力须按本设计的最终目标完整实现：
    - 使用与生产环境一致的数据服务层与 qlib runtime；
    - 支持 RD-Agent 侧 qlib 模型的真实推理流程，而非硬编码的替代逻辑；
    - 在数据接口与存储 schema 上不得引入“只用于 demo 的特例路径”。

#### 7.10.2 AIstock 内部 qlib runtime 与 DataProvider 集成

- **固定版本 qlib runtime（只做模型/数据，不做回测）**
  - AIstock 后端环境中安装固定版本 qlib（例如 `qlib==x.y.z`），用于：
    - 解析 RD-Agent 导出的 qlib `model_conf` / `dataset_conf` / `feature_conf` 等配置；
    - 通过 qlib Model 接口加载已训练模型并执行 `predict`；
  - AIstock 不使用 qlib 自带回测/撮合引擎，回测与执行仍由 RD-Agent 与 AIstock 执行层各自负责.

- **自定义 DataProvider 挂接 AIstock 数据服务层**
  - 实现一个 qlib DataProvider / DataHandler 适配层，内部调用 AIstock 的数据服务层接口：
    - 历史视图：`get_history_window(universe, start/end 或 bars, fields, freq)` 返回 `DataFrame(MultiIndex(datetime, instrument))`；
    - 实时/近实时视图：在需要做当日选股时，可通过 `get_realtime_snapshot` / 增量窗口拼接出完整输入；
  - 该 DataProvider 负责：
    - 将 RD-Agent 在 dataset 配置中要求的字段名/频率，转换为 AIstock 数据服务层支持的请求参数；
    - 确保生成的数据结构与 qlib Model 训练时期使用的一致（包括缺失值处理、对齐规则等）。

- **模型兼容性约定**
  - 只要 RD-Agent 侧模型遵守 qlib Model 契约，并在 loop artifacts 中暴露：
    - `model_type` / `model_conf` / `dataset_conf`；
    - 特征字段名列表、窗口长度、频率等信息；
  - AIstock 便有责任在 Phase 3 中通过上述 qlib runtime + DataProvider 实现该模型在选股/预览场景的直接复用，不再额外限定框架白名单.

#### 7.10.3 loop → log 目录与模型 artifacts 的同步要求

- **loop 日志目录信息（log_dir）**
  - Phase 3 要求 RD-Agent：
    - 在 registry 与 Export 脚本中（如 `export_aistock_loop_catalog.py` / `export_aistock_strategy_catalog.py`），为每个 loop 记录：
      - 归属的 log session 根目录（例如某次实验的时间戳目录或 UUID 目录）；
      - 与该 loop 直接关联的关键日志子目录（如回测日志、模型训练日志等）；
  - AIstock 同步时：
    - 将 `log_dir` 及相关相对路径字段一并导入本地数据库；
    - 在 UI 中为每个 loop 提供“查看 RD-Agent 回测/训练日志”的入口，通过约定的共享文件路径或代理服务跳转到对应目录.

- **模型 artifacts 与元数据**
  - RD-Agent 需要在 Phase 3 内补充/完善模型 registry 与导出逻辑，使每个可复用的 loop 至少包含：
    - 模型文件路径（相对于 workspace 或共享模型仓库的相对路径）；
    - 模型类型（对 qlib 而言的 `class` / `model_type` 名称）；
    - 输入特征 schema（字段名列表、窗口长度、频率等），与 qlib dataset 配置保持一致；
  - Export 脚本需将上述信息汇入 `strategy_catalog.json` / `loop_catalog.json`，并由专门的 `tools/export_aistock_model_catalog.py` 生成第四个 Catalog：`model_catalog.json`，供 AIstock 一并同步（对应只读 API 中的 `/catalog/models`）。

- **AIstock 同步与落库**
  - 在 Phase 3 中，AIstock 的成果同步任务需要在 Phase 2 的基础上扩展：
    - 除了因子/策略/loop 的静态元数据外，必须同步：
      - loop 关联的 `log_dir` 信息；
      - 与 loop 绑定的模型 artifacts 与 schema 信息；
  - 将这些字段落入本地 DB 表，使选股服务与策略预览服务可以：
    - 从 DB 中解析出完整模型加载/数据请求所需的信息；
    - 从 `log_dir` 提供回测/训练日志的溯源入口.

> 通过本节约定，Phase 3 结束时，AIstock 侧应具备：
> - 基于任意合格 loop 进行选股与策略预览的完整链路；
> - 对 RD-Agent 回测/训练日志与模型 artifacts 的直接溯源能力；
> - 一套可长期演进的 qlib runtime + DataProvider 机制，支撑后续引入的新模型类型，而无需在 AIstock 内重复重写模型逻辑.
