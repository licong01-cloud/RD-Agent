# Phase 3 Execution & Strategy Preview Design for AIstock

> 本文是 **AIstock 执行层 + 策略预览 + LiveFeedback 输出** 的工程实现指引，
> 面向 AIstock 团队与在 AIstock 仓库中运行的 AI 助手，作为执行相关设计的 **单一真相源**。
> 若需了解整体架构或 RD-Agent/Qlib 研究侧细节，请参考：
> - 《2025-12-23_RD-Agent_AIstock_TopLevel_Architecture_Design.md》
> - 《2025-12-26_Phase3_Detail_Design_RD-Agent_AIstock_v1.md》

## 1. 角色与边界概览

- **RD-Agent + Qlib**：
-  负责因子/模型/组合的研究与回测，不参与实盘撮合；
-  对外以 JSON/YAML/Artifact 输出实验结果与配置，不暴露内部实现细节。

- **AIstock**：
-  负责数据服务、执行层（预览/模拟盘/实盘）、风控与监控；
-  使用 Phase 2/3 约定的 schema 与 RD-Agent 交互，不直接调用 RD-Agent 运行时代码。

- **miniQMT / xtquant**：
-  唯一真实交易通道（模拟盘 + 实盘），提供下单、撤单、账户、成交与行情能力；
-  AIstock 通过 `backend/infra/qmt_client.py` 与之集成。

- **Qlib 在本项目中的定位**：
-  仅用于研究/回测与离线风险评估，不用于实盘撮合；
-  其组合构建/风险评估逻辑可被移植/借鉴到 AIstock 侧的 `PortfolioEngine` 中。

## 2. 执行层五层架构（AIstock 视角）

1. **执行网关层（Execution Gateway Layer）**
-  实现文件：`backend/infra/qmt_client.py`
-  责任：
-  - 管理与 miniQMT 连接、订阅与会话；
-  - 封装 `place_order` / `cancel_order` 等下单接口；
-  - 承接 XtQuant 回调（订单、成交、持仓、资金）。

2. **账户与市场视图层（Account & Market View Layer）**
-  主要文件：
-  - `backend/data_service/api.py`
-  - `backend/data_service/xtquant_adapter.py`
-  - `backend/data_service/miniqmt_adapter.py`
-  （可选）`backend/services/account_service.py`
-  责任：
-  - 将 miniQMT/xtquant 的账户、持仓、订单、成交、行情映射为 AIstock 自有 DB 表与领域模型；
-  - 向上提供统一的 `get_portfolio_state` / `get_open_orders` / `get_trades` / `get_history_window` 等接口。

3. **组合构建与仓位决策层（Portfolio & Positioning Layer）**
-  建议实现：`backend/execution/portfolio_engine.py`
-  责任：
-  - 从策略信号/因子出发，计算目标权重/目标持仓；
-  - 在实现上可移植/借鉴 Qlib 的组合优化与风险评估逻辑。

4. **订单生成与执行策略层（Order Generation & Execution Policy Layer）**
-  建议实现：`backend/execution/order_planner.py`
-  责任：
-  - 将“当前持仓 + 目标持仓”差分为具体订单集合；
-  - 结合流动性与策略风格，决定下单时机、价格类型与是否启用 miniQMT 智能算法。

5. **风控与监控层（Risk & Monitoring Layer）**
-  建议实现：`backend/execution/risk_engine.py`（命名示例）
-  责任：
-  - 在线风控（单票/组合上限、回撤阈值、黑名单等）；
-  - 指标采集与结构化日志输出，为 LiveFeedback 与事后分析提供数据基础。

## 3. 执行模式：preview / paper / live

- **preview（策略预览）**：
-  - 使用实时行情与策略/因子/模型计算买卖点与虚拟持仓；
-  - 不调用 miniQMT 下单接口，不影响模拟盘/实盘账户；
-  - 结果写入 `preview_*` 系列表，用于页面展示与粗略绩效评估。

- **paper（模拟盘）**：
-  - 通过 miniQMT 模拟账户真实下单；
-  - 复用 `backend/infra/strategy_executor.py::SimpleStrategyExecutor` 与 `qmt_client`；
-  - 作为进入实盘前的正式“沙盒环境”。

- **live（实盘）**：
-  - 通过 miniQMT 真实账户下单；
-  - 在逻辑上与 `paper` 共用同一执行栈，只是账号、限额与审批流程不同；
-  - 上线时需通过审批流显式切换。

## 4. 策略预览模式：DB 设计与刷新机制

### 4.1 预览相关表结构（示意）

- **`preview_strategy`**：预览实例元信息
-  - 典型字段：`id`、`strategy_id`、展示名称、`initial_capital`、状态（running/stopped/archived）、创建/更新时间等。

- **`preview_trade`**：虚拟成交明细
-  - 典型字段：`preview_id`、`trading_day`、`datetime`、`symbol`、`side`（buy/sell）、`price`、`volume`、`amount`、`reason` 等。

- **`preview_position`**：预览持仓视图
-  - 典型字段：`preview_id`、`trading_day`、`symbol`、`volume`、`avg_cost`、`last_price`、`market_value` 等。

- **`preview_account`**：预览账户资金维度
-  - 典型字段：`preview_id`、`trading_day`、`cash`、`total_value`、`daily_pnl`、`cum_pnl` 等。

### 4.2 盘中刷新与显示策略

- **虚拟成交与持仓更新：**
-  - 策略执行路径在 `preview` 模式下不调用 miniQMT；
-  - 由一个 "SignalPreviewEngine"（概念实现）接管：
-    - 接收策略/模型输出的买卖信号或目标持仓；
-    - 使用 `backend/data_service/api.py` 获取的实时行情价作为虚拟成交价；
-    - 写入 `preview_trade`，更新 `preview_position` 与 `preview_account`。

- **1 分钟批量刷新：**
-  - 盘中前端页面不追求 tick 级实时，而是：
-    - 直接展示 DB 中最近一次计算结果；
-    - 通过定时任务（例如每 1 分钟）统一：
-      - 拉取所有活跃预览实例持仓标的的最新行情；
-      - 批量更新 `preview_position.last_price` / `market_value` 与 `preview_account.total_value`；
-  - 要求至少支持 10 个并行预览实例，上限可按资源配置。

### 4.3 收盘结算与净值曲线

- 每个交易日收盘后执行 EOD 任务：
-  - 使用 AIstock 数据服务提供的收盘价/日线行情，为每个 `preview_id` 计算当日收盘总市值与总资产；
-  - 将结果写入当日 `preview_account.total_value`、`daily_pnl`、`cum_pnl`；
-  - 更新历史峰值，用于最大回撤统计。

- 有了 `preview_account` 的逐日记录后：
-  - 前端可以绘制单个或多个预览实例的净值曲线（`total_value / initial_capital`）；
-  - 为审批/对比提供可视化依据。

## 5. 多策略对比与指数基准曲线

- 页面需求：
-  - 支持勾选多个 `preview_strategy` 实例，在同一图表中展示多条净值曲线；
-  - 叠加至少一条指数基线（推荐上证指数 `SH000001`），作为对比对象；
-  - 时间轴与交易日历对齐，支持区间选择与缩放。

- 指数基线获取方式：
-  - 通过数据服务获取指数日 K 线；
-  - 将收盘价序列归一化为净值曲线（首日 = 1.0）；
-  - 前端与策略净值曲线共用坐标系展示。

## 6. LiveFeedback：从执行到研究的闭环规划

- 统一 schema 与阶段划分参考：
-  - 顶层文档中 “Phase 4.x LiveFeedback 统一设计与 Phase 3 准备工作”；
-  - 这里只保留 AIstock 需要实现的要点。

- **Phase 3：准备阶段（AIstock 责任）**
-  - 在模拟盘/实盘执行路径中，确保可以从 DB 中恢复：
-    - 账户与持仓轨迹（含净值曲线）；
-    - 订单/成交轨迹（价格、数量、成交比率、撤单/拒单）；
-    - 风控与异常事件（风控拦截、kill-switch 触发、QMT 异常等）。
-  - 按 LiveFeedback schema 设计内部聚合服务：
-    - 能按策略实例 + 时间区间聚合出收益、回撤、波动率、滑点、拦截次数等指标；
-  - 可在模拟盘路径先输出一个最小版 LiveFeedback JSON 做联调。

- **Phase 4：闭环阶段（AIstock + RD-Agent）**
-  - AIstock：
-    - 周期性生成完整 LiveFeedback JSON（按策略实例与评估周期）；
-    - 写入约定目录或通过 API 推送给 RD-Agent；
-  - RD-Agent：
-    - 从 LiveFeedback 中读取实际表现与限制条件；
-    - 构造带明确优化目标的 ExperimentRequest（例如约束 max_drawdown、提高 Sharpe 等）；
-    - 将这类实验标记为“定向演进”，形成研究—执行—反馈闭环。

## 7. 实施路线建议

- **Phase 3 内必须完成：**
-  - 执行层五层架构的基础实现（至少支持模拟盘）；
-  - 策略预览模式（含 `preview_*` 表、1 分钟刷新、预览页面与指数基线）；
-  - 为 LiveFeedback 准备所需的数据采集与聚合能力（但闭环可推迟到 Phase 4）。

- **Phase 4 及以后：**
-  - 补齐 LiveFeedback 全量生成与回传；
-  - 引入审批流、工作流与多策略/多窗口自动演进能力。
