# RD-Agent × AIstock 演进研发实验室：终极目标与分阶段目标备忘录（草案）

> 目的：对本项目的“终极目标”形成统一认知，并在此基础上设计一个可分阶段落地的演进路线。要求每个阶段都**可实用、可增量演进**，避免走弯路或做大规模重构。

---

## 1. 终极目标愿景（Ultimate Vision）

RD-Agent 与 AIstock 共同构成一个“可定制的量化研发实验室”，支持：

- **多源因子与策略的统一管理与演进**：
  - 包括 RD-Agent 自动演进出来的因子与策略；
  - 也包括外部提供的因子（静态因子表 / 外部 factor.py）与外部策略（已有 Qlib YAML / Python 策略）。

- **全链路闭环**：
  1. AIstock 定义研发方向（StrategyConfig + FactorSetConfig + 目标函数 + 约束），下发给 RD-Agent。
  2. RD-Agent 生成/组合因子与策略，训练模型并回测，产出结构化成果（策略信号、回测指标、因子元数据、因子表现、反馈摘要等）。
  3. AIstock 存储与可视化所有成果，支持检索与比较（因子库 / 策略库 / 实验库）。
  4. AIstock 将选定的策略/因子推入模拟盘或生产环境。
  5. 从模拟盘/生产环境拿到“真实效果”反馈（收益、回撤、滑点、风控事件等），再结构化反馈给 RD-Agent。
  6. RD-Agent 基于这些真实反馈做“**有方向的下一轮演进**”，而非盲目尝试。

- **可视化与可编排**：
  - 所有关键环节（因子生成、组合、训练、回测、反馈、真实表现）在 AIstock UI 中可视化；
  - 后端通过工作流、多 agent 架构与 RAG 等技术，对实验进行智能编排与搜索。

- **从实验室到交易自动化的渐进式落地**：
  - AIstock 在早期以“研发管理端 + 数据汇总端 + 模拟盘验证”为主，逐步演进为面向实盘的交易执行平台；
  - 交易执行能力遵循“可控、可审计、可回滚”的原则，从人工交易辅助开始，逐步过渡到半自动与全自动交易。

简而言之：

> **AIstock：负责“设计研发方向 + 存储与可视化成果”；**  
> **RD-Agent：负责“执行具体实验 + 生成结构化成果 + 按反馈有方向地演进”。**

---

## 2. 核心数据与契约原则

为避免后期大改结构，从一开始就统一几个关键原则：

1. **以 registry.sqlite + artifacts 为“唯一权威源”**：
   - 所有可对外消费的成果（策略信号、回测指标、因子元数据、因子表现、反馈摘要等）必须通过：
     - `task_runs` / `loops` / `workspaces` 表 +
     - `artifacts` / `artifact_files` 记录
   - 暴露给 AIstock，而不是依赖隐式的目录结构或日志约定。

2. **统一 artifacts 生成逻辑**：
   - 在 RD-Agent 内部抽象一个类似 `write_loop_artifacts(...)` 的函数/方法：
     - 输入：task_run、loop、workspace_path、action、has_result 等上下文；
     - 内部：根据 workspace 中已有文件与计算结果，生成并登记所有约定的 artifacts；
     - 所有调用点（包括主流程与回填脚本）只需“填好参数、调用一次”。

3. **区分“内部实现细节”与“对外契约”**：
   - 日志、内部 debug 信息不会直接暴露给 AIstock；
   - 但会从 log/feedback 中“提炼关键信息”，写入结构化的 `feedback.json` 等对外契约文件。

4. **所有新设计的 artifacts / JSON Schema 必须兼容增量演进**：
   - 通过新增字段/新增 artifact 类型来扩展，而不是频繁修改现有结构；
   - 确保早期阶段产出的数据，在后续阶段仍然可读可用。

5. **引入统一数据服务层，屏蔽底层多数据源差异**：
   - AIstock 未来作为“数据服务中枢”，统一封装底层多种数据源：
     - 实时/盘中行情与委托成交：miniQMT（作为实时与交易相关数据的首选来源，降低延迟与不确定性）；
     - 历史行情与衍生数据：TDX、tushare 等第三方数据源；
     - 自有本地化存储：AIstock 自建的 timescaledb 等时序数据库；
   - 在数据服务层之上，对外只暴露**标准化的数据接口与数据集规范**：
     - 面向 Qlib/回测的标准数据集（如 qlib bin、Alpha158 所需字段）；
     - 面向 RD-Agent/因子研发的 pandas/HDF5/Parquet 视图（如 `daily_pv.h5`、`moneyflow.h5`、静态因子表等）；
     - 面向模拟盘与实盘交易的撮合与风控所需视图（包括实时行情、账户/持仓、订单与成交流）；
   - **数据使用方（RD-Agent、Qlib、模拟盘/实盘模块）不需要关注具体数据来源与底层格式差异**，只需声明所需的数据视图与规范，数据服务层负责路由、融合与转换；
   - 唯一显式例外是：针对实时行情与交易执行，默认优先使用 miniQMT 作为底座，以保证低延迟与数据一致性，其它数据源更多用于补充与回溯。

---

## 3. 目标对象：因子、策略、模型、反馈与实盘

整个系统围绕以下五个维度运转：

1. **因子（Factor）**：
   - RD-Agent 自动演进出来的因子；
   - 外部因子（静态因子表、外部 factor.py 等）。

2. **策略（Strategy）**：
   - 当前以 Qlib 的 `TopkDropoutStrategy` 为主；
   - 未来通过 StrategyConfig 支持多策略、多参数；
   - 支持外部策略（自定义 YAML / Python 实现）。

3. **模型（Model）**：
   - LGB baseline / SOTA 模型 / 未来的其他模型族；
   - 作为“因子 → label”学习的引擎，与具体交易策略解耦。

4. **回测与反馈（Backtest & Feedback）**：
   - 统一通过 Qlib pipeline 跑历史回测；
   - RD-Agent 提供结构化的反馈（metrics、图表、critic 总结、告警等）。

5. **模拟盘与真实表现（Live / Paper Trading）**：
   - AIstock 在模拟盘/生产环境中执行策略；
   - 将真实运行结果（收益、回撤、滑点、风险事件）结构化反馈给 RD-Agent，用于定向演进。

6. **交易执行与风控（Execution & Risk）**：
   - AIstock 从“人工交易辅助（信号解释 + 下单建议）”逐步演进到“半自动交易（人审 + 自动下单）”，最终实现“全自动交易（自动下单 + 自动风控 + 自动监控）”；
   - 交易侧产出的真实成交、滑点、风控触发、订单生命周期等数据，作为研发闭环的重要反馈来源。

---

## 4. 分阶段目标设计（不一步到位，逐步可用）

以下阶段不是严格时间线，而是“能力层级”的划分。每一阶段都在**前一阶段的数据与架构基础上增量演进**，避免推翻重来。

### Phase 1：单策略、多因子，打通 registry 与 AIstock（已基本完成）

**目标：**

- RD-Agent 能通过 registry.sqlite + workspace manifest，稳定记录：
  - model 回测成果：`signals.*`, `ret_schema.*`, `qlib_res.csv`, `ret.pkl`；
  - loops / workspaces / artifacts / artifact_files 之间的关联；
- AIstock 能基于 registry：
  - 发现“有结果的回测”（`action='model' AND has_result=1`）；
  - 读取 `signals.*` / `ret_schema.*` / 回测指标用于后续消费。

**关键特征：**

- 历史上当前实现主要使用的策略框架为 TopkDropout（topk=50, n_drop=5, 长多组合），该框架本身几乎不包含仓位管理与风险控制能力；在本备忘录所规划的整体架构中，应逐步以**具备仓位管理与风险控制能力的成熟策略框架**作为基准，而不是长期沿用这一默认框架。
- 因子演进主要在 RD-Agent 内部进行，AIstock 看到的是“单条策略线”的结果。
- 历史 workspace 回填（backfill）已幂等，确保旧成果也能被发现。

**与交易侧的关系（早期）**：

- AIstock 仍以“结果查看与人工决策辅助”为主：
  - 读取 `signals.*`，基于策略与因子的组合生成选股逻辑与候选股票清单；
  - 结合自定义的股票池与过滤条件，对候选股票进行人工审阅与打分；
  - 通过人工交易或模拟盘试运行来验证策略与因子组合的有效性；
  - 不要求自动下单能力，但要求所有输出可追溯、可解释。

> 状态：当前项目已基本满足 Phase 1 要求，是后续阶段的基础。

---

### Phase 2：统一 artifacts 生成函数 + 因子/策略成果结构化暴露

**目标：**

- 在 RD-Agent 内部抽象出统一的 `write_loop_artifacts(...)` 工具，集中处理：
  - workspace_meta / experiment_summary / manifest 的写入；
  - `signals.*` / `ret_schema.*` / `qlib_res.csv` / `ret.pkl` 的登记；
  - 为后续新增 artifacts 预留挂载点。

- 在不破坏现有行为的前提下，逐步扩展 artifacts：
  - `factor_meta.json`：记录因子名称、公式、中文说明、来源（RD-Agent / 外部）、创建时间等；
  - `factor_perf.json`：记录单因子与典型组合的 IC / 年化收益 / 回撤 / Sharpe 等指标，以及适用窗口信息；
  - `feedback.json`：提炼自 RD-Agent 的 CoSTEER / log，包含决策（Decision）、亮点、局限、告警等摘要，是对外暴露“本轮实验如何被评估”的必备信息；
  - 回测图表文件（如 `ret_curve.png`, `dd_curve.png`）：用于在 AIstock 侧直观展示收益/回撤曲线与关键表现，是策略与因子成果对齐的必备可视化 artifacts。

**对 AIstock 的意义：**

- 能够在不读 RD-Agent 内部目录/日志的前提下：
  - 构建因子库视图：查看所有历史演进因子及其表现；
  - 构建策略库视图：结合 strategy 配置与指标；
  - 渲染与 RD-Agent UI 接近的回测曲线与指标面板。

**演进约束：**

- 优先实现 `write_loop_artifacts(...)` 的“重构不改行为”；
- 然后再逐步新增 artifacts 类型，保证每一步都是向后兼容的增量扩展。

---

### Phase 3：StrategyConfig / FactorSetConfig 与外部资产接入

**目标：**

- 引入结构化的：
  - `StrategyConfig`：描述策略类型、参数、回测窗口、目标指标，以及绑定的提示词模板 ID；
  - `FactorSetConfig`：描述使用哪些因子（Alpha158 / RD-Agent 演进因子 / 外部因子），及其来源与标签。

- 支持外部因子与策略的接入：
  - 外部因子：
    - 通过静态因子表（parquet + StaticDataLoader 或 CombinedAlpha158StaticLoader/CombinedAlpha158DynamicFactorsLoader）接入；
    - 或以外部 `factor.py` 的形式接入 RD-Agent factor workspace；
  - 外部策略：
    - 通过额外的 Qlib YAML（`conf_external_strategy_xxx.yaml`）以及对应的 StrategyConfig 接入。

- 由 AIstock 统一管理：
  - 策略库（StrategyConfig 列表，包括外部与内部策略）；
  - 因子库（来自 Phase 2 的因子元数据与表现）；
  - 为每次 RD-Agent 实验下发表达清晰的“实验配置”。

**对 RD-Agent 的意义：**

- 从“模糊自然语言任务”转向“有约束、有方向的 ExperimentRequest”：
  - 包含：StrategyConfig + FactorSetConfig + Train/TestConfig + 目标函数与约束；
- 外部/内部因子与策略统一纳入同一演进管线。

---

### Phase 4：多策略、多窗口与模拟盘反馈闭环

**目标：**

- 支持在多个策略 / 多个回测窗口下，对同一批因子进行对比评估：
  - 多 YAML、多 StrategyConfig；
  - 多时间 regime（牛市 / 熊市 / 震荡等）。

- AIstock 将模拟盘 / 生产环境中的真实表现以结构化 LiveFeedback 的形式反馈给 RD-Agent：

  ```json
  {
    "strategy_instance_id": "xxx",
    "factor_set_id": "SOTA_plus_new_20251223",
    "eval_window": ["2026-01-01", "2026-03-31"],
    "live_metrics": {
      "annual_return": 0.12,
      "max_drawdown": 0.18,
      "sharpe": 1.0,
      "turnover": 1.8,
      "slippage_cost": 0.004,
      "risk_events": ["hit_drawdown_15%", "hit_var_limit"]
    },
    "desired_improvement": {
      "reduce_max_drawdown": true,
      "reduce_slippage": true,
      "keep_return_not_lower_than": 0.1
    }
  }
  ```

- RD-Agent 在接到 LiveFeedback 后：
  - 以之为目标/约束，发起“定向演进实验”，而非盲目尝试；
  - 在 StrategyConfig / FactorSetConfig / Train/TestConfig 约束下，优化因子组合与策略参数。

**成效：**

- 研发与真实表现形成闭环：
  - 研发实验不再只对历史回测指标负责，也对“真实执行效果”负责；
  - RD-Agent 的下一轮演进具有明确改进方向（如降低回撤、降低滑点）。

**交易自动化的阶段性落地（与闭环协同）**：

- 交易执行从“可控的人审模式”开始：
  - AIstock 将策略输出的信号用于模拟盘与人工交易（手动下单或半自动下单）；
  - 将真实成交、滑点、风控触发等反馈给 RD-Agent，驱动“面向执行质量”的定向演进。
- 逐步引入自动化能力：
  - 人审下单（建议单）
  - 半自动（策略自动生成订单，人审确认后执行）
  - 全自动（策略自动下单 + 风控规则自动干预 + 异常自动降级）

---

### Phase 5：工作流、多 agent 与 RAG 驱动的智能实验室

**目标：**

- 在前几期奠定的数据与契约基础上，引入：
  - 工作流引擎：将复杂实验拆成多步（因子生成 → 组合 → 多策略回测 → 汇总 → 建议），并可视化执行状态；
  - 多 agent 架构：
    - 因子 agent：负责在 FactorSetConfig 与反馈约束下设计/修正因子；
    - 策略 agent：负责在 StrategyConfig 空间内搜索合适的策略与参数；
    - 分析 agent：根据历史 experiment + LiveFeedback 生成“下一步实验建议”；
  - RAG：
    - 从历史实验库（registry + artifacts）中检索类似案例与已验证模式，用于
      - 提示词增强、
      - 参数建议、
      - 风险提示。

- AIstock UI 作为“统一控制台”：
  - 一键发起从“策略/因子设想 → 批量实验 → 筛选候选 → 模拟盘 → 再演进”的全流程；
  - 可视化所有阶段的状态与结果。

- 交易与运营可视化（面向生产）：
  - 交易执行状态（订单生命周期、成交分布、滑点分析、失败原因）；
  - 风控面板（限仓、限回撤、VaR/波动控制、黑名单、熔断与降级）；
  - 策略健康度（漂移监控、因子失效监控、回测-实盘偏差监控）。

---

## 5. 设计原则与注意事项

为实现“每阶段成果可向上延伸、避免大改结构”，整体设计需要遵守：

1. **先稳定数据契约，再扩展功能**：
   - 先保证 registry.sqlite + artifacts + JSON schema 的稳定性和可扩展性；
   - 所有新功能优先通过新增 artifacts / 字段来集成。

2. **统一入口函数减少耦合**：
   - 像 `write_loop_artifacts(...)` 这样集中编写输出逻辑，避免分散在多处；
   - 将来需要新增因子元数据、因子表现、反馈摘要、图表，只需修改这一处。

3. **区分“推荐成果”与“所有尝试”**：
   - 通过 Decision / has_result / artifact_type 等标记：
     - 哪些是可供 AIstock 使用的“推荐成果”；
     - 哪些只是内部尝试记录。

4. **外部资源一视同仁**：
   - 外部因子、外部策略只要符合统一的 FactorSetConfig / StrategyConfig 规范，就可以无缝纳入 RD-Agent 的演进管线。

5. **每个阶段都可独立交付价值**：
   - Phase 2：AIstock 能够看到完整的因子与策略成果视图；
   - Phase 3：外部资产可接入，研发空间放大；
   - Phase 4：研发与模拟盘闭环；
   - Phase 5：智能化与自动化程度提高。

   在上述每一个阶段：

   - AIstock 都可以通过**人工或半人工的方式在模拟盘/实盘上做验证与反馈**：
     - 早期阶段以“参考信息”为主，策略信号更多作为人工决策与研究参考，不直接作为最终自动交易决策；
     - 随着阶段推进与风控/监控能力增强，逐步提高自动化程度与信任度，将实验室成果更紧密地连接到真实交易与组合管理中。

---

## 6. 后续步骤（仅方向，不含细节）

在你确认本备忘录后，下一步可以：

1. 基于此文档，起一份“上层架构设计草案”：
   - 定义 ExperimentRequest / ExperimentResult / LiveFeedback / StrategyConfig / FactorSetConfig 的草案 schema；
   - 明确哪些字段由 AIstock 负责，哪些由 RD-Agent 负责。

2. 针对 Phase 2 先行细化：
   - 设计 `write_loop_artifacts(...)` 的职责边界与输入输出；
   - 设计 `factor_meta.json` / `factor_perf.json` / `feedback.json` 的字段与存放路径；
   - 确认与现有 registry / backfill 逻辑的兼容性。

3. 依此将 Phase 3–5 分解成更细的里程碑文档与实现计划。

本备忘录作为“终极目标 + 分阶段目标”的高层框架，后续所有详细设计与实现都应在此框架下展开，以避免日后大规模结构性修改。
