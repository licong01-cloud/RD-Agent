# RD-Agent 多策略回测与 StrategyConfig 备忘录（草案）

> 目标：在不立即改动主流程代码的前提下，梳理未来在 **多策略回测 + 策略配置 + 提示词管理** 方向的演进思路，为后续设计具体方案与接口做准备。

---

## 1. 当前状态与边界

- **回测执行方式**：
  - 由 Qlib YAML 配置控制（如 `conf_baseline.yaml`, `conf_combined_factors_dynamic.yaml`）。
  - RD-Agent 侧通过 `QlibFBWorkspace.execute(qlib_config_name=...)` 调用 `qrun conf_xxx.yaml` 触发回测。

- **主策略框架（当前）**：
  - `TopkDropoutStrategy`，`topk=50`, `n_drop=5`。
  - 长多组合（long-only），无显式多空/行业/风格约束。
  - 固定交易成本（如 `open_cost=close_cost=0.0001`），回测窗口集中在 `2021-01-01 ~ 2025-11-28`。

- **因子与特征管线**：
  - 动态因子由 LLM 生成/修复的 `factor.py` 输出（`result.h5` → DataFrame）。
  - 结合已有因子（SOTA 因子）与新因子，在 `factor_runner.py` 中做：
    - `SOTA_factor` + `new_factors` → 去重（IC 阈值）→ `combined_factors_df.parquet`。
  - 回测特征由 `CombinedAlpha158DynamicFactorsLoader` 统一加载：
    - Alpha158 固定特征 + 动态因子（`combined_factors_df.parquet`）。

- **演进重心（当前）**：
  - 主要集中在 **因子设计 / 特征组合 / 模型迁移** 上。
  - 策略形态与参数（TopkDropout + 固定 topk/n_drop）基本不变，主要作为统一评价框架。

---

## 2. 多策略回测：仅通过 YAML 扩展的空间

在不改 Python 代码的前提下，仍有如下可行空间：

- **复制/修改 YAML，定义多条“策略通道”**：
  - 例如在 `factor_template` 目录下，基于 `conf_combined_factors_dynamic.yaml` 派生：
    - `conf_combined_factors_dynamic.yaml`（现有 Topk50 baseline）。
    - `conf_combined_factors_topk20.yaml`（更集中持仓，高风险高收益）。
    - `conf_combined_factors_defensive_2023_2025.yaml`（低换手 + 短窗口防御型）。

- **可调整维度（不改代码，仅改 YAML）**：
  - `TopkDropoutStrategy` 的参数：
    - `topk`：控制持仓集中度（10 / 20 / 50 / 100 等）。
    - `n_drop`：控制换手率（0 = 几乎 buy&hold，高值 = 高频调仓）。
  - 回测窗口：
    - 按时间 regime 拆分多份：
      - 例如 2010–2014、2015–2018、2019–2020、2021–2022、2023–2025 等。
  - 调仓频率 / 其他 kwargs（在策略/handler 支持范围内）。

- **使用方式（当前阶段）**：
  - RD-Agent 主流程仍自动使用既定 YAML（如 baseline/dynamic）。
  - 额外策略 YAML 可在 workspace 内 **手工 `qrun conf_xxx.yaml`** 运行，用于离线比较不同策略框架下的表现。
  - 如需纳入 registry，需要后续代码/设计迭代，本备忘录暂不展开。

---

## 3. 多策略是否需要专用提示词？

从长期效果角度，**多策略演进非常建议配合专用提示词**，原因包括：

- **策略不同 → 因子/模型关注点不同**：
  - 中长周期多头策略：更关注持有期收益的稳定性、低换手、回撤控制。
  - 高频/短周期策略：更关注短期价量微结构、反应速度、交易成本敏感性。
  - 行业中性 / 市场中性：需要显式控制行业/风格因子暴露，多空配对结构。

- **评价指标偏好不同**：
  - 有的策略以年化超额收益为主目标；
  - 有的更强调最大回撤、波动率、Sharpe、换手等风险/成本指标；
  → 提示词中应写清“优化目标”和“权重”，引导 LLM 生成更契合策略的因子/模型。

- **数据与约束侧重点不同**：
  - 是否允许做空、是否有杠杆、是否有行业/因子暴露上限、是否有仓位/风险预算等约束，宜在 prompt 中明确说明。

结论：

- 策略形态、回测配置、评价指标和提示词之间，应该是一一（或一对多）映射关系，而不是共用一套通用 prompt。

---

## 4. StrategyConfig：策略配置与提示词的统一抽象（设想）

为实现 AIstock 与 RD-Agent 在“多策略 + 提示词”上的统一管理，可以引入结构化的 `StrategyConfig` 与 `PromptTemplate` 抽象。

### 4.1 StrategyConfig（示意 JSON 模型）

```json
{
  "strategy_id": "long_topk50_baseline",
  "name": "多头 Topk50 指数增强",
  "type": "long_only",
  "universe": "CN_STOCK_ALL",
  "benchmark": "000300.SH",
  "horizon": "medium_term",
  "backtest_windows": [
    {
      "train": ["2010-01-07", "2018-12-31"],
      "valid": ["2019-01-01", "2020-12-31"],
      "test": ["2021-01-01", "2025-12-01"]
    }
  ],
  "execution": {
    "strategy_class": "TopkDropoutStrategy",
    "topk": 50,
    "n_drop": 5,
    "rebalance_freq": "day"
  },
  "objectives": {
    "primary": "excess_return",
    "secondary": ["max_drawdown", "turnover"]
  },
  "prompt_template_id": "factor_long_topk50_v1"
}
```

要点：

- `strategy_id`：策略的唯一标识，AIstock 策略库管理用。
- `execution`：与 Qlib YAML 中 `port_analysis_config.strategy` + `backtest` 的关键参数对应。
- `backtest_windows`：定义训练/验证/测试/回测的时间窗口（可扩展为多段）。
- `objectives`：明确优化目标（收益 / 风险 / 成本），为提示词与反馈提供锚点。
- `prompt_template_id`：与提示词模板管理模块绑定，用于选择合适的 prompts。

### 4.2 PromptTemplate（提示词模板管理）

提示词模板管理模块可以按以下方式扩展：

- 以 `prompt_template_id` 为主键，存储一组模板：
  - 背景说明（策略、市场、数据源等）。
  - 接口规范（因子函数签名、输出格式、约束条件等）。
  - 目标与评价指标偏好（如更重视 IC/IR 或更重视回撤/换手）。

示例：

- `factor_long_topk50_v1`：针对长周期多头 Topk50 指数增强策略的因子设计模板。
- `factor_long_topk20_concentrated_v1`：针对持仓更集中的高 conviction 策略。
- `factor_market_neutral_pair_trading_v1`：针对市场中性多空策略等（未来扩展）。

RD-Agent 在构造场景（如 `QlibFactorScenario`）时，根据 `prompt_template_id` 选择对应模板，拼出最终的 system prompt / background / interface / output_format 等。

---

## 5. AIstock 在多策略与提示词管理中的角色设想

AIstock 可以承担“策略定义中心 + 任务调度方”的角色：

1. **策略配置管理**：
   - 维护一张“策略库”：
     - 策略 ID、名称、自然语言描述。
     - 技术参数：universe、benchmark、时间窗口、Topk/n_drop、调仓频率等。
     - 对应的 `prompt_template_id`。

2. **提示词模板映射**：
   - 策略库中的每条策略记录，绑定一个或多个提示词模板 ID。
   - 便于在前端展示：“该策略使用的内部提示词模板版本为 XXX”。

3. **为每次 RD-Agent 演进任务定义目标**：
   - 当用户在 AIstock 中选择某个策略并发起“因子/模型演进”任务时：
     - AIstock 将 `StrategyConfig` + 任务目标（如：提高收益 / 降低回撤 / 降低换手 / 提升某段时间表现）打包发送给 RD-Agent。
   - RD-Agent 根据这些结构化信息：
     - 选用对应 YAML（或基于模板生成临时 YAML）。
     - 选用对应的提示词模板，生成 factor/model 代码。
     - 按策略配置回测，并将结果写入 registry。

---

## 6. 固定 TopkDropout 情况下的演进空间（补充）

即使策略类暂时固定为 `TopkDropoutStrategy`，仍有多维度演进空间：

- **参数层面**：
  - `topk`：集中度调节。
  - `n_drop`：换手率调节。
  - 回测/训练窗口划分：不同市场 regime 的稳健性测试。

- **因子与策略耦合层面**：
  - 在提示词中明确：
    - 更侧重截面排序能力（适用于 Topk 策略）。
    - 更重视成本/换手（鼓励稳健低频信号）。
  - 在反馈中增强对成本、换手、回撤等指标的权重，引导 LLM 生成更贴合当前策略特性的因子。

- **多策略版本的准备**：
  - 先通过 YAML + PromptTemplate 设计多套“候选策略 + 提示词”组合；
  - 等未来需要时，再逐步在 RD-Agent 中将这些策略纳入自动 pipeline 和 registry 体系。

---

## 7. 后续演进方向（仅做占位，具体方案待设计）

- 设计正式的 `StrategyConfig` 与 `PromptTemplateConfig` Schema：
  - 明确哪些字段由 AIstock 管理，哪些字段由 RD-Agent 消费。
  - 明确与 Qlib YAML 的映射关系（包括「多 YAML / 多回测窗口」的映射）。

- 规划“多策略自动回测 + 结果入库”的版本路线：
  - v1：手工 YAML + 手工 qrun（当前）。
  - v2：AIstock 通过 API 下发 StrategyConfig，RD-Agent 生成/选择 YAML，但暂不改变 registry 结构，仅产出额外 artifacts。
  - v3：将多策略回测结果统一纳入 registry schema（增加策略维度字段、artifacts 命名规范等）。

- 建立策略/提示词版本管理机制：
  - 便于在回溯某次因子/模型实验时，明确其所处的“策略配置 + 提示词版本”组合。

本备忘录仅作为方向性整理，具体接口、Schema 与实现细节可在后续设计阶段进一步细化。
