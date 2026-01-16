# RD-Agent + AIstock + Qlib 集成问题与方案备忘录

> 更新时间：2025-12-07

---

## 1. 场景与目标概述

- **RD-Agent**：自动化研究代理，用 Qlib 作为量化平台，结合 CoSTEER 知识库迭代策略与因子。
- **AIstock snapshot**：`C:\Users\lc999\NewAIstock\AIstock\qlib_snapshots\qlib_export_20251206`
  - 核心文件：`daily_pv.h5`（日线价量行情）、`instruments/all.txt`（全市场股票列表）。
- **目标**：
  - 让 RD-Agent 在 **完全基于 AIstock 数据** 的前提下，完成因子研发、模型训练和回测；
  - 避免依赖本地/线上其它 Qlib 数据集；
  - 支持未来扩展到美股、资金流向、筹码峰等 **Qlib 原生不支持的数据**。

---

## 2. 关键错误与日志现象

### 2.1 Dataset 为空：Alpha158 / DatasetH 下游拿不到数据

- 现象：`debug_dataset_pipeline.py` 输出：

  ```text
  raw train shape: (0, 21)
  raw valid shape: (0, 21)
  raw test shape:  (0, 21)
  ```

- 意味着：
  - 无论是否使用 processors（DropNaLabel / CSRankNorm），
  - **Alpha158 handler 在 train/valid/test 三个 segment 上一开始拿到的就是空 DataFrame**；
  - 问题不在后处理，而在最源头的数据获取（handler.fetch）。

### 2.2 `D.list_instruments("all")` 抛出 TypeError

- 在 `debug_qlib_features.py` 中：

  ```python
  from qlib.data import D
  D.list_instruments("all")
  ```

- 报错：

  ```text
  TypeError('string indices must be integers')
  ```

- 与此同时，AIstock snapshot 中的 `instruments/all.txt` 内容格式为：

  ```text
  000001.SZ\t2010-01-07\t2025-12-01
  000002.SZ\t2010-01-07\t2025-12-01
  ...
  ```

- 该格式 **本身符合 Qlib instruments 文本规范**（`code TAB start_date TAB end_date`）。

### 2.3 LLM 生成因子脚本时的列名 KeyError

- 现象：
  - 早期 LLM 生成的因子代码里引用了 `'$close'` 等列；
  - AIstock 的 HDF5 实际列为 `close, open, high, low, volume, amount, ...`；
  - 导致 `KeyError: '$close'`。

- 已处理：
  - 在 prompts 中明确要求“去掉 `$` 前缀，使用不带 `$` 的列名”；
  - 因子脚本中统一直接使用 `close` 等列名。

### 2.4 instruments / market 命名不一致

- 现象：
  - 早期 YAML 中出现 `market: aistock`、`instruments: aistock`；
  - 实际 AIstock snapshot 下只有 `instruments/all.txt`；
  - 会导致 Qlib 尝试读取不存在的 `aistock.txt` 等文件。

- 已处理：
  - 统一将 RD-Agent Qlib 场景下的 `market` / `instruments` 改为 `all`；
  - 但由于 Qlib 对 `"all"` 的解析仍未配置到这套 snapshot 上，`D.list_instruments("all")` 仍然报错。

---

## 3. 问题诊断过程与中间结论

### 3.1 为什么 Dataset 一开始就是空？

1. 使用 `debug_dataset_pipeline.py` 直接打印：
   - raw data（handler 输出）、
   - processor 之后的数据尺寸。

2. 发现：
   - `raw train/valid/test` 全部是 `(0, 21)`；
   - 不论是否开启 DropNaLabel / CSRankNorm，结果都一样；

3. 由此判断：
   - **问题在于 handler 无法基于当前 Qlib 配置从 snapshot 中取到任何数据**，
   - 而不是处理器把数据“洗光”。

### 3.2 `D.list_instruments("all")` TypeError 的根因

1. `instruments/all.txt` 已存在且格式正确；
2. 调用 `D.list_instruments("all")` 却报 `TypeError('string indices must be integers')`；
3. 结合 Qlib 的内部实现推断：
   - `"all"` 被当成某种结构（dict/配置项）来索引；
   - 但实际拿到的是一个纯字符串，导致按 `[...]` 取字段时抛 `TypeError`；
4. 中间结论：
   - **当前这套 Qlib 初始化时，并没有把 `market="all"` 正确映射到 `instruments/all.txt`**；
   - 虽然 `all.txt` 在文件系统里存在，但 Qlib **逻辑上的“all 市场”并未配置好**。

### 3.3 AIstock snapshot 与“完整 Qlib bin 数据”的差异

通过现象反推：

- AIstock snapshot 当前提供：
  - `daily_pv.h5`：原始或半加工的日线价量数据；
  - `instruments/all.txt`：标的列表；
- 没有看到典型 Qlib bin 目录应有的结构：
  - `calendars/`（交易日历）、
  - `features/`（Alpha158 等特征存储）、
  - `fields` 的多级目录等。

因此可以判断：

- 这套 snapshot **不是 Qlib 官方 `dump_bin` 输出的完整数据目录**；
- 它更像是：
  - “给自定义 factor.py / HDF5 读取用的原始行情库”；
  - 而不是“供 Alpha158 / DatasetH / D.features 直接食用的标准 Qlib 数据栈”。

### 3.4 对 RD-Agent 当前 Qlib 场景的影响

综合以上：

- **Alpha158 / DatasetH / baseline LGBModel 等默认流水线在当前 AIstock snapshot 上不可用**：
  - handler 拿不到数据 → Dataset 为空 → 训练/回测都失败；
- 但 **基于 `daily_pv.h5` 的自定义因子计算（CoSTEER 生成 factor.py）已验证可用**：
  - 如 `Momentum_10D` 等价量类因子可以成功算出结果；
- 因此：
  - RD-Agent 的“因子研发能力”在方案 A 路线下是可用的；
  - 只是 **不能依赖 Alpha158 这一套官方因子与数据处理流水线**。

---

## 4. 方案分析：A / B 两条大路线

### 4.1 方案 A：维持自定义数据格式（HDF5/Parquet）+ 自己的因子/模型

**定义**：

- 各类数据（AIstock 日线、美股、资金流向、筹码峰等）均按自定义 schema 存在 HDF5/Parquet 中；
- 因子脚本（`factor.py` 等）直接使用 pandas 读取数据、merge、计算指标；
- 模型训练可以：
  - 使用 sklearn / LightGBM / PyTorch 直接在 pandas 上训练；
  - 或通过简单适配，使用 Qlib 部分模型组件，但不依赖 DatasetH/Alpha158。

**优点**：

- **灵活性极高**：
  - 可轻松支持 Qlib 原生没有的数据，比如资金流向、筹码峰、外部指数、新闻情绪等；
  - 数据结构不受 Qlib bin 格式约束，适合复杂/稀疏/高维数据。
- **已验证链路可用**：
  - 目前 `daily_pv.h5 + factor.py` 已经能够稳定输出因子。
- **调试简单**：
  - 问题集中在 pandas 级别，而不是 Qlib 内核。

**缺点**：

- **无法直接使用 Alpha158 / Alpha360 等内置因子库**；
- Qlib 的 DatasetH / qrun / 标准回测流水线利用率较低，需要自己接一层 glue code。

**风险**：

- 如果没有统一数据 schema，未来多数据源（A 股、美股、资金流、筹码）容易出现 join/对齐混乱；
- 需要自己制定并坚持内部命名规范（时间列名、标的列名、频率、缺失值处理等）。

### 4.2 方案 B：把所有数据标准化为 Qlib bin 格式

**定义**：

- 将 AIstock、未来美股、资金流、筹码等数据，尽可能映射为 Qlib 官方支持的数据形态：
  - 价格时间序列 → 走 `dump_bin.py` 生成多市场、多频率数据；
  - 扩展特征 → 自定义 Handler，将外部数据作为 fields 写入或按需加载。

**优点**：

- **高度复用 Qlib 生态**：
  - 可以直接使用 Alpha158/Alpha360 等官方因子；
  - DatasetH、ModelZoo、回测引擎、策略配置都高度标准化；
  - 多项目、多团队共享配置和数据更方便。

**缺点**：

- **前期成本高**：
  - 每多一种“非典型数据”，都要考虑如何抽象为适合 Qlib handler 的形态；
  - 筹码、资金流等数据并不天然适合塞进日度面板，可能需要大量 glue code；
- **心智负担大**：
  - 需同时理解业务数据和 Qlib 抽象；
  - Qlib 更新时，定制部分可能要适配。

**风险**：

- 数据量一旦扩展到全市场+多频率，bin 生成和维护成本高；
- 万一 Qlib 内部改动较大，已有 handler 可能需要重写或迁移。

---

## 5. 回测相关的路径设计

### 5.1 方案 A 下是否还能用 Qlib 回测？

- **严格的方案 A（完全不生成 Qlib bin 行情）**：
  - Qlib 的 DataProvider 无法读到行情；
  - Qlib 回测引擎无法计算每日盈亏；
  - 只能用自写回测框架或其他第三方工具。

- **折中：方案 A + 轻量 B（推荐）**：
  - 为 **价格数据** 单独生成一份“最小 Qlib 行情目录”：
    - 只包含日线（后续可扩展到分钟）、必要的价量字段和 instruments/calendar；
  - 因子/信号计算全部在 Qlib 外完成，形成 `date, instrument, target_weight/signal`；
  - Qlib 只负责根据“信号 + 行情”做回测和绩效评估。

### 5.2 不用 Qlib 回测，RD-Agent 还能演进吗？

- 如果完全不用 Qlib，又 **不给 RD-Agent 任何“机器可读的回测指标”**：
  - RD-Agent 无法自动判断策略好坏，只能做静态因子/代码生成；
  - “自动演进”功能基本失效。

- 如果使用 **自写回测脚本 + 统一指标输出（JSON/CSV）**：
  - RD-Agent 完全可以：
    1. 生成/修改策略 & 信号生成代码；
    2. 调用你的回测脚本；
    3. 解析 JSON/CSV 中的 `ann_return, max_drawdown, sharpe, ic, ...`；
    4. 基于这些指标做效果判断和下一步优化建议。
  - 本质上，只要有稳定的“实验 → 指标文件”接口，RD-Agent 的演进能力就可以维持。

### 5.3 分钟级策略的扩展

- 若未来要测试分钟级策略（例如 1min/5min）：
  - **只需要将分钟行情转成 Qlib 可读的分钟级数据目录**：
    - 分钟级 calendar、instruments、features（含价量字段）；
    - 在配置中设置 `freq: 1min`；
  - 因子（包括日内资金流、微观结构等）仍可以在 Qlib 外部计算；
  - 最终回测仍是：
    - Qlib 读分钟线 + `datetime, instrument, target_weight` 信号；
    - 完成分钟级撮合与绩效分析。

---

## 6. 关于“因子变化 & 买卖点触发”在回测中的实现

### 6.1 实现方式（与 Qlib 是否参与无关）

典型实现流程：

1. **计算因子时间序列**（在 Qlib 外部）：

   ```python
   panel = merge_on_date_instrument(price_df, flow_df)
   panel["flow_factor"] = panel["main_inflow"] / panel["float_mv"]
   panel["flow_change"] = panel.groupby("instrument")["flow_factor"].diff()
   ```

2. **写触发规则，生成信号**：

   ```python
   panel["signal"] = 0
   panel.loc[panel["flow_change"] > 0.01, "signal"] = 1   # 买入
   panel.loc[panel["flow_change"] < -0.01, "signal"] = -1  # 卖出/减仓
   ```

3. **将信号转换为持仓/权重时间序列**：

   ```python
   weights = build_target_weight_from_signal(panel)

- **逻辑生成/修改**：
  - 根据自然语言需求，生成或调整上述“因子 + 信号”代码；
  - 例如加入止损、持仓上限、多因子打分等逻辑。

- **效果评估与演进**：
  - 读取回测结果（无论来自 Qlib 还是自写框架）；
  - 依据收益/风险/稳定性指标做判断，提出下一轮实验修改建议；
  - 形成“代码 → 回测 → 评价 → 改代码”的闭环。

### 6.3 关于 10 日动量因子实现的两个注意点

- **[注意点 1] `shift(10)` 近似 10 个交易日**：
  - 当前 10 日动量因子实现采用：
    - `prev_close = df.groupby(level="instrument")["close"].shift(10)`
    - `Momentum_10D = df["close"] / prev_close - 1`
  - 在按日期排序的日线数据中，这对应“向前回看 10 行数据”，在大多数股票上等价于回看 10 个交易日；
  - 对于部分上市初期样本（历史不足 10 条）或存在个别日期缺失的情况，该实现是按“数据行”近似 10D，而非严格意义上的 10 个交易日；
  - 在当前 AIstock+Qlib 环境下：
    - 日历已按交易日对齐，ST/退市股票已在导出阶段剔除，股票池整体连续；
    - 因而这种近似在实际影响上可以接受，后续如需更严谨的“精确 10 交易日”实现，可再基于 Qlib calendar 或完整日历对齐改进。

- **[注意点 2] 前 10 日的 NaN 保留是预期行为**：
  - 对每只股票，前 10 个交易日缺少 `t-10` 的价格，因此 `Momentum_10D` 必然为 NaN；
  - 当前实现直接输出完整时间范围的因子序列（包含这些 NaN 行），以便在 Qlib 中与其他因子和标签对齐；
  - 下游 DatasetH / DataHandlerLP / processors（如 `DropnaLabel`、`Fillna` 等）会在训练与回测阶段统一处理这些 NaN；
  - 因此，因子结果中存在首 10 日的 NaN 并非实现错误，而是有意保留的行为特性。

---

## 7. 综合结论与推荐路线

### 7.1 关键结论汇总

- **当前 AIstock snapshot 不是完整 Qlib bin 数据**：
  - 尽管有 `daily_pv.h5` 和 `instruments/all.txt`，
  - 但缺少标准化的 calendar/features 结构，
  - Qlib 的 Alpha158 / DatasetH 无法正常运行，`D.list_instruments("all")` 也未成功配置。

- **RD-Agent 的因子研发能力在 AIstock HDF5 上是可用的**：
  - 通过 CoSTEER 生成 factor.py，直接用 pandas 读 HDF5 计算因子可以成功；
  - 列名问题（`'$close'` vs `close`）已通过 prompt 规范解决。

- **回测与策略演进的关键依赖是“稳定的指标输出”，而非 Qlib 本身**：
  - 只要有统一格式的回测结果（JSON/CSV），RD-Agent 就能使用它来驱动策略迭代；
  - Qlib 是一个强大的回测/评估平台，但不是唯一选择。

### 7.2 推荐路线（当前阶段）

1. **短期：采纳“方案 A + 轻量 B”的混合策略**：
   - 上游因子/特征：
     - 全部在 Qlib 外部用 HDF5/Parquet + pandas 计算，保持高自由度；
   - 下游回测：
     - 为 AIstock 日线行情构建一份“最小 Qlib 日线数据目录”，仅含必要价量信息；
     - 用 Qlib 回测“日线级策略”（包括资金流/筹码等因子生成的信号）。

2. **中期：为 RD-Agent 接入统一“回测接口”**：
   - 无论底层是 Qlib 回测，还是自写 backtest，引擎都输出统一格式的指标文件；
   - RD-Agent 只依赖这份指标文件做策略评价与演进；
   - 这样未来替换/并行多种回测引擎也更灵活。

3. **长期：按需要选择性地向完整方案 B 靠拢**：
   - 若未来要大规模复用 Alpha158/Alpha360 等 Qlib 生态：
     - 再考虑将部分通用数据（各市场日/分钟价量）按 Qlib bin 标准全量构建；
   - 对于高度非结构化的数据（筹码分布、高频盘口、资金流分解），
     - 依然可以坚持“在 Qlib 外部计算因子 + 只把信号/权重喂给 Qlib 回测”的方式。

---

## 8. AIstock Qlib bin 导出与 RD-Agent 长周期实验验证

### 8.1 AIstock 侧 Qlib bin 导出（含 ST / 退市过滤）

- 通过 AIstock `/qlib` 页面新增的 CSV→bin 导出功能，完成了从 TimescaleDB 到 Qlib bin 的全链路：
  - 导出日期区间：`2020-01-06 ~ 2025-12-01`；
  - 交易所范围：上交所 (SH) / 深交所 (SZ) / 北交所 (BJ) 全部勾选；
  - 样本过滤：
    - 勾选【排除所有有过 ST 记录的股票（包括当前 ST）】；
    - 勾选【排除退市 / 当前暂停上市的股票】；
  - 导出结果：
    - Snapshot ID: `qlib_bin_20251209`；
    - CSV 目录: `C:\Users\lc999\NewAIstock\AIstock\qlib_csv\qlib_bin_20251209`；
    - bin 目录: `C:\Users\lc999\NewAIstock\AIstock\qlib_bin\qlib_bin_20251209`；
    - `dump_bin.py dump_all` 执行成功，未再出现 symbol 转 float 等错误；
    - `check_data_health.py --qlib_dir ... --freq day` 健康检查通过。
- Qlib 健康检查脚本输出的样本：
  - `instrument='DAILY_ALL'`，`datetime` 从 `2020-01-06` 到 `2025-12-01`；
  - 共 1431 行，包含 `open, high, low, close, volume, factor` 六列，价量无明显缺失。

### 8.2 RD-Agent 侧 Qlib bin 集成与长周期回测验证

1. 修改 RD-Agent Qlib 场景 YAML 的数据目录：
   - 将如下模板的 `qlib_init.provider_uri` 改为指向 AIstock 生成的 bin：
     - `rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml`
     - `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml`
     - `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml`
     - `rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml`
     - `rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml`
   - 统一设置：
     ```yaml
     qlib_init:
         provider_uri: "/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209"
         region: cn
     ```

2. 为当前这份 bin 数据配置合理的时间段：
   - `data_handler_config`：
     ```yaml
     start_time: 2020-01-06
     end_time: 2025-12-01
     fit_start_time: 2020-01-06
     fit_end_time: 2021-12-31
     instruments: all
     ```
   - `segments`（以 `conf_baseline.yaml` 为例）：
     ```yaml
     train: [2020-01-06, 2023-12-31]
     valid: [2024-01-01, 2024-12-31]
     test:  [2025-01-01, 2025-12-01]
     ```
   - `benchmark` 由于当前数据集中只有 `DAILY_ALL`，因此从 `SH000300` 改为：
     ```yaml
     market: &market all
     benchmark: &benchmark DAILY_ALL
     ```

3. 解决 Qlib 回测在最后一个交易日的日历越界问题：
   - 在 `port_analysis_config.backtest` 中，将回测结束日期从包含最后一个交易日改为略提前：
     ```yaml
     backtest:
         start_time: 2021-01-01
         # 当前 day 频数据最后一行是 2025-12-01，为避免 calendar_index+1 越界，将回测截止日设置到 2025-11-28
         end_time: 2025-11-28
     ```

4. 在与 RD-Agent 相同的 Conda 环境中运行 Qlib 实验：
   ```bash
   conda activate rdagent-gpu
   cd /mnt/c/Users/lc999/NewAIstock/AIstock/RD-Agent-main

   # 基线因子实验（Alpha158 + LGBModel）
   qrun rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml
   ```

5. 本次实验的关键日志与结论：
   - Qlib 初始化：
     - `qlib successfully initialized based on client settings.`
     - `data_path={'__DEFAULT_FREQ': PosixPath('/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209')}`
   - 数据加载与处理：
     - `Loading data Done` 后，`FilterCol / RobustZScoreNorm / Fillna / DropnaLabel / CSRankNorm` 均正常执行；
   - 模型训练：
     - LGBModel 成功完成训练，虽然当前超参较简单（早停在第 1 轮），但说明训练/验证数据管道已经正常；
   - 预测与信号：
     - `Signal record 'pred.pkl' has been saved`，能看到 2025 年测试集上的若干 `DAILY_ALL` 预测分数；
   - 回测：
     - `backtest loop` 在 2021-01-01 ~ 2025-11-28 区间顺利完成；
     - 生成 `port_analysis_1day.pkl` 与 `indicator_analysis_1day.pkl` 两个核心 artifact；
   - 指标示例（单标的 DAILY_ALL 回测，仅作链路验证）：
     - 基准（DAILY_ALL）日频：
       ```text
       annualized_return  ≈ 0.0948
       information_ratio  ≈ 0.50
       max_drawdown       ≈ -0.39
       ```
     - 策略超额收益（含/不含成本）也顺利输出，但由于当前策略与因子配置较简陋，年化与 IR 为负，仅作为“回测正常运行”的证明。

6. 综合结论：
   - 在不修改 RD-Agent 源码逻辑的前提下，仅通过：
     - 在 AIstock 侧导出带 ST/退市过滤的 Qlib bin 数据集；
     - 在 RD-Agent Qlib 场景 YAML 中更新 `provider_uri` / 时间区间 / 基准；
   - 已经成功让 RD-Agent 使用 AIstock 作为唯一数据源，完成 Alpha158 + LGBModel 的长周期实验和完整回测流程。

### 8.3 bin 与 HDF5 在当前方案中的分工

- **Qlib bin 目录（`qlib_bin_20251209`）**：
  - 作为 RD-Agent + Qlib 的“标准日线行情视图”，用于：
    - 运行 Alpha158 / LGBModel / GeneralPTNN 等默认 Qlib 实验；
    - 在 DatasetH + 回测引擎上完成长周期回测与绩效评估。
  - 特点：结构固定、读写高效、适合作为统一回测底座，但不承载复杂业务侧数据。

- **HDF5/Parquet 数据集（`daily_pv.h5` 及未来扩展）**：
  - 作为 CoSTEER 因子研发的“原始/中间视图”，用于：
    - 在 pandas 层灵活构造各类新因子（价量、波动、横截面回归等）；
    - 未来接入资金流、筹码峰、新闻情绪等 Qlib 原生未覆盖的数据源。
  - 当前 `daily_pv.h5` 与 `qlib_bin_20251209` 在 **时间与股票池范围上已经对齐**：
    - 时间区间均覆盖 2010–2025；
    - 股票集合与全市场一致，便于“在 HDF5 上做因子、在 bin 上做回测”的一一映射。

- **未来扩展方向**：
  - 在 HDF5/Parquet 侧按需新增多组主题数据，如：
    - `capital_flow_daily.h5`（资金流入/流出）、`chip_profile_daily.h5`（筹码峰）、`fundamental.parquet` 等；
  - CoSTEER 因子脚本负责在这些 HDF5/Parquet 上完成多源数据对齐和因子计算；
  - Qlib bin 目录继续专注于“撮合 + 收益曲线计算”，消费由因子/信号脚本产出的信号或权重。

### 8.4 本次集成版本提交记录

- **Git 分支与提交信息**：
  - 分支：`main`
  - 提交信息：`Integrate AIstock Qlib bin 20251209 with RD-Agent and align Qlib experiments`
  - 推送到远程：`origin/main`

- **本次提交涵盖的关键内容**：
  - **Qlib bin 集成与配置**：
    - 在 RD-Agent Qlib 实验模板 YAML（`conf_baseline*`、`conf_combined_factors*`、`conf_*factors_model.yaml`）中统一：
      - `provider_uri: /mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209`
      - `market: all` 与 `benchmark: DAILY_ALL`
      - 数据起止时间统一到 `2010-01-07 ~ 2025-12-01`，回测截止日期为 `2025-11-28`，避免 calendar 越界。
  - **AIstock 导出链路与配置管理**：
    - 新增 Qlib CSV→bin 导出与 ST/退市过滤逻辑（`backend/qlib_exporter/*`、`backend/infra/wsl_qlib_runner.py`）。
    - 将 WSL/Conda/QLIB 路径配置纳入后端配置管理与前端 `/config` 页面，支持通过 UI 管理 `QLIB_BIN_ROOT_WIN` 等环境变量。
  - **前端与调度扩展（与本备忘录主题直接相关部分）**：
    - `/qlib` 页面支持从 AIstock 一键导出 HDF5 与 bin，并在 UI 上暴露 ST/退市过滤选项。
    - `/local-data`、`/scheduler`、`/smart-monitor` 等页面为后续 RD-Agent + AIstock 全流程接入预留了入口和布局。

- **注意事项**：
  - 大体量 CSV 目录 `qlib_csv/` 已通过 `.gitignore` 排除，避免超过 GitHub 单文件 100MB 限制；
  - 当前远程仓库中的 `qlib_bin/qlib_bin_20251209` 目录可作为 RD-Agent + Qlib 长周期回测的标准基准数据源，后续新的 snapshot 建议使用新目录名保留历史版本。

---

## 9. 后续可执行的具体工作（建议）

> 以下是建议的下一步行动，不代表已经在代码中实现。

- **[建议 1] 设计统一数据 schema 与 loader**：
  - 统一 `instrument` / `date` / `datetime` / `close` 等字段命名；
  - 为每类数据（价量、资金流、筹码、美股等）写一个小型 loader，而不是在每个 factor.py 里随意 `read_hdf`.
  - 为每类数据（价量、资金流、筹码、美股等）写一个小型 loader，而不是在每个 factor.py 里随意 `read_hdf`。

- **[建议 2] 构建“最小 Qlib 日线行情目录”**：
  - 从 `daily_pv.h5` 提取 `close/open/high/low/volume/amount`；
  - 生成符合 Qlib 要求的 calendar/instruments/features；
  - 在 RD-Agent Qlib 场景 YAML 中将 `provider_uri` 指向该目录。

- **[建议 3] 定义统一回测结果格式**：
  - 例如固定为一个 JSON：`{"ann_return": ..., "max_drawdown": ..., "sharpe": ..., ...}`；
  - RD-Agent 只依赖这份 JSON 来做策略好坏判断和演进决策。

- **[建议 4] 如需分钟级策略，再扩展分钟级 Qlib 行情目录**：
  - 先在少量标的/时间段上做 POC，评估数据体量和回测性能；
  - 保持“因子外部计算 + Qlib 只负责撮合与评估”的思路不变。

---

## 9. 升级选项：让 RD-Agent 自动编排外部因子/信号脚本

### 9.1 升级目标概述

- 在“方案 A + 轻量 B”的基础上，进一步升级为：
  - 由 RD-Agent/CoSTEER 自动生成和维护一个外部因子/信号脚本（例如 `generate_signals.py`）；
  - 该脚本负责：
    - 从各类 HDF5/Parquet 数据源加载价量、资金流、筹码、美股等数据；
    - 计算自定义因子；
    - 根据因子变化和策略规则生成 `signal/target_weight` 时间序列；
  - 回测引擎（Qlib 或自定义）只需要消费：
    - 一份“最小 Qlib 行情目录”（日线/分钟线价量）；
    - 一份外部脚本输出的信号文件（CSV/Parquet）。

### 9.2 相对现有方案的优势

- **优势 1：彻底解耦 Qlib 数据格式**
  - 因子和信号计算完全在外部脚本中进行：
    - 可以任意组合 AIstock 行情、美股指数、资金流向、筹码峰、新闻情绪等数据；
    - 不必为每个数据源写 Qlib Handler 或强制塞进 Qlib bin 结构。
  - Qlib 只负责“撮合与绩效评估”，数据接入逻辑高度自由。

- **优势 2：策略逻辑表达能力很强**
  - 外部脚本可以使用完整 Python 生态：pandas、numpy、scipy、sklearn、PyTorch 等；
  - 易于实现：
    - 多因子打分、风险约束、仓位管理、分层持仓等复杂逻辑；
    - 跨市场联动（A 股策略中使用美股/指数因子）、跨资产策略；
    - 滚动窗口、分组回归、时间序列模型等。

- **优势 3：RD-Agent 真正参与“写逻辑”而不仅是改 YAML**
  - 通过自然语言描述因子和买卖规则，RD-Agent 可以：
    - 自动生成/修改 `generate_signals.py` 中的因子与信号部分；
    - 根据回测结果自动调整条件、阈值、特征组合；
  - 让 CoSTEER 的“策略演进能力”扩展到完整的因子+信号链路，而不仅是 Qlib handler 配置层。

- **优势 4：未来替换或并行多回测引擎更容易**
  - 只要所有回测后端遵守同一输入/输出协议：
    - 输入：行情 + 信号文件；
    - 输出：统一格式的指标 JSON/CSV；
  - RD-Agent 只需要调用不同的“回测适配器”，上层演进逻辑几乎无需改动。

### 9.3 相对现有方案的劣势与代价

- **劣势 1：复杂度明显高于“只改 YAML”**
  - 当前 RD-Agent 主要负责改 YAML 和少量因子配置；
  - 引入外部脚本自动编排后，需要：
    - 维护脚本骨架与数据约定；
    - 处理更多运行时错误（数据对齐、空值、类型等）。

- **劣势 2：调试难度增加**
  - 错误来源从 Qlib handler/YAML 扩展到：
    - 数据源加载失败；
    - merge 对齐错误；
    - 不同频率/日历错位；
  - 需要在脚本中设置清晰的日志与中间结果检查点。

- **劣势 3：结果复现成本提高**
  - 每一轮实验可能对应一个略有不同的 `generate_signals.py`；
  - 若不做版本管理和快照（例如按实验 ID 存脚本与配置），后续复现会变得困难。

### 9.4 初步工作量评估

相对“仅 Qlib YAML 改动”的当前模式，增加的工作大致包括：

- **[W1] 定义统一的信号文件与回测结果格式**
  - 信号文件：如 `date/datetime, instrument, target_weight` 或 `signal_score`；
  - 回测结果：统一用 JSON/CSV 包含 `ann_return, max_drawdown, sharpe, turnover, ic` 等指标；
  - 这是 RD-Agent 与回测引擎之间的“接口契约”，应尽量长期稳定。

- **[W2] 设计 `generate_signals.py` 标准骨架**
  - 例如包含：
    - `load_xxx_data()`：加载各类 HDF5/Parquet 数据；
    - `compute_factors(df)`：计算基础因子；
    - `generate_signals(df)`：根据因子与规则产出信号/权重；
    - `main()`：串联并输出信号文件；
  - RD-Agent 主要在 `compute_factors` 和 `generate_signals` 两个区域内做自动修改。

- **[W3] 在 RD-Agent Qlib 场景中插入“外部脚本执行”步骤**
  - 新的执行顺序：
    1. 生成/修改 YAML；
    2. 调用 `python generate_signals.py` 生成信号文件；
    3. 调用 Qlib 回测（或自定义回测）；
    4. 读取回测结果 JSON/CSV 供 CoSTEER 使用。

- **[W4] 加强日志与错误捕获**
  - 为 `generate_signals.py` 统一日志格式；
  - 在 RD-Agent 执行链中捕获：
    - 数据加载/merge 错误；
    - 信号文件缺失或格式不符；
    - 回测失败信息；
  - 给 LLM 足够的上下文，支持自动诊断并修复脚本。

整体看，这是一轮“中等偏大型”的工程升级，不是几行 patch 能完成，但可以分阶段实现并逐步验证。

---

## 10. 升级选项对未来合并上游 RD-Agent 仓库的影响

### 10.1 三种改动方式及影响

从“如何影响未来与上游仓库同步”的角度，可以把升级方案分成三种实现路径：

1. **方式 A：完全在仓库外/脚本层做适配（影响最小）**
   - 不改动 `rdagent/` 源码，仅在 `scripts/` 或其他目录新增：
     - `scripts/generate_signals.py`
     - `scripts/run_backtest_with_signals.py` 等；
   - 用人工或简单脚本把这些与 RD-Agent 原有 Qlib 场景串联起来；
   - 上游更新时：
     - 绝大多数改动集中在 `rdagent/`，你新增的 `scripts/` 不受影响；
     - 基本不会有 merge 冲突。

2. **方式 B：在 RD-Agent 内新增模块 + 轻量修改场景 executor（中等影响，较推荐）**
   - 在 `rdagent/` 下新增如：
     - `rdagent/custom_signal_pipeline/loader.py`
     - `rdagent/custom_signal_pipeline/executor.py`；
   - 在 Qlib 场景的执行链中，仅做一层“插针”：
     - 原来：`生成 YAML → qrun → 读结果`；
     - 现在：`生成 YAML → generate_signals.py → qrun → 读结果`；
   - 如果通过配置开关控制是否启用外部信号（默认保持原行为），则：
     - 上游更新大多不会影响新增模块；
     - 偶尔需要在场景 executor 上解决冲突，但范围有限且可控。

3. **方式 C：深度修改 RD-Agent 核心执行逻辑（影响最大，不建议）**
   - 大面积改动通用调度器、公共工具模块等；
   - 将外部信号逻辑硬编码进大量现有代码；
   - 后果：
     - 几乎每次从上游拉取更新都会出现大量冲突；
     - 一旦上游进行大版本重构，你的定制部分可能需要重写。

### 10.2 整体影响评估

- 若采用 **方式 A（外部脚本）**：
  - 几乎不影响与上游同步，merge 冲突风险极低；
  - 但 RD-Agent 本身对外部脚本“不可见”，自动化程度有限。

- 若采用 **方式 B（新增模块 + 轻量 hook）**：
  - 是在“自动化能力”与“上游兼容性”之间较好的折中：
    - 能在 RD-Agent 内部集成外部信号流水线；
    - 冲突主要集中在少量 executor 文件，人工处理成本可接受。

- 若采用 **方式 C（重度 fork）**：
  - 长期维护成本高，对上游依赖会大幅减弱；
  - 不符合当前“希望持续跟进上游 RD-Agent”的诉求，一般不推荐。

### 10.3 实践建议

- **建议 1：优先考虑方式 B，但设计为“增强而非替代”**
  - 通过配置项控制是否启用外部信号流水线；
  - 默认保留原有 Qlib 场景完整可用，以便与上游行为保持一致。

- **建议 2：将改动集中在少数入口文件和新模块中**
  - 尽量“多加文件、少改旧文件”；
  - 把自定义逻辑都包在独立的 package 下，避免污染通用工具代码。

- **建议 3：维护一个“纯上游”主干分支**
  - 例如：
    - `upstream-main`：只跟进官方 RD-Agent；
    - `aistock-custom`：在此基础上叠加 AIstock + 外部信号的定制；
  - 每次更新流程：
    - 先把上游同步到 `upstream-main`；
    - 再 merge 到 `aistock-custom`，仅在少量文件上解决冲突。

整体结论：
- 在合理设计的前提下，“自动编排外部因子/信号脚本”这一升级选项，
- 对未来跟进上游 RD-Agent 仓库的影响可以控制在 **中等、可接受** 的水平，
- 不必演化成一个难以维护的重度 fork。

---

## 10.x 本次集成与实验中的典型错误与防范要点（补充）

- **[坑 1] StaticDataLoader 直接读取 HDF5 导致 `UnpicklingError: 'H'`**
  - 表现：`StaticDataLoader` 的 `config` 指向 `result.h5`，Qlib 内部用 `pickle.load` 打开 HDF5 文件头 `HDF5...`，抛出 `invalid load key, 'H'`。
  - 防范：预计算因子脚本统一输出 `result.h5 + result.pkl + result.parquet`，YAML 中一律指向 `result.pkl`，绝不直接引用 `.h5`。

- **[坑 2] WSL 下输出路径未做 C:/ → /mnt/c/ 映射**
  - 表现：输入路径已做 `C:/` → `/mnt/c/` 映射，但输出仍用 `Path("C:/...")`，在 WSL 内会写到本地文件系统的 `C:/...`，`/mnt/c/...` 下看不到文件。
  - 防范：输出路径与输入同样使用字符串 + 判断 `os.name != "nt"` 的方式做映射，统一写到 `/mnt/c/...` 对应的 Windows 盘挂载点。

- **[坑 3] 因子输出索引未遵守 MultiIndex(datetime, instrument) contract**
  - 表现：
    - LLM 生成的因子实现中，先用 `groupby(level="instrument").rolling(...)`，再随意 `reset_index(level=0, drop=True)`，导致结果索引从两层 MultiIndex 变成单层 Index；
    - 后续又手工写 `result_df.index.names = ['datetime', 'instrument']`，pandas 抛出 `ValueError: Length of new names must be 1, got 2`；
    - 或 rolling 后索引层级顺序变为 `(instrument, datetime)`，下游 Qlib 在按日期切片时触发 `Not allowed to merge between different levels` / `UnsortedIndexError`。
  - 防范：
    - 明确 **因子输出 contract**：所有预计算因子表（含 factor.py 产物）必须满足：
      - 索引为 `MultiIndex(datetime, instrument)`，且已按索引排序；
      - 列为一列或多列浮点因子，命名遵守统一前缀（如 `mf_*`、`db_*`、`ae_*` 等）。
    - 实现时：
      - 尽量使用 `result_df.index = df.index`、`result_df.index.names = df.index.names`，而不是手写 `['datetime', 'instrument']`；
      - 如需 `groupby + rolling`，推荐模式：
        1. `series = df['x'].groupby(level='instrument').rolling(window=K, min_periods=K).func(...)`；
        2. `series = series.reset_index(level=0, drop=True)` 恢复与原索引对齐；
        3. `result_df = pd.DataFrame(index=df.index); result_df['FACTOR'] = series; result_df.index.names = df.index.names`；
      - 避免在结果上重复 `reset_index(drop=True)` 或随意 `swaplevel` 导致索引结构与原始数据不一致。

### 10.y 因子脚本模板与 LLM 创新空间的权衡

- **固定的是“协议”，不是“算法”**
  - 为避免上面的索引类错误，推荐为所有因子脚本提供统一骨架：
    - 读取与整理数据（`read_hdf`、重命名、`df = df.sort_index()` 等）；
    - 输出部分：
      - `result_df = pd.DataFrame(index=df.index)`；
      - `result_df['FACTOR_NAME'] = series`；
      - `result_df.index.names = df.index.names`；
      - `result_df.to_hdf("result.h5", key="data", mode="w")`。
  - LLM 只被允许在标记好的“因子计算区域”内修改逻辑，用 `df` 计算出一个和 `df.index` 对齐的 `series`；
  - 这样锁定的是输入/输出格式和索引 contract，相当于固定了 HTTP/函数签名，而不是限制内部算法。

- **LLM 的创新空间集中在“怎么算”，而非“怎么写文件/索引”**
  - 在因子计算区域内，LLM 仍然可以自由设计：
    - 使用哪些字段（价量、daily_basic、moneyflow、AE 因子等）；
    - 采用什么统计操作（多窗口 rolling、横截面标准化、行业中性化、回归残差等）；
    - 如何组合多源数据和多因子做打分或信号；
    - 如何做稳健处理（去极值、变换、缺失值填充等）。
  - 被“收紧”的只是容易出低级错的协议层细节（索引层数/名称、存储格式），这些并不产生 Alpha，只会增加 debug 成本。

- **长期效果：减少无意义多样性，提升有效探索密度**
  - 统一模板后：
    - 绝大多数因子脚本天然满足 `MultiIndex(datetime, instrument)` contract，不再频繁触发 pandas/qlib 的 MergeError/UnsortedIndexError；
    - RD-Agent / CoSTEER 的“创意预算”从“怎么写 index.names”集中到“因子结构与策略逻辑”上；
    - 回测与分析侧可以假定所有因子表满足统一 schema，方便做自动化评估与组合优化。
  - 真正需要改变“协议层”（例如未来支持多频率输出、多列输出）的场景，可以通过升级这套模板和 contract 来实现，而不是在每次因子生成中随机突破约定。

---

## 11. 稳定数据管线与策略演进路线（补充）

### 11.1 当前推荐的数据/回测管线

- **数据与因子层**
  - 行情：使用 AIstock 导出的 Qlib bin 目录 `qlib_bin_20251209`，作为日线行情与 Alpha158 的标准数据源；
  - 预计算因子：
    - AE 重构误差：`/mnt/c/Users/lc999/NewAIstock/AIstock/factors/ae_recon_error_10d/result.pkl`；
    - daily_basic 因子：`/mnt/c/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors/result.pkl`；
    - 其它通过 CoSTEER 生成的因子表，统一输出为 MultiIndex(`datetime`, `instrument`) 的 `result.pkl`/`result.parquet`；
  - 线下合并：通过 `tools/merge_static_factors_to_parquet.py`，将上述静态因子线下合并为单一表：
    - `/mnt/c/Users/lc999/NewAIstock/AIstock/factors/combined_static_factors.parquet`；
    - 保证索引为 MultiIndex(`datetime`, `instrument`)，避免 NestedDataLoader 在多路 merge 时触发 index level 相关的 `MergeError`。

- **Qlib handler 与数据加载**
  - 在 `conf_combined_factors.yaml` 中，`data_handler_config.data_loader.kwargs.dataloader_l` 推荐配置为：
    - `Alpha158DL`：负责 label 与基础价量因子；
    - **单一 `StaticDataLoader`**：`config` 指向 `combined_static_factors.parquet`；
  - 不再在 `NestedDataLoader` 内同时挂 3 个或更多的 `StaticDataLoader`，而是在线下完成列合并，Qlib 只消费一张大表，显著降低 pandas/qlib merge 触发 `Not allowed to merge between different levels` 的概率。

### 11.2 RD-Agent 在因子与策略演进中的定位

- **RD-Agent/CoSTEER 主要做的事**
  - 自动生成和迭代因子脚本（基于 HDF5/Parquet）：
    - 约束输出 contract：MultiIndex(`datetime`, `instrument`)、日频、浮点列，并提供 `result.pkl`/`result.parquet`；
    - 避免在因子脚本中随意修改 Qlib handler 或回测配置；
  - 自动提出“因子组合/标签/规则”层面的改进建议：
    - 在 prompts 中强调优先调因子和标签，其次调股票池与组合规则，最后才考虑复杂模型（GRU/Transformer）。

- **人手/工程侧负责的稳定部分**
  - 维护 Qlib YAML（行情目录、时间段、benchmark、手续费等），作为“金标准”回测配置；
  - 维护 `tools/merge_static_factors_to_parquet.py` 这类线下工具脚本，将通过验收的因子统一合并到 `combined_static_factors.parquet`；
  - 维护一套稳定的回测接口（Qlib 或自写 backtest），对 RD-Agent 暴露统一格式的指标文件（JSON/CSV）。

### 11.3 面向实盘的策略评估标准（建议）

在当前 A 股日线多因子选股场景下，可将“可上实盘策略”的核心评估指标约定为：

- **收益与风险**
  - 年化收益率：> 5%–8%（取决于基准与风险偏好）；
  - 最大回撤：< 10%–15%；
  - IC/Rank IC：整体显著为正，且分年度/分市场环境不过度失效。

- **稳定性与鲁棒性**
  - 分子样本（按年份、牛/熊/震荡市）策略表现不过度依赖单一阶段；
  - 对手续费/滑点假设不极端敏感；
  - 因子贡献可解释（避免完全黑箱）。

- **实盘可行性**
  - 持仓集中度合理（如 30 只等权，每只权重上限控制）；
  - 换手率与冲击成本可接受；
  - 真实模拟了停牌/涨跌停、下单价格（收盘价或 VWAP）、单笔止盈/止损规则等执行细节.

### 11.4 策略演进路线（概要）

1. **稳定 baseline 管线**
   - 固定 Qlib 日线 bin 数据与 Alpha158 handler 配置；
   - 使用单一 `combined_static_factors.parquet` 承载所有静态因子，避免 NestedDataLoader 多路 merge 错误；
   - 保证每轮实验的“数据/回测底座”完全一致，可重复.

2. **因子层优先演进**
   - 先在价量/波动/daily_basic 因子上做系统性迭代，比较不同窗口、去极值/标准化方式；
   - 逐步引入 AE 等更复杂因子，明确其对 IC、收益、回撤的边际贡献；
   - 严格检查因子在不同行业/市值分组和时间窗口的稳定性.

3. **规则与组合层演进**
   - 在“好因子”的基础上，先用简单模型（线性、树模型、浅层 MLP）做打分/预测；
   - 明确组合规则：持仓数、权重上限、止盈止损、持有期等，并在回测执行层真实模拟；
   - 通过多周期、多假设（成本/滑点）回测验证策略的鲁棒性.

4. **复杂模型与特殊场景**
   - 在有稳定 Tabular baseline 策略后，才考虑在局部引入 GRU/Transformer 等时间序列模型；
   - 所有复杂模型实验都要以 baseline 策略为对照，不得在没有改善的情况下不断增加复杂度；
   - 对表现良好的策略，进一步通过模拟盘/小资金试运行验证实盘可行性.

整体上，本节的建议是：
- 通过「**稳定的 Qlib 日线回测底座 + 线下合并的静态因子大表 + RD-Agent 因子脚本自动化**」这一组合，
- 让策略演进重点落在 **因子设计、标签/规则调整与风险控制** 上，
- 将工程侧的数据格式、回测接口和路径问题尽量一次性固化、长期复用.

- 方案：
  1. **离线预训练一个全局共享自编码器（AE）**：
     - 在 AIstock 导出的 `daily_pv.h5`（如 `qlib_export_20251209/daily_pv.h5`）上，使用所有股票的历史 10 日价格序列作为训练样本；
     - 训练一个简单但全局共享的 AE 模型（例如 input_dim=10，latent_dim=16），学习“常见 10 日形态”。

2. **预计算一张全市场 AE 重构误差因子表**：
   - 固定好训练后的 AE 权重和标准化参数；
   - 在一个脚本中仅做 forward：对每个 (datetime, instrument) 的最近 10 日窗口计算重构误差；
   - 将结果写成一张标准因子表（MultiIndex(datetime, instrument)，一列 `ae_recon_error_10d`）。

3. **在 Qlib YAML 中把这张表作为额外因子源挂入**：
   - 在 `rdagent/scenarios/qlib/experiment/conf_combined_factors.yaml` 等模板中，新增一条数据源配置；
   - 使得每个自动生成的 workspace 在回测时，都能自动加载这列 AE 因子，无需在 workspace 内重复训练或计算.
  2. **预计算一张全市场 AE 重构误差因子表**：
     - 固定好训练后的 AE 权重和标准化参数；
     - 在一个脚本中仅做 forward：对每个 (datetime, instrument) 的最近 10 日窗口计算重构误差；
     - 将结果写成一张标准因子表（MultiIndex(datetime, instrument)，一列 `ae_recon_error_10d`）。
  3. **在 Qlib YAML 中把这张表作为额外因子源挂入**：
     - 在 `rdagent/scenarios/qlib/experiment/conf_combined_factors.yaml` 等模板中，新增一条数据源配置；
     - 使得每个自动生成的 workspace 在回测时，都能自动加载这列 AE 因子，无需在 workspace 内重复训练或计算。

这样可以：

- 显著降低单次实验的计算负担（因子阶段只 forward，不训练）；
- 保证因子值可复现（固定权重 + 统一归一化）；
- 让 AE 因子作为“额外视角”参与 Alpha 组合，而不是把每个窗口当一个新模型来训练。

### 11.2 当前实现形态（脚本层）

为便于验证，该方案首先以两份脚本的形式落地在 RD-Agent 仓库根目录（不会影响现有流程）：

- **预训练脚本：`train_ae_10d.py`**
  - 功能：
    - 从指定的 `daily_pv.h5` 读取数据（默认使用重命名后的 `close` 列）；
    - 构造所有股票的滚动 10 日窗口序列；
    - 子采样（默认最大 50 万样本），做全局标准化；
    - 用 PyTorch 训练一个简单 AE（输入维度=10，latent_dim=16）；
    - 将 `state_dict`、窗口长度、特征列表、mean/std 等打包保存到 `models/ae_10d.pth`.

- **因子计算脚本：`factor_ae_recon_error_10d.py`**
  - 功能：
    - 在某个因子 workspace 目录下：
      - 从当前目录的 `daily_pv.h5` 读取数据（按规范重命名列）；
      - 从环境变量 `AE10D_MODEL_PATH`（或默认 `models/ae_10d.pth`）加载预训练 AE；
      - 为每个 (datetime, instrument) 构造最近 10 日窗口，按训练时相同的方式标准化；
      - 向量化 forward 计算重构误差，作为异常程度；
      - 生成 `result.h5`，索引为 MultiIndex(datetime, instrument)，列名为 `ae_recon_error_10d`.
  - 代码层面严格遵守现有约束：
    - 仅使用 `pandas.read_hdf` / `to_hdf` 读写 HDF5；
    - 不显式导入 `h5py`；
    - 不在因子脚本中训练模型，只做 forward；
    - 不使用随机数据作为真实特征.

这两份脚本目前主要作为“POC + 模板”，还未直接接入 RD-Agent 自动流水线.

### 11.3 将 AE 因子作为预计算表接入 Qlib（路线 1）

目标：

- 不改 RD-Agent 核心逻辑，只通过 **“预计算 + Qlib YAML 模板”**，让所有新建 workspace 的回测自动拥有 AE 因子.

步骤示意：

1. **离线在固定目录预计算 AE 因子表**：
   - 选择一个长期存放因子表的位置，例如：
     - `C:/Users/lc999/NewAIstock/AIstock/factors/ae_recon_error_10d/result.h5`
   - 在 WSL 中运行：
     ```bash
     conda activate rdagent-gpu
     cd /mnt/c/Users/lc999/RD-Agent-main

     # 已完成的预训练
     python train_ae_10d.py \
       --h5-path /mnt/c/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_20251209/daily_pv.h5 \
       --window 10 \
       --features close \
       --output models/ae_10d.pth \
       --epochs 20 \
       --batch-size 512 \
       --max-samples 500000

     # 在全量 daily_pv.h5 上运行因子计算脚本，输出 result.h5
     export AE10D_MODEL_PATH=/mnt/c/Users/lc999/RD-Agent-main/models/ae_10d.pth
     python factor_ae_recon_error_10d.py
     # 然后将生成的 result.h5 拷贝到固定因子目录
     ```

2. **在 Qlib 实验模板 YAML 中挂入这张因子表**：
   - 在 `rdagent/scenarios/qlib/experiment/conf_combined_factors.yaml` 等模板中：
     - 在 `data_handler_config` 或 `feature` 相关配置里增加一条数据源，指向上述固定路径的 `result.h5`；
     - 将 `ae_recon_error_10d` 列纳入 features 列表或 NestedDataLoader 的合并规则中.
   - 这样，每次 RD-Agent 创建新 workspace 并基于这些模板生成 `conf_*.yaml` 时，AE 因子都会自动包含在内，无需在 workspace 内重复运行因子脚本.

3. **注意事项**：
   - 需要保证 AE 因子表与 Qlib bin 在：
     - 时间范围上对齐（例如均为 2010-01-07 ~ 2025-12-01）；
     - 股票池上对齐（同样剔除 ST/退市/停牌，并使用相同 instruments 列表）；
   - 若未来生成新的 snapshot（例如 `qlib_export_2026xxxxx`），需要重新跑一次预训练 + 全量因子计算脚本，生成对应版本的 AE 因子表.

### 11.4 作为 RAG 知识库的示例内容

这套 AE 因子方案本身也是一个很好的 **RAG 示例模板**，可写入 AIstock RAG 知识库，帮助 LLM 避免“每窗口训练一次 AE”这类错误：

- 可以在 RAG 文档中增加：
  - 《全局自编码器因子设计示例》：
    - 说明为什么“每个滑动窗口单独训练 AE”不可行；
    - 说明全局预训练 + 预计算因子的思路与优点；
    - 给出高层伪代码（而非具体实现）.
  - 在“因子实现常见错误案例集”中加入：
    - “自编码器每窗口训练导致过拟合与计算爆炸”的负面示例；
    - 本节方案作为推荐修正路径.

- 在 RD-Agent 的因子生成/修正 prompt 中，通过 RAG 检索：
  - 当用户或系统提出“异常检测因子 / 自编码器因子”需求时，优先将上述文档片段作为上下文返回给 LLM；
  - 引导 LLM 避免重新走“每个窗口训练一个 AE”的老路，而是直接采用“全局 AE + 预计算表”类的设计.

### 11.5 综合收益

- **训练效率**：大部分计算集中在一次性预训练和一次性全市场预计算上，演进过程中每个实验只需读表，不再重复训练；
- **结果稳定性**：因子值由固定权重 + 统一归一化生成，可控且易于复现；
- **演进空间**：
  - RD-Agent 可以在“是否使用 AE 因子、如何与其他因子组合、如何在回测中解释其贡献”等维度上自动探索；
  - 结合 RAG，后续 LLM 在提出新异常检测型因子时，会更多复用这一成熟模式，而不是从零开始瞎试.

---

## 12. 基于 `moneyflow.h5` 的主力资金因子接入与策略演进

### 12.1 数据约定回顾（来自 H5 备忘录）

- `AIstock_Qlib_数据集使用备忘录.md` 中对 `moneyflow.h5` 的约定：
  - 路径示例：`C:/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_YYYYMMDD/moneyflow.h5`；
  - 索引：`MultiIndex(datetime, instrument)`；
  - 列名前缀：`mf_*`，典型字段包括：
    - `mf_net_vol`, `mf_net_amt`（净流入量/额）；
    - 各档买卖量/额：`mf_sm_buy_vol`, `mf_lg_buy_amt` 等；
  - 单位：`_vol` 为股、`_amt` 为元。

整体上，`moneyflow.h5` 与 `daily_pv.h5` 在索引上完全对齐，可以视为“资金流向面板”，与价格面板按 `(datetime, instrument)` 直接 join。

### 12.2 Route：先预计算资金流因子表，再挂入 Qlib

与 AE 因子类似，这里也推荐 **“预计算因子表 + Qlib YAML 挂载”** 的路线，而不是在每个 workspace 里临时从 `moneyflow.h5` 开始算起：

1. **在 AIstock/H5 侧预计算资金流因子表**：
   - 在 WSL 中编写一个独立脚本（可以由 RD-Agent/CoSTEER 生成框架）：
     - 读取：`daily_pv.h5` + `moneyflow.h5`；
     - 在 `(datetime, instrument)` 上 inner join；
     - 构造标准资金流因子列，例如：
       - `mf_net_amt_5d = rolling_sum(mf_net_amt, 5)`；
       - `mf_big_buy_ratio = (mf_lg_buy_amt + mf_elg_buy_amt) / amount`；
       - `mf_net_amt_mv = mf_net_amt / db_total_mv`（结合 `daily_basic.h5` 时）；
     - 输出一张标准因子表：
       - 索引：`MultiIndex(datetime, instrument)`；
       - 列：若干因子列（命名约定如 `mf_net_amt_5d`, `mf_big_buy_ratio` 等）；
       - 文件格式：`result.h5` 或 `capital_flow_factors.h5`。
   - 将结果放入长期目录，例如：
     - `C:/Users/lc999/NewAIstock/AIstock/factors/capital_flow_daily/result.h5`
     - WSL 路径：`/mnt/c/Users/lc999/NewAIstock/AIstock/factors/capital_flow_daily/result.h5`。

2. **在 Qlib YAML 中通过 StaticDataLoader/NestedDataLoader 挂入**：
   - 参照 11.3 中 AE 因子表的接入方式，在：
     - `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml` 等模板里，
     - 在 `data_handler_config.data_loader` 的 `NestedDataLoader.dataloader_l` 列表中，
       再新增一个 `StaticDataLoader`，指向资金流因子表路径：
       ```yaml
       - class: qlib.data.dataset.loader.StaticDataLoader
         kwargs:
           config: "/mnt/c/Users/lc999/NewAIstock/AIstock/factors/capital_flow_daily/result.h5"
       ```
   - 这样 Qlib 的 DatasetH 在构造特征时，会自动把这张资金流因子表与现有 Alpha158 / 联合因子 / AE 因子一起拼接到特征空间中。

3. **保持与 bin 行情和其它 H5 的对齐**：
   - 时间区间上与 `qlib_bin_20251209`、`daily_pv.h5`、`daily_basic.h5` 对齐；
   - 股票池上与 `instruments/all.txt` 以及 ST/退市过滤后的 universe 对齐；
   - 列命名保持 `mf_*` 前缀，避免与 price/basic 因子混淆。

### 12.3 RD-Agent 如何基于资金流因子演进策略

在上述预计算 + YAML 挂载完成后：

- **现有 RD-Agent Qlib 场景**：
  - 在使用联合因子模板的任务中，模型输入将自然包含：
    - 价格/波动/形态类因子（Alpha158 + H5 因子）；
    - 资金流因子（来自 `capital_flow_daily/result.h5`）；
    - AE 异常检测因子（来自 AE 因子表）。
  - RD-Agent 无需感知资金流因子的具体计算过程，只需在模型/策略演进时观察：
    - 引入/剔除资金流因子对 IC、IR、收益回撤的影响；
    - 不同 moneyflow 因子在不同市场风格阶段的贡献度。

- **因子与策略层面的演进方向示例**：
  - 在 CoSTEER 的因子/策略生成提示中，鼓励 LLM：
    - 利用 `mf_net_amt_5d`, `mf_big_buy_ratio`, `mf_net_amt_mv` 等字段构造：
      - 资金持续流入驱动的多头信号；
      - 高位大额净流出触发的减仓/止盈信号；
    - 将资金流因子与价格动量、估值因子（`db_*`）联合作为多因子打分输入；
  - 在回测结果分析阶段，引导 LLM：
    - 对比“含资金流因子”和“剔除资金流因子”两组模型的收益/风险；
    - 针对极端行情（大跌、大涨）分析资金流因子的保护/放大作用。

### 12.4 提示词与 RAG 的配合建议

结合 `AIstock_Qlib_数据集使用备忘录.md` 第 3 节的约定，建议在 RD-Agent/CoSTEER 的 prompt/RAG 中：

- **在数据说明部分显式声明**：
  - 已有 `moneyflow.h5`，索引为 `MultiIndex(datetime, instrument)`，列为 `mf_*` 系列；
  - 已有一张预计算的资金流因子表（例如 `capital_flow_daily/result.h5`），已在 Qlib YAML 中挂载为特征源；
  - LLM 设计资金流相关因子和策略时，应优先基于这些现有字段和因子，而非重新拉取原始数据。

- **在策略需求中鼓励“价格 + 资金 + 基本面 + 预计算因子”的多模态组合**：
  - 明确指出希望比较：
    - 纯价格因子 vs 价格 + 资金流因子；
    - 价格 + 资金流 + 估值因子 vs 价格 + AE 异常检测因子等不同组合；
  - 让 RD-Agent 在回测与演进过程中，系统性地探索“主力资金流向因子”在组合中的边际贡献。

### 12.5 关于 `daily_basic.h5` 预计算因子表的接入说明

- 通过脚本 `precompute_daily_basic_factors.py`，已在 AIstock 侧预计算一张每日指标因子表：
  - 输入：`qlib_export_20251209/daily_basic.h5`（`db_*` 列，如 pe/pb/市值/换手率/量比等）；
  - 输出：`C:/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors/result.h5`，索引为 `MultiIndex(datetime, instrument)`，列包括 `value_pe_inv/value_pb_inv/size_log_mv/liquidity_turnover/liquidity_vol_ratio` 等；
  - 不需要在 RD-Agent workspace 间拷贝，使用固定路径即可。

- 在 Qlib 联合因子模板（`conf_combined_factors.yaml`）中，已通过 `StaticDataLoader` 挂载该表：

  ```yaml
  - class: qlib.data.dataset.loader.StaticDataLoader
    kwargs:
      config: "/mnt/c/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors/result.h5"
  ```

  因此：

  - 所有基于该模板生成的 RD-Agent workspace，在训练/回测时都会自动读到这些 `daily_basic` 预计算因子列；
  - 无需在每个 workspace 内重复运行预计算脚本，也无需手动拷贝 `result.h5`。

- 提示词层面：
  - **不是必须修改** 才能使用这些因子，Qlib 已经会把它们作为特征传给模型；
  - 建议在 Qlib 场景或因子/策略生成 prompt 的“已有数据资产”部分补一句：
    - “除了 AE 因子和资金流因子外，还预先计算了一张 daily_basic 因子表（估值、市值、流动性等），已在 Qlib YAML 中挂载，可直接视为‘风格与风险因子’加入模型。”
  - 同时，在 `rdagent/scenarios/qlib/experiment/prompts.yaml` 的 `qlib_factor_experiment_setting` 段落中，已明确提示：在策略演进时，应**优先利用 daily_basic 预计算因子对股票做基础筛选与风险控制（估值、市值、流动性等），再在筛选后的股票池上结合价量因子、AE 因子及未来的资金流因子做多因子建模与选股**，而不是在全市场原始股票池上直接暴力选股。

### 12.6 已实现的资金流向因子预计算与合并脚本（当前实践）

- **脚本 1：预计算个股资金流向因子表 `precompute_moneyflow_factors.py`**

  - 位置：`tools/precompute_moneyflow_factors.py`
  - 主要逻辑：
    - 从 AIstock Qlib snapshot 中读取：
      - `moneyflow.h5`（含 `net_mf_amount/net_mf_vol/buy_elg_*/sell_elg_*` 等字段）；
      - `daily_pv.h5`（含 `amount/volume` 等基础价量字段）；
    - 在 `(datetime, instrument)` 上 inner join，并统一索引为 `MultiIndex(datetime, instrument)`；
    - 构造一组带 `mf_` 前缀的资金流因子，包括：
      - 超大单维度：`mf_elg_net_amt`、`mf_elg_net_amt_ratio`、`mf_elg_net_vol_ratio`、`mf_elg_dominance`；
      - 整体净流入维度：`mf_net_amt`、`mf_net_amt_ratio`、`mf_net_vol_ratio`；
      - 主力 vs 整体偏离：`mf_elg_vs_all`；
      - 5 日滚动累计/均值：`mf_elg_net_amt_5d`、`mf_net_amt_5d`、`mf_elg_net_amt_ratio_5d`、`mf_net_amt_ratio_5d`、`mf_elg_vs_all_5d`。
    - 输出因子表到：
      - Windows 路径：`C:/Users/lc999/NewAIstock/AIstock/factors/moneyflow_factors/result.pkl`；
      - WSL 路径：`/mnt/c/Users/lc999/NewAIstock/AIstock/factors/moneyflow_factors/result.pkl`。

  - 推荐运行时机：
    - **在当前一轮 RD-Agent Qlib 长周期实验（不含资金流因子）跑完之后** 再执行；
    - 避免在关键 baseline 验证过程中引入新因子源，保证对照实验的可比性。

  - 推荐命令（在 WSL 内）：

    ```bash
    conda activate rdagent-gpu
    cd /mnt/c/Users/lc999/RD-Agent-main

    python tools/precompute_moneyflow_factors.py
    ```

- **脚本 2：将资金流因子并入统一静态因子表 `merge_static_factors_to_parquet.py`**

  - 位置：`tools/merge_static_factors_to_parquet.py`
  - 目前已支持自动合并以下来源（若存在）：
    - 代表性 workspace 下的 `combined_factors_df.parquet`；
    - AE 因子表：`/mnt/c/Users/lc999/NewAIstock/AIstock/factors/ae_recon_error_10d/result.pkl`；
    - daily_basic 因子表：`/mnt/c/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors/result.pkl`；
    - **资金流因子表**：`/mnt/c/Users/lc999/NewAIstock/AIstock/factors/moneyflow_factors/result.pkl`（由脚本 1 生成）；
  - 合并方式：
    - 对各因子表调用统一的 `_ensure_multiindex` 逻辑，修正为 `MultiIndex(datetime, instrument)` 并排序；
    - 使用 `pd.concat(dfs, axis=1)` 按列拼接，随后再次按 `(datetime, instrument)` 排序，保证下游切片/merge 不再触发 `UnsortedIndexError`/`MergeError`；
    - 输出到统一路径：
      - `/mnt/c/Users/lc999/NewAIstock/AIstock/factors/combined_static_factors.parquet`。

  - 推荐运行顺序：
    1. 先运行脚本 1 生成 `moneyflow_factors/result.pkl`；
    2. 再运行脚本 2 重新生成 `combined_static_factors.parquet`；
    3. 保持 Qlib YAML 中的 `static_path` 不变（仍指向 `combined_static_factors.parquet`），下一轮实验自然会看到新增的 `mf_*` 因子列。

  - 推荐命令（在 WSL 内）：

    ```bash
    conda activate rdagent-gpu
    cd /mnt/c/Users/lc999/RD-Agent-main

    python tools/merge_static_factors_to_parquet.py
    ```

---

## 2025-12-14 更新：因子演进（static_factors.parquet + schema + rolling moneyflow + 重试修复）

1. **统一静态因子表产出与字段白名单**：
   - 在 repo 内生成/维护：
     - `git_ignore_folder/factor_implementation_source_data/static_factors.parquet`
     - `git_ignore_folder/factor_implementation_source_data/static_factors_schema.csv`
     - `git_ignore_folder/factor_implementation_source_data/static_factors_schema.json`
   - 目的：让因子脚本（`factor.py`）在运行时可以通过 `read_parquet + join` 使用 `daily_basic.h5`（`db_*`）与 `moneyflow.h5`（`mf_*`）相关字段，并且通过 schema 文件实现“字段白名单约束”，减少 LLM 编造字段名导致的 KeyError。

2. **资金流 rolling 派生特征纳入静态因子表**：
   - 将资金流相关的 rolling 聚合与派生列（例如 `*_5d` / `*_20d`）一并写入 `static_factors.parquet`，并同步反映在 schema 中，方便 LLM 在可用列集合内做选择。

3. **因子执行数据目录同步与 schema 注入**：
   - 在 RD-Agent 生成因子执行数据目录时，优先同步 repo 生成的 `static_factors.parquet` 及 schema（csv/json）到运行目录（含 debug 目录）。
   - 同时补齐了数据文件描述能力：schema 文件（`.csv`/`.json`）能被正确读取并注入到 prompt 中（避免“文件类型不支持”的异常）。

4. **FactorAutoRepair（失败自动修复）触发范围扩展**：
   - 扩展了失败签名匹配范围，覆盖更常见的失败类型（例如缺列、空结果、全 NaN 等），以便在演进轮次中实际触发重试修复。

5. **关联备忘录与提示词归档**：
   - 备忘录：`docs/20251214_因子演进_debug_static_factors_rolling_retry备忘录.md`
   - 提示词全量 dump 与诊断：`docs/20251214_QLib因子全量提示词_dump与诊断.md`
