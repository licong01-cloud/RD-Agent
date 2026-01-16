# AIstock + Qlib “方案 A + 轻量 B” 设计文档

> 版本：v1.0  
> 日期：2025-12-08  
> 适用范围：AIstock + Qlib + RD-Agent 集成，在 A 股（日线/分钟线）与多源数据（资金流向、筹码峰、基本面、美股等）场景下的整体技术方案。

---

## 1. 总体目标

在不牺牲数据源灵活性的前提下：

- **上游（因子与信号研发）**：
  - 全部多源数据（AIstock 日线、美股、资金流向、筹码峰、基本面等）
  - 统一按自定义 schema 存在 HDF5/Parquet 中，由 Python/pandas/factor.py 直接加载与加工；

- **下游（回测与执行）**：
  - 仅将 **日线 / 分钟线价量数据** 用 Qlib 官方工具从 CSV 导出为标准 bin 目录；
  - 使用 Qlib 的回测、策略和风险评估框架（`qrun` / Strategy / Portfolio / Executor）进行日线/分钟级回测。

- **智能体（RD-Agent / CoSTEER）**：
  - 通过 LLM 生成和修改因子/信号脚本（例如 `factor.py` / `generate_signals.py`），直接操作 HDF5/Parquet；
  - 调用 Qlib 回测（基于 bin 数据）并解析统一结构的指标输出，驱动策略演进。

此方案即为：**“方案 A + 轻量 B” 的具体落地版本**。

---

## 2. 架构与数据流概览

### 2.1 逻辑组件

- **AIstock 数据服务层**：
  - 来源为数据库（或原始 HDF5），对外导出：
    - 日线价量（可扩展分钟线）：导出为 CSV，供 Qlib `dump_bin.py` 使用；
    - 资金流向、筹码峰、基本面、美股指数等：导出为 HDF5/Parquet 面板表。

- **Qlib 数据层（bin）**：
  - `provider_uri_daily`: 标准 Qlib 日线 bin 目录（通过 CSV + `dump_bin.py` 生成）；
  - `provider_uri_minute`: 标准 Qlib 分钟级 bin 目录（可选，未来扩展）；
  - 内含 `calendars/`, `instruments/`, `features/`, `fields/` 等结构。

- **外部特征 & 因子层（HDF5/Parquet）**：
  - `daily_pv.h5`（或等价）作为原始/中间价量库；
  - `capital_flow.h5` / `chip_features.parquet` / `fundamental.parquet` 等；
  - 由自定义脚本（`factor.py` / `generate_signals.py`）统一读取与合并。

- **RD-Agent & CoSTEER 层**：
  - 负责：
    - 因子脚本与策略逻辑生成/修改；
    - 实验配置（YAML）生成与更新；
    - 调用 Qlib 回测（通过 `QlibFBWorkspace`）并解析结果；
    - 利用指标 + 知识库进行策略演进。

- **回测与执行层（Qlib）**：
  - 依赖 `provider_uri_(daily|minute)` 的 bin 数据；
  - 接收外部生成的信号/权重文件；
  - 执行回测、统计指标、输出曲线与报告。

### 2.2 数据流（简化顺序）

1. **数据库 → CSV / HDF5**：
   - 日线/分钟线价量 ⇒ 导出标准 CSV；
   - 其他数据源（资金流、筹码、基本面等） ⇒ 导出 HDF5/Parquet。

2. **CSV → Qlib bin（日线/分钟线）**：
   - 使用 Qlib 官方 `dump_bin.py` / DUMP_ALL 脚本，生成：
     - `provider_uri_daily`（freq = 1d）；
     - `provider_uri_minute`（freq = 1min/5min...，可选）。

3. **HDF5/Parquet → 因子/信号**：
   - RD-Agent 生成/维护 `factor.py` / `generate_signals.py`：
     - pandas 读取 HDF5/Parquet；
     - 计算因子 & 策略信号；
     - 输出 `signals.csv` 或统一格式的权重文件。

4. **Qlib 回测（依赖 bin + 信号）**：
   - Qlib `qrun` 场景：
     - 从 bin 读取价量；
     - 从信号文件读取 `date/datetime, instrument, target_weight`；
     - 进行日线/分钟级回测；
     - 输出指标（年化、回撤、IC、换手等）与净值曲线。

5. **RD-Agent 演进**：
   - 读取回测指标文件（JSON/CSV）；
   - 基于预设目标（收益/风险/稳定性）评价策略；
   - 结合知识库（CoSTEER）提出修改建议并更新脚本与配置，进入下一轮。

---

## 3. 日线/分钟线：从数据库到 Qlib bin 的设计

### 3.1 CSV 结构约定

以日线为例，推荐 CSV 字段格式（列名尽量贴近 Qlib 常用习惯）：

```text
instrument,date,open,high,low,close,volume,amount
000001.SZ,2010-01-04,12.34,12.80,12.20,12.50,12345678,1.23e8
...
```

- **instrument**：证券代码（如 `000001.SZ`），与 `instruments/all.txt` 保持一致；
- **date**：交易日期，建议 `YYYY-MM-DD` 格式；
- **open/high/low/close/volume/amount**：标准日线价量字段；
- 可选扩展字段：复权价、停牌标记、涨跌停标记等，也可以先不进入 bin，只在 HDF5 里使用。

分钟线 CSV 类似，只是：

```text
instrument,datetime,open,high,low,close,volume,amount
000001.SZ,2010-01-04 09:31:00,12.34,12.40,12.30,12.35,123456,1.23e7
...
```

- 时间字段可命名为 `datetime` 或 `date`（但需在 dump 配置中一致）；
- 精度到分钟级（或更高）。

### 3.2 使用 Qlib `dump_bin.py` 的基本思路

> 注意：这里给的是概念性设计，不直接耦合到具体路径/脚本名，由 AIstock 数据侧实现具体代码。

1. **准备 Qlib 数据配置**（简化示意）：

   ```python
   # pseudo-code for dump config
   DUMP_CONFIG = {
       "csv_path": "path/to/daily.csv",
       "qlib_dir": "path/to/qlib_bin_ai_stock_daily",
       "symbol_field": "instrument",
       "date_field": "date",
       "freq": "day",  # 或者 "1min" 等，对分钟线另配
       "fields": ["open", "high", "low", "close", "volume", "amount"],
       "instruments": "path/to/instruments/all.txt",
       # 其他如 timezone、start/end 日期等按 Qlib 推荐配置
   }
   ```

2. **AIstock 数据服务提供一个脚本/工具**，内部调用 Qlib 官方接口：

   - 输入：数据库连接或 CSV 路径；
   - 输出：符合 Qlib 规范的 bin 目录（`calendars/`, `instruments/`, `features/`, `fields/` 等）。

3. **验证**：

   - 使用 Qlib 的 `D.list_instruments`, `D.features` 在 Python 里做一组 sanity check：
     - 能列出 `"all"` 市场的 instruments；
     - 对任一标的在给定日期区间能取出 `open/close` 等字段；
   - 使用简单的 Alpha158 / DatasetH 实验跑一个最小样例，确认 Qlib 标准 pipeline 能工作。

### 3.3 bin 目录命名与版本管理

建议在 AIstock 路径下为不同频率/版本建立独立目录，例如：

```text
qlib_bin_ai_stock_daily_v1/
qlib_bin_ai_stock_minute_v1/
```

- RD-Agent 的 Qlib YAML 中 `provider_uri` 指向对应版本；
- 未来新增数据或修复时，可增加 `v2`, `v3` 等版本，旧版本仍可用于结果复现；
- 可在 AIstock 侧维护一个小的元数据文件（如 JSON），记录：
  - 生成时间、使用的源数据版本、字段映射规则、频率等信息。

---

## 4. 其他数据（资金流向、筹码峰、基本面等）的 HDF5/Parquet 设计

### 4.1 面板化为 `instrument × date` 或 `instrument × datetime`

目标是：

- 所有扩展数据，无论来源多复杂，**最终都统一成“面板表”**：

  ```text
  instrument, date, feature1, feature2, ...
  ```

示例：资金流向日度表：

```text
instrument,date,net_inflow,main_inflow,main_inflow_ratio,...
000001.SZ,2020-01-02,1.23e8,8.8e7,0.35,...
...
```

筹码峰处理流程：

1. 原始筹码分布可能是 `price_bin` × `volume` 的结构；
2. 在 AIstock 或因子脚本中先转换为若干日度特征：
   - `chip_concentration`（持仓集中度）、
   - `avg_cost`（筹码加权平均成本）、
   - `trapped_ratio`（高于现价的筹码比例）等；
3. 存成面板表：

   ```text
   instrument,date,chip_concentration,avg_cost,trapped_ratio,...
   ```

### 4.2 存储格式与路径建议

- 推荐使用 HDF5 或 Parquet：
  - 日度数据体量大但结构规整 ⇒ Parquet/HDF5 都适合；
  - 分钟级或更高频可按分区（按日期/按标的）拆分以便加载。
- 路径示例：

  ```text
  AIstock/
    factors_raw/
      capital_flow_daily.parquet
      chip_features_daily.parquet
      fundamentals_quarterly.parquet
      us_index_daily.parquet
      ...
  ```

### 4.3 与因子/信号脚本的接口

- 因子脚本（`factor.py` / `generate_signals.py`）中：
  - 使用统一的 loader 函数封装数据访问：

    ```python
    def load_daily_price():
        ...  # from daily_pv.h5 or equivalent

    def load_capital_flow():
        ...  # from capital_flow_daily.parquet

    def load_chip_features():
        ...

    def load_fundamentals():
        ...
    ```

  - 这样未来换存储格式、改路径时，只改 loader，不改所有因子逻辑。

---

## 5. 策略逻辑与信号生成

### 5.1 信号格式约定

无论是日线还是分钟线策略，最终给 Qlib 回测的信号统一约定为：

```text
date/datetime, instrument, target_weight
```

或（少用）

```text
date/datetime, instrument, signal_score
```

- `target_weight`：
  - 表示该时刻目标持仓权重（可正负，sum=1 或不超过某阈值）；
- `signal_score`：
  - 若使用 Ranking/TopK 策略，可只输出得分，由 Strategy 负责转权重。

### 5.2 信号生成脚本的角色

- 由 RD-Agent / CoSTEER 生成和维护，例如 `generate_signals.py`：
  1. 通过 loader 读取 HDF5/Parquet 数据；
  2. 计算因子（价量、资金流、筹码、基本面等）；
  3. 根据策略规则或模型输出计算 `target_weight`；
  4. 输出信号文件（CSV/Parquet）给 Qlib 使用。

- 将来可以逐步演化为：
  - 由 RD-Agent 自动改写 `compute_factors` 和 `generate_signals` 两个核心函数；
  - 保持脚本骨架稳定，从而控制复杂度与可维护性。

---

## 6. RD-Agent 集成设计

### 6.1 当前 Qlib 场景与 workspace

- RD-Agent Qlib 场景使用 `QlibFBWorkspace`（见 `rdagent/scenarios/qlib/experiment/workspace.py`）：
  - 将模板 YAML 和脚本注入 workspace 目录；
  - 调用 Qlib 环境（conda/docker）运行：

    ```bash
    qrun conf.yaml
    python read_exp_res.py
    ```

  - 从 `qlib_res.csv` / `ret.pkl` 中提取指标和回测曲线。

### 6.2 在“方案 A + 轻量 B”下的调整点

1. **YAML 中的 `provider_uri` 指向新的 bin 目录**：

   - 日线：`provider_uri: path/to/qlib_bin_ai_stock_daily_v1`
   - 分钟：`provider_uri: path/to/qlib_bin_ai_stock_minute_v1`（如有）

2. **保持因子/信号脚本在 workspace 内或外部脚本中执行**：

   - 简单版本：
     - 因子/信号脚本在 RD-Agent 外执行，Qlib 场景只消费已有信号文件；
   - 升级版本：
     - 在 Qlib 场景执行链中插入步骤：
       1. 先运行 `python generate_signals.py`（由 RD-Agent 控制/生成）；
       2. 再运行 `qrun` 回测。

3. **统一回测结果格式**：

   - 不管是 Qlib 标准回测，还是未来的自定义回测脚本，
     - 都输出统一结构的 JSON/CSV 指标文件；
   - RD-Agent 仅读取这份指标做策略好坏判断与演进决策。

---

## 7. 策略演进、回测与可开发策略的广度分析

### 7.1 策略演进

- 有了标准 Qlib bin（至少日线）：
  - 回测链路更加稳定（避免 Dataset 为空、D.features 失败）；
  - 可以按统一的方式获取年化收益、回撤、Sharpe、IC 等指标；
  - RD-Agent 的演进机制可以长期依赖这套指标，不必频繁改接口。

- 上游因子/信号仍由 HDF5/Parquet + pandas 管理：
  - 能快速试验新数据源和特征；
  - RD-Agent 通过自然语言指导 + 代码生成，持续改进因子和策略逻辑。

### 7.2 回测能力

- 日线回测：完全用 Qlib 的回测引擎；
- 分钟级回测：在构建分钟级 bin 后，也可以用 Qlib 进行分钟级撮合与评估；
- 支持：单因子、多因子、规则策略、事件驱动策略以及模型驱动策略等多种形态。

### 7.3 可开发策略的广度

- 价量类因子：可直接基于 Qlib bin 或 HDF5 计算；
- 资金流向、筹码峰、基本面、美股联动、情绪等：
  - 在 HDF5/Parquet 中任意组合与加工；
  - 只要最终归约为 `instrument × date/datetime` 面板即可。
- Qlib 端只关心：有行情（bin）+ 有信号（权重），不限制上游逻辑复杂度。

---

## 8. 实现与迭代路线建议

### 8.1 阶段 1：日线 bin POC

- 目标：
  - 从数据库/CSV 构建一套日线 Qlib bin `qlib_bin_ai_stock_daily_v1`；
  - 使用最简单的单因子或均值回归策略，在该 bin 上成功跑通 Qlib 回测。

- 步骤：
  1. 选取部分标的（如 300–500 只股票）和有限年份（如 2016–2020）导出为 CSV；
  2. 配置并运行 `dump_bin.py` 生成 bin；
  3. 编写最小化 Qlib YAML（或复用现有模板简化版），验证：
     - `D.list_instruments("all")` 正常；
     - `D.features` 能获取价量字段；
     - 简单双均线/单因子策略能成功回测。

### 8.2 阶段 2：全市场日线 bin + RD-Agent 接入

- 将导出范围扩展到全 A 股（或你定义的高质量 Universe）；
- 更新 RD-Agent Qlib 场景 YAML 的 `provider_uri`，指向新 bin 目录；
- 在 RD-Agent 中用现有 Qlib 场景跑几轮因子/模型实验，确认：
  - 回测指标稳定输出；
  - Dataset 不再为空；
  - 演进链路可用。

### 8.3 阶段 3：分钟级 bin 与更复杂策略

- 选择关键标的或指数构建分钟级 bin（为控制数据量可先做子集）；
- 设计并调试分钟级信号生成与回测流程；
- 在 RD-Agent 场景中增加分钟级策略配置（可先手工，再逐步自动化）。

### 8.4 阶段 4：自动信号脚本编排（可选增强）

- 将 `generate_signals.py` 纳入 RD-Agent 控制：
  - 由 CoSTEER 生成/修改因子与信号规则；
  - 由场景 executor 先执行信号脚本，再调用 Qlib 回测；
- 在保持上游 HDF5/Parquet 自由度的前提下，
  - 达成“数据 → 因子 → 信号 → 回测 → 指标 → 改脚本”的全自动迭代闭环。

---

## 9. 总结

在本设计中：

- **行情数据（日线/分钟线）** 完全按照 Qlib 官方工具从 CSV 导出为 bin：
  - 确保与 Alpha158 / DatasetH / 标准回测引擎的最大兼容性；
- **其他扩展数据（资金流向、筹码峰、基本面、美股等）** 保持 HDF5/Parquet 格式：
  - 保持数据源和特征工程的高度灵活性；
- **RD-Agent 与 CoSTEER**：
  - 专注在因子与策略逻辑的生成与演进；
  - 通过统一信号格式和回测指标接口，解耦上游数据复杂度与下游回测实现；
- 整体方案兼顾：
  - 回测能力（借力 Qlib bin 与回测框架）；
  - 策略开发的广度（多源数据 + 自由因子工程）；
  - 实现与维护成本（仅标准化行情一块，其他数据不过度套 Qlib 模型）。

该方案在当前与可预见中期内，可视为 AIstock + Qlib + RD-Agent 集成的**推荐主线架构**。

---

## 10. Qlib Snapshot 管理（新增导出 bin 页签）

在现有 Qlib Snapshot 管理功能上新增一个“导出为 Qlib bin”页签，保留原有功能不变，新增能力与流程如下：

- **导出流程**：  
  1) 先从 AIstock 数据库导出 K 线数据为 CSV（符合 3.1 的字段约定）；  
  2) 调度 Qlib 官方工具（如 `dump_bin.py`）将 CSV 转换为标准 bin 目录；  
  3) 生成的 bin 目录可供 Qlib 回测、RD-Agent YAML 配置引用。

- **CSV 导出选项（对导出的证券集合做过滤）[已实现]：**  
-  1) **剔除曾经 ST 或当前 ST 的股票**：勾选后，后端在构建日线 Qlib 宽表时，会查询 `stock_st` 表，凡是在 `stock_st` 中出现过的 `ts_code` 一律从样本集中剔除（包括当前仍为 ST 的股票）；  
-  2) **剔除退市/暂停上市的股票**：勾选后，后端会在选取样本代码时查询 `stock_basic` 表，过滤掉 `list_status IN ('D','P')` 的证券，避免退市和当前暂停上市标的进入导出样本；  
-  3) 以上两个选项同时作用于 Qlib Snapshot 日线导出与 CSV→bin 导出链路，保证 RD-Agent 使用的 bin 数据与 AIstock 的高质量 Universe 一致。

- **UI 交互要点**：  
  - 新页签名称建议：“导出 Qlib bin（CSV→bin）”；  
  - 保留原有 Snapshot 导出页签，新增页签仅负责 CSV→bin 转换；  
  - 显示当前过滤条件摘要（是否剔除 ST、是否剔除退市/暂停上市），导出时需明确提示过滤后的样本数量；  
- **后端处理要点**：  
  - 导出成功后展示 bin 目录路径与校验提示（如样本数、日期范围、字段列表）。

 - **后端处理要点**：  
  - CSV 导出时按勾选条件过滤证券；  
  - 调用 Qlib dump 工具时确保 symbol_field/date_field/freq 与 CSV 一致；  
  - 不改动原有 Snapshot 导出逻辑，新增一条独立的 CSV→bin 生成链路。

### 10.1 CSV 字段与 qlib `dump_bin.py` 的匹配要求

`C:/Users/lc999/RD-Agent-main/scripts/dump_bin.py` 基于 `DumpDataBase`/`DumpDataAll` 等类实现 CSV→bin 转换，关键参数如下：

- `data_path`：CSV 所在目录（或单一 CSV 文件路径）；  
- `qlib_dir`：目标 bin 目录；  
- `freq`：`day`/`1min` 等；  
- `date_field_name`：CSV 中日期字段名，默认 `date`；  
- `symbol_field_name`：证券代码字段名，默认 `symbol`；  
- `file_suffix`：文件后缀，默认 `.csv`；  
- `exclude_fields` / `include_fields`：需要排除或仅保留的字段；  
- `max_workers`：多进程并发度。

**对 AIstock 导出的 CSV 的要求：**

- CSV 可有两种组织方式（二选一）：  
  1) **每个 symbol 一个 CSV 文件**：文件名为代码（如 `000001.SZ.csv`），可以没有 `symbol` 列；dump 工具会从文件名推断。  
  2) **单/少量大 CSV 文件**：必须包含 `symbol_field_name` 对应的列（如 `symbol` 或 `instrument`），且同一个文件内可含多只股票。

- 字段命名：  
  - 日期列推荐名：`date`（与默认 `date_field_name` 一致）或在调用 `dump_bin.py` 时通过 `--date_field_name` 显式指定；  
  - 证券列推荐名：`symbol`；若使用 `instrument`，需在调用时指定 `--symbol_field_name instrument`；  
  - 其他数值列（`open, high, low, close, volume, amount`）可按文档 3.1 的约定输出，dump 工具会按 `exclude_fields/include_fields` 过滤后，为每个字段生成对应的 `*.bin` 文件。

- **强约束**：  
  - 每只股票在同一 CSV 中的日期不能重复，dump 过程中会用 `drop_duplicates(date_field_name)` 去重；  
  - `date_field_name` 列必须可以被 `pd.to_datetime` 正确解析；  
  - 全部 CSV 合起来的日期集合用于生成 `calendars/{freq}.txt`，因此日期范围要连续/合理。

在 AIstock 的实现中，推荐：

- 导出为“每只股票一个 CSV”的形式，文件名为 `ts_code`（如 `000001.SZ.csv`），并使用：  
  - `symbol_field_name = "symbol"` 或保留默认；  
  - `date_field_name = "date"`；  
  - `file_suffix = ".csv"`。

### 10.2 `dump_bin.py` 输出的 bin 目录结构与附加文件

`dump_bin.py` 在 `qlib_dir` 下自动生成标准 Qlib bin 结构，无需 AIstock 预先准备 `all.txt` 或 `calendars` 文件：

- `calendars/` 目录：  
  - 文件：`{freq}.txt`（如 `day.txt`），每行一个交易日（或时间戳），由所有 CSV 的日期集合推导而来；

- `instruments/` 目录：  
  - 文件：`all.txt`，默认分隔符为制表符 `\t`，格式为：  
    - 列：`symbol, start_datetime, end_datetime`；  
    - `start_datetime/end_datetime` 由每只股票在 CSV 中出现的最早/最晚日期计算；  
    - 写入时会将 symbol 转换为 Qlib 约定的形式（`fname_to_code(...).upper()`）。

- `features/` 目录：  
  - 子目录：每个 symbol 一个子目录（如 `sz000001` 等），命名由 `code_to_fname` 转换；  
  - 文件：`{field_name}.{freq}.bin`（如 `close.day.bin`），每个字段一个 bin 文件，按日历顺序存储浮点数组；

因此：

- AIstock 只需提供 **CSV 目录** 与 **目标 bin 目录**；  
- `calendars/day.txt` 与 `instruments/all.txt` **由 dump 工具自动生成**，无需自行维护；  
- Qlib / RD-Agent 的 `provider_uri` 只需指向 `qlib_dir` 根目录即可。

### 10.3 `check_data_health.py` 的检查逻辑与使用方式

`C:/Users/lc999/RD-Agent-main/scripts/check_data_health.py` 支持两种检查模式：

1. **直接检查 CSV 目录**：  
   - 参数：`csv_path` 指向一个包含若干 `.csv` 的目录；  
   - 每个 CSV 需包含至少 `open, high, low, close, volume` 等列；  
   - 会检查：缺失数据、价格/成交量的异常跳变、必需列是否缺失、`factor` 列是否存在且非全空。

2. **检查 Qlib bin 目录**：  
   - 参数：`qlib_dir` 指向一个已经由 `dump_bin.py` 生成的 bin 目录；  
   - 内部会执行：  
     - `qlib.init(provider_uri=qlib_dir)`；  
     - 通过 `D.instruments(market="all")` + `D.list_instruments(...)` 获取标的列表；  
     - 调用 `D.features([instrument], ["$open", "$close", "$low", "$high", "$volume", "$factor"])` 拉取数据；  
     - 将 `$open/$close/.../$factor` 重命名为普通列名，再执行同样的缺失值与异常跳变检查。

在“方案 A + 轻量 B”的推荐流程中：

- 在运行 `dump_bin.py` 生成 bin 后，**优先使用 `qlib_dir` 模式**调用 `check_data_health.py`：  
  - 可在 AIstock 的 Qlib Snapshot 页面中提供一个“导出并检查”按钮，按顺序执行：  
    1. 导出 CSV；  
    2. 调 `dump_bin.py` 生成 bin；  
    3. 调 `check_data_health.py` 对 `qlib_dir` 做全面检查；  
    4. 将检查结果摘要展示在前端（缺失列、异常跳变、因子缺失等）。

### 10.4 在 WSL + conda 环境中调度 qlib 工具的统一配置

考虑到 Qlib 及其依赖目前安装在 WSL 的 `rdagent-gpu` conda 环境中，推荐在 AIstock 集成的 RD-Agent `.env` 中增加如下配置项：

```env
# ========== Qlib bin 导出 & 健康检查（通过 WSL+conda 调度） ==========
QLIB_WSL_DISTRO=Ubuntu
QLIB_WSL_CONDA_SH=~/miniconda3/etc/profile.d/conda.sh
QLIB_WSL_CONDA_ENV=rdagent-gpu

QLIB_RDAGENT_ROOT_WIN=C:/Users/lc999/RD-Agent-main
QLIB_RDAGENT_ROOT_WSL=/mnt/c/Users/lc999/RD-Agent-main
QLIB_SCRIPTS_SUBDIR=scripts

QLIB_BIN_ROOT_WIN=C:/Users/lc999/NewAIstock/AIstock/qlib_bin
QLIB_CSV_ROOT_WIN=C:/Users/lc999/NewAIstock/AIstock/qlib_csv
```

AIstock 后端通过通用的调度函数在 WSL 中执行 qlib 脚本，例如（伪代码）：

```python
def run_qlib_script_in_wsl(script_name: str, args: list[str]) -> str:
    scripts_dir = f"{QLIB_RDAGENT_ROOT_WSL}/{QLIB_SCRIPTS_SUBDIR}"
    bash_cmd = " && ".join([
        f"source {QLIB_WSL_CONDA_SH}",
        f"conda activate {QLIB_WSL_CONDA_ENV}",
        f"cd {scripts_dir}",
        "python " + " ".join([script_name, *args]),
    ])
    # 通过 wsl.exe 调用指定发行版
    result = subprocess.run(
        ["wsl", "-d", QLIB_WSL_DISTRO, "bash", "-lc", bash_cmd],
        capture_output=True,
        text=True,
    )
    ...
```

在“导出 Qlib bin（CSV→bin）”子页面的后端实现中，可以：

1. 使用 AIstock 自有 DB → CSV 导出逻辑，将日线数据导出到 `QLIB_CSV_ROOT_WIN/{bin_id}`；  
2. 将该 Windows 路径转换为 WSL 路径（或直接在配置中维护对应的 WSL 路径）；  
3. 通过 `run_qlib_script_in_wsl("dump_bin.py", [...])` 执行 CSV→bin；  
4. 再通过 `run_qlib_script_in_wsl("check_data_health.py", ["--qlib_dir", bin_dir_wsl])` 做健康检查；  
5. 将 bin 目录路径（Windows 形式）和检查结果摘要返回前端展示.

### 10.5 qlib dump_bin/check_data_health CSV requirements, generated bin directory structure, and WSL+conda orchestration and .env configuration

The qlib dump_bin and check_data_health scripts have specific requirements for the CSV files and the generated bin directory structure. The CSV files should have the following columns:

* date: the date column, which should be in the format 'YYYY-MM-DD'
* symbol: the symbol column, which should be in the format 'XXXX.SZ' or 'XXXX.SH'
* open: the open price column
* high: the high price column
* low: the low price column
* close: the close price column
* volume: the volume column
* amount: the amount column

The generated bin directory structure should have the following subdirectories:

* calendars: this subdirectory should contain a file named 'day.txt' or '1min.txt', which contains the trading dates or timestamps
* instruments: this subdirectory should contain a file named 'all.txt', which contains the symbol, start date, and end date for each instrument
* features: this subdirectory should contain subdirectories for each symbol, which should contain files named '{field_name}.{freq}.bin', where field_name is the name of the field (e.g. 'open', 'high', etc.) and freq is the frequency (e.g. 'day', '1min', etc.)

The WSL+conda orchestration and .env configuration should be set up as follows:

* QLIB_WSL_DISTRO: the WSL distribution to use (e.g. 'Ubuntu')
* QLIB_WSL_CONDA_SH: the path to the conda shell script (e.g. '~/miniconda3/etc/profile.d/conda.sh')
* QLIB_WSL_CONDA_ENV: the conda environment to use (e.g. 'rdagent-gpu')
* QLIB_RDAGENT_ROOT_WIN: the path to the RD-Agent root directory on Windows (e.g. 'C:/Users/lc999/RD-Agent-main')
* QLIB_RDAGENT_ROOT_WSL: the path to the RD-Agent root directory on WSL (e.g. '/mnt/c/Users/lc999/RD-Agent-main')
* QLIB_SCRIPTS_SUBDIR: the subdirectory containing the qlib scripts (e.g. 'scripts')
* QLIB_BIN_ROOT_WIN: the path to the Qlib bin root directory on Windows (e.g. 'C:/Users/lc999/NewAIstock/AIstock/qlib_bin')
* QLIB_CSV_ROOT_WIN: the path to the Qlib CSV root directory on Windows (e.g. 'C:/Users/lc999/NewAIstock/AIstock/qlib_csv')

The run_qlib_script_in_wsl function should be used to run the qlib scripts in WSL, and should be configured to use the correct WSL distribution, conda environment, and script directory.

### 10.6 Detailed design notes

The detailed design notes for the qlib dump_bin and check_data_health scripts are as follows:

* The dump_bin script should take the following arguments:
	+ csv_path: the path to the CSV file or directory
	+ qlib_dir: the path to the Qlib bin directory
	+ freq: the frequency of the data (e.g. 'day', '1min', etc.)
	+ date_field_name: the name of the date field in the CSV file
	+ symbol_field_name: the name of the symbol field in the CSV file
	+ file_suffix: the suffix of the CSV file (e.g. '.csv')
	+ exclude_fields: a list of fields to exclude from the bin file
	+ include_fields: a list of fields to include in the bin file
	+ max_workers: the number of worker processes to use
* The check_data_health script should take the following arguments:
	+ qlib_dir: the path to the Qlib bin directory
	+ csv_path: the path to the CSV file or directory
	+ freq: the frequency of the data (e.g. 'day', '1min', etc.)
	+ date_field_name: the name of the date field in the CSV file
	+ symbol_field_name: the name of the symbol field in the CSV file
	+ file_suffix: the suffix of the CSV file (e.g. '.csv')
* The scripts should be run in WSL using the run_qlib_script_in_wsl function, which should be configured to use the correct WSL distribution, conda environment, and script directory.
* The scripts should be run with the correct arguments, which should be configured based on the requirements of the qlib dump_bin and check_data_health scripts.
* The output of the scripts should be checked for errors and warnings, and any issues should be reported to the user.
* The scripts should be run in a way that allows for easy debugging and troubleshooting, such as by using print statements or a debugger.
* The scripts should be run in a way that allows for easy testing and validation, such as by using unit tests or integration tests.
