# AIstock Qlib H5 数据集使用备忘录

> 本备忘录用于记录 **AIstock 中已有的 Qlib 相关 H5 数据集** 如何在 **Qlib** 与 **RD-Agent** 中合理使用。
>
> 每增加一个新数据集（新的 *.h5 或 bin 目录），请在本文件中补充一个小节：说明数据结构、推荐使用方式、提示词建议与策略演进方向。

---

## 1. 当前 Snapshot/H5 数据集总览

AIstock 的 Qlib Snapshot 目前主要落在：

- 根目录（Windows）：`C:/Users/lc999/NewAIstock/AIstock/qlib_snapshots/<snapshot_id>/`
- 常见 snapshot 结构示例：
  - `daily_pv.h5`：日频价格与成交量（Qlib 标准日线）
  - `minute_1min.h5`：1 分钟频率价格数据（如已导出）
  - `moneyflow.h5`：个股资金流向（基于 `market.moneyflow_ts`）
  - `daily_basic.h5`：Tushare daily_basic 股票每日指标
  - `instruments/all.txt`：标的与起止日期
  - `calendars/day.txt`：交易日历

本备忘录先覆盖以下 H5：

1. `daily_pv.h5`  —— 日线行情与成交量
2. `moneyflow.h5` —— 个股资金流向（moneyflow_ts 汇总）
3. `daily_basic.h5` —— 股票每日指标（Tushare daily_basic）

后续如新增：

- 板块日线/板块成分 H5
- 自定义因子 H5（如 AE 重构误差汇总表等）

请在第 5 节之后继续扩展。

---

## 2. `daily_pv.h5` —— 日线行情（Qlib 标准价格面板）

### 2.1 数据结构

- 文件路径示例：
  - `C:/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_YYYYMMDD/daily_pv.h5`
- 存储格式：
  - HDF5，键名通常为 `data`
- DataFrame 索引与列：
  - Index：`MultiIndex(datetime, instrument)`
  - Columns：
    - `$open`、`$high`、`$low`、`$close`、`$volume` 等价格/成交量字段
    - 可能包含 `$factor`（前复权因子）等

> 这是 Qlib 与 RD-Agent 的“主行情面板”，大多数价格相关策略都会直接或间接基于这个数据集。

### 2.2 在 Qlib 中的推荐用法

- 作为主数据源挂入 Qlib `DataHandler`：
  - 通常通过已有配置模板（RD-Agent 场景中的 conf_*yaml）自动加载。
- 特征示例：
  - 简单因子：`$close`、`$volume`、`$high/$low`、日收益率、波动率等
  - 技术指标：
    - MA、EMA、MACD、RSI、布林带等
  - 价格结构：
    - 过去 N 日高低点、趋势斜率、波动率聚类等

### 2.3 在 RD-Agent 中的推荐用法

- **训练与回测的主输入**：
  - 模型输入 feature 通常包含 `$close`、`$volume` 及其派生特征。
- 与其它 H5（moneyflow/daily_basic）组合：
  - 将 `daily_pv.h5` 作为“价格基准”，用 instrument+datetime 与其它面板 join。

### 2.4 提示词修改建议（针对 LLM）

在向 RD-Agent / LLM 描述需求时，建议 **显式告知已有的 daily_pv 面板**：

- 强调：
  - “已有 Qlib 标准的 `daily_pv.h5`，包含 `$open/$high/$low/$close/$volume` 等字段。”
  - “请直接基于这些字段构造价格与成交量相关特征，不要重复造轮子。”
- 示例提示语：
  - “请基于 Snapshot 中的 `daily_pv.h5`（Qlib 标准格式，字段 `$open/$close/$high/$low/$volume`）设计日频因子，不要重新从 CSV 导入数据。”

### 2.5 策略演进方向建议

- 从简单到复杂：
  - 起步：单因子动量/均值回归策略（如 20 日动量、均值回归区间）。
  - 进阶：多因子线性模型或树模型（将价格特征与资金流、基本面因子组合）。
  - 高级：时序模型（LSTM/Transformer）捕捉价格路径信息。
- 日后可结合资金流 (`moneyflow.h5`) 与基本面 (`daily_basic.h5`) 做多模态 Alpha。

---

## 3. `moneyflow.h5` —— 个股资金流向

### 3.1 数据来源与结构

- 源库表：`market.moneyflow_ts`（Tushare moneyflow_ts 聚合）
- 导出文件：`moneyflow.h5`
- DataFrame：
  - Index：`MultiIndex(datetime, instrument)`
  - Columns：`mf_*` 系列字段：
    - `mf_sm_buy_vol`, `mf_sm_sell_vol`, `mf_md_buy_amt`, `mf_lg_buy_vol`, ...
    - `mf_net_vol`, `mf_net_amt` 等净流入量/额
  - 单位：
    - `_vol`：股
    - `_amt`：元

### 3.2 Qlib 中的使用方式

- 作为 **附加因子面板**：
  - 与日线行情对齐（相同的 datetime/instrument 组合）。
  - 可通过自定义 `FeatureD`、`StaticDataLoader` 或 DataHandler 扩展配置，将 `mf_*` 列加入 factor pipeline。
- 典型特征构造：
  - 当日/多日累积净流入：`rolling_sum(mf_net_amt, N)`
  - 大单/超大单占比：`(mf_lg_buy_amt + mf_elg_buy_amt) / total_amount`
  - 资金流量相对市值：`mf_net_amt / db_total_mv`（结合 daily_basic）。

### 3.3 RD-Agent 中的使用方式

- 在因子 workspace 中：
  - 读取 `moneyflow.h5` 与 `daily_pv.h5`，按 index join。
  - 将 `mf_*` 作为新增特征喂给模型（例如 `mf_net_amt_5d` 等）。

### 3.4 提示词建议

- 明确告诉 LLM：
  - “已经有 `moneyflow.h5`，字段为 `mf_*` 系列，索引与 `daily_pv.h5` 对齐。”
  - “请基于这些资金流特征构造短期资金驱动型因子，与价格趋势结合。”
- 避免：
  - 重新访问数据库/外部 API 抓 `moneyflow`；要求 LLM 直接用现有 H5。

### 3.5 策略演进方向

- 短线情绪+资金驱动：
  - 资金不断流入且价格稳步上行 → 多头信号。
  - 资金大幅流出但价格尚未下跌 → 预警信号。
- 与成交量/换手率结合：
  - 高换手 + 正净流入 → 强势接力；
  - 高换手 + 负净流入 → 高位出货风险。

---

## 4. `daily_basic.h5` —— 每日基本面指标

### 4.1 数据来源与结构

- 源库表：`market.daily_basic`（由 `scripts/ingest_tushare_daily_basic.py` 写入）
- 字段（核心）：
  - `trade_date`, `ts_code`
  - `close`
  - `turnover_rate`, `turnover_rate_f`, `volume_ratio`
  - `pe`, `pe_ttm`, `pb`, `ps`, `ps_ttm`
  - `dv_ratio`, `dv_ttm`
  - `total_share`, `float_share`, `free_share`, `total_mv`, `circ_mv`
- 导出文件：`daily_basic.h5`
- 导出后的 DataFrame：
  - Index：`MultiIndex(datetime, instrument)`
  - Columns：`db_*` 系列：
    - `db_pe`, `db_pb`, `db_ps`, `db_ps_ttm` 等估值因子
    - `db_total_mv`, `db_circ_mv` 等市值因子（已从万元转换为元）
    - `db_turnover_rate`, `db_volume_ratio` 等流动性因子

### 4.2 Qlib 中的推荐用法

- 作为 **基本面/估值因子面板**：
  - 与 `daily_pv.h5` 按 index 完全对齐；
  - 可以通过 StaticDataLoader 或自定义 DataHandler 添加到特征空间。
- 典型用法：
  - 估值分层：以 `db_pe`、`db_pb` 构造低估/高估分组；
  - 市值风格：以 `db_circ_mv` 构造大盘/中盘/小盘因子；
  - 流动性筛选：`db_turnover_rate`、`db_volume_ratio` 作为流动性安全因子。

### 4.3 RD-Agent 中的推荐用法

- 将 `daily_basic.h5` 视为 **全局静态/慢变量因子表**：
  - 在因子 workspace 里读取 `daily_pv.h5` + `daily_basic.h5`；
  - 在特征工程阶段 join 后直接把 `db_*` 列喂进模型。
- 例子：
  - 模型输入 = 价格动量（来自 `daily_pv`） + 估值与市值（来自 `daily_basic`） + 资金流（来自 `moneyflow`）。

### 4.4 提示词建议

- 在策略/因子生成提示中加入：
  - “已有 `daily_basic.h5`，列名为 `db_*` 系列，包含 pe/pb/市值/换手率等基本面与流动性指标。”
  - “请结合 `daily_pv.h5` 与 `moneyflow.h5` 设计多因子模型，其中 `daily_basic.h5` 主要提供慢变量（估值/市值/流动性）。”
- 显式要求：
  - 不要重复从 Tushare 拉每日指标，而是优先使用 `daily_basic.h5`。

### 4.5 策略演进方向

- 风格因子维度：
  - 价值 vs 成长：低 `db_pe/db_pb` + 高盈利增长（后续可扩展盈利因子）。
  - 大盘 vs 小盘：按 `db_circ_mv` 分 bucket。
- 风险控制：
  - 根据 `db_turnover_rate`、`db_circ_mv` 剔除流动性差的个股；
  - 使用估值因子作为 risk factor，在组合构建时做中性化处理。

---

## 5. 对 LLM 提示词与策略演进的总建议

1. **始终声明现有数据资产**：
   - 在任何 Qlib/RD-Agent 相关需求的 prompt 中，优先说明：
     - 已有：`daily_pv.h5`、`moneyflow.h5`、`daily_basic.h5`、以及预计算因子表（如 AE 重构误差）。
     - 希望 **基于这些现有 H5** 构建因子与策略，而不是重新从外部数据源抓数据。

2. **明确索引和列命名约定**：
   - 索引：`MultiIndex(datetime, instrument)`；
   - 行情列：`$open/$close/...`；
   - 资金流列：`mf_*`；
   - 基本面列：`db_*`；
   - 预计算因子列：按因子名前缀（如 `ae_recon_error_10d`）。

3. **鼓励“多面板融合”而非单一面板**：
   - 在 prompt 中建议 LLM：
     - “请将不同 H5（价格、资金、基本面、预计算因子）视为多个 feature 面板，通过 datetime+instrument join 后建模。”

4. **避免的坑**：
   - 每个滑动窗口训练一个新模型（例如自编码器） → 计算爆炸、无法复现；
   - 重复拉取相同原始数据，而不是利用现有的 snapshot/H5；
   - 忽略 ST/退市过滤逻辑，导致回测与真实可交易标的不一致。

5. **演进路线建议**：
   - 第一步：单一面板（`daily_pv`）上做经典技术因子策略；
   - 第二步：加入 `moneyflow` 作为资金流因子，构建“价格+资金”双因子模型；
   - 第三步：再将 `daily_basic` 纳入，形成“价格+资金+基本面”三维因子；
   - 第四步：在此基础上尝试预计算因子（如全局 AE 重构误差、情绪因子等）。

---

## 6. 未来扩展约定

- 每当在 AIstock 中新增一个 H5 数据集或新的因子表（例如：板块日线、行业因子、宏观因子、AE/自监督因子等），请在本备忘录中新增一个小节，遵循以下结构：

1. 数据来源与库表 / 脚本位置；
2. H5 结构（索引 + 列 + 单位/前缀约定）；
3. 在 Qlib 中的典型使用模式；
4. 在 RD-Agent 中的典型使用模式；
5. 对提示词的补充建议；
6. 推荐的策略演进方向。

- 这样可以让未来的你（以及 LLM）在扩展策略与因子时，始终围绕“已有数据资产”来设计，避免重复建设与无效计算。

---

## 2025-12-14 更新：static_factors.parquet/schema 与资金流 rolling 特征接入（RD-Agent/Qlib 因子演进）

1. **统一静态因子表（repo 侧产出）**：
   - 新增/更新 repo 内的 `git_ignore_folder/factor_implementation_source_data/static_factors.parquet`，并配套输出：
     - `static_factors_schema.csv`
     - `static_factors_schema.json`
   - 该表用于把 `daily_basic.h5`（`db_*`）与 `moneyflow.h5`（`mf_*`）等字段以统一方式提供给因子脚本（`factor.py`）在运行时可选 join。

2. **资金流 rolling 派生特征**：
   - 在静态因子表中补齐了资金流相关的派生列（例如 `*_5d` / `*_20d` 这类 rolling 聚合列），并通过 schema 文件显式列出字段白名单，避免 LLM 继续编造字段名。

3. **因子运行目录的数据拷贝与 schema 注入**：
   - RD-Agent 在准备因子执行数据目录时，会优先拷贝 repo 生成的 `static_factors.parquet` 与 schema（csv/json）到因子运行目录（包含 debug 目录），确保 LLM 能在 prompt 中看到 schema 描述，并在 `factor.py` 中按需读取 join。

4. **失败重试（FactorAutoRepair）与典型失败类型**：
   - 扩展了自动修复触发的失败签名覆盖面，便于在“缺列 / 全 NaN / 空结果”等场景触发修复重试。

5. **关联备忘录与提示词全量打印**：
   - 备忘录：`docs/20251214_因子演进_debug_static_factors_rolling_retry备忘录.md`
   - 提示词 dump 与诊断：`docs/20251214_QLib因子全量提示词_dump与诊断.md`
