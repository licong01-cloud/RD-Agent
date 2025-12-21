# RD-Agent 预计算/派生因子说明（由 RD-Agent 脚本生成）

本文件用于说明：
- `C:\Users\lc999\NewAIstock\AIstock\factors` 下由 RD-Agent 相关脚本生成的预计算因子；
- `C:\Users\lc999\NewAIstock\AIstock\qlib_snapshots\qlib_export_20251209` 中 `static_factors.parquet` 合并出来的预计算/派生字段的含义与计算口径。

**重要澄清**：AIstock 在本链路中仅负责按 RD-Agent 的要求导出基础数据（如 `daily_pv.h5`、`daily_basic.h5`、`moneyflow.h5` 以及 qlib bin/metadata）。本文件描述的所有“因子生成/加工/合并”，均由 RD-Agent 仓库内脚本完成。

---

## 1. 因子生成与合并链路（对应本地脚本）

### 1.1 daily_basic 预计算因子
- **脚本**：`RD-Agent-main/precompute_daily_basic_factors.py`
- **输入**：`<snapshot_root>/daily_basic.h5`（默认 `C:/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_20251209/daily_basic.h5`）
- **输出**：
  - `C:/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors/result.h5`
  - `C:/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors/result.pkl`
  - `C:/Users/lc999/NewAIstock/AIstock/factors/daily_basic_factors/result.parquet`

### 1.2 moneyflow 预计算因子
- **脚本**：`RD-Agent-main/tools/precompute_moneyflow_factors.py`
- **输入**：
  - `<snapshot_root>/moneyflow.h5`
  - `<snapshot_root>/daily_pv.h5`（用于提供 `amount/volume`）
- **输出**：
  - `C:/Users/lc999/NewAIstock/AIstock/factors/moneyflow_factors/result.pkl`

### 1.3 AE 重构误差因子（ae_recon_error_10d）
- **因子计算脚本（CoSTEER 因子形式）**：`RD-Agent-main/factor_ae_recon_error_10d.py`
- **格式转换脚本**：`RD-Agent-main/convert_ae_factor_h5_to_parquet.py`
- **输入**：
  - `daily_pv.h5`（从当前目录读取，字段会把 `$open/$high/...` 统一重命名为 `open/high/...`）
  - 预训练模型文件：`models/ae_10d.pth`（由环境变量 `AE10D_MODEL_PATH` 指定，默认该路径）
- **输出**：
  - `result.h5`（单列 `ae_recon_error_10d`）
  - 之后可转换为 `result.pkl/result.parquet` 供静态加载

### 1.4 static_factors.parquet 合并脚本
- **脚本**：`RD-Agent-main/tools/generate_static_factors_bundle.py`
- **输入**：
  - snapshot：`daily_basic.h5`、`moneyflow.h5`
  - 可选预计算表（存在则合并）：
    - `.../factors/daily_basic_factors/result.pkl`
    - `.../factors/moneyflow_factors/result.pkl`
    - `.../factors/ae_recon_error_10d/result.pkl`
    - `.../factors/combined_static_factors.parquet`
  - 并在合并后**额外派生**一组 `mf_*` 的稳健字段（见第 3 节）
- **输出**：
  - `static_factors.parquet`
  - `static_factors_schema.json`
  - `static_factors_schema.csv`

---

## 2. daily_basic_factors：估值/规模/流动性预计算因子

> 来源：`precompute_daily_basic_factors.py`

### 2.1 value_pe_inv
- **中文说明**：倒数市盈率（估值因子）。数值越大通常代表越“便宜”。
- **公式/口径**：
  - 若存在 `db_pe_ttm`：`value_pe_inv = 1 / db_pe_ttm`（`db_pe_ttm=0` 视为缺失）
  - 否则若存在 `db_pe`：`value_pe_inv = 1 / db_pe`（`db_pe=0` 视为缺失）
- **输入字段**：`daily_basic.h5` 中的 `db_pe_ttm` 或 `db_pe`

### 2.2 value_pb_inv
- **中文说明**：倒数市净率（估值因子）。数值越大通常代表越“便宜”。
- **公式/口径**：`value_pb_inv = 1 / db_pb`（`db_pb=0` 视为缺失）
- **输入字段**：`daily_basic.h5` 中的 `db_pb`

### 2.3 size_log_mv
- **中文说明**：市值对数（规模因子）。
- **公式/口径**：
  - `mv_col` 优先取 `db_circ_mv`，否则取 `db_total_mv`
  - `size_log_mv = log(mv_col)`（仅对 `mv_col>0` 取对数；非正值视为缺失）
- **输入字段**：`daily_basic.h5` 中的 `db_circ_mv` 或 `db_total_mv`

### 2.4 liquidity_turnover
- **中文说明**：换手率（流动性因子）。
- **公式/口径**：`liquidity_turnover = db_turnover_rate`
- **输入字段**：`daily_basic.h5` 中的 `db_turnover_rate`

### 2.5 liquidity_vol_ratio
- **中文说明**：量比（流动性因子）。
- **公式/口径**：`liquidity_vol_ratio = db_volume_ratio`
- **输入字段**：`daily_basic.h5` 中的 `db_volume_ratio`

---

## 3. moneyflow_factors：资金流向预计算因子

> 来源：`tools/precompute_moneyflow_factors.py`
> 
> 注意：该脚本会把 instrument 从 `SH600000/SZ000001` 映射成 `600000.SH/000001.SZ` 以便与 qlib 对齐。

脚本使用 `moneyflow.h5` 的资金流字段 + `daily_pv.h5` 的成交额/成交量，生成以下因子：

### 3.1 mf_total_net_amt
- **中文说明**：全市场口径的资金净流入金额（当日）。
- **公式/口径**：`mf_total_net_amt = mf_net_amt`
- **输入字段**：`moneyflow.h5` 的 `mf_net_amt`

### 3.2 mf_total_net_vol
- **中文说明**：全市场口径的资金净流入量（当日）。
- **公式/口径**：`mf_total_net_vol = mf_net_vol`
- **输入字段**：`moneyflow.h5` 的 `mf_net_vol`

### 3.3 mf_total_net_amt_ratio
- **中文说明**：净流入金额强度（净流入金额/成交额）。
- **公式/口径**：`mf_total_net_amt_ratio = mf_net_amt / amount`
- **输入字段**：`mf_net_amt` + `daily_pv` 的 `amount/$amount`

### 3.4 mf_total_net_vol_ratio
- **中文说明**：净流入量强度（净流入量/成交量）。
- **公式/口径**：`mf_total_net_vol_ratio = mf_net_vol / volume`
- **输入字段**：`mf_net_vol` + `daily_pv` 的 `volume/$volume`

### 3.5 mf_main_net_amt
- **中文说明**：主力净流入金额（大单+超大单）。
- **公式/口径**：
  - `mf_main_net_amt = (mf_lg_buy_amt + mf_elg_buy_amt) - (mf_lg_sell_amt + mf_elg_sell_amt)`

### 3.6 mf_main_net_vol
- **中文说明**：主力净流入量（大单+超大单）。
- **公式/口径**：
  - `mf_main_net_vol = (mf_lg_buy_vol + mf_elg_buy_vol) - (mf_lg_sell_vol + mf_elg_sell_vol)`

### 3.7 mf_main_net_amt_ratio
- **中文说明**：主力净流入金额强度（主力净流入金额/成交额）。
- **公式/口径**：`mf_main_net_amt_ratio = mf_main_net_amt / amount`

### 3.8 mf_main_net_vol_ratio
- **中文说明**：主力净流入量强度（主力净流入量/成交量）。
- **公式/口径**：`mf_main_net_vol_ratio = mf_main_net_vol / volume`

### 3.9 mf_elg_net_amt
- **中文说明**：超大单净流入金额。
- **公式/口径**：`mf_elg_net_amt = mf_elg_buy_amt - mf_elg_sell_amt`

### 3.10 mf_elg_net_vol
- **中文说明**：超大单净流入量。
- **公式/口径**：`mf_elg_net_vol = mf_elg_buy_vol - mf_elg_sell_vol`

### 3.11 mf_elg_net_amt_ratio
- **中文说明**：超大单净流入金额强度（超大单净流入金额/成交额）。
- **公式/口径**：`mf_elg_net_amt_ratio = mf_elg_net_amt / amount`

### 3.12 mf_elg_net_vol_ratio
- **中文说明**：超大单净流入量强度（超大单净流入量/成交量）。
- **公式/口径**：`mf_elg_net_vol_ratio = mf_elg_net_vol / volume`

### 3.13 mf_elg_share_in_main_amt
- **中文说明**：超大单在“主力净流入金额”中的占比（用来刻画主力结构，强调超大单）。
- **公式/口径**：`mf_elg_share_in_main_amt = mf_elg_net_amt / mf_main_net_amt`（主力净流入为 0 时记缺失）

### 3.14 mf_elg_share_in_main_vol
- **中文说明**：超大单在“主力净流入量”中的占比。
- **公式/口径**：`mf_elg_share_in_main_vol = mf_elg_net_vol / mf_main_net_vol`（主力净流入为 0 时记缺失）

---

## 4. generate_static_factors_bundle.py：额外派生的 mf_* 稳健字段（静态 bundle 内）

`tools/generate_static_factors_bundle.py` 在合并 `moneyflow.h5`（原始买卖拆单字段）后，会额外派生一组更“schema-friendly”的资金流字段（并包含 5D/20D 滚动聚合）。

这些字段与第 3 节存在“同名但口径不同”的风险：
- 第 3 节来自 `mf_net_amt/mf_net_vol` + `daily_pv` 的成交额/量；
- 本节派生使用的是 `moneyflow.h5` 内部的拆单买卖额/量自行拼出的“买卖总额/量”，分母为“买入额+卖出额”。

在 `static_factors.parquet` 内，如果列名重复，代码会采用 **keep="last"** 的策略保留“最后合并进来”的那一列。

本节字段包括（按脚本实现）：
- `mf_total_net_amt` / `mf_total_net_vol`
- `mf_total_net_amt_ratio` / `mf_total_net_vol_ratio`（分母：全档买卖额/量之和）
- `mf_main_net_amt` / `mf_main_net_vol`
- `mf_main_net_amt_ratio` / `mf_main_net_vol_ratio`（分母：主力买卖额/量之和）
- `mf_elg_net_amt` / `mf_elg_net_vol`
- `mf_elg_net_amt_ratio` / `mf_elg_net_vol_ratio`（分母：超大单买卖额/量之和）
- `mf_elg_share_in_main_amt` / `mf_elg_share_in_main_vol`（注意：该实现是“超大单买入占主力买入比”，并非 net 比）
- 5D/20D：
  - `mf_total_net_amt_{5,20}d`
  - `mf_main_net_amt_{5,20}d`
  - `mf_elg_net_amt_{5,20}d`
  - `mf_total_net_amt_ratio_{5,20}d`
  - `mf_main_net_amt_ratio_{5,20}d`
  - `mf_elg_net_amt_ratio_{5,20}d`

（这些字段的中文 meaning 已在脚本 `_build_schema()` 中内置）

---

## 5. ae_recon_error_10d：自编码器重构误差因子

> 来源：`factor_ae_recon_error_10d.py`

- **中文说明**：10 日窗口的价格/成交特征序列自编码器重构误差。数值越大，表示该股票在该窗口内的行为模式越“异常/不可被常见模式解释”。
- **核心口径**：
  - 把每只股票的时间序列按 window=10（由模型文件内的 `window` 指定）构造成样本序列；
  - 使用预训练 AE 做重构；
  - 因子值为：`mean((x - AE(x))^2)`（对输入维度做均值）。
- **输入字段**：来自 `daily_pv.h5`（具体使用哪些字段由模型文件 `payload["features"]` 决定）

---

## 6. 未确认来源字段（需要进一步溯源）

以下字段在 `static_factors_schema.json` 中出现，但在上述 3 个预计算脚本中没有直接公式实现：

- `PriceStrength_10D`

**说明**：该字段更可能来自以下路径之一：
- `NewAIstock/AIstock/factors/combined_static_factors.parquet`（由 `tools/merge_static_factors_to_parquet.py` 合并时带入）
- workspace 内的 `combined_factors_df.parquet`（历史运行产物）

后续需要通过“列级溯源”（检查 `combined_static_factors.parquet` 与各输入表的列集合差异）来确认其公式与生成脚本。

---

## 7. 与 static_factors_schema 对齐建议

- 对于 `meaning` 为空的原始字段（如部分 `db_*`、`mf_sm_*` 等），可通过 `aistock_field_map.csv` 导出的字段注释补全。
- 对于预计算字段（本文件第 2/3/4/5 节），建议以本文件为权威来源，写入长期维护的文档（即本文件）。
