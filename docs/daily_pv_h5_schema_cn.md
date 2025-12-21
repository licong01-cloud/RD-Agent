# daily_pv.h5 字段口径说明（供 RD-Agent/LLM 使用）

本文档根据 `docs/AIstock_日频H5导出规范.md` 抽取整理，用于说明 RD-Agent 因子执行/回测中使用的 `daily_pv.h5` 数据字段含义、单位、复权口径与计算过程。

## 1. 文件结构与索引

- **H5 key**：`data`
- **写入方式**：`pandas.DataFrame.to_hdf(..., key="data", format="fixed", mode="w")`
- **Index**：`MultiIndex(["datetime", "instrument"])`
  - `datetime`：交易日（`pd.Timestamp`）
  - `instrument`：Tushare `ts_code` 格式（如 `000001.SZ`、`600000.SH`、`430047.BJ`）

## 2. 数据来源（逻辑层）

- 日线未复权行情：`market.kline_daily_raw`（配置项 `DAILY_RAW_TABLE`）
- 复权因子：`market.adj_factor`（配置项 `ADJ_FACTOR_TABLE`）

## 3. 单位与复权口径（关键假设）

### 3.1 价格单位

数据库价格字段单位为“厘”（元 × 1000），导出时换算为“元”。

- `PRICE_UNIT_DIVISOR = 1000.0`

### 3.2 复权策略：前复权

导出时基于 Tushare `adj_factor` 计算前复权因子：

- `qfq_factor = adj_factor / latest_adj_factor`

其中 `latest_adj_factor` 默认取每只股票在导出区间内的最大 `adj_factor`（通常对应最新日期）。

### 3.3 成交量复权处理

- 原始成交量字段为 `volume_hand`（手，1手=100股）
- 导出时先转为股，再按前复权因子做反向调整：
  - `_volume_shares = volume_hand * 100`
  - `$volume = _volume_shares / qfq_factor`

## 4. 字段定义与计算公式

| 字段名 | 中文说明 | 单位/口径 | 计算公式（从原始字段推导） | 备注 |
|---|---|---|---|---|
| `$open` | 当日开盘价（前复权） | 元 | `$open = open_li/1000 * $factor` | `open_li` 为“厘” |
| `$high` | 当日最高价（前复权） | 元 | `$high = high_li/1000 * $factor` | `high_li` 为“厘” |
| `$low` | 当日最低价（前复权） | 元 | `$low = low_li/1000 * $factor` | `low_li` 为“厘” |
| `$close` | 当日收盘价（前复权） | 元 | `$close = close_li/1000 * $factor` | `close_li` 为“厘” |
| `$volume` | 当日成交量（复权调整后） | 股 | `$volume = (volume_hand*100)/$factor` | `volume_hand` 为“手” |
| `$factor` | 前复权因子（qfq_factor） | 无 | `$factor = adj_factor/latest_adj_factor` | 由 `adj_factor` 推导 |
| `$amount` | 当日成交额（若导出包含该列） | 元 | 取决于导出 SQL 是否选择 `amount_li` | **可选列**，默认导出通常不包含 |

## 5. RD-Agent 使用注意事项

- **价格已前复权**：`$open/$high/$low/$close` 已经乘了 `$factor`
- **可还原未复权价格**：例如 `raw_close = $close / $factor`
- **成交量已做复权反向调整**：例如 `raw_volume_shares = $volume * $factor`
- **下游统一使用 ts_code**：不要再假设 `SH600000` 这种 Qlib 旧风格代码

## 6. amount 字段治理策略（强制写清楚，避免误用）

### 6.1 字段语义

`amount` 表示**当日成交额（现金成交金额）**，属于“金额口径”，建议用于衡量当日交易规模。

### 6.2 单位与复权口径

- **单位**：推荐统一为 `元`
- **是否复权**：**不复权**（保持当日真实成交额语义）

> 说明：复权因子主要用于拆分送转导致的价格/股数口径可比性。成交额属于当日真实发生的现金金额，通常不应随复权因子调整。

### 6.3 缺失/占位策略

- **优先策略**：从上游数据源导出**真实成交额**
- **若无法获取真实值**：
  - H5 侧推荐使用 `NaN` 表示缺失
  - 不建议用 `0` 伪装真实成交额（会静默污染所有 amount-based 因子/特征）

## 7. H5 与 bin/CSV 的字段映射与兼容策略（概念规范）

### 7.1 命名差异

- H5（Qlib/RD-Agent 风格）通常使用 `$open/$close/$volume/$factor`
- bin/CSV（dump_bin）通常使用 `open/close/volume/amount` 且不带 `$`

推荐做法：允许存储层命名不同，但在 RD-Agent 数据入口层做 **canonicalization**（统一映射到内部使用的一套字段语义）。

### 7.2 factor 与 amount

- `factor`：建议在 H5 权威数据链路中保留（用于复权一致性诊断、还原未复权值）
- `amount`：
  - 若 bin/CSV 由于格式要求必须补齐该列，但真实值缺失，则允许输出 `0` 作为占位
  - 必须在 schema/文档中明确标注“占位/不可用”，避免下游把 0 当真实成交额
