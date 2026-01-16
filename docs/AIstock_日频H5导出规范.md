# AIstock 日频 H5 导出规范（供 RD-Agent 分析使用）

本文描述 AIstock 后端将日频数据导出为 HDF5（`.h5`）文件的**实际文件结构**、**复权处理**与**字段/表结构映射关系**。

适用范围：

- `daily_pv.h5`（RD-Agent 因子数据：日线 OHLCV + 复权因子）
- `daily_basic.h5`（Tushare `daily_basic` 指标：`db_*` 字段）
- `moneyflow.h5`（Tushare `moneyflow_ts` 指标：`mf_*` 字段）

> 说明：本文描述的是当前实现逻辑（代码为准）。H5 均使用 `pandas.DataFrame.to_hdf(..., key="data", format="fixed")` 写入。

---

## 1. 文件位置与 H5 Key

所有导出文件均位于：

- `<QLIB_SNAPSHOT_ROOT>/<snapshot_id>/`

其中 `QLIB_SNAPSHOT_ROOT` 由：

- 环境变量 `QLIB_SNAPSHOT_ROOT`（优先）
- 否则为项目根目录下 `qlib_snapshots/`

H5 写入约定：

- **H5 key**：固定为 `data`
- **写入模式**：全量导出 `mode="w"`
- **HDF format**：`format="fixed"`

相关实现：

- `backend/qlib_exporter/snapshot_writer.py::SnapshotWriter.write_factor_data`

---

## 2. 统一索引（Index）结构

所有日频 H5（`daily_pv.h5 / daily_basic.h5 / moneyflow.h5`）在写入前都应整理为：

- Index：`MultiIndex(["datetime", "instrument"])`

其中：

- `datetime`：交易日（`pd.Timestamp`）
- `instrument`：股票标识，统一为 **Tushare `ts_code`** 格式，例如：
  - `000001.SZ`
  - `600000.SH`
  - `430047.BJ`

> 注：当前实现中 `instrument` 明确使用 `ts_code`（而非 `SH600000`）。

---

## 3. `daily_pv.h5`（RD-Agent 因子数据）

### 3.1 文件语义

`daily_pv.h5` 用于给 RD-Agent 提供日线基础特征（可作为因子研究/回测的输入）：

- OHLC（复权后）
- 成交量（按复权因子做反向调整）
- 复权因子 `$factor`

导出入口：

- `backend/qlib_exporter/exporter.py::QlibFactorExporter.export_full`

写入实现：

- `backend/qlib_exporter/snapshot_writer.py::SnapshotWriter.write_factor_data`

### 3.2 源数据表

- 日线不复权行情：`market.kline_daily_raw`（配置项 `DAILY_RAW_TABLE`）
- 复权因子：`market.adj_factor`（配置项 `ADJ_FACTOR_TABLE`）
  - 若表不存在或无数据：可使用 Tushare API 回退（受 `TUSHARE_TOKEN` 影响）

### 3.3 价格单位

数据库价格字段单位为 **“厘”**（元 × 1000）：

- `PRICE_UNIT_DIVISOR = 1000.0`

导出时会先将价格从 “厘” 换算为 “元”。

### 3.4 复权策略（前复权）

复权因子来源：Tushare `adj_factor`（概念上属于后复权因子）。

在导出中会计算前复权因子：

- `qfq_factor = adj_factor / latest_adj_factor`
  - `latest_adj_factor` 默认取**每只股票在给定区间内的最大 adj_factor（通常对应最新日期）**

相关实现：

- `backend/qlib_exporter/adj_factor_provider.py::AdjFactorProvider.calculate_qfq_factor`

合并要求：

- 日线价格表与复权表按 `(ts_code, trade_date)` 左连接
- 若合并后仍存在缺失复权因子：**抛出 RuntimeError（严格模式，不允许继续导出）**

### 3.5 导出字段（列）定义

`daily_pv.h5` 的 DataFrame 列如下：

- `$open`：前复权开盘价（元）
- `$high`：前复权最高价（元）
- `$low`：前复权最低价（元）
- `$close`：前复权收盘价（元）
- `$volume`：成交量（股，按前复权因子反向调整）
- `$factor`：前复权因子（`qfq_factor`）

> 备注：实现中 `$amount` 是可选列（如果查询结果包含 `amount_li` 才会写入）。当前日线查询 SQL 未选择 `amount_li`，因此通常不会出现 `$amount`。

### 3.5.1 字段中文说明（RD-Agent 口径）

| 字段名 | 中文说明 | 单位/口径 | 备注 |
|---|---|---|---|
| `$open` | 当日开盘价（前复权） | 元 | `open_li`（厘）÷1000×`$factor` |
| `$high` | 当日最高价（前复权） | 元 | `high_li`（厘）÷1000×`$factor` |
| `$low` | 当日最低价（前复权） | 元 | `low_li`（厘）÷1000×`$factor` |
| `$close` | 当日收盘价（前复权） | 元 | `close_li`（厘）÷1000×`$factor` |
| `$volume` | 当日成交量（复权调整后） | 股 | `volume_hand`（手）×100 ÷`$factor` |
| `$factor` | 前复权因子 | 无 | `adj_factor / latest_adj_factor` |

### 3.6 字段计算公式

#### 3.6.1 价格（前复权）

数据库字段（以 `_li` 结尾）单位为 “厘”，先转为元，再乘以前复权因子：

- `$open  = open_li  / 1000 * qfq_factor`
- `$high  = high_li  / 1000 * qfq_factor`
- `$low   = low_li   / 1000 * qfq_factor`
- `$close = close_li / 1000 * qfq_factor`

#### 3.6.2 成交量

数据库成交量字段：

- `volume_hand`：手（1 手 = 100 股）

导出逻辑：

- `_volume_shares = volume_hand * 100`
- `$volume = _volume_shares / qfq_factor`

#### 3.6.3 复权因子

- `$factor = qfq_factor`

相关实现：

- `backend/qlib_exporter/db_reader.py::DBReader.load_qlib_daily_data`

---

## 4. `daily_basic.h5`（Tushare daily_basic 指标）

### 4.1 源数据表

- `market.daily_basic`

### 4.2 数据区间与过滤（与 UI 参数一致）

- `trade_date >= start`
- `trade_date <= end`
- 可选按交易所过滤：基于 `ts_code` 后缀（`.SH/.SZ/.BJ`）
- `exclude_st=True`：排除 `market.stock_st` 中出现过的股票
- `exclude_delisted_or_paused=True`：排除 `market.stock_basic.list_status IN ('D','P')`

### 4.3 导出字段（db_*）映射

`daily_basic.h5` 输出列固定为下列 `db_*`：

| DB 列名 | H5 列名 |
|---|---|
| `close` | `db_close` |
| `turnover_rate` | `db_turnover_rate` |
| `turnover_rate_f` | `db_turnover_rate_f` |
| `volume_ratio` | `db_volume_ratio` |
| `pe` | `db_pe` |
| `pe_ttm` | `db_pe_ttm` |
| `pb` | `db_pb` |
| `ps` | `db_ps` |
| `ps_ttm` | `db_ps_ttm` |
| `dv_ratio` | `db_dv_ratio` |
| `dv_ttm` | `db_dv_ttm` |
| `total_share` | `db_total_share` |
| `float_share` | `db_float_share` |
| `free_share` | `db_free_share` |
| `total_mv` | `db_total_mv` |
| `circ_mv` | `db_circ_mv` |

### 4.4 字段中文说明（来自 `aistock_field_map.csv`）

| 字段名 | 中文说明 | 单位 | 来源表 | 备注 |
|---|---|---|---|---|
| `db_circ_mv` | 流通市值(万元) |  | `daily_basic` | 流通市值(万元) |
| `db_close` | 当日收盘价 |  | `daily_basic` | 当日收盘价 |
| `db_dv_ratio` | 股息率(%) |  | `daily_basic` | 股息率(%) |
| `db_dv_ttm` | 股息率(TTM)(%) |  | `daily_basic` | 股息率(TTM)(%) |
| `db_float_share` | 流通股本(万股) |  | `daily_basic` | 流通股本(万股) |
| `db_free_share` | 自由流通股本(万) |  | `daily_basic` | 自由流通股本(万) |
| `db_pb` | 市净率(总市值/净资产) |  | `daily_basic` | 市净率(总市值/净资产) |
| `db_pe` | 市盈率(总市值/净利润, 亏损的PE为空) |  | `daily_basic` | 市盈率(总市值/净利润, 亏损的PE为空) |
| `db_pe_ttm` | 市盈率(TTM,亏损的PE为空) |  | `daily_basic` | 市盈率(TTM,亏损的PE为空) |
| `db_ps` | 市销率 |  | `daily_basic` | 市销率 |
| `db_ps_ttm` | 市销率(TTM) |  | `daily_basic` | 市销率(TTM) |
| `db_total_mv` | 总市值(万元) |  | `daily_basic` | 总市值(万元) |
| `db_total_share` | 总股本(万股) |  | `daily_basic` | 总股本(万股) |
| `db_turnover_rate` | 换手率(%) |  | `daily_basic` | 换手率(%) |
| `db_turnover_rate_f` | 换手率(自由流通股) |  | `daily_basic` | 换手率(自由流通股) |
| `db_volume_ratio` | 量比 |  | `daily_basic` | 量比 |

类型规范：

- 输出统一转换为 `float32`

相关实现：

- `backend/qlib_exporter/db_reader.py::DBReader.load_daily_basic_panel`

---

## 5. `moneyflow.h5`（Tushare moneyflow_ts 指标）

### 5.1 源数据表

- `market.moneyflow_ts`

### 5.2 数据区间与过滤（与 UI 参数一致）

- `trade_date >= start`
- `trade_date <= end`
- 可选按交易所过滤：基于 `ts_code` 后缀（`.SH/.SZ/.BJ`）
- `exclude_st=True`：排除 `market.stock_st`
- `exclude_delisted_or_paused=True`：排除 `market.stock_basic.list_status IN ('D','P')`

### 5.3 导出字段（mf_*）映射

`moneyflow.h5` 输出列为 `mf_*` 系列，来自 `moneyflow_ts` 字段重命名：

| DB 列名 | H5 列名 |
|---|---|
| `buy_sm_vol` | `mf_sm_buy_vol` |
| `sell_sm_vol` | `mf_sm_sell_vol` |
| `buy_sm_amount` | `mf_sm_buy_amt` |
| `sell_sm_amount` | `mf_sm_sell_amt` |
| `buy_md_vol` | `mf_md_buy_vol` |
| `sell_md_vol` | `mf_md_sell_vol` |
| `buy_md_amount` | `mf_md_buy_amt` |
| `sell_md_amount` | `mf_md_sell_amt` |
| `buy_lg_vol` | `mf_lg_buy_vol` |
| `sell_lg_vol` | `mf_lg_sell_vol` |
| `buy_lg_amount` | `mf_lg_buy_amt` |
| `sell_lg_amount` | `mf_lg_sell_amt` |
| `buy_elg_vol` | `mf_elg_buy_vol` |
| `sell_elg_vol` | `mf_elg_sell_vol` |
| `buy_elg_amount` | `mf_elg_buy_amt` |
| `sell_elg_amount` | `mf_elg_sell_amt` |
| `net_mf_vol` | `mf_net_vol` |
| `net_mf_amount` | `mf_net_amt` |

### 5.4 字段中文说明（来自 `aistock_field_map.csv`）

| 字段名 | 中文说明 | 单位 | 来源表 | 备注 |
|---|---|---|---|---|
| `mf_elg_buy_amt` | 特大单买入金额（元） |  | `moneyflow` | 特大单买入金额（万元） |
| `mf_elg_buy_vol` | 特大单买入量（股） |  | `moneyflow` | 特大单买入量（手） |
| `mf_elg_sell_amt` | 特大单卖出金额（元） |  | `moneyflow` | 特大单卖出金额（万元） |
| `mf_elg_sell_vol` | 特大单卖出量（股） |  | `moneyflow` | 特大单卖出量（手） |
| `mf_lg_buy_amt` | 大单买入金额（元） |  | `moneyflow` | 大单买入金额（万元） |
| `mf_lg_buy_vol` | 大单买入量（股） |  | `moneyflow` | 大单买入量（手） |
| `mf_lg_sell_amt` | 大单卖出金额（元） |  | `moneyflow` | 大单卖出金额（万元） |
| `mf_lg_sell_vol` | 大单卖出量（股） |  | `moneyflow` | 大单卖出量（手） |
| `mf_md_buy_amt` | 中单买入金额（元） |  | `moneyflow` | 中单买入金额（万元） |
| `mf_md_buy_vol` | 中单买入量（股） |  | `moneyflow` | 中单买入量（手） |
| `mf_md_sell_amt` | 中单卖出金额（元） |  | `moneyflow` | 中单卖出金额（万元） |
| `mf_md_sell_vol` | 中单卖出量（股） |  | `moneyflow` | 中单卖出量（手） |
| `mf_net_amt` | 净流入额（元） |  | `moneyflow` | 净流入额（万元） |
| `mf_net_vol` | 净流入量（股） |  | `moneyflow` | 净流入量（手） |
| `mf_sm_buy_amt` | 小单买入金额（元） |  | `moneyflow` | 小单买入金额（万元） |
| `mf_sm_buy_vol` | 小单买入量（股） |  | `moneyflow` | 小单买入量（手） |
| `mf_sm_sell_amt` | 小单卖出金额（元） |  | `moneyflow` | 小单卖出金额（万元） |
| `mf_sm_sell_vol` | 小单卖出量（股） |  | `moneyflow` | 小单卖出量（手） |

### 5.5 单位口径说明（重要）

`aistock_field_map.csv` 中 moneyflow 的中文说明里同时出现“元”和“万元/手”的表述：

- `meaning_cn` 目前写为“（元）/（股）”
- `comment` 里保留了原始表注释“（万元）/（手）”

以导出代码为准：

- `*_amt`：导出为数据库读取到的数值（当前代码未对金额做额外倍率换算），RD-Agent 侧如需要统一为“元”，应以实际 DB 存储单位为准。
- `*_vol`：导出为数据库读取到的数值（当前代码未对量做额外倍率换算），RD-Agent 侧如需要统一为“股”，应以实际 DB 存储单位为准。

类型规范：

- 输出统一转换为 `float32`

相关实现：

- `backend/qlib_exporter/db_reader.py::DBReader.load_moneyflow_panel`

---

## 6. RD-Agent 使用建议（关键假设）

- **价格已前复权**：`$open/$high/$low/$close` 已经乘了 `$factor`
- **原始未复权价格可还原**：
  - `raw_close = $close / $factor`
- **成交量已经做了复权反向调整**：
  - `raw_volume_shares = $volume * $factor`
- **instrument 统一为 ts_code**：RD-Agent 下游不应再假设 `SH600000` 格式

---

## 7. 相关代码索引

- `backend/qlib_exporter/exporter.py`
  - `QlibFactorExporter.export_full`（导出 `daily_pv.h5`）
- `backend/qlib_exporter/snapshot_writer.py`
  - `SnapshotWriter.write_factor_data`（H5 写入）
- `backend/qlib_exporter/db_reader.py`
  - `DBReader.load_qlib_daily_data`（日线计算 + 复权）
  - `DBReader.load_daily_basic_panel`（daily_basic -> db_*）
  - `DBReader.load_moneyflow_panel`（moneyflow_ts -> mf_*）
- `backend/qlib_exporter/adj_factor_provider.py`
  - `AdjFactorProvider.get_adj_factor` / `calculate_qfq_factor`
- `backend/qlib_exporter/config.py`
  - `DAILY_RAW_TABLE` / `ADJ_FACTOR_TABLE` / `PRICE_UNIT_DIVISOR`
