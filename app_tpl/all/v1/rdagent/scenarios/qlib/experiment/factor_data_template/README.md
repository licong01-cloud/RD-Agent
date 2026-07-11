# 数据文件读取方法

读取 HDF5 文件示例：

```python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```

**注意：所有 HDF5 文件的 key 统一为 `"data"`。**

所有 H5 文件的索引均为 MultiIndex(`datetime`, `instrument`)。

# 数据文件概览

| 文件名 | 说明 | 字段前缀 |
|--------|------|----------|
| `daily_pv.h5` | 复权日频价量数据 | 无前缀 |
| `daily_basic.h5` | 每日基本面指标（静态PE、市净率、换手率、市值等） | `db_` |
| `moneyflow.h5` | 每日资金流向（各档买卖量额、净流入等） | `mf_` |
| `bak_basic.h5` | 历史基本面（动态PE、每股收益、营收/利润同比、毛利率等） | `bb_` |
| `cyq_perf.h5` | 筹码分布绩效（历史最高/最低价、成本分位、胜率等） | `cp_` |
| `sector_data.h5` | 申万二级行业数据（行业指数行情、行业资金流向、行业估值） | `sw2_` |
| `static_factors.parquet` | 预合并的静态因子表，包含 db_/mf_/bb_/cp_/sw2_ 全部字段及预计算衍生因子 | 混合 |

# 各数据集字段详细说明

## daily_pv.h5 — 复权日频价量数据（7个字段）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `open` | float | 开盘价 |
| `close` | float | 收盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `volume` | float | 成交量 |
| `amount` | float | 成交额 |
| `factor` | float | 复权因子 |

## daily_basic.h5 — 每日基本面指标（16个字段，前缀 `db_`）

| 字段名 | 类型 | 说明 | 单位 |
|--------|------|------|------|
| `db_close` | float32 | 当日收盘价 | 元 |
| `db_turnover_rate` | float32 | 换手率 | % |
| `db_turnover_rate_f` | float32 | 换手率（自由流通股） | % |
| `db_volume_ratio` | float32 | 量比 | — |
| `db_pe` | float32 | 市盈率（总市值/净利润，亏损的PE为空）⚠️ 这是**静态**PE，与 `bb_pe_dyn` 动态PE不同 | — |
| `db_pe_ttm` | float32 | 市盈率TTM（亏损的PE为空） | — |
| `db_pb` | float32 | 市净率（总市值/净资产） | — |
| `db_ps` | float32 | 市销率 | — |
| `db_ps_ttm` | float32 | 市销率TTM | — |
| `db_dv_ratio` | float32 | 股息率 | % |
| `db_dv_ttm` | float32 | 股息率TTM | % |
| `db_total_share` | float32 | 总股本 | 万股 |
| `db_float_share` | float32 | 流通股本 | 万股 |
| `db_free_share` | float32 | 自由流通股本 | 万 |
| `db_total_mv` | float32 | 总市值 | 万元 |
| `db_circ_mv` | float32 | 流通市值 | 万元 |

## moneyflow.h5 — 每日资金流向（18个字段，前缀 `mf_`）

| 字段名 | 类型 | 说明 | 单位 |
|--------|------|------|------|
| `mf_sm_buy_vol` | float32 | 小单买入量 | 股 |
| `mf_sm_buy_amt` | float32 | 小单买入金额 | 元 |
| `mf_sm_sell_vol` | float32 | 小单卖出量 | 股 |
| `mf_sm_sell_amt` | float32 | 小单卖出金额 | 元 |
| `mf_md_buy_vol` | float32 | 中单买入量 | 股 |
| `mf_md_buy_amt` | float32 | 中单买入金额 | 元 |
| `mf_md_sell_vol` | float32 | 中单卖出量 | 股 |
| `mf_md_sell_amt` | float32 | 中单卖出金额 | 元 |
| `mf_lg_buy_vol` | float32 | 大单买入量 | 股 |
| `mf_lg_buy_amt` | float32 | 大单买入金额 | 元 |
| `mf_lg_sell_vol` | float32 | 大单卖出量 | 股 |
| `mf_lg_sell_amt` | float32 | 大单卖出金额 | 元 |
| `mf_elg_buy_vol` | float32 | 特大单买入量 | 股 |
| `mf_elg_buy_amt` | float32 | 特大单买入金额 | 元 |
| `mf_elg_sell_vol` | float32 | 特大单卖出量 | 股 |
| `mf_elg_sell_amt` | float32 | 特大单卖出金额 | 元 |
| `mf_net_vol` | float32 | 净流入量 | 股 |
| `mf_net_amt` | float32 | 净流入额 | 元 |

## bak_basic.h5 — 历史基本面（15个字段，前缀 `bb_`）

| 字段名 | 类型 | 说明 | 单位 |
|--------|------|------|------|
| `bb_pe_dyn` | float32 | **动态**市盈率 ⚠️ 与 `db_pe` 静态PE不同：`db_pe`=总市值/净利润（静态），`bb_pe_dyn`=基于最新财报滚动年化（动态） | — |
| `bb_total_assets` | float32 | 总资产 | 亿 |
| `bb_liquid_assets` | float32 | 流动资产 | 亿 |
| `bb_fixed_assets` | float32 | 固定资产 | 亿 |
| `bb_reserved` | float32 | 公积金 | 亿元 |
| `bb_reserved_pershare` | float32 | 每股公积金 | 元 |
| `bb_eps` | float32 | 每股收益 | 元 |
| `bb_bvps` | float32 | 每股净资产 | 元 |
| `bb_undp` | float32 | 未分配利润 | 亿元 |
| `bb_per_undp` | float32 | 每股未分配利润 | 元 |
| `bb_rev_yoy` | float32 | 收入同比 | % |
| `bb_profit_yoy` | float32 | 利润同比 | % |
| `bb_gpr` | float32 | 毛利率 | % |
| `bb_npr` | float32 | 净利润率 | % |
| `bb_holder_num` | float32 | 股东人数 | 人 |

## cyq_perf.h5 — 筹码分布绩效（9个字段，前缀 `cp_`）

| 字段名 | 类型 | 说明 | 单位 |
|--------|------|------|------|
| `cp_his_high` | float32 | 历史最高价（自上市以来） | 元 |
| `cp_his_low` | float32 | 历史最低价（自上市以来） | 元 |
| `cp_cost_5pct` | float32 | 5%分位成本 | 元 |
| `cp_cost_15pct` | float32 | 15%分位成本 | 元 |
| `cp_cost_50pct` | float32 | 50%分位成本（中位成本） | 元 |
| `cp_cost_85pct` | float32 | 85%分位成本 | 元 |
| `cp_cost_95pct` | float32 | 95%分位成本 | 元 |
| `cp_weight_avg` | float32 | 加权平均成本 | 元 |
| `cp_winner_rate` | float32 | 胜率（当前价格高于持仓成本的比例） | % |

## sector_data.h5 — 申万二级行业数据（23个字段：22个 `sw2_` + `l2_code_id`）

> `l2_code_id`：int16 申万 L2 稳定整数编码；unknown/未匹配=-1；映射源 `market.sw_index_classify` L2 `index_code ASC`（禁 factorize）；离散分组键非连续特征；与 `static_factors.parquet` 的 `l2_code_id` 逐行一致。

每只股票按其所属的申万二级行业分类，映射到该行业的行情和资金流数据。可用于构建行业相对因子（个股 vs 行业对比）、行业动量/反转、行业资金流强度等。

| 字段名 | 类型 | 说明 | 单位 |
|--------|------|------|------|
| `sw2_close` | float32 | L2行业指数收盘价 | — |
| `sw2_open` | float32 | L2行业指数开盘价 | — |
| `sw2_high` | float32 | L2行业指数最高价 | — |
| `sw2_low` | float32 | L2行业指数最低价 | — |
| `sw2_pct_change` | float32 | L2行业涨跌幅 | % |
| `sw2_vol` | float32 | L2行业成交量 | 万股 |
| `sw2_amount` | float32 | L2行业成交额 | 万元 |
| `sw2_pe` | float32 | L2行业市盈率 | — |
| `sw2_pb` | float32 | L2行业市净率 | — |
| `sw2_total_mv` | float32 | L2行业总市值 | 万元 |
| `sw2_mf_net_amt` | float32 | L2行业净资金流入额（成分股聚合） | 万元 |
| `sw2_mf_net_vol` | float32 | L2行业净资金流入量（成分股聚合） | 手 |
| `sw2_mf_buy_elg_amt` | float32 | L2行业超大单买入额（成分股聚合） | 万元 |
| `sw2_mf_buy_elg_vol` | float32 | L2行业超大单买入量（成分股聚合） | 手 |
| `sw2_mf_sell_elg_amt` | float32 | L2行业超大单卖出额（成分股聚合） | 万元 |
| `sw2_mf_sell_elg_vol` | float32 | L2行业超大单卖出量（成分股聚合） | 手 |
| `sw2_mf_buy_lg_amt` | float32 | L2行业大单买入额（成分股聚合） | 万元 |
| `sw2_mf_sell_lg_amt` | float32 | L2行业大单卖出额（成分股聚合） | 万元 |
| `sw2_mf_buy_md_amt` | float32 | L2行业中单买入额（成分股聚合） | 万元 |
| `sw2_mf_sell_md_amt` | float32 | L2行业中单卖出额（成分股聚合） | 万元 |
| `sw2_mf_buy_sm_amt` | float32 | L2行业小单买入额（成分股聚合） | 万元 |
| `sw2_mf_sell_sm_amt` | float32 | L2行业小单卖出额（成分股聚合） | 万元 |

### 行业数据典型用法

- **行业相对收益**: 个股收益率 - sw2_pct_change → 剥离行业β
- **行业资金流强度**: mf_net_amt / sw2_mf_net_amt → 个股在行业中的资金流占比
- **行业估值偏离**: db_pe / sw2_pe → 个股相对行业的估值溢价
- **行业动量**: sw2_close 的 N 日收益率 → 行业趋势信号

## ⚠️ 重要区分：静态PE vs 动态PE

- `db_pe`：**静态**市盈率 = 总市值 / 最近年报净利润，更新频率低，反映历史盈利
- `bb_pe_dyn`：**动态**市盈率 = 基于最新季报滚动年化计算，更新频率高，反映最新盈利预期
- 设计因子时需明确选择使用哪个PE口径，或构造两者差异/比值作为信号