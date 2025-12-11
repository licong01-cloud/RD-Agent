# Qlib 指数 bin 导出规范（提交给 AIstock 使用）

> 目标：让 AIstock 根据本文档，**一次性正确导出指数数据到 Qlib bin**，可直接供 RD-Agent / Qlib 回测与评估使用。
>
> 当前股票 bin：`C:/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209`
>
> 数据区间：`2010-01-07 ~ 2025-12-01`

---

## 1. 需要支持的指数列表

至少导出以下日频指数（可按需扩展）：

- **核心基准指数（至少 1 条）**
  - 沪深 300：`000300.SH`（推荐）
  - 或 上证综指：`000001.SH`

- **推荐额外指数（便于风格分析，可选但推荐）**
  - 深证成指：`399001.SZ`
  - 创业板指：`399006.SZ`
  - 中证 500：`000905.SH`
  - 中证 1000：`000852.SH`

> 说明：代码格式可采用 `000300.SH`/`399001.SZ` 这一套，只要在 **features/h5 与 instruments 中保持完全一致**，Qlib 端即可使用。

---

## 2. 存储位置与目录结构

指数数据需要与当前股票 bin 共用同一个目录结构：

```text
qlib_bin_20251209/
  instruments/
    all.txt              # 股票池定义（已存在）
    index.txt            # （可选）指数池定义，需新增
  features/
    <hash0>.h5
    <hash1>.h5
    ...                  # 指数数据与股票数据混存于同一 features 集合
  # 或者：
  # daily_pv.h5 / moneyflow.h5 等（取决于你们当前实现）
```

### 2.1 HDF5 文件（features 或 daily_pv.h5）格式

无论是拆分为多个 `features/*.h5`，还是集中放在一个 `daily_pv.h5` 中，指数部分的 **DataFrame 结构必须满足以下条件**：

- **索引类型**：`pandas.MultiIndex`
  - `index.names == ["datetime", "instrument"]`
  - `datetime`：`pd.Timestamp`，精确到日，无时区。
  - `instrument`：指数代码字符串，如 `000300.SH`, `399001.SZ`。

- **列字段（columns）**：
  - 必须包含：`"$close"`
  - 建议包含：`"$open"`, `"$high"`, `"$low"`, `"$volume"`
  - 字段名需带 `$` 前缀，与当前股票数据保持一致。

示例（逻辑结构）：

```text
index:
  datetime   instrument
  2010-01-07 000300.SH    -> 行 1
  2010-01-08 000300.SH    -> 行 2
  ...

columns:
  ["$open", "$high", "$low", "$close", "$volume"]
```

以单个文件 `daily_pv.h5` 为例，推荐写入方式（伪代码）：

```python
import pandas as pd

# df: MultiIndex(index=[datetime, instrument], columns=["$open", "$high", "$low", "$close", "$volume"])

df.to_hdf("daily_pv.h5", key="data")
```

> 要求：AIstock 导出的指数数据与现有股票数据在存储结构上完全一致，**不要为指数单独设计另一套结构**。

---

## 3. 指数行情数据要求

### 3.1 时间区间

- 起始日期：`2010-01-07`
- 结束日期：`2025-12-01`
- 精确到交易日。非交易日无需补数据（即使中间有节假日空档）。

### 3.2 字段含义

- `$open`：指数开盘价（浮点数）
- `$high`：当日最高价
- `$low`：当日最低价
- `$close`：收盘价（用于回测、基准比较等）
- `$volume`：成交量（若数据源无量，可设为 0，但字段仍需存在）

### 3.3 数据质量

- 不要求对指数进行 ST/退市过滤，但需保证：
  - 时间序列连续、没有明显的乱序；
  - 若某日无有效收盘价，可使用 `NaN`，但尽量少。
- 后续由 Qlib / RD-Agent 在回测时自行处理少量缺失值。

---

## 4. instruments 中注册指数

为使 Qlib 能在回测中使用指数作为 benchmark，需在 `instruments` 目录下注册指数。

### 4.1 指数池定义（推荐）

新增一个指数池文件，例如：

**`qlib_bin_20251209/instruments/index.txt`**：

```text
000300.SH
000001.SH
399001.SZ
399006.SZ
000905.SH
000852.SH
```

- 格式：每行一个代码，无额外字段；
- Qlib 侧可以通过：

  ```python
  D.instruments("index")
  ```

  取得该指数池。

### 4.2 是否将指数放入 all 池？

- 可选：是否也把上述指数加入 `all` 池，取决于你们对 `all` 的定义。
- 对 RD-Agent 来说，**不要求指数在 `all` 中**，但必须能在某个池或直接通过代码让 Qlib 读到。

关键是：

- 在 index bin 导出完毕后，Qlib 初始化后应满足：

  ```python
  import qlib
  from qlib.data import D

  qlib.init(provider_uri=".../qlib_bin_20251209", region="cn")

  # 1) 直接按代码读取指数行情
  df_idx = D.features(["000300.SH"], ["$close"],
                      start_time="2016-01-01", end_time="2016-01-10", freq="day")

  # 2) 通过 index 股票池展开
  pool_cfg = D.instruments("index")
  insts = list(D.list_instruments(pool_cfg,
                                  start_time="2010-01-07",
                                  end_time="2025-12-01"))
  # insts 中至少包含上面列出的指数代码
  ```

---

## 5. 导出后的自检脚本（AIstock 侧建议执行）

AIstock 在生成好指数 bin 数据后，建议用如下脚本做一次 **本地自检**（可在你们自己的环境中运行）：

```python
import qlib
from qlib.data import D

provider_uri = "/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209"  # 按实际路径替换

qlib.init(provider_uri=provider_uri, region="cn")

print("== 检查指数池 'index' ==")
pool_cfg = D.instruments("index")
print("pool_cfg:", pool_cfg)

insts = list(D.list_instruments(pool_cfg, start_time="2010-01-07", end_time="2025-12-01"))
print("index 池中的 instrument 数量:", len(insts))
print("前若干个代码:", insts[:10])

print("\n== 抽查 000300.SH 行情数据 ==")
try:
    df = D.features(["000300.SH"], ["$close", "$volume"],
                    start_time="2016-01-01", end_time="2016-01-10", freq="day")
    print("000300.SH 特征 shape:", df.shape)
    print(df.head())
except Exception as e:
    print("读取 000300.SH 失败:", repr(e))
```

若上述检查均通过，则说明 **指数 bin 数据在结构与内容上已经符合 Qlib 要求**，RD-Agent 侧即可在回测配置中放心使用：

```yaml
benchmark: &benchmark 000300.SH
...
backtest:
  benchmark: *benchmark
```

---

## 6. 小结（给 AIstock 的执行要点）

- **要做的事**：
  - 在当前股票 bin 所在目录下，按 Qlib CN 日频格式增加一组指数数据；
  - 指数包括至少一条全局基准（建议 000300.SH），以及若干常用指数（399001.SZ 等）；
  - 把这些代码注册到 `instruments/index.txt` 等股票池文件中。

- **必须满足**：
  - 指数数据与股票数据共享相同的 HDF5 存储结构：MultiIndex(datetime, instrument) + `$close` 等字段；
  - Qlib 初始化后，`D.features(["000300.SH"], ["$close"], ...)` 可直接读取到数据；
  - `D.instruments("index")` + `D.list_instruments(...)` 能展开出这批指数代码。

按以上规范导出后，RD-Agent 端即可：

- 使用这些指数作为回测基准，计算超额收益与相对表现；
- 同时保持与当前股票 bin 完全兼容，无需修改现有数据结构。 
