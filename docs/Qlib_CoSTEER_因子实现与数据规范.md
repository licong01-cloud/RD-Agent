# Qlib / CoSTEER 因子实现与数据规范（草案）

> 版本：v0.1  
> 作用：统一因子实现对 `daily_pv.h5` / Qlib 数据的使用方式，避免常见实现错误，为 RAG 提供“因子代码必须遵守的硬规则”。

---

## 1. 数据文件与路径

### 1.1 HDF5：daily_pv.h5

- 所有 CoSTEER / RD-Agent 因子实现统一从当前工作目录下的 `daily_pv.h5` 读取数据：
  - 索引：`MultiIndex(['datetime', 'instrument'])`
  - 列名采用 Qlib 风格：`$open`, `$high`, `$low`, `$close`, `$volume`, `$amount`, `$factor`，以及可能扩展的资金流/筹码字段（如 `$mf_net_inflow_1d` 等）。
- 该文件通常通过 `.env` 中的 `FACTOR_CoSTEER_data_folder` 找到：
  - `FACTOR_CoSTEER_data_folder=/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_snapshots/qlib_export_20251209`

### 1.2 Qlib bin：provider_uri

- 回测和模型训练使用 Qlib bin 数据：
  - Windows：`C:/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209`
  - WSL：`/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209`
- Qlib YAML 模板中的 `qlib_init.provider_uri` 已统一指向该目录。

### 1.3 股票池对齐与过滤

- `daily_pv.h5` 和 Qlib bin 在导出阶段已经：
  - 剔除了所有 ST / *ST 股票；
  - 剔除了所有已退市或当前暂停上市的股票；
  - 股票池在两个数据源之间完全对齐。
- 因子实现默认只在这个可交易股票池上工作，**无需在代码中再次按 ST/退市状态过滤**。

---

## 2. 因子实现 IO 规范

### 2.1 读取 daily_pv.h5

- 推荐的读取与重命名模板（Python 示意）：

```python
import pandas as pd

# 读取数据并按索引排序
df = pd.read_hdf("daily_pv.h5", key="data").sort_index()

# 将带 $ 前缀的列统一重命名为无前缀业务名
rename_map = {
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$close": "close",
    "$volume": "volume",
    "$amount": "amount",
    "$factor": "factor",
    # 扩展字段示例
    "$mf_net_inflow_1d": "mf_net_inflow_1d",
    "$mf_net_inflow_5d": "mf_net_inflow_5d",
}

df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
```

- 后续所有代码中 **只允许使用不带 `$` 的列名**（如 `df["close"]`），禁止混用 `df["$close"]` 和 `df["close"]`。

### 2.2 因子输出 result.h5

- 因子计算结果必须保持索引结构：
  - 索引：`MultiIndex(['datetime', 'instrument'])`
  - `index.names == ["datetime", "instrument"]`
- 输出示例：

```python
result_df.to_hdf("result.h5", key="data", mode="w")
```

- 列要求：
  - 一列或多列因子值；
  - 每列类型为 `float64`；
  - 列名为因子英文名（如 `vol_adj_momentum_20d`）。

### 2.3 程序结构约束

- 建议（但不强制）的结构：
  - 顶部 import（`pandas`, `numpy` 等）；
  - 一个主函数 `calculate_{factor_name}()`；
  - `if __name__ == "__main__":` 调用主函数。
- 禁止：
  - 在主逻辑外层用大范围 `try/except` 吞掉所有异常；
  - 随意导入当前环境中不保证存在的第三方库。

---

## 3. 常见时间序列操作规范

### 3.1 时间窗口与前视偏差

- 因子设计中必须显式考虑时间窗口与方向，例如：
  - 20 日动量因子：`Momentum_20D = Close_t / Close_{t-20} - 1`；
  - 20 日波动率应基于过去 20 日收益率，如 `R_{t-1} ... R_{t-20}`，而不是包含当日收益 `R_t`。
- 推荐实践：
  - 对收益率/特征先 `shift(1)`，再做 `rolling(window=n)`；
  - 避免使用未来信息（`t+1`、未来财报等）。

### 3.2 数值稳定性

- 对含除法的因子（如收益 / 波动率、价差 / 成交额等），必须处理分母为 0 的情况，例如：

```python
eps = 1e-12
ratio = numerator / (denominator + eps)
```

- 对极端值可采用简单的 winsorize 或剪裁：

```python
q_low, q_high = result_df.quantile([0.01, 0.99])
result_df = result_df.clip(q_low, q_high)
```

（具体剪裁规则可以根据场景调整。）

### 3.3 MultiIndex 操作示例

- 获取某只股票的时间序列：

```python
one_stock = df.xs("SH600000", level="instrument")
```

- 对每个股票做 rolling 计算：

```python
close = df["close"].unstack("instrument")
ret = close.pct_change()
vol_20 = ret.rolling(window=20).std()
# 计算后再 stack 回 MultiIndex
vol_20 = vol_20.stack().to_frame("vol_20")
```

---

## 4. 允许与禁止的库

### 4.1 允许使用（示例）

- 标准库：`math`, `itertools`, `datetime` 等；
- 科学计算：`numpy`, `pandas`；
- 对于特定场景，可使用 `scipy`（需在项目依赖中已安装）。

### 4.2 禁止使用或需谨慎的库

- 不允许直接导入：`h5py`；
- 不允许用不在项目依赖中的大型框架：例如再引入一个新的深度学习框架、数据库驱动等；
- 对于 `sklearn` 等，若未在当前环境中确认安装，也应避免默认使用。

---

## 5. 与 Qlib 回测的协同

- 因子在 HDF5 上计算完成后，会被 RD-Agent / Qlib 自动接入，形成因子表并参与回测：
  - Qlib 会按配置的 `segments` 拆分 train/valid/test 区间；
  - 使用指定的模型（如 LGBM、GRU、GeneralPTNN 等）进行训练和预测；
  - 最终输出组合绩效指标（年化、Sharpe、max drawdown 等）。
- 因子实现应尽量：
  - 在样本内外都有稳定表现，而不是只在局部时间段“刷分”；
  - 避免依赖难以扩展/解释的黑箱特征（过于复杂的非线性变换、特别是基于未来信息的）。

---

## 6. RAG 中的使用建议

- 当 RAG 为“因子实现生成 / 修改任务”提供上下文时，应至少检索：
  - 本文档；
  - AIstock-RD-Agent 集成快速规范；
  - 常见错误案例集（如果已建立）。
- LLM 在编写/修改因子代码时，应优先遵守本规范中的：
  - 数据路径与表结构；
  - IO 规范；
  - 时间序列与数值稳定性约束；
  - 允许/禁止库列表。
