# Qlib bin 数据问题与重新导出指引（针对 qlib_bin_20251209）

## 1. 当前问题概述

当前 RD-Agent 在 Qlib 环境下使用的数据目录为：

```text
/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209
```

经过一系列调试脚本检查（`debug_dataset_pipeline.py`、`debug_instruments.py`），发现 **该 bin 数据在 `market: all` 下仅包含一个聚合标的 `DAILY_ALL`**，而不是正常意义上的“多只 A 股股票池”。

### 1.1 关键证据

使用 `debug_instruments.py` 的输出：

```text
stock pool config for 'all': {'market': 'all', 'filter_pipe': []}
展开后的真实股票数量: 1 只
前 1 个示例:
DAILY_ALL
```

以及 `debug_dataset_pipeline.py` 的输出（以训练段为例）：

```text
raw train shape: (731, 21)
label cross-section @ 2016-01-04: n=1, mean=..., std=nan, ...
...
prepared train size: (731, 21), index_levels=2
```

说明：

- 时间维度约 731 个交易日，但 **每个交易日只有 1 条 (datetime, instrument) 记录**；
- 该唯一 instrument 为 `DAILY_ALL`，是某种“全市场聚合序列”，而非单只股票；
- 在任意给定日期，label 的横截面样本数 `n=1`，标准差 `std=NaN`，这会直接导致横截面 IC 难以定义。

### 1.2 对 RD-Agent / Qlib 实验的影响

在这种数据形态下：

- **横截面 IC 必然为 NaN**：
  - Qlib 的 IC 是按横截面相关系数定义的，当某日只有 1 只标的时，分母为 0、相关系数不可定义；
  - 这解释了多轮实验中“IC 始终为 NaN”的现象。

- **所有模型 / 因子 / 策略回测高度相似且表现极差**：
  - 回测实际上是在单一时间序列 `DAILY_ALL` 上调整仓位，而不是在多只股票之间做选股和权重分配；
  - 不同模型/因子输出再怎么变化，最终只影响“这一条曲线上的持仓节奏”，很难产生稳定的正收益，且不同方案之间差异度极低；
  - 这与当前观测到的现象一致：
    - 年化收益大多为负（例如 -9.48%、-26.08%），最大回撤极大；
    - 不同 loop 的表现高度一致，即使模型从 GRU 换到 Transformer / LSTM / MLP，结果仍然几乎相同。

- **自动演进机制被彻底限制**：
  - RD-Agent 设计的初衷是利用多因子 + 多模型在“多股票横截面”上演进；
  - 目前的 bin 数据退化为“单资产时间序列”，自动演进几乎无法从 IC / 选股效果中学习任何有用反馈。

结论：

> 在修复 bin 数据之前，继续做因子 / 模型 / 策略演进，结论都不具备参考价值。

---

## 2. 重新导出 bin 时需要的改动要点

目标：

- 导出一个新的 Qlib bin 数据目录（例如 `qlib_bin_2025XXXX`），
- 其中 `market` 对应的是 **真实的多股票池**（如 A 股全市场或自定义股票列表），
- 从而使 `D.instruments("all")` 或相应股票池展开后得到的是“数百/上千只股票”，而不是 `DAILY_ALL`。

下面从 **AIstock 导出侧** 角度列出需要注意的改动点。

### 2.1 明确股票池定义（stock pool）

1. **不要再使用仅包含 `DAILY_ALL` 的聚合池作为训练股票池**。

2. 选择一个明确的股票池方案，例如：
   - **全 A 股**：所有非 ST、非退市且满足一定流动性条件的股票；
   - **指数成分股**：如 `CSI300`（沪深 300）、`CSI500` 等；
   - **自定义白名单**：由 AIstock 或研究侧维护的 `aistock.txt` 等股票列表。

3. 在导出配置中，将该股票池注册为一个命名池（如 `aistock_all`），确保：
   - `D.instruments("aistock_all")` 返回一个包含 `market`、`filter_pipe` 的 pool 配置 dict；
   - `D.list_instruments(D.instruments("aistock_all"), ...)` 展开后得到的是“多只股票代码”（如 `000001.SZ`, `600000.SH` 等）。

### 2.2 导出 bin 时的参数设置

以 Qlib 官方导出流程为例（伪代码，仅示意）：

```bash
# 伪代码示意：实际命令请按你们当前使用的 export 脚本为准
python your_export_script.py \
  --provider-uri /path/to/qlib_bin_2025XXXX \
  --region cn \
  --instruments-config /path/to/aistock_pool_config.yaml
```

关键点：

- **instruments-config / 股票池配置** 中，应明确指向多只股票，而不是 `DAILY_ALL`；
- 如果使用的是 `csv` / `txt` 列表（如一行一个股票代码），需要在导出脚本中将其转换为 Qlib 支持的 instruments 定义；
- 导出完成后，建议在 AIstock 侧用一个极简脚本做 sanity check：

  ```python
  import qlib
  from qlib.data import D

  qlib.init(provider_uri="/path/to/new_qlib_bin", region="cn")

  pool_cfg = D.instruments("aistock_all")
  insts = list(D.list_instruments(pool_cfg, start_time="2016-01-01", end_time="2016-01-10"))
  print(len(insts), insts[:20])
  ```

  - 若 `len(insts)` 很小（例如 1）或只有 `DAILY_ALL` 等聚合标的，说明导出配置仍然有问题；
  - 期望值应为“几十到几千只股票不等”，视选用的股票池而定。

### 2.3 与 RD-Agent / Qlib YAML 的对齐

RD-Agent 侧当前的 Qlib 配置大致如下（示意）：

```yaml
qlib_init:
  provider_uri: "/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209"
  region: cn

market: &market all
benchmark: &benchmark DAILY_ALL

data_handler_config:
  instruments: *market
  ...
```

在 AIstock 重新导出新的 bin 之后，需要协调做以下对齐：

1. **provider_uri**：
   - RD-Agent 侧改为新导出的目录，例如：
     ```yaml
     qlib_init:
       provider_uri: "/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_2025XXXX"
       region: cn
     ```

2. **market / instruments**：
   - 将 `market: &market all` 替换为新股票池名称，例如：
     ```yaml
     market: &market aistock_all
     ...
     data_handler_config:
       instruments: *market
     ```
   - 保证 `D.instruments("aistock_all")` + `D.list_instruments(...)` 能展开为多只股票。

3. **benchmark**：
   - 如有需要，可将 `DAILY_ALL` 替换为更合理的基准指数代码（如 `SH000300`），前提是该指数在新 bin 中也被导出；
   - 若暂时仍使用 `DAILY_ALL` 作为基准，对 IC 计算影响不大，但对策略回测的对照意义有限。

---

## 3. 验证步骤（AIstock 侧建议）

在重新导出并更新 YAML 后，建议在 AIstock / RD-Agent 侧做以下验证：

1. **股票池验证**：
   - 使用更新后的 `debug_instruments.py`，确认：
     - `stock pool config for 'aistock_all'` 合理；
     - `展开后的真实股票数量` 显著大于 1，且前若干示例为真实股票代码。

2. **label / 样本量验证**：
   - 运行更新后的 `debug_dataset_pipeline.py`：
     - `raw train/valid/test shape` 中的行数应 ≈ “交易日数 × 股票数”；
     - label 截面统计中，`n` 应明显 > 1，`std` 应不再是 NaN。

3. **小规模 RD-Agent 回测验证**：
   - 只跑 1~2 个 loop，检查：
     - IC 是否不再长期为 NaN；
     - 年化收益 / 最大回撤是否摆脱“极端负值 + 全部 loop 几乎一样”的模式。

当上述三点都通过后，才有意义继续做大规模的因子 / 模型 / 策略自动演进实验。

---

## 4. 总结

- 当前 `qlib_bin_20251209` 在 `market: all` 下实际只包含一个聚合标的 `DAILY_ALL`，导致：
  - 横截面 IC 基本不可用（NaN）；
  - 多因子 / 多模型回测退化为在单一指数时间序列上调仓，结果极差且高度一致；
  - RD-Agent 的自动演进在这种数据上无法获得有意义的反馈。

- 要恢复正常的多股票横截面研究能力，必须：
  1. 在 AIstock 侧重新导出包含真实股票池（多只股票）的 Qlib bin 数据；
  2. 在 RD-Agent 的 Qlib YAML 中，将 `provider_uri`、`market` / `instruments` 等设置与新股票池对齐；
  3. 通过调试脚本与小规模回测验证数据与配置的正确性。

这份文档可作为 AIstock 侧修改导出流程、协调 RD-Agent 配置调整的操作说明。
