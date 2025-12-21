# RD-Agent(Q) + Qlib + 多源数据 备忘录

## 1. 当前环境与基础状态

- **运行环境**
  - WSL（Ubuntu）+ Conda 环境：`rdagent-gpu`
  - RD-Agent 已安装，验证：
    - `python -m rdagent.app.cli health_check ...` 通过
    - `python -m rdagent.app.cli fin_quant ...` 可运行
  - GPU：RTX 2060，`torch.cuda.is_available() = True`，可用于 PyTorch

- **LLM 访问**
  - DeepSeek 官方：
    - API Base: `https://api.deepseek.com`
    - 模型：`deepseek-chat`
    - 性能稳定：平均 ~2 秒级，401 说明网络可达（需带正确 API Key）
  - SiliconFlow：
    - API Base: `https://api.siliconflow.cn/v1`
    - 模型测试结果：
      - `Pro/deepseek-ai/DeepSeek-V3.1-Terminus`
        - avg ≈ 1.5s, p50 ≈ 1.1s, p90 ≈ 1.9s
        - 典型比 DeepSeek 官方快，有少量波动
      - `Pro/deepseek-ai/DeepSeek-V3.2-Exp`
        - 单次 17s+，多次超时（APITimeoutError）
        - 当前环境下不适合作为主力 Chat 模型

---

## 2. LLM 性能对比与建议

### 2.1 测试脚本说明

脚本：`tools/llm_speed_compare.py`（根目录 `llm_speed_compare.py` 为兼容入口）

- 功能：
  - 对同一 prompt 调用多个 LLM 若干次，统计：
    - 每次耗时
    - 输出长度
    - 平均延迟 / p50 / p90
- DeepSeek：
  - 通过 `DEEPSEEK_API_KEY` / `DEEPSEEK_API_BASE` / `DEEPSEEK_MODEL` 配置
- SiliconFlow：
  - 固定 base：`https://api.siliconflow.cn/v1`
  - 测试模型：
    - `Pro/deepseek-ai/DeepSeek-V3.1-Terminus`
    - `Pro/deepseek-ai/DeepSeek-V3.2-Exp`
  - 只从 `.env` 读取 `OPENAI_API_KEY`

### 2.2 实测结论

- **DeepSeek 官方 `deepseek-chat`**
  - avg ≈ 1.9–2.4 秒
  - p50 ≈ 1.8–2.2 秒
  - 波动小，稳定

- **SiliconFlow `Pro/deepseek-ai/DeepSeek-V3.1-Terminus`**
  - avg ≈ 1.5 秒
  - p50 ≈ 1.1 秒
  - p90 ≈ 1.9 秒
  - 典型响应更快，偶发 2–3 秒慢次

- **SiliconFlow `Pro/deepseek-ai/DeepSeek-V3.2-Exp`**
  - 17 秒+ 且多次超时（>15s）
  - 当前网络/账号下不适合在线高频调用

**推荐：**

- **Chat 主力：**
  - 保持 DeepSeek 官方 `deepseek-chat` 作为主通道；
  - 或在特定场景下（对时延敏感）考虑使用 SiliconFlow V3.1。
- **Chat 不推荐：**
  - SiliconFlow `Pro/deepseek-ai/DeepSeek-V3.2-Exp` 仅适合作为离线实验模型。
- **Embedding：**
  - 继续使用 Qwen3-Embedding-0.6B（SiliconFlow），已验证性能和批量能力。

---

## 3. RD-Agent CLI 与 GPU 使用

### 3.1 CLI 调用方式

避免使用可能连错 Python 的 `rdagent` 可执行脚本，统一使用：

```bash
conda activate rdagent-gpu
cd /mnt/c/Users/lc999/RD-Agent-main

python -m rdagent.app.cli health_check --check-env --no-check-docker --no-check-ports
python -m rdagent.app.cli fin_quant --loop-n 3
```

- `--loop-n` （中划线）是正确的参数名。

### 3.2 GPU 使用与模型代码约束

当前环境中：

- `torch.cuda.is_available() = True`，GPU 可用；
- 但 Qlib 的很多计算默认在 CPU；
- 真正用 GPU 的部分主要是 **PyTorch 模型**，需要在生成代码中强制使用 `cuda`：

建议在 `prompts.yaml` 中加入硬约束，让 LLM 生成模型代码时：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch = batch.to(device)
```

从而提升 PyTorch 模型自动利用 GPU 的概率。

---

## 4. Qlib 数据：下载与扩展

### 4.1 数据目录规划

统一使用：

```text
~/.qlib/qlib_data/cn_data
# 实际路径：/home/lc999/.qlib/qlib_data/cn_data
```

RD-Agent 的 Qlib 场景默认也是使用该目录。

### 4.2 官方 cn_data 下载（解决网络问题后的标准方案）

由于 WSL → GitHub 超时，采用 **Windows 浏览器下载 + WSL 手动解压**：

1. 在 Windows 浏览器打开：
   ```text
   https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_cn_1d_0.9.6.zip
   ```
2. 下载后将 `qlib_data_cn_1d_0.9.6.zip` 拷贝到 WSL：
   ```bash
   mkdir -p ~/.qlib/qlib_data/cn_data
   cp /mnt/c/Users/lc999/Downloads/qlib_data_cn_1d_0.9.6.zip \
      ~/.qlib/qlib_data/cn_data/
   ```
3. 在 WSL 解压：
   ```bash
   cd ~/.qlib/qlib_data/cn_data
   unzip qlib_data_cn_1d_0.9.6.zip
   rm qlib_data_cn_1d_0.9.6.zip
   ```

完成后，`calendars/`, `features/`, `instruments/` 等目录即可被 Qlib/RD-Agent 使用。

---

## 5. 因子实现问题与修正建议

### 5.1 自动生成因子存在的问题

示例问题代码：

- 在 `calculate_sma_10_main` 中：
  - 硬编码 `data_file = 'data.csv'`，`result.h5`；
  - 在 RD-Agent 流程中会导致：
    - `FileNotFoundError: Data file 'data.csv' not found.`
  - 违背接口要求：因子函数应只使用传入的 DataFrame，不自行读写文件。

- 实现上使用手写 for 循环 + `np.mean` 计算 SMA：
  - 性能差；
  - 容易出错，对缺失值和时间对齐处理不规范。

### 5.2 正确接口与实现风格

**接口要求：**

```python
def calculate_sma_10(data: pd.DataFrame) -> pd.DataFrame:
    """
    输入:
      - data: MultiIndex (datetime, instrument) 的 DataFrame, 至少包含 'close'
    输出:
      - 同样索引的 DataFrame, 仅含一列 'SMA_10'
    """
```

**推荐实现：**

```python
factor_name = "SMA_10"

if "close" not in data.columns:
    raise ValueError("input data must contain 'close' column")

close = data["close"]
sma = (
    close.groupby(level="instrument")
         .rolling(window=10, min_periods=10)
         .mean()
)
sma = sma.droplevel(0)  # 还原 MultiIndex (datetime, instrument)

result = pd.DataFrame(sma)
result.columns = [factor_name]
result = result.sort_index()
return result
```

- 不读写任何本地文件；
- 使用 `groupby + rolling` 而非手写循环；
- 自然处理窗口不足为 NaN。

### 5.3 对 prompts.yaml 的约束增强

为避免未来 LLM 再写出 `data.csv` 一类代码，建议在 `prompts.yaml` 中增加明确条款：

- **禁止：**
  - `pd.read_csv`, `pd.read_hdf`, `open`, `h5py.File` 等在因子实现中读取外部文件；
  - `to_csv`, `to_hdf`, `to_excel` 等在因子实现中写结果文件。
- **强制使用：**
  - pandas `groupby(level="instrument").rolling(...).mean()` 计算滚动因子；
  - 输入只依赖 `data: DataFrame`，输出只返回 DataFrame，不做 IO。

---

## 6. .env 关键配置说明

### 6.1 Qlib Quant 场景（QLIB_QUANT_ 前缀）

- `env_prefix = QLIB_QUANT_` 只是内部规则，不需要在 `.env` 中设置。
- 你需要设置的是具体字段，例如：
  - `QLIB_QUANT_EVOLVING_N=3`
    - 演化轮数上限（调试 3，正式可 10+）
  - `QLIB_QUANT_ACTION_SELECTION=bandit`
    - 动作选择策略：
      - `bandit`（推荐）、`llm`、`random`

### 6.2 因子 CoSTEER 行为（FACTOR_CoSTEER_ 前缀）

常用字段：

- `FACTOR_CoSTEER_MAX_LOOP=10`
  - 协同编码迭代次数
- `FACTOR_CoSTEER_DATA_FOLDER=git_ignore_folder/factor_implementation_source_data`
- `FACTOR_CoSTEER_DATA_FOLDER_DEBUG=git_ignore_folder/factor_implementation_source_data_debug`
- `FACTOR_CoSTEER_FILE_BASED_EXECUTION_TIMEOUT=3600`
- `FACTOR_CoSTEER_SELECT_METHOD=random`
- `FACTOR_CoSTEER_PYTHON_BIN=python`

---

## 7. 多源数据接入：资金 / 筹码 / 板块 / 指数 / 新闻

### 7.1 共通原则

无论是哪类数据，最终都应尽量转成：

- 日频（或分钟频）DataFrame；
- 索引为 `(datetime, instrument)`；
- 列为数值特征，如：
  - `fund_main_net`, `fund_main_ratio`
  - `chip_avg_cost`, `chip_concentration`
  - `sector_mom_20d`, `sector_money_net`
  - `index_sh000001_ret_1d`, `us_sp500_ret_1d`
  - `news_sentiment`, `news_volume`, `news_negative_ratio`
  - `fed_cut_prob_6m` 等（宏观）

然后按股票分组，将这些列 join 进 Qlib `features/day/*.h5` 文件中。

### 7.2 新闻与宏观因子

- 没有历史新闻数据：
  - 无法在历史样本上训练“新闻因子”的有效性；
  - 新闻可以作为未来实时辅助信息，但无法直接参与历史回测。
- 未来若有历史新闻：
  1. 文本 → 数值信号
     - 情绪评分、主题标签、利率预期概率等；
  2. 聚合到 `(datetime, instrument)` 或宏观层面；
  3. 写入 Qlib 数据目录；
  4. 在 `prompts.yaml` 中描述字段含义与用法。

---

## 8. 实盘架构与 Qlib / 自有数据源

### 8.1 日频实盘（最推荐的起步方案）

- **数据更新：**
  - 每个交易日收盘后，将当日行情 + 自有因子写入 Qlib 日频目录。
- **策略运行：**
  - 使用 RD-Agent 研发好的策略代码与模型；
  - 读取截至当日收盘的特征，生成第二日目标持仓；
  - 通过 QMT 执行。

优点：

- 工程简单，易落地；
- 回测/实盘使用同一特征接口，偏差小；
- 足以支撑中低频多因子策略。

### 8.2 分钟级执行策略（下层）

- 上层：日频多因子给出“买/卖什么、多少”；
- 下层：分钟级执行/择时策略，根据：
  - 分钟线价量 / 技术指标；
  - 盘口或流动性特征；

决定当日具体执行时机与价格。
可以用 RD-Agent 在分钟数据场景下单独训练，只在上层做出调仓决策后介入执行，不独立决定多空方向。

### 8.3 完全自有数据源，不用 Qlib（远期）

- 研发阶段仍用 Qlib + RD-Agent；
- 策略定型后，在自有 TSDB/API 上重写特征计算 & 模型预测；
- 适合高频/更复杂数据平台，但工程和维护成本高。

---

## 9. 实施优先级建议

1. **短期（当前阶段）**
   - 完成 Qlib cn_data 历史数据准备；
   - 在 `rdagent-gpu` 中稳定运行 Q 模式（`fin_quant`）；
   - 用日频数据完成一条“多因子策略研发 → 回测 → 模拟盘”的完整闭环。

2. **中期**
   - 逐步将自有数值数据（资金、筹码、板块、指数等）接入 Qlib 特征；
   - 在 `prompts.yaml` 中暴露这些新因子，引导 RD-Agent 使用；
   - 评估多源因子对策略表现的提升。

3. **中长期**
   - 设计并训练独立的“分钟级执行策略”，与日频多因子策略解耦；
   - 根据需求评估是否将部分或全部实盘数据管线迁移到自有 TSDB/API 框架，Qlib 主要保留为离线研究环境。

---

**状态总结：**

- LLM 通路：DeepSeek 官方 & SiliconFlow V3.1 均可用，性能已测；
- GPU：`rdagent-gpu` 环境与 PyTorch CUDA 正常，可用于模型训练；
- Qlib：已明确数据目录与离线下载方案；下一步重点是完成 cn_data 历史数据部署与自有因子接入；
- RD-Agent：Q 模式（fin_quant）已能跑通，后续重点是：
  - 约束因子/模型生成风格（无硬编码路径、充分使用 pandas & CUDA）；
  - 扩展多源因子与实盘支持方案。

## 10. Qlib 与数据库 / 实盘的关系设计（问答整理）

### 10.1 RD-Agent、Qlib、AIstock 的分工

- **RD-Agent**：
  - 只做“研究引擎”：生成因子/模型/回测脚本。
  - 假设数据已准备好（Qlib 目录 / HDF5 / Parquet），不直接管理数据库或下载数据。
- **Qlib**：
  - 作为“回测与研究视图”，消费已经准备好的行情+因子视图（如 cn_data、daily_pv.h5）。
  - 不负责长期维护数据库或增量补数，由外部数据中台控制。
- **AIstock（数据中台 + 调度/UI 中枢）**：
  - 负责对接所有真实数据源（行情库、资金流、筹码、新闻、分钟线等）。
  - 统一实现：清洗、复权、因子计算、标签构造、特征存储（DB）。
  - 负责把数据库数据导出成 Qlib 可用的“快照”（Snapshot），供 RD-Agent/Qlib 回测使用。
  - 负责实盘调度与 UI：触发数据同步、训练、回测、上线与监控。

结论：**数据接入与维护逻辑尽量都放在 AIstock 中实现，RD-Agent 只通过标准数据视图/配置使用这些数据。**

### 10.2 单一数据源：数据库 + Qlib 只做回测视图

- **唯一权威数据源：数据库**
  - 行情（含复权价）、成交量、基本面、资金流、筹码、行业、指数、新闻、labels 全部存入 DB。
  - 训练、实盘、回测都应从 DB（或由 DB 导出的视图）获得输入，避免多套数据源不一致。
- **Qlib 的角色**：
  - 仅作为“回测引擎 + 研究视图”，读取由 AIstock 导出的 Qlib 视图：
    - 标准 `cn_data` 目录结构，或
    - 定制的 HDF5 / Parquet（如 `daily_pv.h5`，MultiIndex (datetime, instrument) + 多列因子/label）。
  - 不直接连 DB，也不直接负责增量补数。
- **快照（Snapshot）概念**：
  - 某一次回测需要的时间区间、股票池、因子集合，从 DB 中抽取后，导出为一套 Qlib 数据视图目录/文件：
    - 例如 `/qlib_snapshots/quant_2025-01-01_to_2025-06-30_allA/`；
  - RD-Agent/Qlib 的配置 (`conf.yaml` / `QLIB_DATA_PATH`) 指向此快照目录进行回测。

好处：

- 数据治理集中在 DB/AIstock，Qlib 不“握真数”；
- 回测/训练/实盘查看的是同一份 DB 派生的视图，最大程度减少数据偏差；
- 支持多套 Snapshot 并存，便于对比不同样本区间与股票池。

### 10.3 训练 vs 实盘：数据源不同会有多大影响？

关键不是“Qlib vs DB”，而是：**训练/回测时看到的特征分布，与实盘时的输入是否一致**。主要风险点：

- **字段含义与单位**：
  - 训练时 `volume` 是“股数”，实盘时如果从 DB 取成交额 `amount` 并命名成 `volume`，分布完全不同，会导致模型行为偏移。
- **频率与采样时刻**：
  - 日频策略训练时使用“收盘后完整日线”，实盘若在盘中用未完成的 K 线计算同名因子，则属于 out-of-distribution 输入。
- **预处理方式**：
  - 训练时用前复权价，实盘用未复权现价；
  - 训练时用某种缺失值填充/标准化方式，实盘若不一致，都会改变特征分布。
- **信息时点 / 延迟**：
  - 财报、公告、新闻等在训练/回测时如果当作“即时可见”，而实盘实际上有披露延迟，会引入未来函数。

如果上述环节在 DB→Qlib 快照 和 DB→实盘 数据流水线中处理一致，则：

- 数据源（Qlib vs DB）只是“存取方式不同”，对策略影响相对有限；
- 真正的影响主要来自市场变化、滑点、手续费等现实因素。

### 10.4 字段命名与映射：只改名字是否有影响？

- **字段名本身不是问题**，只要：
  - 含义、单位、频率、预处理方式都完全一致；
  - 例如：DB 中 `last_price_adj` → 实盘适配层映射为 `close`，与训练/回测使用的 `close` 完全同义。
- 风险在于“改名同时悄悄改了含义”：
  - 例如用 `turnover` 冒充 `volume`，或用未复权价冒充复权价，会导致特征分布漂移。

推荐做法：

- 在策略内部固定一套“逻辑字段名”（如 `open`, `high`, `low`, `close`, `volume`, `factor_x`...）。
- 在数据适配层维护一份显式映射表（YAML 或 Python dict）：

  ```python
  FIELD_MAPPING = {
      "close": "last_price_adj",   # DB 列名 → 策略内部逻辑名
      "volume": "trade_volume",
      # ...
  }
  ```

- 训练/回测与实盘都通过同一份配置完成字段映射，便于审计与排错。

### 10.5 因子存储：Qlib 中是否已有因子？

- 标准 Qlib `cn_data`：
  - 主要是行情基础字段与少量内建特征，并不包含 RD-Agent 本轮演化出的多因子库。
- RD-Agent Qlib 场景：
  - 在 `rdagent/scenarios/qlib/experiment/factor_data_template` 中存在一份预生成的 SOTA 因子数据：
    - `generate.py` 生成 `daily_pv_all.h5`, `daily_pv_debug.h5`；
    - 我们已经让 `generate_data_folder_from_qlib()` 在本地运行 `generate.py`，并复制为 `daily_pv.h5` 供场景使用。
  - 这部分可以视为“模板因子库（SOTA 因子）”，但是场景私有，不在全局 `cn_data` 中。
- RD-Agent 本轮演化出的新因子：
  - 在运行时根据各轮的 `factor.py` 实时计算；
  - 结果写入各自 experiment workspace 下的 `combined_factors_df.parquet` / HDF5；
  - **不会自动写回到全局 Qlib `cn_data` 目录或统一因子库**。

如果未来希望把优秀因子长期沉淀，需要在 AIstock 中设计“因子登记 & 存库”机制：

- RD-Agent 输出的 `factor.py`/因子描述 → 提交给 AIstock 审核/登记；
- 通过统一的 `factor_engine` 对 DB 中历史数据批量计算 → 写入 `factor_daily` 表；
- 供训练/回测/实盘统一使用。

### 10.6 实盘因子是否需要预先计算？

- 对于**日频策略**（多因子选股/调仓）：
  - 强烈建议：大部分因子在收盘后批处理计算好，并写入 DB：
    - 表结构示例：`factor_daily(symbol, date, factor_x, factor_y, ...)`；
  - 实盘（收盘后/开盘前）只需从 DB 读取最新一行因子数据，不在实盘通道里重算全历史窗口。
- 对于**分钟级/更高频策略**：
  - “重因子”（长窗口、依赖大量历史）的部分仍宜离线批量计算或准实时滚动更新；
  - “轻因子”（短窗口、当日内计算）可以在实盘引擎中基于实时行情做滚动更新；
  - 前提是训练/回测阶段要模拟同样的计算方式和信息时点。

整体建议：

- 建立统一的 `factor_engine` 包：
  - 输入：DB 中的 OHLCV + 其他特征表；
  - 输出：标准化因子表（MultiIndex 或宽表），可写入 DB 或导出为 Qlib HDF5。
- 训练/回测：
  - 从 DB 因子表导出为 Qlib 视图（Snapshot）供 RD-Agent/Qlib 使用。
- 实盘：
  - 从同一因子表直接读取最新因子，用于策略决策。

### 10.7 Qlib Snapshot 导出与实盘适配层（高层方案）

- **QlibSnapshotExporter（AIstock 侧）**：

  ```python
  class QlibSnapshotExporter:
      def export_range(self, start_date, end_date, symbols, factors, labels):
          # 1. 从 DB 取行情 + 因子 + label
          # 2. 清洗、复权，与日历对齐
          # 3. 生成符合 Qlib/rdagent 场景要求的 HDF5 / Parquet
          # 4. 写入 /qlib_snapshots/{snapshot_id}/...
  ```

- **实盘数据适配器（AIstock 侧）**：

  ```python
  class LiveDataAdapter:
      def get_feature_df(self, symbols, now, window, freq="1d") -> pd.DataFrame:
          # 1. 从 DB 取历史 K 线 + 因子
          # 2. （可选）从实时行情接口补最新一根 bar
          # 3. 做复权/对齐/缺失值处理
          # 4. 映射成与训练/回测时相同的列结构
          return df
  ```

这样：
- RD-Agent/Qlib 在研究阶段通过 Snapshot 使用 Qlib 视图；
- 实盘策略通过 `LiveDataAdapter` 从 DB + 实时行情获得与 Snapshot 同结构的特征表；
- 因子/模型逻辑可以在训练/回测/实盘三种场景中高度复用。

### 10.8 RD-Agent 如何“看到”与“自动使用”新数据集

- **数据发现方式**：
  - RD-Agent 不会自动遍历目录猜测有哪些列。
  - 训练/回测时，`factor.py` / `model.py` 只看到一份 DataFrame：
    - index 通常是 `(datetime, instrument)`；
    - columns 来自 Qlib/HDF5 视图中实际存在的字段。
  - LLM 对数据的“理解”完全来自：
    - 场景 `prompts.yaml` 中的字段说明；
    - 代码模板中对数据结构的约定。

- **切换到新的数据集**：
  - 用 AIstock 从 DB 导出新的 Qlib 目录或 HDF5 视图：
    - 例如 `~/.qlib/qlib_data/cn_data_aistock_v1` 或新的 `daily_pv.h5`。
  - 通过 `.env` 或场景配置切换：
    - `QLIB_DATA_PATH=~/.qlib/qlib_data/cn_data_aistock_v1`；
    - 或在实验配置中指定新的 HDF5 路径。
  - 重启 `rdagent fin_quant` 的运行进程，新的训练/回测自动基于新数据源执行。

- **补充资金流等新字段后让 RD-Agent 自动使用**：
  - 在 AIstock 的导出逻辑中，将新字段（如 `fund_main_net`, `fund_main_ratio`）
    - 写入训练/回测用的 HDF5 / Parquet（或 Qlib Snapshot）；
  - 在 `prompts.yaml` 的数据背景中显式描述这些字段：
    - 名称、含义、单位、典型用法；
  - 确保数据加载代码能在 DataFrame 中提供这些列：
    - 例如在 `build_training_table()` 中统一对齐并检查必需列是否存在。
  - 这样 LLM 在生成因子/模型代码时会自然把这些列作为候选特征使用，
    不需要单独改 RD-Agent 核心逻辑。

## 11. HMM 模型与 PyTorch / RD-Agent 的使用备忘

- **PyTorch 官方对 HMM 的支持情况**
  - PyTorch 核心库本身没有内置 `HMMModel` 之类的官方模块。
  - 社区中存在若干基于 PyTorch 的 HMM 实现，例如：
    - `lorenlugosch/pytorch_HMM`（约 100+ star，教学/示例性质）；
    - `TreB1eN/HiddenMarkovModel_Pytorch`（实现了 Viterbi、Forward-Backward、Baum-Welch）；
    - `torchmm`, `gmmhmm-pytorch` 等小型项目。
  - 总体来看，这些项目 **可作为参考或研究用途，但尚不是“工业级标准库”**。

- **自己在 PyTorch 中实现 HMM 的可行性与风险**
  - 从数学上看，HMM 的定义清晰：初始分布 π、转移矩阵 A、发射分布 B，以及前向/后向/Viterbi/EM 等算法。
  - 严格按教科书公式在 PyTorch 中实现（注意 log-sum-exp 与数值稳定），理论上可以得到“准确的 HMM”。
  - 主要风险：
    - 数值稳定性（下溢、归一化）；
    - 索引/维度错误等实现 Bug；
    - 训练方式选择（EM vs 直接最大似然 + 梯度下降）。
  - 必须通过 **合成数据和与其他库对比的单元测试** 验证前向概率、Viterbi 路径等是否正确，才能在交易策略中放心使用。

- **在 RD-Agent / Qlib 中引入 HMM 的两种主要方式**
  1. **HMM 作为因子生成器（推荐的第一步）**
     - 使用成熟 HMM 库（如 `hmmlearn`）在因子脚本中：
       - 从 K 线、资金流等特征序列拟合 HMM；
       - 输出“隐状态序列”、“某状态概率”等作为新因子列。
     - RD-Agent 后续仍用 LGBM/GRU/MLP 等模型对这些因子进行预测与回测。
     - 优点：
       - 不需要改 RD-Agent/Qlib 的训练框架；
       - HMM 训练逻辑由外部库负责，工程风险较低。

  2. **HMM 作为 Qlib 训练的模型本身**
     - 编写 `HMMModel(torch.nn.Module)`，满足 RD-Agent/Qlib 的模型接口：
       - 继承 `nn.Module`，实现 `forward`，输入 `(batch, features)` 或 `(batch, time, features)`，输出 `(batch, 1)`；
       - 在 `model.py` 中暴露 `model_cls = HMMModel`。
     - 训练方式：
       - 可以在 `forward` 中实现可微分近似，使用梯度下降；
       - 或在模型内部封装 EM/前向后向，但此时与 Qlib 的优化器/epoch 循环衔接会更复杂。
     - 这种做法不需要修改 RD-Agent 框架代码，但实现复杂度高，需谨慎测试。

- **当前阶段的决策**
  - 现阶段暂不在 RD-Agent 中优先推进 HMM 的深度集成：
    - 首先专注于利用 AIstock 提供的多源数据 + 经典多因子/深度模型，完成一条稳定可复现的选股/调仓策略链路；
    - 若后续在外部实验中验证 HMM 对某些市场状态/风格切换有明显增益，再考虑：
      - 先以“因子生成器”的形式接入；
      - 进一步评估是否需要实现 `HMMModel(nn.Module)` 以纳入 RD-Agent 模型搜索空间。

- **AIstock 扩展数据源 + RD-Agent 自动因子/模型演化的角色分工**
  - AIstock 侧：
    - 负责对接多源数据：主力资金流、板块轮动、美股/期货、宏观指标等；
    - 将这些数据统一清洗为可被 Qlib/因子脚本消费的表结构（如日频 `(datetime, instrument)` + 特征列）。
    - 根据业务需求，将“策略目标 + 约束条件”整理成结构化信息，再自动生成适合 RD-Agent 的自然语言提示词（prompt 模板）。
  - RD-Agent / Qlib 侧：
    - 在已经准备好的多源数据视图上，自动生成/迭代因子和模型：
      - 因子：基于 K 线 + 资金流 + 板块 + 宏观等特征，设计与组合新的 Alpha 因子；
      - 模型：在树模型、MLP、RNN/GRU、Transformer 等候选之间尝试与调整；
      - 策略：针对持仓个数、持有期、止盈止损、换手约束等进行参数搜索与改进。
    - 通过多轮回测反馈，逐步筛选出在历史数据上表现较好的候选策略。

- **整体目标与现实边界**
  - 在“AIstock 提供多源高质量数据 + 优化过的中文提示词”的前提下：
    - RD-Agent 可以在一个大得多的特征和模型空间中自动探索与演化策略；
    - 相比单一 K 线数据 + 固定因子/模型，显著提升发现复杂 Alpha 的能力。
  - 但仍需认识到：
    - 这是一种“带约束的自动化搜索与迭代”，**不能保证找到数学意义上的全局最优策略**；
    - 过拟合风险依然存在，需要结合人工审查、稳健性测试和风控体系综合评估策略是否可上线。

## 12. 趋势跟踪 + 多因子策略与纯选股统计型策略

- **在 RD-Agent/Qlib 中研发“趋势跟踪 + 多因子”的复合策略**
  - RD-Agent 的本质：在固定回测引擎（Qlib）的前提下，由 LLM 自动设计因子/模型/策略代码，并通过多轮回测迭代优化。
  - 可以将「趋势跟踪」设为基础策略逻辑，再叠加多因子过滤/打分：
    - 趋势信号因子示例：均线突破（MA cross）、通道突破、ATR 波动通道、ADX 趋势强度等；
    - 多因子扩展：估值（PE/PB）、成长（ROE/ROA/利润增速）、资金流（主力净流入）、板块强弱、情绪等；
    - 组合方式：
      - 先用趋势信号筛出“顺势股票池”；
      - 再用多因子打分排序，从池中选出 TopK 持仓；
      - 或趋势因子参与模型输入，由模型自行学习权重。
  - 在 `prompts.yaml` 中可以通过策略描述显式告诉 LLM：
    - 「基础是趋势跟踪逻辑，在此基础上结合多因子优化选股与仓位管理」；
    - 鼓励生成/改进趋势相关因子与多因子组合，而不是只做静态多因子 TopK。
  - Qlib 场景依然是“多因子选股 + 组合回测”框架，但可以通过因子与策略规则，模拟出较为复杂的趋势+多因子复合逻辑。

- **只做“选股质量评估”的统计型策略（关注 1/5/10 日上涨表现）**
  - 需求：
    - 不以组合年化收益/夏普为主要目标；
    - 更关心「策略选出的股票，在未来 1 天、5 天、10 天上涨的统计表现」。
  - 在 RD-Agent/Qlib 中的实现思路：
    - 训练/预测层：
      - 模型仍然可以输出未来收益或上涨概率（标签可以是 1d/5d/10d 收益或多周期标签）；
      - 在 prompt 中明确要求：
        - 「策略的核心目标是提高选中股票在未来 1/5/10 日内上涨的概率和平均涨幅，而不是单纯最大化组合年化」；
    - 评估/统计层：
      - 编写（或由 RD-Agent 生成）自定义评估脚本，对每次回测结果做选股命中率分析：
        - 对每日选股集合 S，统计：
          - 在 T+1、T+5、T+10 的上涨比例（hit ratio）；
          - 平均涨幅 / 中位数涨幅；
          - 相对基准指数的超额收益分布；
        - 可以按分位数（Top10、Top20 等）或打分区间进行分组统计。
      - 在 `read_exp_res.py` 或额外的结果分析脚本中，把这类统计作为主要输出展示给用户；
      - 在 prompt 中强调「以上述选股统计指标为主要优化方向」，引导 RD-Agent 在迭代中优先改进这些指标。
  - Qlib 默认的年化收益、最大回撤等指标仍会自动给出，但可以通过自定义统计与提示，
    让 RD-Agent 把“选股命中率 / N 日涨幅统计”视为更重要的评价标准。
