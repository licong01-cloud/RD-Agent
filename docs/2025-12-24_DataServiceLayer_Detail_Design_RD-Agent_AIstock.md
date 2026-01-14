# 数据服务层详细设计（RD-Agent × AIstock）

> 重点：从“未来数据消费需求与使用方法”的角度，定义数据服务层对上暴露的标准视图、接口与约束。
> 具体实现机制（存储引擎、缓存、容灾等）由 AIstock 侧在后续文档中补充和落地。

---

## 1. 设计目标与边界

### 1.1 设计目标

- 为 **RD-Agent 演进出来的因子/策略** 以及 **AIstock 自有策略** 提供统一的数据入口：
  - 离线训练/回测：继续支持 snapshot HDF5 / qlib bin 等文件视图；
  - 在线模拟盘/实盘：提供实时行情与账户视图的内存接口，避免任何 h5/bin 中转。
- 对上层（RD-Agent / AIstock 策略引擎）屏蔽底层数据源差异：
  - miniQMT、TDX、tushare、自有 DB、时序库（如 TimescaleDB）等；
  - 统一字段名、时间/标的索引规范。
- 支持未来的扩展场景：
  - 多市场、多时区；
  - 多频率（1m/5m/15m/日线等）；
  - 异构资产（股票、基金、期货等）。

### 1.2 边界与非目标

- 本设计只定义：
  - **对 RD-Agent / AIstock 上层开放的接口 & 视图**；
  - **字段与索引的规范、使用方式与限制**；
  - **与顶层架构 Phase1–5 的关系**。
- 本设计**不约束**具体实现方式：
  - 是否采用微服务、单体模块或库函数；
  - 是否使用 Redis/Kafka/TimescaleDB 等具体组件；
  - 具体的运维、高可用与监控方案。

这些实现机制由 AIstock 侧在后续“数据服务层实现设计”文档中补充。

---

## 2. 对上标准视图：离线 vs 在线

### 2.1 离线视图（供 RD-Agent/Qlib 使用）

- 离线训练/回测阶段，数据服务层对上暴露的仍是 **文件型视图**，包括但不限于：
  - `daily_pv.h5` / `moneyflow.h5` / `daily_basic.h5`；
  - `instruments/all.txt`、`calendars/day.txt`；
  - `static_factors.parquet`（静态因子表）；
  - `qlib_bin`（标准 Qlib bin 数据目录）。
- 这些文件：
  - 由数据服务层从 AIstock 自有 DB/行情源中导出；
  - 对 RD-Agent/Qlib 来说是“research/backtest-ready”的标准快照。
- 约束：
  - 文件格式、字段名、索引规范在《AIstock 因子数据链路全流程》文档中定义，不在此重复；
  - RD-Agent/Qlib 只依赖这些标准视图，不直接访问原始行情库或外部网关。

### 2.2 在线视图（供模拟盘/实盘策略使用）

- 在线场景（模拟盘/实盘），数据服务层对上只暴露 **内存视图 / 流式接口**：
  - 不再生成任何 h5/bin 参与在线决策；
  - 所有实时决策（包括 RD-Agent 演进策略在 AIstock 侧的落地）必须通过这些接口消费实时数据。

- 核心在线视图包括：
  1. **实时行情视图**：当前 Snapshot + 流式报价；
  2. **历史行情窗口视图**：用于在线因子计算或增量更新；
  3. **账户/持仓视图**：当前资金、持仓、订单、成交；
  4. **交易日历与交易规则视图**：是否交易日、开收盘时间、涨跌停等（可后续扩展）。

---

## 3. 实时行情与历史窗口视图接口

### 3.1 实时 Snapshot 接口

面向使用方的语义：

- 获取某一时刻（或“最近一次可用”）的标准化行情切片：

```python
get_realtime_snapshot(
    universe: list[str],
    *,
    fields: list[str] | None = None,
    level: str = "stock",        # 预留：未来支持 ETF/期货等
    freq: str = "1d",            # 预留：1m/5m/15m
) -> pd.DataFrame
```

返回：

- `DataFrame`，index: `instrument`，columns 至少包含：
  - `datetime`（或单独返回一个 snapshot_ts 字段）；
  - `open`, `high`, `low`, `close`, `volume`, `amount`；
  - 可扩展字段：`vwap`, `turnover`, `limit_up`, `limit_down` 等。

字段命名与含义应尽量 **与 `daily_pv.h5` 保持一致**，避免因子/策略在离线和在线之间切换时需要重写逻辑。

### 3.2 报价流接口

- 为高频/事件驱动型策略提供连续报价流：

```python
class QuoteBatch:
    timestamp: datetime
    data: pd.DataFrame  # index: instrument, columns: price, volume, ...


def stream_quotes(
    universe: list[str],
    *,
    fields: list[str] | None = None,
    level: str = "stock",
    freq: str = "tick"  # 或 "1s"/"1m" 等
) -> Iterator[QuoteBatch]:
    ...
```

使用方式：

- 在线策略引擎可以：
  - 在主循环/事件回调中从 `stream_quotes` 取出最新一个或一批 `QuoteBatch`；
  - 将其与既有因子值、账户状态一起输入策略函数，生成调仓/下单信号。

### 3.3 历史窗口视图

- 用于在线因子计算或增量更新：

```python
def get_history_window(
    universe: list[str],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    bars: int | None = None,      # 指定 bars 时，start/end 可空，由数据服务层向前/向后回溯
    fields: list[str] | None = None,
    freq: str = "1d",
) -> pd.DataFrame:
    ...
```

返回：

- `DataFrame`，MultiIndex: `(datetime, instrument)`；
- columns = 请求的字段集合，字段含义与 offline snapshot 保持一致。

典型用法：

- 当某个在线因子需要最近 N 天的窗口：

```python
window_df = get_history_window(
    universe=universe,
    bars=60,
    fields=["close", "volume"],
    freq="1d",
)

factor_series = factor_x_from_df(window_df)
```

其中 `factor_x_from_df` 是 RD-Agent 演进因子在“在线版本”中的统一接口形式（见第 5 章）。

---

## 4. 账户与持仓视图接口

在线策略与风控需要访问当前账户状态。本设计只定义上层语义与字段，不固化具体实现。

### 4.1 账户与持仓

```python
@dataclass
class Position:
    instrument: str
    volume: float
    available: float
    avg_price: float
    market_value: float


@dataclass
class PortfolioState:
    cash: float
    equity: float
    positions: list[Position]
    timestamp: datetime


def get_portfolio_state() -> PortfolioState:
    ...
```

要求：

- 字段意义与实际券商/模拟盘保持一致；
- `equity` 应等于 `cash + sum(positions.market_value)`（允许微小误差）。

### 4.2 订单与成交

用于策略检查挂单/成交状态与风控决策：

```python
@dataclass
class Order:
    order_id: str
    instrument: str
    side: str       # "buy" / "sell"
    volume: float
    price: float | None
    status: str     # "pending", "filled", "canceled", ...
    created_at: datetime


@dataclass
class Trade:
    trade_id: str
    order_id: str
    instrument: str
    side: str
    volume: float
    price: float
    traded_at: datetime


def get_open_orders() -> list[Order]:
    ...


def get_trades(start: datetime | None = None, end: datetime | None = None) -> list[Trade]:
    ...
```

这两类接口主要由 AIstock 的交易通道实现，本设计只约定形态和字段，方便 RD-Agent 演进策略在在线回测/模拟盘中复用相同结构。

---

## 5. 因子与策略的在线消费方式

### 5.1 因子函数的统一形态

为满足“离线用文件、在线用视图，同一套逻辑可复用”的目标，RD-Agent 演进出来的因子建议遵循以下形态：

```python
def factor_x_from_df(df: pd.DataFrame) -> pd.Series:
    """df: MultiIndex(datetime, instrument)，columns 为标准行情字段。

    返回：index 同 df，或为 (datetime, instrument) 的子集，值为该因子数值。
    """
    ...
```

使用方式：

- 离线：

```python
df = pd.read_hdf("daily_pv.h5", key="data")
series = factor_x_from_df(df)
series.to_hdf("result.h5", key="data")
```

- 在线：

```python
window_df = get_history_window(universe, bars=60, fields=["close", "volume"], freq="1d")
series = factor_x_from_df(window_df)
# 在线只在内存中使用，不写回文件
```

### 5.2 策略函数的统一形态

在线策略引擎消费“因子视图 + 实时行情视图 + 账户视图”，建议标准策略接口为：

```python
def strategy_long_topk_with_risk(
    *,
    factors: pd.DataFrame,       # index: instrument, columns: 因子名
    prices: pd.DataFrame,        # index: instrument, columns: 当前价/盘口等
    portfolio: PortfolioState,
    context: dict | None = None, # 预留：回测 vs 模拟盘 vs 实盘 等上下文
) -> dict[str, float]:           # instrument -> target_weight
    ...
```

数据服务层在这里的职责是：

- 通过 `get_realtime_snapshot` / `get_history_window` / `get_portfolio_state` 提供上述 inputs 所需的数据；
- 在线策略引擎负责将这些数据整合后，调用策略函数，生成目标权重/订单。

RD-Agent 演进出来的策略：

- 可被翻译为上述形态的策略函数；
- 或者导出训练好的模型（如 LightGBM/XGBoost/NN 权重），在线策略函数在内部调用模型 `predict`，对数据服务层提供的因子窗口进行推断。

---

## 6. 与分阶段实施的关系

### 6.1 Phase 1–2：以离线链路为主

- 数据服务层主要负责：
  - 从自有 DB/行情源导出 snapshot HDF5 / qlib bin；
  - 确保字段/索引规范与本设计保持一致；
- 在线接口在 Phase 1–2 阶段可以先以 PoC 形式做早期验证，
  但进入 Phase 2/3 验收与生产使用时，必须按照本设计与 REQ-DATASVC-P2-001/P3-001 所定义的接口和数据形态完成生产级实现，
  不得以 PoC 或“最小可用实现”为理由保留精简路径进入执行栈。

### 6.2 Phase 3：数据服务层模块化与标准接口固化

- 在 AIstock 后端中：
  - 把上述接口实现为清晰可复用的模块/服务；
  - 完成对 miniQMT / TDX / DB 的适配；
  - 完成与账号/风控模块的基本集成。

### 6.4 与 qlib runtime 集成及模型复用（面向 AIstock 的硬性要求）

- **统一数据形态：tabular MultiIndex 因子矩阵**
  - 本设计明确要求：
    - 无论后续在 RD-Agent/qlib 侧使用何种模型（线性、树模型、深度模型、序列模型等），
    - AIstock 数据服务层对上暴露的**唯一标准数据形态**均为：
      - `pd.DataFrame`，MultiIndex: `(datetime, instrument)`；
      - columns：标准行情字段 + 因子字段（包括 Alpha158 与 RD-Agent 演进因子）；
    - 更复杂的张量结构（如 K 线图像、序列张量）须在上层通过该 tabular 矩阵自行转换获得。
  - 因此，本文件中定义的 `get_history_window` / `get_realtime_snapshot` 等接口，不是建议性的 PoC，而是 AIstock 在 Phase 2–3 内必须落地的**生产级接口契约**。

- **固定版本 qlib runtime + 自定义 DataProvider**
  - 为保证 RD-Agent 侧使用的全部 qlib 模型可以在 AIstock 侧“零重写”复用，AIstock 后端需：
    - 集成固定版本的 qlib runtime（例如 `qlib==x.y.z`），用于：
      - 解析 RD-Agent 导出的 `model_conf` / `dataset_conf` / `feature_conf` 等；
      - 加载与运行符合 qlib Model 接口的已训练模型；
    - 实现自定义 DataProvider / DataHandler，将 qlib 的数据请求（字段名、频率、窗口长度等）
      映射到本文件定义的数据服务层接口：
      - 历史视图：通过 `get_history_window` 获取指定 universe、时间窗口、字段集合；
      - 实时/近实时视图：通过 `get_realtime_snapshot` 等接口补齐当日数据；
    - 确保 DataProvider 输出的数据结构在索引、字段名、缺失值处理等方面，与 RD-Agent 在科研/回测中使用 qlib 时保持一致。

- **模型复用范围与约束**
  - RD-Agent 侧所有希望在 AIstock 生产侧复用的模型必须：
    - 遵守 qlib Model 接口契约（`fit` / `predict` / `save` / `load` 等）；
    - 在 loop 的 artifacts/registry 中完整记录：
      - 模型类型（对 qlib 而言的模型类名或 `model_type` 字段）；
      - 对应的 `model_conf` / `dataset_conf` / 特征字段名列表 / 窗口长度 / 频率等；
  - 在上述前提下，AIstock 必须保证：
    - 只要模型在 RD-Agent + qlib 环境中已通过训练和回测验证，
    - 即可在 AIstock 侧通过固定版本 qlib runtime + 本数据服务层接口直接运行，
      不得以“模型类型不在白名单”等理由简化或拒绝实现。

- **Alpha158 与 RD-Agent 因子支持**
  - 数据服务层必须完整支持：
    - Alpha158 因子所需的基础行情字段、时间索引与数据频率；
    - RD-Agent 演进因子在离线视图与在线视图中的消费需求（参见因子共享包与成果导出设计文档），
      包括近 N 日窗口的滚动计算；
  - 具体实现路径可以灵活选择（RD-Agent 预计算、AIstock 内部 Python 实现、或基于 qlib 表达式），
    但对上层使用者而言，必须通过本数据服务层提供统一、稳定的字段视图。

- **不得使用临时/精简数据路径进入生产执行栈**
  - 所有进入模拟盘/实盘执行栈的策略/模型，在消费行情与因子数据时：
    - 一律通过本设计中定义的数据服务层接口访问数据；
    - 不得绕过数据服务层，直接访问底层行情源或临时缓存；
  - 面向研究/PoC 的特殊数据路径（如直接读取某个测试文件）
    只能用于本地验证，不得进入正式运行环境，以免导致与 RD-Agent/qlib 回测环境的数据不一致。

> 本小节的约束是面向 AIstock 实现团队的硬性要求，目的是：
> - 确保 RD-Agent × qlib 侧所有已验证模型与因子，在 AIstock 生产环境中可以在统一的数据形态与接口下复用；
> - 避免由于临时/精简实现导致的回测环境与生产环境行为偏差。

### 6.3 Phase 4–5：在线策略与真实反馈闭环

- RD-Agent 演进策略在 AIstock 落地：
  - 因子函数改造为对 DataFrame 纯函数形态；
  - 策略函数改造为消费数据服务层视图的标准接口；
- 模拟盘与实盘：
  - 所有实时决策一律通过数据服务层消费数据，不再使用 h5/bin。

---

## 7. 约束与建议

- **一致性约束**：
  - 相同字段在离线视图与在线视图中含义必须一致（如 `close`、`volume`）；
  - 相同频率/市场的时间索引必须可对齐，避免窗口对齐问题。

- **解耦建议**：
  - RD-Agent / 在线策略引擎只依赖本设计中定义的接口与字段；
  - 不依赖任何具体实现细节（Redis/Kafka/具体表结构等），由 AIstock 侧自由演进。

- **可观测性建议**（供 AIstock 侧实现时参考）：
  - 每个接口应记录调用日志与耗时；
  - 对关键数据（如 snapshot 与 portfolio）提供简单的健康检查与可视化工具。

---

## 8. 硬性要求（REQ Checklist，按 2025-12-30 项目规范对齐）

> 本节将数据服务层的关键约束 ID 化，便于在 RD-Agent × AIstock 联合开发中作为硬性契约执行。

- **REQ-DATASVC-P2-001：统一数据形态（离线/研究场景）**  
  数据服务层对上暴露的历史窗口与快照接口，必须以 `pd.DataFrame`（MultiIndex `(datetime, instrument)`）
  作为唯一标准数据形态，字段名与含义与离线视图（如 `daily_pv.h5`、Alpha158 因子）保持一致。

- **REQ-DATASVC-P3-001：仅通过 DataService 获取执行栈数据**  
  进入模拟盘/实盘执行栈的策略/模型在消费行情与因子数据时，必须通过本文件定义的数据服务层接口，
  禁止直接访问底层行情源、临时缓存或测试文件。

- **REQ-DATASVC-P3-002：与 qlib runtime 的数据契约**  
  自定义 DataProvider / DataHandler 必须通过 DataService 接口获取数据，并保证输出的数据结构在
  索引、字段命名与缺失值处理等方面与 RD-Agent/qlib 回测环境保持一致，以支持模型零重写复用。

- **REQ-MODEL-P3-010：模型复用范围与约束（数据服务视角）**  
  所有希望在 AIstock 生产环境复用的 RD-Agent 模型，必须通过 DataService + qlib runtime 的组合
  进行预测，不得引入绕过 DataService 的“临时/精简数据路径”。

---
