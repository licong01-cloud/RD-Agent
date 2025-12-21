# 20251218 RD-Agent × Qlib 数据访问链路与数据准备差距分析备忘录

## 目标
- 明确 RD-Agent 在 **因子研发** 与 **回测/训练** 两条链路中分别访问哪些数据、访问顺序如何。
- 列出当前已落地的数据准备产物（H5、parquet、schema、README、治理 schemas 等）。
- 基于代码事实，指出目前仍可能缺失/不完整之处与质量提升点。
- 给出可执行的解决方案建议与验收标准。

## 一、数据访问顺序与范围（按链路拆解）

### 1. 因子研发链路（LLM 生成并执行 factor.py）

#### 1.1 data_folder 准备（一次性生成/拷贝）
- **入口**：`rdagent/scenarios/qlib/experiment/utils.py::generate_data_folder_from_qlib()`
- **生成/写入**：运行 `rdagent/scenarios/qlib/experiment/factor_data_template/generate.py` 生成行情 H5。
- **拷贝/落盘**：将 H5、README、静态因子 parquet/schema 等复制到：
  - `git_ignore_folder/factor_implementation_source_data`（全量）
  - `git_ignore_folder/factor_implementation_source_data_debug`（debug 子集）

#### 1.2 因子执行（workspace 内运行 factor.py）
- **入口**：`rdagent/components/coder/factor_coder/factor.py::FactorFBWorkspace.execute()`
- **行为**：
  - 将 data_folder 下所有文件 link 到因子 workspace 目录
  - `python factor.py` 在 workspace 内执行
  - 读取 `result.h5` 作为因子输出

#### 1.3 LLM 背景数据说明（data_folder_intro）
- **入口**：`rdagent/scenarios/qlib/experiment/factor_experiment.py`（`get_data_folder_intro()`）
- **说明生成**：`rdagent/scenarios/qlib/experiment/utils.py::get_file_desc()`
- **要点**：
  - schema（csv/json）会被用于生成字段 meaning 预览，直接影响 LLM 对字段含义与可用性的理解。

> 因子研发链路的数据范围主要是：`daily_pv.h5`、`static_factors.parquet`、其 schema（csv/json）、README，以及可选的治理 schemas。

---

### 2. 回测/训练链路（Qlib qrun + provider_uri 指向的 bin 数据）

#### 2.1 qrun 执行与结果读取
- **入口**：`rdagent/scenarios/qlib/experiment/workspace.py::QlibFBWorkspace.execute()`
- **执行目录**：`self.workspace_path`
- **典型输出/读取**：
  - 执行：`qrun <yaml>`
  - 读取：`ret.pkl`、`qlib_res.csv`
  - MLflow：在 workspace 下创建/写入 `mlruns/`

#### 2.2 bin 数据访问
- 发生在 `qrun` 内部。
- **数据源路径**：YAML 内 `qlib_init.provider_uri`。
- **数据范围**：instruments/market 股票池、calendar、features/labels 依赖的底层字段等。

> 回测/训练链路是否“可演进”，关键取决于 bin 数据的股票池覆盖与字段完整性。

---

## 二、当前已落地的数据准备产物（清单）

### 1. 基础行情 H5（供因子研发读取）
- **产物**：`daily_pv.h5`
- **生成脚本**：`rdagent/scenarios/qlib/experiment/factor_data_template/generate.py`
- **数据来源**：Qlib bin（`qlib.init(provider_uri=...)` + `D.features(...)`）
- **字段规范**：写出前将 `$open/$close/...` 清洗为 `open/close/...`（不带 `$`）
- **落点**：
  - `git_ignore_folder/factor_implementation_source_data/daily_pv.h5`
  - `git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5`

### 2. 静态因子表（parquet）与 schema
- **产物**：
  - `static_factors.parquet`
  - `static_factors_schema.csv`
  - `static_factors_schema.json`
- **落点**：
  - `git_ignore_folder/factor_implementation_source_data/`（全量）
  - `git_ignore_folder/factor_implementation_source_data_debug/`（debug）
- **注入策略（代码事实）**：
  - 优先使用 repo 内 `git_ignore_folder/factor_implementation_source_data/static_factors.parquet` 及其 schema
  - 若 repo 内不存在 parquet，则 fallback 到 `AISTOCK_FACTORS_ROOT`（默认 `/mnt/f/Dev/AIstock/factors`）寻找 parquet（但 fallback 分支仅复制 parquet，未必复制 schema）

### 3. README（供 LLM 理解数据目录）
- **产物**：`README.md`
- **落点**：data_folder 与 data_folder_debug
- **已对齐**：字段名说明不再使用 `$open` 等旧形式

### 4. provider_uri 一致性与可配置
- **生成 H5 使用的 provider_uri**：`factor_data_template/generate.py` 支持从环境变量获取（并提供默认值与兜底）
- **qrun 回测使用的 provider_uri**：由各模板 YAML 的 `qlib_init.provider_uri` 决定

### 5. 可选治理 schemas 注入
- **逻辑位置**：`generate_data_folder_from_qlib()`
- **来源目录**：`AISTOCK_DATA_GOVERNANCE_DIR`（默认 `/mnt/f/Dev/AIstock/data_governance`）下的 `schemas/`
- **落点**：复制到 data_folder 与 debug_data_folder 下的 `schemas/`

---

## 三、当前可能欠缺/未准备的数据与质量改进点

### A. 影响“回测有效性/演进能力”的高优先级缺口
1. **Qlib bin 股票池/market 覆盖可能退化**
   - 风险表现：`market: all` 仅返回单一聚合标的（如 `DAILY_ALL`），导致横截面指标（IC/RankIC）失效，模型结果高度相似。
   - 结论：这是 bin 数据导出/覆盖质量问题，不是 RD-Agent 侧纯代码改动能彻底解决的。

2. **因子研发链路（H5/parquet）与回测链路（bin）口径不一致的风险**
   - 可能体现在：股票池、时间段、停牌/缺失值处理、字段可用性等。

### B. 主要影响“LLM 生成质量/稳定性”的中优先级缺口
3. **schema 的“必达性”不足**
   - 当 static_factors 走 fallback parquet 时，逻辑可能只复制 parquet，不复制 schema。
   - 影响：LLM 背景信息缺少字段 meaning，导致生成不稳定。

4. **`static_factors_schema.csv` 的中文 meaning 覆盖仍不完整**
   - 典型集中在：预计算/派生字段（如 rolling/ratio/PriceStrength_* 等）。
   - 影响：治理一致性与 LLM 可读性。

5. **debug vs 非 debug 版本的一致性风险**
   - 若 schema 或字段集合不同，容易出现“debug 能跑，full 出错/漂移”。

### C. 体验/维护层面的低优先级改进
6. **schema 搜索多源导致说明口径漂移**
   - intro 生成会尝试多个候选 schema 路径；若版本不一致，会造成 LLM 看到的说明漂移。

---

## 四、建议解决方案（按优先级）

### 1) 先修复/验证 Qlib bin 的 market 覆盖（最高优先级）
- **目标**：保证 `market: all` 下 instruments 为真实多标的股票池，而非单一聚合标的。
- **验收标准**：
  - `D.instruments("all")` 返回多标的
  - 横截面指标不再系统性 NaN
  - 不同因子/模型回测结果具备可区分性

### 2) 强制 data_folder 内 parquet 与 schema 成对出现（提升 LLM 稳定性）
- **做法**：在 `generate_data_folder_from_qlib()` 中，fallback 到外部 parquet 时：
  - 同步寻找并复制对应 schema（csv/json）
  - 若外部无 schema，则自动生成最小 schema（至少 name/dtype），并优先合并治理 schemas/field_map
- **验收标准**：两套 data_folder 永远同时存在 parquet + schema（csv/json）。

### 3) 产线化补齐中文 meaning（治理与 LLM 可读性）
- **做法**：
  - 用 `aistock_field_map.csv` 覆盖 daily_basic/moneyflow 原始字段
  - 对派生字段（rolling/ratio/PriceStrength_*、*_5d/*_20d 等）使用规则生成 meaning
  - 对明确预计算因子从定义脚本/定义集中提取 meaning_cn/formula 并写入 schema
- **验收标准**：schema 中 meaning 空值比例显著下降，且 debug/full 一致。

### 4) 固化“权威 schema 来源”与复制策略
- **做法**：明确“权威生成点”并复制到所有运行时目录，避免 intro 多源漂移。
- **验收标准**：LLM 看到的 schema 与执行时实际字段口径一致。

---

## 五、下一步建议的执行顺序（可作为短期 roadmap）
1. 先验证并修复 qlib_bin 的 market/instruments 覆盖（否则回测不可用）。
2. 修正 data_folder 复制逻辑：fallback 时也确保 schema 到位。
3. 继续完善 `static_factors_schema.csv` 的中文 meaning（尤其预计算/派生列）。
4. 对齐 debug/full 一致性，减少漂移。

