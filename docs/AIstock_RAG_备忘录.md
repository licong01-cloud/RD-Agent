# AIstock RAG 备忘录

> 版本：v0.1（草稿）  
> 目标：梳理 AIstock 在接入 RAG（Retrieval-Augmented Generation）时的业务需求与技术方案，为后续实现提供统一参考。

---

## 1. 场景与目标

### 1.1 总体场景

- **上游：AIstock**
  - 接收用户的自然语言策略描述（股票/量化策略、研究问题等）；
  - 提供与本地时序数据库（行情、基本面、资金流、筹码等）的统一访问；
  - 管理 RD-Agent 集成、Qlib 场景、实验结果等。

- **下游：RD-Agent + Qlib + CoSTEER**
  - RD-Agent 负责因子/模型/策略的自动研究与演进；
  - Qlib 负责标准化回测与绩效评估（使用 AIstock 导出的 Qlib bin）；
  - CoSTEER 负责在 HDF5/Parquet 上做高自由度因子研发。

- **中间层：AIstock RAG + LLM**
  - 通过检索 RD-Agent 文档、AIstock 集成文档、Qlib 模板、量化知识文档等，为 LLM 提供上下文；
  - 将自然语言需求转换为：
    - RD-Agent 所需的 StrategyConfig / 场景配置 / 提示词；
    - 量化研究建议（因子构思、调参建议、诊断分析等）。

### 1.2 关键目标

- **G1：自然语言 → RD-Agent 可执行配置**
  - 用户在 AIstock 用自然语言描述策略、市场、时间段、风险偏好；
  - 通过 RAG + LLM，自动生成：
    - RD-Agent 场景类型（fin_quant / fin_factor / fin_model 等）；
    - 对应的 StrategyConfig（JSON）与 Qlib YAML 模板选择与填充；
    - CoSTEER 因子实现所需提示词与约束。

- **G2：支持策略研究与问答**
  - 用户针对某支股票、某类因子、某个市场现象提出问题；
  - RAG 引入量化研究文献、内部备忘录与实验结果；
  - LLM 结合本地时序数据（通过 API 访问）给出研究建议和解释。

- **G3：保持系统规则的严谨性**
  - 在生成 RD-Agent 提示词和配置时，必须严格遵守：
    - RD-Agent 官方文档与接口规范；
    - AIstock 集成文档中约定的路径、market/segments、bin/HDF5 分工；
    - Qlib/CoSTEER 的字段命名/数据约束。
  - 外部量化资料只能作为“知识参考”，不得覆盖本系统的硬约束。

---

## 2. 需求梳理

### 2.1 文档与知识需求

- **系统规范文档（最高优先级）**：
  - RD-Agent 官方文档：`docs/*.rst`（尤其是 `scens/*.rst`、`installation_and_configuration.rst`、`ui.rst`）。
  - AIstock 集成文档：
    - `docs/AIstock_Qlib_方案A_轻量B_设计文档.md`
    - `docs/RD-Agent_AIstock_Qlib_备忘录.md`
    - `docs/qlib_bin_export_plan.md`
    - `docs/rdagent_scheduler_design.md`
  - Qlib 场景模板与 prompts：
    - `rdagent/scenarios/qlib/experiment/*.yaml`
    - `rdagent/scenarios/qlib/prompts.yaml`
    - `rdagent/scenarios/qlib/experiment/prompts.yaml`

- **量化研究文档（次优先）**：
  - 因子类型与构造：动量、反转、价值、质量、情绪、资金流、筹码等；
  - 风险控制与组合构建经验；
  - A 股/资金流/筹码峰相关研究报告或高质量博客；
  - Qlib 社区的实验案例与经验总结。

- **本地数据与 API 说明（中高优先级）**：
  - 本地时序数据库表结构、字段含义、时间粒度；
  - AIstock 后端提供的统一数据访问 API：
    - 示例：`get_price_series`、`get_fund_flow_series`、`get_fundamental_series` 等；
  - 典型查询范式：单票日线窗口、财报对齐、滚动因子计算等。

### 2.2 功能性需求

- **R1：策略配置生成（严谨模式）**
  - 输入：自然语言策略描述 + 用户在前端勾选的模型/数据区间等；
  - 输出：
    - StrategyConfig JSON；
    - RD-Agent 场景选择与参数（loop_n、step_n 等）；
    - Qlib YAML 模板选择与关键字段覆盖；
    - CoSTEER 因子研发提示词（明确 HDF5 列名、路径、限制等）。

- **R2：因子与策略研究问答（研究模式）**
  - 输入：关于某支股票/某类因子的提问（可能包含时间区间、市场设定）；
  - 行为：
    - 通过 RAG 检索量化文献、内部文档、实验结果；
    - 通过数据访问层调用本地时序库 API，拉取实际行情/基本面/资金流数据；
    - 结合两者，输出分析和研究建议（非直接可执行配置）。

- **R3：多模式支持**
  - 同一个 RAG 知识库，支持不同“工作模式”：
    - 模式 A：RD-Agent 提示词/配置编辑（强约束、结构化输出）；
    - 模式 B：通用量化研究问答（弱约束、综合分析输出）。

---

## 3. 架构分层与职责划分

整体分为三层：**知识层（RAG） → 数据访问层 → LLM 推理层**。

### 3.1 知识层（RAG）

- **职责**：为 LLM 提供“如何正确使用系统”和“量化领域知识”的文本上下文。
- **内容类型**：
  - `rdagent_spec`：RD-Agent 官方文档、API 参考、场景说明；
  - `aistock_integration`：AIstock + RD-Agent + Qlib 集成规范（当前几份备忘录和设计文档）；
  - `qlib_templates`：Qlib YAML 模板、prompts、配置片段；
  - `quant_research`：外部量化研究文档；
  - `code_entry`（可选）：少量入口/协议相关源码（`cli.py`、`qlib_rd_loop/*.py`、`llm_conf.py` 等）的说明性片段。

- **RAG 的职责边界**：
  - RAG ≠ 数据库：
    - 不直接承载高频更新的时序数据；
    - 不负责精确数值查询与聚合。
  - 重点在于：
    - 让 LLM 知道“有哪些场景、模板、约束、字段、API”；
    - 帮忙生成正确的配置/查询/因子逻辑。

### 3.2 数据访问层（本地时序数据库 API）

- **职责**：
  - 对接 Timescale/ClickHouse 等本地时序库、基础数据仓库；
  - 提供一组稳定的函数/HTTP API：
    - `get_price_series(ts_code, start, end)`
    - `get_fund_flow_series(ts_code, start, end)`
    - `get_fundamental_series(ts_code, metrics, start, end)`
    - `get_stock_snapshot(ts_code, as_of_date)`
    - ……
  - 实际执行 SQL/时序查询，处理缓存、性能等问题。

- **LLM 的使用方式**：
  - 通过“工具调用/函数调用”模式：
    - LLM 先基于 RAG 文档知道有哪些 API、如何调用；
    - 然后生成带参数的 API 调用；
    - 系统执行后，将数值结果返回给 LLM 继续推理与解释。

### 3.3 LLM 推理层

- **职责**：
  - 根据任务模式（RD-Agent 配置 / 量化研究）调用 RAG & 数据访问层；
  - 生成结构化的 StrategyConfig / 提示词，或非结构化的研究建议。

- **模式区分**：
  - **模式 A：配置生成模式**
    - System Prompt：强调遵守 RD-Agent/AIstock 规范，严格输出固定 JSON/YAML 结构；
    - RAG 检索：主要来自 `rdagent_spec` + `aistock_integration` + `qlib_templates`；
    - 外部量化文档权重低，仅作补充说明。
  - **模式 B：研究分析模式**
    - System Prompt：强调综合分析、提出研究建议，不必输出可执行配置；
    - RAG 检索：`quant_research` + 内部备忘录 + 必要的 `rdagent_spec` 说明；
    - 强化与数据访问层协作，从本地 DB 拉取实际数据做诊断。

---

## 4. 自建 RAG vs 使用开源 RAG 组件

### 4.1 自建完整 RAG 的利弊

- **优点**：
  - 逻辑简洁，完全贴合 AIstock + RD-Agent 的需求；
  - 不会引入大而全框架的额外复杂度。

- **缺点**：
  - 需自行处理：文档切分、embedding 调用、向量存储、检索打分、重排序等；
  - 一旦场景扩展（多数据源、多类型索引），后续维护成本增加。

### 4.2 使用开源 RAG 组件（推荐思路）

- 基础设施层：
  - 使用现成组件解决：
    - 文档加载与切分；
    - 向量化（复用现有 SiliconFlow embedding 等）；
    - 向量存储（FAISS/Chroma/SQLite+HNSW 等）；
    - 基础检索接口。

- 业务逻辑层（必须自研）：
  - 文档打标签与分层：`rdagent_spec` / `aistock_integration` / `quant_research` 等；
  - 模式 A/B 下的检索策略（过滤、权重）；
  - 用于生成 StrategyConfig / 提示词的 System Prompt、JSON Schema、模板。

### 4.3 结论

- 当前阶段（单系统、少用户、不考虑复杂权限）：
  - **不需要完整“RAG 产品”，也不建议从零造轮子**；
  - 更适合：
    - 选一个轻量的向量检索方案（或 SDK/FW）；
    - 在其上方搭建一个“AIstock + RD-Agent 定制的知识/模式层”。

---

## 5. 与 RD-Agent 的深度集成

### 5.1 RD-Agent 场景与 StrategyConfig 映射

- RAG 应优先索引的 RD-Agent 内容：
  - `docs/scens/*.rst`：各类场景的用途、输入输出；
  - `docs/installation_and_configuration.rst`、`docs/ui.rst`：环境变量/启动方式；
  - `rdagent/app/cli.py`：CLI 入口（`fin_quant` / `fin_factor` / `fin_model` 等）；
  - `rdagent/app/qlib_rd_loop/*.py`：各研究 loop 的参数（loop_n、step_n、all_duration 等）；
  - `rdagent/oai/llm_conf.py` 与 backend：LLM 调用约束、response_format 支持等。

- StrategyConfig 层建议：
  - 单独起草一份 `docs/aistock_strategy_interface.md`（未来可补）：
    - 明确 StrategyConfig JSON Schema；
    - 描述场景字段（任务类型、市场、时间段、benchmark、模型列表、主指标等）；
    - 给出若干“自然语言 → StrategyConfig 示例”。
  - RAG 检索时将该文档设为生成 StrategyConfig 的最高优先级依据。

### 5.2 RD-Agent 提示词与因子实现

- RAG + LLM 帮助：
  - 根据策略描述和已有实验结果，为 CoSTEER 生成/修改因子提示词：
    - 明确 HDF5 列名（`close/open/high/low/volume/amount/...`）；
    - 避免错误（如 `$close` vs `close`）；
    - 结合量化研究文档提出更合理的因子组合。

- 硬约束：
  - 始终以 `rdagent/scenarios/qlib/experiment/prompts.yaml` 中的约束为准：
    - 不允许乱 import 未安装库；
    - 不允许用 try/except 吞掉错误；
    - 必须在当前工作目录下按规范读写 HDF5 等。

---

## 6. 与本地时序数据库的结合

### 6.1 RAG 不直接承载时序数据

- 时序数据（行情、基本面、资金流、筹码等）的特点：
  - 高维、高频更新、需要精确数值查询与聚合；
  - 不适合全部 embedding 进向量库（体量大且更新频繁）。

- 设计原则：
  - **RAG 存“怎么查、查什么、字段代表什么”**；
  - **数据库/API 负责“真实查数值数据”**。

### 6.2 数据访问方式

- 面向 LLM 暴露一组受控 API：
  - `get_price_series(ts_code, start_date, end_date)`
  - `get_fund_flow_series(ts_code, start_date, end_date)`
  - `get_fundamental_series(ts_code, metrics, start_date, end_date)`
  - …

- 流程：
  1. LLM 通过 RAG 获取表结构/API 文档与使用示例；
  2. 构造具体 API 调用（或 SQL 模板 + 参数）；
  3. 后端执行查询，返回结构化数据；
  4. LLM 基于查询结果输出分析或因子代码。

---

## 7. 多模式 RAG 的控制策略

### 7.1 单一向量库，多种“模式”

- 不必为每个场景建独立向量库；
- 建议：
  - 单库存放所有文档；
  - 通过 `type/tag` 标记文档类别；
  - 在检索时按模式做过滤与权重控制。

### 7.2 模式 A：RD-Agent 配置/提示词模式

- System Prompt：
  - 强调“输出必须是可执行的 StrategyConfig/提示词/配置，遵守如下 JSON/YAML Schema”；
  - 当外部文档（quant_research）与本地规范冲突时，以本地规范为准。

- 检索策略：
  - `filter: type in ["rdagent_spec", "aistock_integration", "qlib_templates"]`；
  - `quant_research` 权重降低或不使用。

### 7.3 模式 B：量化研究/问答模式

- System Prompt：
  - 强调综合分析与研究建议，不强求输出特定结构；
  - 可以引用更多量化研究与外部文献。

- 检索策略：
  - `filter: type in ["quant_research", "aistock_integration", "rdagent_spec"]`；
  - 提高 `quant_research` 的权重。

---

## 8. 实施优先级与路线图（建议）

### 8.1 第一阶段：RAG for RD-Agent 集成

- 建立最小 RAG：
  - 语料：现有 RD-Agent `.rst` 文档 + AIstock 集成备忘录 + Qlib 场景 YAML / prompts；
  - 建库与检索：选用轻量向量库与 SDK（FAISS/Chroma + LangChain/LlamaIndex 的子集）；
  - 模式：先实现“模式 A”（StrategyConfig/提示词生成）。

- 输出能力：
  - 能将自然语言策略转换成：
    - RD-Agent 场景选择；
    - StrategyConfig JSON；
    - Qlib YAML 覆盖片段；
    - CoSTEER 因子提示词草稿。

### 8.2 第二阶段：接入时序数据库与研究模式

- 实现统一数据访问 API；
- 在 RAG 语料中加入：
  - 表结构与字段说明；
  - 典型查询模式；
- 实现“模式 B”：
  - 支持结合真实数据做单票/多票的诊断分析与研究建议。

### 8.3 第三阶段：扩大量化知识库与结果反馈

- 引入更多高质量量化研究文档；
- 将 RD-Agent 回测结果摘要、实验备忘录等纳入 RAG；
- 让 LLM 在提出新因子/策略时参考历史实验效果，形成更“闭环”的研究流程。

---

## 9. 结语

- 本备忘录的核心观点：
  - **RAG 是 AIstock + RD-Agent 的“知识/规范层”，不是“数据仓库”**；
  - **系统规范和本地文档优先，外部量化资料作为增强**；
  - **通过多模式控制与 API 工具调用，将“知识检索”和“数据访问”组合起来，为因子研发与策略实验提供更智能的支持**。

- 后续可以在此基础上：
  - 进一步细化 StrategyConfig Schema 文档；
  - 确定选用的向量库/SDK；
  - 在 AIstock 后端实现最小可用的 RAG + 数据访问一体化服务。

## 10. RAG 对 RD-Agent 因子演进质量的影响（补充）

### 10.1 当前因子错误与低收益的原因拆分

- **LLM 量化专业度有限的影响**：
  - 对时间窗口对齐、前视偏差（look-ahead bias）、除零处理等细节缺乏“肌肉记忆”，容易写出形式正确但细节错误的因子实现。
  - 对 A 股特有约束（T+1、涨跌停、ST 处理、资金风格等）认识不足，初始策略往往偏简单或忽略关键风险约束。

- **自动演进早期的“正常现象”**：
  - RD-Agent 从 0 开始自动探索，天然会产生大量收益低甚至失败的实验，这一点在人类量化团队中同样存在，只是人类会在脑中提前过滤一部分垃圾方向。
  - 目前演进轮数和每轮 step 数有限，一些有潜力的方向可能尚未经过足够迭代就被丢弃。

- **系统目标设定较为苛刻**：
  - 例如：短周期下要求年化收益 >10%、最大回撤 <10%、单笔止盈/止损严格等，这会放大“策略一般”带来的收益不佳感受。

结论：当前看到的“因子错误多、收益低”并不意味着数据或 RD-Agent 框架有根本问题，更多是 **LLM 缺乏专业量化经验 + 自动探索早期试错不可避免** 的综合结果。

### 10.2 引入 RAG 后可以改善的方面

- **减少常识性实现错误**：
  - RAG 中存放经过审校的“正确因子实现示例”和“错误案例 + 修正方案”，LLM 生成新代码时可参照这些模式，降低诸如：
    - 时间窗口错位（例如动量与波动率窗口不对齐）；
    - 使用未来信息（未来收益/未来财报字段）；
    - 未处理除零、极端值等数值稳定性问题。

- **更专业的初始假设与方向选择**：
  - 通过检索量化研究文档和内部备忘录，让 LLM 知道：
    - 常见有效因子家族（动量、反转、质量、价值、情绪、流动性等）；
    - A 股中常见的组合形态（行业中性、多因子框架、交易成本/换手约束等）。
  - 这样 RD-Agent 在提出新因子/模型假设时，不再完全“白纸起步”，而是更接近有一定经验的初级量化研究员。

- **每轮演进的“有效实验比例”提升**：
  - 有了 RAG 后，更多实验会集中在“有一定理论/经验支撑”的方向上，完全无意义或明显错误的实验占比下降，从而提高 GPU/CPU 的使用效率。

需要强调的是：

- RAG **不能直接保证出现高收益策略**，它更多是提升“候选方向的平均质量”和“单轮迭代的增益稳定性”；
- Alpha 仍然主要依赖真实数据上的回测结果筛选和样本外验证。

### 10.3 两种接入方案与技术复杂度

#### 方案 A：轻量级 Prompt 层接入 RAG（推荐优先实施）

- **核心思路**：不大改 RD-Agent 主流程，只在调用 LLM 生成因子/模型/行动建议前，多做一步“知识检索”：
  1. 根据当前任务上下文（市场、时间段、目标、已有实验结果）构造查询；
  2. 调用 RAG 服务检索若干条相关片段（内部规范、因子示例、量化经验等）；
  3. 将检索结果以 `{{ knowledge_context }}` 的形式注入现有的 `prompts.yaml` 模板（例如因子生成、模型生成、action_gen 提示）；
  4. 其余 RD-Agent 调度逻辑保持不变。

- **对 RD-Agent 源码的改动点（示意）**：
  - 在构造 LLM 调用参数的地方（通常是某个 Scenario / Loop 的 prompt 渲染逻辑）：
    - 增加一小段：`knowledge = rag_client.retrieve(context)`；
    - 将 `knowledge` 作为额外字段注入到 Jinja 模板上下文中。
  - 增加一个独立的 `rag_client` 模块，负责：
    - 接收查询结构（任务类型、关键词等）；
    - 发起 HTTP/SDK 请求到 AIstock RAG 服务；
    - 返回整理好的文本片段列表。

- **复杂度评估**：
  - 代码量：几十到一两百行以内，主要集中在：
    - RAG Client 封装；
    - 向 prompt 渲染函数注入 `knowledge_context`。
  - 对现有 RD-Agent 逻辑的侵入性：低；失败时可以回退为“无 RAG 模式”继续使用。

#### 方案 B：深度集成为前置“策略生成模块”

- **核心思路**：
  - 在 AIstock 侧或独立服务中，构建一套“自然语言策略 → StrategyConfig → RD-Agent 场景”的 RAG+LLM 生成链；
  - RD-Agent 更专注于执行结构化的 StrategyConfig，而不是直接面对用户的自然语言与所有量化知识。

- **实现要点**：
  - 定义统一的 StrategyConfig Schema（任务类型、市场、时间段、benchmark、候选模型列表、因子偏好、风险约束等）；
  - RAG+LLM 根据：
    - 用户高层策略描述；
    - RAG 检索到的量化知识与历史实验结果；
    - 内部规范文档；
    生成一个高质量的 StrategyConfig；
  - RD-Agent 在入口处接收该 StrategyConfig，将其映射到：
    - 具体的 Qlib YAML 模板与覆盖字段；
    - 每轮演进的 loop_n/step_n 设置；
    - 因子/模型候选集与优先级。

- **对 RD-Agent 源码的改动点**：
  - `fin_quant` 等 CLI 入口函数，需要支持从外部 JSON/配置文件读取 StrategyConfig；
  - 场景初始化逻辑需要增加一层“StrategyConfig → 内部参数对象”的映射；
  - 部分 prompt 构造逻辑改为基于 StrategyConfig，而不是完全从固定 YAML 中硬编码。

- **复杂度评估**：
  - 代码改动：中等偏大（数百行级别，涉及 CLI、配置对象、部分 Scenario 初始化逻辑）；
  - 更适合作为第二阶段，在方案 A 跑通并验证有效以后再推进。

### 10.4 建议的推进顺序

- **短期（1–2 周）**：
  - 优先落地方案 A：
    - 在因子生成、模型生成、action_gen 等关键节点前挂载 RAG 检索；
    - 用少量代码改动，观察：
      - 因子实现错误率是否下降；
      - 每轮演进中“略有改进”的实验数量是否增加；
      - 回测结果分布是否更健康（极端差策略比例是否减少）。

- **中期（1–2 月）**：
  - 在验证 RAG 对演进质量有正向作用后，逐步引入方案 B：
    - 将“自然语言策略 → StrategyConfig”固定为一条标准链路；
    - 让 RD-Agent 尽量只接收结构化任务配置，成为执行引擎 + 回测工厂。

- **长期**：
  - 结合 RD-Agent 历史实验结果和 Qlib 回测摘要，将“因子/策略 → 结果反馈”也纳入 RAG 语料，形成闭环：
  - 新一轮策略生成时，RAG+LLM 不仅参考规则和文献，还参考“该账号在本地真正跑出来的结果”；
  - 逐渐朝着“本地量化研究知识库 + 自动化科研助手”的方向演进。

## 11. RAG 工具选型与结论（补充：优先提升因子研发效率）

本节补充对主流 RAG 组件与框架的对比，并给出在当前阶段（优先解决 RD-Agent 因子研发效率、短期不做 KG）的落地选型与工程边界建议。

### 11.1 当前最优先要解决的问题定义

- **目标**：降低因子研发中“反复 coding + 反复报错/返工”的时间成本，使 LLM 更稳定地产出符合本项目约束的因子实现。
- **问题根因（高频）**：
  - LLM 缺少项目内“硬约束”上下文（输出格式、路径、数据集对齐、不可重复过滤股票池等）；
  - LLM 缺少“预计算因子表数据字典”，导致重复计算或与标准化预计算口径不一致；
  - 错误往往属于“运行不报错但逻辑不符合定义”，需要 reviewer 才能发现。

结论：短期最有效的 RAG 不是“引入更多外部量化知识”，而是让 LLM 在生成与评审时能够检索到并遵守 **项目内规范 + 预计算表字典 + 常见错误模式与修复模板**。

### 11.2 组件分层：向量库 vs RAG 编排框架 vs 端到端系统

- **向量库（存储/检索层）**：Chroma、FAISS 等。
  - 职责：向量索引、相似度检索、metadata 过滤（视实现而定）.
  - 非职责：不负责业务规则、提示词注入、工作流编排.
- **RAG 编排框架（索引/检索抽象层）**：LlamaIndex、LangChain.
  - 职责：文档加载/切分、向量化、检索器、上下文拼装、（可选）重排等.
  - 非职责：不应直接替代 AIstock/RD-Agent 的业务编排与任务执行.
- **端到端 RAG 系统（平台/服务）**：LightRAG.
  - 职责：包含更完整的存储与服务化形态（如多类存储、API/WebUI、可观测等）.
  - 代价：索引流程与运维面更重，且对索引期 LLM 能力要求更高.

### 11.3 候选方案对比：LightRAG vs LangChain/LlamaIndex + Chroma/FAISS

#### 11.3.1 LightRAG（HKUDS/LightRAG）

- **优势**：
  - 提供更完整的系统能力（服务化接口、WebUI、引用、评测/Tracing、删除与一致性维护等）；
  - 支持图存储与多种向量/KV 后端；
  - 检索模式更丰富（local/global/hybrid/mix 等）并强调 reranker.
- **劣势（与当前阶段目标冲突点）**：
  - 索引流程更重、成本更高；
  - 需要更强的抽取模型与更成熟的 schema 治理；
  - 当前最紧迫问题是因子实现的工程约束与口径一致性，向量检索注入即可显著改善.

结论：LightRAG 更适合作为后续“RAG 平台化/治理 UI”方向的候选方案，而非当前“提升因子研发效率”的第一优先落地.

#### 11.3.2 LangChain

- **优势**：
  - 生态广、通用工作流编排能力强，适合未来在 AIstock 上扩展阶段 B（自然语言到策略的多步骤编排）.
- **劣势**：
  - 抽象面更大，短期仅做检索注入时容易引入不必要的复杂度；
  - 若不控制框架渗透，长期可能出现“业务代码里到处是 chain/agent”的边界漂移.

结论：LangChain 在阶段 B 的价值更高；阶段 A 可用但需严格限制使用范围.

#### 11.3.3 LlamaIndex

- **优势**：
  - 更聚焦 RAG 的索引/检索路径，概念更少，适合阶段 A 的“检索注入 + 提示词增强”；
  - 更容易保持系统边界（只提供 `retrieve()->contexts[]`），不与 AIstock/RD-Agent 的编排逻辑冲突.
- **劣势**：
  - 若未来阶段 B 需要复杂多步骤编排，仍可能需要额外的 orchestrator（自研或引入 LangChain）.

结论：在“优先提升因子研发效率”的短期目标下，LlamaIndex 更容易快速见效.

### 11.4 最终选型（当前阶段）

- **选型结论**：采用 **LlamaIndex + Chroma**.
- **理由（与当前目标强相关）**：
  - 最快把“项目内规范/字典/修复模式”变成可检索上下文，减少因子开发返工；
  - 嵌入式落地简单，代码侵入 RD-Agent/AIstock 低；
  - 不做 KG 也能实现核心价值（减少实现偏差、提高一次通过率）.

### 11.5 KG（知识图谱）在本项目的价值与短期取舍

- **价值**：KG 更适合支撑“策略/因子逻辑分析 UI”“组件化组合与约束筛选”等中长期能力.
- **短期不做 KG 的原因**：
  - 索引流程更重、成本更高；
  - 需要更强的抽取模型与更成熟的 schema 治理；
  - 当前最紧迫问题是因子实现的工程约束与口径一致性，向量检索注入即可显著改善.

### 11.6 索引期 LLM 类型与 token 成本注意事项

- **索引期模型建议**：优先使用“指令遵循强、结构化输出稳定”的模型，不建议使用偏 reasoning 的模型做批量抽取.
- **token 成本构成**：输入（prompt + chunk）+ 输出（抽取结果/摘要）+ 重试（JSON 不合规等）.
- **降本要点**：
  - 控制 chunk 大小与数量，减少重复 prompt 开销；
  - 对静态规范类文档可采用更少切分、更高复用的策略；
  - 对预计算因子表字典优先生成结构化索引（列名/含义/来源），减少“长文本反复读”.

### 11.7 KB 治理：文档删除/禁用、可追溯与可审计

即使不引入 LightRAG，也必须具备知识库治理能力，避免“错误文档污染生成”。建议：

- **文档唯一标识**：所有入库文档需有稳定的 `doc_id`.
- **软删除优先**：通过 registry 将 `doc_id` 标记为 disabled，检索时过滤；保留可回滚与审计记录.
- **引用（citation）与审计落盘**：每次生成/评审应保存 `retrieved_contexts[]`（含 doc_id、source_path、chunk_id、score、kb_version）.
- **KB 与 Index 分离**：KB 可版本化、Index 可重建，便于回滚与一致性管理.

#### 11.7.1 文档管理策略（推荐软删除优先）

- **软删除/禁用（P0 必做）**：
  - 引入 `doc_registry`（doc_id -> status/enabled/disabled + source_path + hash + tags + kb_version）。
  - 当发现文档不准确或会污染生成时，将其标记为 disabled；检索阶段必须过滤掉 disabled 文档。
  - 优点：可回滚、可审计、实施成本低。
- **硬删除（P1 可选）**：
  - 对 Chroma 中对应 doc_id 的向量做物理删除（或触发重建索引）。
  - 适用：确定永久删除且需要回收空间时。

### 11.8 阶段 A/B 的架构边界与后续演进（避免复杂化）

- **阶段 A（当前）**：RAG 仅负责 `retrieve()->contexts[]`, 注入因子生成与 critic 评审提示词.
- **阶段 B（未来 AIstock NL2Strategy 多步骤）**：
  - 不强制替换 LlamaIndex；可以自研轻量 orchestrator，或在确实需要复杂工作流时引入 LangChain 作为编排层；
  - 关键是始终通过统一接口隔离底层实现：
    - `RetrieverService.retrieve(query, filters) -> contexts[]`
    - orchestrator 只依赖该接口与 RD-Agent 执行接口，不直接依赖底层框架对象.

### 11.9 与因子研发效率直接相关的落地清单（建议优先级）

- **P0：入库三类权威语料**：
  - 因子开发硬约束（输出格式、股票池假设、文件写入协议等）；
  - 预计算因子表数据字典（表名/列名/含义/口径/路径）；
  - 常见错误模式与修复模板（例如“因子定义为 A_minus_B 则必须直接引用预计算 A/B 列”）.
- **P0：最小校验器（validator）**：
  - 自动检查 MultiIndex 与 `to_hdf` 协议；
  - 自动提示“应优先使用预计算列而非重算”的规则；
  - 自动处理/提示除零与 inf 风险.
- **P1：检索策略**：
  - 模式 A（严谨）：只检索 `rdagent_spec/aistock_integration/qlib_templates/factor_spec` 等强约束文档；
  - 外部量化文献仅作为补充，不得覆盖本系统硬约束.

### 11.10 设计方案更新：因子开发使用“RAG 增强的 LLM 调用链”

本项目在因子开发阶段仍然由 LLM 负责编写代码；“使用 RAG”指在调用 LLM 之前检索项目内权威上下文，并将检索结果注入同一次 LLM 请求中，从而提高一次通过率并降低返工。

#### 11.10.1 标准调用链（retrieve → augment → LLM → validator → 迭代）

- **Step 1：构造查询**：基于因子定义、当前场景（数据约束/输出协议）、以及最近一次失败反馈（traceback/critic/validator 输出）生成检索 query。
- **Step 2：RAG 检索**：使用 LlamaIndex 从 Chroma 向量库检索 `top_k` 片段，强制优先命中：
  - 因子输出协议（MultiIndex、`result.h5`、`to_hdf` 写入规则）；
  - 预计算因子表数据字典（表名/列名/口径/路径）；
  - 常见错误模式与修复模板（例如“定义要求用预计算列则不得重算口径”）。
- **Step 3：上下文拼装**：将检索结果格式化为固定结构的 `knowledge_context`（带 citation：doc_id/source_path/chunk_id），并注入因子编码提示词。
- **Step 4：调用 LLM 生成代码**：LLM 输出因子实现代码。
- **Step 5：执行与校验**：运行因子代码并执行 validator（输出格式、预计算列引用要求、除零/inf 风险等）。
- **Step 6：失败则迭代**：把 validator 的结构化失败原因（而非长日志）与必要的 traceback 作为新的输入，再走一轮 Step 1-5。

#### 11.10.3 Retry 策略（含“硬失败 gate”）

为避免长时间无效循环（例如长达数小时仍无法进入回测），建议将失败分为“可修复失败（允许 retry）”与“不可修复失败（直接终止）”，并对 retry 次数设置硬上限（例如 1-2 次）。

- **可修复失败（允许 retry，建议 1-2 次）**：
  - 代码可通过修改直接修复的问题：
    - 输出协议/格式错误（MultiIndex、`result.h5`、`to_hdf` key、dtype 等）；
    - 列名/字段引用错误或缺失检查不足（例如定义要求必须使用 `mf_main_net_amt_ratio_5d` 但代码使用了 `mf_main_net_amt_ratio`）；
    - LLM 输出结构不合规（JSON 解析失败等）。
  - 对应策略：将 validator 的结构化错误摘要作为下一轮输入，必要时补充一次检索，触发“重写/修复代码”。

- **不可修复失败（直接终止 abort，不进入 retry）**：
  - 环境/数据/依赖/路径类问题（LLM 重写代码无法解决）：
    - 数据文件缺失或路径错误；
    - 依赖未安装、权限不足；
    - 回测入口配置错误导致流程无法启动等。
  - 对应策略：输出明确的“系统修复项”并终止，待人工/系统修复后再运行。

#### 11.10.2 “必须引用预计算列（当定义如此要求时）”的落地方式

- 当因子定义/任务描述明确点名需要使用某些已预计算字段（例如 `mf_main_net_amt_ratio_5d`、`mf_main_net_amt_ratio_20d`）时：
  - 因子实现必须直接读取这些字段并组合计算，不得从原始明细字段重新推导等价滚动指标，以避免与预计算口径不一致。
- 建议在因子任务结构中显式声明依赖字段（便于 validator 强制检查），而不是仅依赖自然语言解析。

### 11.11 工程隔离与落地形态（LlamaIndex 进仓库，Chroma 数据不入 git）

#### 11.11.1 代码与数据的边界

- **LlamaIndex 作为 RD-Agent 的必须组件可以进入 git 仓库**：用于提供统一的 `RetrieverService` 接口与实现。
- **Chroma/向量索引产物必须不入 git**：索引数据体积大、易变、不可审计；应放入可丢弃产物目录（例如 `git_ignore_folder`）并通过环境变量配置。

#### 11.11.2 建议的目录与接口边界

- 建议在 `rdagent/` 下新增独立子包（示例命名）：
  - `rdagent/rag/`：只负责检索与上下文格式化
    - `RetrieverService.retrieve(query, filters, top_k) -> contexts[]`
    - `contexts[]` 仅包含纯数据（text/doc_id/source_path/score/kb_version），禁止向外暴露 LlamaIndex Node/Document 对象
  - `rdagent/factor_dev/`（或因子开发组件内部）：仅调用 `RetrieverService` 获取 `knowledge_context`，再调用现有 LLM coder
  - `rdagent/validators/`：集中放置 validator，失败时给出结构化可回灌的错误摘要

#### 11.11.3 存储路径建议

- 建议使用可配置存储目录（不入 git）：
  - `RAG_STORAGE_DIR=git_ignore_folder/rag_storage`
  - Chroma 持久化：`git_ignore_folder/rag_storage/chroma`
  - 文档 registry 与索引元信息（doc_id、kb_version、embedding 配置等）同目录保存，便于回滚与重建

#### 11.11.4 低侵入接入点

- 对 RD-Agent 的接入建议控制在极少数位置：
  - 在因子生成/因子实现/critic 评审的 prompt 渲染处增加 `knowledge_context`；
  - 其他业务逻辑（回测、workspace、loop 调度）保持不变。
- 必须保留开关（例如 `RAG_ENABLED`），确保随时可退化为“无 RAG 模式”以排障或对比效果。

### 11.12 P0 金标测试用例：强制引用 `mf_main_net_amt_ratio_5d`

为验证“RAG + validator + retry gate + trace”闭环在因子研发中的实际效果，P0 阶段引入一个可复现、可判定的金标用例。

#### 11.12.1 用例背景

- 在实际运行中发现同一类错误会在多个 Evolving Rounds 中重复出现：
  - 因子定义明确要求使用预计算列 `mf_main_net_amt_ratio_5d`（来源：`static_factors.parquet` 或等价静态因子表）；
  - 但实现代码错误地使用原始字段 `mf_main_net_amt_ratio` 并自行计算 5 日移动平均，从而与预计算口径不一致。

#### 11.12.2 金标判定标准（必须满足）

- **必须项**：
  - 代码必须显式读取并使用 `mf_main_net_amt_ratio_5d` 作为输入；
  - 代码必须对 `mf_main_net_amt_ratio_5d` 的存在性做检查：缺失时抛出明确错误（不可静默降级到其他字段）。
- **禁止项**：
  - 不允许使用 `mf_main_net_amt_ratio` 通过 rolling/mean 等方式替代 `mf_main_net_amt_ratio_5d`。

该用例不要求因子收益表现（alpha）好坏，仅用于验证“实现是否符合定义口径”与“自动修复闭环是否生效”。

### 11.13 整体架构（内嵌 RD-Agent，低侵入）

#### 11.13.1 模块与职责

- **RAG 检索层（内嵌）**：
  - `rdagent/rag/`：LlamaIndex + Chroma 的封装与 `RetrieverService` 接口
  - 输出：结构化 `contexts[]` 与可注入的 `knowledge_context`
- **因子代码生成器（generator）**：
  - `FactorCodeGenerator.generate(task) -> code`
  - 内部编排：`retrieve → augment → LLM → run → validator → retry/abort`
- **校验器（validators）**：
  - `output_format_validator`：MultiIndex、`result.h5`、dtype、inf
  - `precomputed_ref_validator`：当定义要求时强制引用预计算列（P0 金标用例即该 validator）
- **Trace 记录与诊断（workspace）**：
  - 每次 attempt 记录 query、命中 citation、validator 结果、决策等

#### 11.13.2 对 RD-Agent 侵入最小化的接入点

- P0 仅接入两处：
  - **因子写代码**（factor coder）
  - **critic 评审**（evaluator）
- 接入方式：仅在 prompt 渲染上下文中增加 `knowledge_context`（当 `RAG_ENABLED=0` 时为空字符串）。

### 11.14 实现方式（关键接口与数据结构）

#### 11.14.1 `RetrieverService` 输出结构（对外契约）

- `retrieve(query, filters, top_k) -> contexts[]`
- `contexts[]` 每条至少包含：
  - `text`
  - `doc_id`
  - `source_path`
  - `chunk_id`
  - `score`
  - `type`（语义类型）
  - `tags`（标签数组）
  - `kb_version`
  - `doc_format`

#### 11.14.2 generator 的结构化 trace（必须落盘）

- trace 存储位置：workspace 内固定文件，默认采用 `rag_trace.jsonl`（一行一个 attempt，便于流式写入与诊断）。
- 每个 attempt 记录以下字段（最小集）：
  - `attempt_idx`, `query`, `top_k`, `kb_version`
  - `retrieved_contexts`（citation：doc_id/source_path/chunk_id/score/type）
  - `validator_results`（name/passed/error_code/error_summary）
  - `decision`（accept/retry/abort）
  - `timing_ms`（retrieve/llm/run/validate）

该 trace 是诊断 RAG 是否起作用、以及定位“坏文档污染”的首要依据。

#### 11.14.3 validator + retry gate（P0 必须实现）

- validator 将失败分类为两类：
  - **可修复失败**（触发 retry 1-2 次）：代码/输出协议/列名/格式
  - **不可修复失败**（直接 abort）：环境/数据/依赖/路径
- P0 金标用例的 validator 建议输出结构化摘要（用于回灌）：
  - `error_code: PRECOMPUTED_COLUMN_REQUIRED`
  - `required_columns: ["mf_main_net_amt_ratio_5d"]`
  - `forbidden_columns: ["mf_main_net_amt_ratio"]`
  - `fix_instruction`: 必须直接读取 required_columns，不得 rolling 替代；缺失则抛 ValueError

### 11.15 落地步骤（P0）与验收标准

#### 11.15.1 P0 步骤（按顺序）

1. **确定 embedding provider 与模型**：通过 `.env` 配置 embedding API 与模型名（建议优先使用 `BAAI/bge-m3`）。
2. **建立 P0 KB 语料清单**：仅索引 `docs/ + prompts + 规范片段`，并将“预计算列必须引用”的规则与反例纳入语料。
3. **实现 `rdagent/rag/` 最小可用版本**：Chroma 持久化目录位于 `git_ignore_folder/rag_storage/chroma`。
4. **实现 `FactorCodeGenerator`（封装调用链）**：输出结构化 trace，并支持 `RAG_ENABLED` 回退。
5. **接入两处 prompt**：因子写代码 + critic，注入 `knowledge_context`。
6. **实现 P0 validators**：输出协议 validator + 预计算引用 validator（覆盖金标用例）。
7. **接入 retry gate**：对可修复失败触发 1-2 次自动修复；不可修复失败直接 abort。
8. **实现文档管理（P0：软删除/禁用）**：落地 `doc_registry`，支持将某个 doc_id 标记为 disabled，并在检索阶段强制过滤。
9. **跑金标用例验证**：触发一次“错误实现（使用 mf_main_net_amt_ratio rolling）”并确认系统能在一次 retry 内修正为引用 `mf_main_net_amt_ratio_5d`。

#### 11.15.2 P0 验收标准

- **A1：金标用例收敛**：同一因子在一次失败后，下一次 retry 能修复为“直接使用 `mf_main_net_amt_ratio_5d` 并做存在性检查”。
- **A2：trace 完整**：workspace 中可看到每次 attempt 的 query、citation、validator 结果、以及 retry/abort 决策。
- **A3：回退可用**：`RAG_ENABLED=0` 时，系统行为与原流程一致，不引入额外失败。
- **A4：无效长循环被阻断**：当出现环境/数据/依赖/路径类错误时，流程能快速 abort 并给出明确修复项，不再无限重试。
- **A5：文档可禁用**：将某个 doc_id 标记为 disabled 后，该文档不再出现在 `retrieved_contexts` 中；取消禁用后可恢复命中。
