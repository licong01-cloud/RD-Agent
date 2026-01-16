# 磁盘迁移与 AIstock 交互方案备忘录

## 背景与目标

- RD-Agent（本仓库）将迁移到新硬盘，并需要长期稳定的目录结构。
- AIstock 的 Windsurf/IDE 需要频繁访问 RD-Agent 的“成果产物”，未来可能还需要访问 RD-Agent 源代码进行检索/对照。
- 短期不修改 RD-Agent/AIstock 程序代码来做深度集成，希望先通过“共享目录 + 约定文件格式 + 工作区配置”实现协作。
- 中长期希望支持：

- AIstock 读取 RD-Agent 产出（报告、回测结果、因子/模型配置等）用于展示与监控。
- AIstock 将自然语言策略描述转为结构化任务，投递给 RD-Agent 做演进（先离线投递文件，后续再升级为 API/队列）。
- AIstock 外部行情/更多数据集可供 RD-Agent 使用。
- RD-Agent 产出可用于模拟盘/实盘（策略包/参数包可追溯可回滚）。

## 总体原则（长期可维护）

- 代码仓库逻辑上解耦：AIstock 与 RD-Agent 分开维护版本、依赖、发布节奏。
- 数据与产物以共享路径联动：通过稳定的共享目录结构与清晰的“产物契约”实现互通。
- IDE/LLM 工具访问通过“多根工作区（multi-root workspace）/Windows junction/符号链接”解决，而不是把两个仓库强行合并为单仓库。
- 产物可追溯：每次 run/每个策略包，都要可定位输入数据版本、配置、代码版本（commit）、回测假设。

## 推荐目录布局（迁移到新硬盘后）

以下以 `D:` 为例（盘符可替换），重点是“代码并列 + 数据与产物独立域”。

- `D:\repos\AIstock\`
- `D:\repos\RD-Agent\`

共享域（不属于任何一个仓库）：

- `D:\quant_data\`：数据域（AIstock 主要写入，RD-Agent 主要读取）
- `D:\quant_artifacts\`：产物域（RD-Agent 主要写入，AIstock 主要读取）

建议细分：

- `D:\quant_data\raw\`：外部/下载原始数据
- `D:\quant_data\curated\`：清洗/对齐后的标准化数据（可导出为 Qlib/provider 或其他形式）
- `D:\quant_data\qlib\`：Qlib provider 标准目录（如果采用 Qlib 格式）
- `D:\quant_data\versions\manifest.json`：数据版本/字段变更记录

- `D:\quant_artifacts\rdagent\runs\<run_id>\...`：每次 RD-Agent 研究/演进产物
- `D:\quant_artifacts\strategies\<strategy_id>\...`：可投产“策略包”（用于模拟盘/实盘）

## 短期（不改代码）实现两项目交互的方案

### 1) AIstock 读取 RD-Agent 产物：统一导出/同步目录

目标：AIstock 不需要理解 RD-Agent 内部 log/目录结构，只需要读取稳定的导出目录。

建议约定：

- RD-Agent 每次运行结束后，将关键文件“复制/同步”到：

- `D:\quant_artifacts\rdagent\runs\<run_id>\`

关键文件最小集合（建议逐步补齐）：

- `qlib_res.csv`（或等价回测指标文件）
- `factor_list.json`（本次使用/保留的因子列表与表达式摘要）
- `model_config.json`、`dataset_config.json`（训练/数据配置摘要）
- `run_meta.json`（run_id、时间范围 segments、market/universe、benchmark、成本/滑点关键参数、git commit、数据版本号）
- `error_summary.json`（如有失败/auto-repair，记录失败类型与修复结果）

短期实现方式：

- 用 PowerShell/bat 同步脚本从 RD-Agent 的 `log/`、`runner_results/` 等目录抓取并复制到 `quant_artifacts`。
- 在 AIstock 中只读取 `quant_artifacts`，不要直接依赖 RD-Agent 内部目录布局。

### 2) AIstock 将自然语言策略投递给 RD-Agent：inbox 文件投递（离线）

目标：先把“任务协议”固化下来，即使短期 RD-Agent 不能自动读取，也能人工/脚本拼接进入 RD-Agent 的运行输入。

建议约定：

- AIstock 写入：

- `D:\quant_artifacts\rdagent\inbox\<task_id>.json`

建议字段（最小可用）：

- `task_id`
- `nl_strategy`：自然语言策略描述
- `objective`：优化目标（IC/RankIC/return/drawdown 等）
- `market`：all/csi300 等
- `segments`：train/valid/test 起止日期
- `data_version`：指向 `quant_data\versions` 的版本号
- `field_whitelist_ref`：字段白名单/映射文件引用（避免字段幻觉）

RD-Agent 执行者（人工/脚本）读取该 JSON，并将关键信息填入对应运行参数/配置。

## 共享 RAG（AIstock 与 RD-Agent 共用）对目录与迁移方案的影响

目标：AIstock 与 RD-Agent 都需要大量调用 LLM，且希望基于同一套“事实知识库/规则库/字段口径”减少冲突与幻觉。这里的关键是把 RAG 拆成两层：

- 知识库内容（KB）：markdown/JSON 等可审计的“真相来源”。
- 索引与检索层（Index/Retriever）：向量库/倒排索引/检索服务，负责加速与召回。

建议把 KB 与 Index 分离存储与管理：KB 更像代码与文档（可 review、可版本化），Index 更像缓存（可重建）。

### 共享 KB 的推荐放置位置与版本化

推荐把共享 KB 放在共享域而不是某个仓库内部路径，避免“哪个仓库是主仓”的争议，并方便两端读取：

- `D:\quant_artifacts\rag\kb\`：共享 KB 根目录
- `D:\quant_artifacts\rag\kb_versions\`：版本化快照（可选）

实践建议：

- KB 内容用 Git 管理，但仓库可以独立于 AIstock/RD-Agent：例如一个单独的 `quant-kb` 仓库；迁移到新盘时仍放在 `D:\repos\quant-kb\`。
- AIstock 与 RD-Agent 运行时统一读取 `D:\quant_artifacts\rag\kb\current`（或 junction 指向当前版本）。
- 重要口径变更（字段名、回测假设、策略约束）必须走 review，并更新版本号。

### 索引/向量库（Index）应该放哪里

向量索引通常体积大、可重建，建议放共享域并按“KB 版本号”隔离：

- `D:\quant_artifacts\rag\index\<kb_version>\...`

这样可以做到：

- KB 升级时不影响旧 run 的可复现（旧 run 仍可用旧索引）。
- 索引损坏/迁移时可直接删掉重建，不影响 KB 资产。

### 三种部署方式对比（从“零改代码”到“服务化”）

#### 方式 A：共享文件夹 KB（零改代码起步）

做法：

- 两端（人/脚本）在调用 LLM 时，手动或半自动把 `rag/kb` 中相关文件拼接进 prompt。

优点：

- 立刻可用，工程成本最低。
- 对工具链（Windsurf/IDE）非常友好：就是读文件。

缺点：

- 召回不自动，依赖人工选择或固定“检索包”。

#### 方式 B：共享 Index + 轻量检索脚本（小改动、但不一定要改主程序）

做法：

- 新增一个独立的小工具（可在 AIstock 侧或共享脚本侧），输入 query 输出若干 KB 片段，再由两端把片段注入 prompt。

优点：

- 召回自动化、内容一致，减少“两个系统各说各话”。

缺点：

- 需要维护索引构建与检索工具（但不必立刻改 RD-Agent 主流程）。

#### 方式 C：统一 RAG 服务（长期推荐）

做法：

- 以一个独立进程/服务提供：

- `/retrieve?kb_version=...&query=...` -> 返回片段
- `/ingest` -> 更新 KB/索引（可选）

优点：

- 两端调用方式一致，便于做：缓存、鉴权、审计、成本控制。

缺点：

- 需要服务治理（部署、日志、稳定性）。

### 两端大量调用 LLM 时，建议提前统一的“治理能力”

即使短期不做服务化，也建议在共享域先落地这些约定，后续可以平滑升级为服务：

- `D:\quant_artifacts\llm_cache\`：Prompt/Response 缓存（按模型与 prompt hash 分桶）
- `D:\quant_artifacts\llm_audit\`：审计日志（时间、调用方、token、费用、失败原因）
- `D:\quant_artifacts\rag\metrics\`：RAG 命中率、片段长度、引用来源统计（可选）

注意：

- 缓存与审计是“跨系统协同”的关键，否则两个系统会重复烧 token 且难以对齐事实。
- 审计里需要记录 `kb_version` 与 `index_version`，保证可复现与可追责。

### 补充约束：AIstock 未来会引入多种策略来源（不止 RD-Agent）

如果 AIstock 未来不仅消费 RD-Agent 演进出来的策略，还会接入其他策略（外部策略、人工策略、其他研究引擎），并希望通过 RAG 完成“策略描述 -> 策略代码化/结构化”的工作，则共享 RAG 的定位会从“RD-Agent 辅助能力”升级为“AIstock 的通用策略中台能力”。这会带来以下影响：

- 共享 KB 的内容应当分层：

- 通用层：策略语言/术语、交易约束、回测假设、风险约束、数据字段口径等（与具体引擎无关）。
- 引擎层：RD-Agent/Qlib 特有的配置模板、字段映射、因子实现契约等（与具体引擎强相关）。

- KB 的“主归属”应更偏向独立仓库或共享域：避免把通用策略知识绑定在 RD-Agent 仓库里，导致 AIstock 接入新引擎时必须跟随 RD-Agent 的发布节奏。

对应到前文 A/B 的选择：

- A（KB 起点在 RD-Agent 仓库）更适合作为短期过渡：可以快速积累 RD-Agent 相关的高价值知识卡片。
- B（独立 `quant-kb`）在该约束下更适合作为长期目标：把“通用策略知识”放在独立 KB 中，并通过版本化与 review 支撑多引擎复用。

实践建议（不改主程序也可先落地）：

- 共享 KB 目录内建议按层级组织：

- `kb/core/`：通用策略知识
- `kb/engines/rdagent_qlib/`：RD-Agent/Qlib 专用
- `kb/engines/other_engine_x/`：其他引擎专用（预留）

- AIstock 的“策略代码化”建议产出一个中间表示（IR）：例如 `strategy_ir.json`，由不同引擎的适配器（adapter）负责转换为可执行配置/代码。这样 RAG 的核心知识可稳定复用，新引擎只需新增 adapter 与少量引擎专用 KB。

### KB 是否共同贡献？向量库/索引是否要分开？（推荐的多租户模型）

结合“AIstock 多策略来源 + RD-Agent 研究引擎 + 双端高频 LLM 调用”的场景，推荐采用：

- KB：共同贡献，但分层治理（通用层共享 + 引擎层隔离）。
- Index：共享底层基础设施，但逻辑上分租户/分命名空间（可同一个向量库实例、不同 collection/namespace；也可不同物理库）。

原因与收益：

- 共同贡献 KB 可以让两端对“字段口径/交易假设/风险约束/术语”保持一致，减少跨系统冲突与幻觉。
- 引擎层隔离可以避免 RD-Agent/Qlib 的细节污染 AIstock 的通用策略代码化流程，也避免其他策略引擎的知识误导 RD-Agent。
- Index 分命名空间可以让不同调用方按需召回（不同 prompt、不同任务类型召回不同知识集），同时共享构建与运维能力（缓存、审计、版本化）。

建议的 KB 组织（允许重叠但不完全相同）：

- `kb/core/`：通用知识（AIstock 主导，RD-Agent 可提 PR）
- `kb/engines/rdagent_qlib/`：RD-Agent 专用知识（RD-Agent 主导，AIstock 可提 PR）
- `kb/engines/<other_engine>/`：其他引擎专用知识

建议的 Index 组织（按用途隔离召回面）：

- `index/core/<kb_version>/`：只索引 `kb/core`
- `index/rdagent_qlib/<kb_version>/`：索引 `kb/core + kb/engines/rdagent_qlib`
- `index/aistock_codegen/<kb_version>/`：索引 `kb/core +（AIstock 策略代码化相关专用 KB）`

这样可以实现：

- RD-Agent 做因子/回测/修复时，默认用 `rdagent_qlib` 的索引，召回更精准。
- AIstock 做“策略描述 -> 代码化/结构化”时，默认用 `aistock_codegen` 的索引，避免被 RD-Agent 的实现细节干扰。
- 通用约束始终来自 `core`，保证跨系统一致。

治理建议：

- KB 允许重叠，但同一条“硬事实”（字段含义、交易成本模型、benchmark、时间段口径）必须在 `kb/core` 有唯一表述；引擎层只能补充“如何映射/如何落地”。
- 所有 LLM 调用（尤其是产出策略 IR/代码）都应记录 `kb_version` 与所用 `index_namespace`，便于复现与排障。

## Windsurf/IDE 需要频繁访问 RD-Agent 成果甚至源码：更好的解决方案

“完全隔离目录”指的是“仓库分开”，并不意味着 Windsurf 不能访问另一个仓库。更推荐通过以下方式解决访问问题。

### 方案 A（推荐）：多根工作区（Multi-root Workspace）

在 VSCode/Windsurf 中把两个仓库同时加入一个 workspace：

- Root1：AIstock
- Root2：RD-Agent

优点：

- 不需要改变 Git 结构
- Windsurf 可以同时检索、读取两个项目文件
- 路径清晰，不需要在磁盘上“硬合并”

注意：

- 多根工作区下，配置文件（比如 tasks、python interpreter）要明确绑定到各自的 root。

### 方案 B（推荐）：Windows junction/symlink 让“看起来在一起”，但仓库仍独立

如果 Windsurf 或某些工具强依赖“同一 root 下的相对路径”，可以在 AIstock 目录下创建一个指向 RD-Agent 的目录链接：

- `D:\repos\AIstock\external\RD-Agent` -> `D:\repos\RD-Agent`

优点：

- 目录层面“近似放在 AIstock 子目录”，但实际仍是独立仓库
- 不影响各自 Git

风险与注意：

- 需要注意避免在 AIstock 仓库里把链接目标当作普通目录误提交。
- 部分工具对 symlink/junction 的行为不同（建议优先 junction）。

### 方案 C（备选）：Git submodule（不太推荐）

优点：

- 形式上变成 AIstock 的子目录，且版本可固定

缺点：

- submodule 对日常开发不友好，容易踩坑
- 多人协作时经常出现“忘记更新 submodule”问题

### 方案 D（不建议）：强行 monorepo 合并

除非你们已经确定长期以一个大仓库统一版本管理，否则不建议。

## 面向模拟盘/实盘的产物设计建议（现在就该按这个方向落地）

建议在 `D:\quant_artifacts\strategies\<strategy_id>` 下形成“策略包”最小结构：

- `strategy.json`：策略描述（可读）+ 参数（可执行）
- `features/`：因子清单与表达式摘要（或引用 run_id 产物）
- `model/`：模型类型与关键参数摘要（或引用 run_id 产物）
- `risk/`：风控参数（仓位、行业暴露、最大回撤阈值等）
- `backtest/`：最后一次回测指标与关键图表/CSV
- `provenance.json`：来源（RD-Agent run_id、git commit、数据版本、时间段、benchmark、成本参数）

这样 AIstock 交易/模拟模块只需要消费策略包，不必依赖 RD-Agent 内部实现。

## 决策建议（推荐组合）

- 代码仓库：RD-Agent 与 AIstock 分开（并列），避免依赖/版本绑定。
- 工具访问：优先使用 multi-root workspace；若工具必须单 root，再加 junction/symlink。
- 交互方式：短期用 `quant_artifacts` 的导出目录 + inbox 任务投递；中长期再升级为 API/队列。

## 待确认项（用于最终落地）

- 新硬盘盘符与计划路径（是否用 `D:\repos` / `D:\quant_data` / `D:\quant_artifacts`）。
- Windsurf 当前是以哪个目录作为 root（AIstock 根目录还是某个上层目录）。
- AIstock 希望读取的“成果文件类型”优先级：json/csv/html/markdown。
- 数据集计划采用 Qlib provider 格式还是 HDF5/CSV（以及数据版本管理方式）。
