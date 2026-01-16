# RD-Agent × AIstock 顶层架构补充说明：因子共享包与成果同步 API 设计（Phase 2/3 增量）

> 本补充文档基于《2025-12-23_RD-Agent_AIstock_TopLevel_Architecture_Design.md》，用于细化“因子/策略/模型成果如何从 RD-Agent 侧交付给 AIstock 侧”的方案，特别是：
> - 因子实现源码的组织与共享方式（因子共享包）；
> - RD-Agent 只读成果 API 的角色与边界；
> - Phase 2 / Phase 3 在“成果导出”和“在 AIstock 侧执行”上的责任划分。
> 本文件不改变原有阶段划分和目标，只做增量约束与实现细节补充。

---

## 1. 因子实现的资产化方式：因子共享包（Factor Package）

### 1.1 目标与约束

- **目标**：
  - 将 RD-Agent 侧可复用的因子实现（包括部分 Alpha 因子与演进因子）整理为**结构化、可版本化的代码资产**；
  - 既支持 RD-Agent 在研发/回测中直接使用，也便于 AIstock 在 Phase 3 中迁移到自有执行引擎；
  - 全过程**不依赖 GitHub 等外部托管服务**，仅依赖本机文件系统与本地 API。

- **约束**：
  - 实盘/模拟盘执行链路始终在 AIstock 侧，**不直接调用 RD-Agent 进程**；
  - 因子共享包作为“参考实现库”，Phase 2 中只用于“查看与对齐”，**Phase 3 才在 AIstock 执行引擎中落地等价实现**；
  - AIstock 可以为适配自有数据服务重写实现，但**须在标准测试集上与 RD-Agent 参考实现数值一致**，保证逻辑含义不变。

### 1.2 因子共享包的组织形式

- 在本地文件系统中维护一个独立的 Python 包目录，例如：

  ```text
  F:\Dev\rd-factors-lib\
    rd_factors_lib\
      __init__.py
      alpha158.py        # Alpha158 相关（如有需要）
      momentum.py        # 一批动量类因子
      volume.py          # 一批量价/流动性因子
      cross_section.py   # 一批截面类因子
      generated.py       # RD-Agent 自动演进的因子实现
      VERSION            # 内部版本号标记，如 "1.0.7"
  ```

- **单个因子**在包中表现为模块内的一个函数或类，而不是独立包/独立顶层目录，例如：

  ```python
  # rd_factors_lib/momentum.py
  def factor_resi5(df):
      ...
  ```

- RD-Agent 与 AIstock **都从同一本地路径安装该包**，例如：

  ```bash
  pip install -e F:\Dev\rd-factors-lib
  ```

- 版本管理：
  - 包内部维护一个版本号文件（如 `VERSION`，或在 `__init__.py` 中定义 `__version__`）；
  - 该版本号会在 RD-Agent 的因子元数据中以 `impl_version` 字段返回。

### 1.3 因子与共享包之间的映射

- 在 `factor_meta.json` / `factor_catalog.json` 以及只读成果 API 的返回中，为每个因子增加实现指针字段：

  - `impl_module: str`  
    - 因子实现所在的 Python 模块路径，例如：`"rd_factors_lib.momentum"`；
  - `impl_func: str`  
    - 因子函数名，例如：`"factor_resi5"`；
  - `impl_version: str`  
    - 当前共享包版本号，例如：`"1.0.7"`。

- Alpha158 因子：
  - 以 `expression` 字段给出 Qlib 表达式；
  - 如需在共享包中提供 Alpha158 的参考实现，可在 `alpha158.py` 中提供相应函数，并同样在元数据中给出 `impl_module/impl_func`；
  - **RD-Agent 不强制在自身重写 Alpha158 实现**，可仅依赖 Qlib + `alpha158_meta.json`；AIstock 在 Phase 3 中可结合 Qlib 做解析与验证。

- RD-Agent 自演进因子：
  - 当某个演进因子被标记为“成功/可参考”时，RD-Agent 会在演进流程的收尾阶段：
    - 将该因子的实现自动写入/更新到 `rd_factors_lib/generated.py` 中，形成稳定的函数接口；
    - 更新共享包的版本号；
    - 在 `factor_meta` 中记录上述 `impl_module/impl_func/impl_version` 信息。

---

## 2. RD-Agent 只读成果 API：角色与边界

### 2.1 API 的定位

- 只读成果 API 的定位是**“研发成果同步渠道”**，而非交易接口：
  - 面向 AIstock 侧：用来拉取 RD-Agent 的任务、循环、因子、策略、模型训练结果等结构化数据；
  - **不**返回实时信号/订单，**不**参与模拟盘/实盘执行链路；
  - 故障只影响“新成果同步的时效”，**不影响交易安全性**。

### 2.2 API 提供的核心资源（示意）

> 具体路径与参数在 Phase 2 详细设计中给出，这里只定义资源范畴与职责。

- **Catalog 视图**：
  - `GET /catalog/factors` → 逻辑等价于当前 `factor_catalog.json`；
  - `GET /catalog/strategies` → 逻辑等价于 `strategy_catalog.json`；
  - `GET /catalog/loops` → 逻辑等价于 `loop_catalog.json`。

- **实验与 artifacts 视图**：
  - `GET /task_runs` / `GET /loops` / `GET /workspaces`：封装 registry.sqlite 中的任务与循环信息；
  - `GET /loops/{id}/artifacts`：列出指定 loop 的关键 artifacts 路径与摘要（factor_meta / factor_perf / feedback / 图表等）。

- **因子实现指针与 Alpha158 元信息**：
  - `GET /factors/{name}`：返回单个因子在 `factor_meta` 中的完整记录，包含：
    - 名称、来源、描述、标签、表现指标；
    - `impl_module` / `impl_func` / `impl_version` 等实现指针（如存在）；
  - `GET /alpha158/meta`：返回通过 `export_alpha158_meta.py` 生成的 Alpha158 全量元信息。

- **可选：源码 bundle 归档接口**：
  - 如需对某个版本的因子共享包做归档，可提供：
    - `GET /factor_package/bundle?version=1.0.7` → 返回一个 tar/zip 包；
  - 用于人工/离线备份，不作为日常同步的主通道。

### 2.3 通信与环境假设

- API 服务部署在 RD-Agent 所在环境（例如 WSL 内部），监听本机端口；
- AIstock 通过本机 HTTP（如 `http://localhost:xxxx`）调用该服务：
  - 实现“无文件系统挂载依赖”的成果同步方式；
  - 所有通信限制在本机或受控内网中，不涉及外网。

---

## 3. Phase 2 / Phase 3 在执行责任上的划分

### 3.1 Phase 2：成果导出与可视化（不负责执行）

- **RD-Agent 侧责任（补充）**：
  - 通过统一写入函数与 backfill 工具，保证：
    - 所有 `has_result=1` 的 loop×workspace 均产出完整的 Phase 2 artifacts（factor_meta / factor_perf / feedback / 图表等）；
    - registry.sqlite 中 artifacts / artifact_files 记录完备；
  - 除已有 JSON/Catalog 外，新增：
    - 因子实现指针元数据：`impl_module` / `impl_func` / `impl_version`；
    - 模型训练 run 的元信息（模型类型、结构、训练数据区间等）；
  - 提供只读成果 API，封装 registry 与 catalog 的读取；
  - 在演进流程中自动维护因子共享包（写入 `rd_factors_lib/generated.py` 并 bump 版本）。

- **AIstock 侧责任（补充）**：
  - 实现与 RD-Agent API 的交互客户端：
    - 定期/按需拉取因子、策略、loop、Alpha158 以及因子实现指针与模型训练元数据；
  - 将上述数据导入 AIstock 本地数据库与文件系统：
    - 因子库：包含元信息 + 表现指标 + 实现指针；
    - 策略库：包含配置与模型信息；
    - 实验库：包含 loop 指标、反馈、关键 artifacts；
  - 通过 UI 提供：
    - RD-Agent 演进因子与策略成果的浏览/筛选/标记功能；
    - Alpha158 全量信息的查看；
  - **不在 Phase 2 中直接在线执行 RD-Agent 模型/策略/因子**，仅作为研究与决策参考。

### 3.2 Phase 3：在 AIstock 执行栈中运行等价策略/因子/模型

- **RD-Agent 侧（保持研发工厂角色）**：
  - 继续通过 ExperimentRequest 等机制，接受 AIstock 提交的策略/因子组合与约束；
  - 完成离线演进、回测与成果总结；
  - 保持因子共享包与只读成果 API 的迭代。

- **AIstock 侧（新增执行责任）**：
  - 基于 Phase 2 引入的因子共享包与元数据：
    - 在 AIstock 自有数据服务之上，实现等价的因子计算逻辑；
    - 在本地策略/模型执行引擎中实现与 RD-Agent 使用的策略/模型配置等价的执行路径；
  - 对于被选中的“可上线组合”（某因子集 + 策略 + 模型）：
    - 使用与 RD-Agent 相同的模型结构/超参，在 AIstock 本地重新训练模型；
    - 或根据标准模型导出规范加载 RD-Agent 导出的模型；
    - 用统一的回归测试框架验证：
      - 同一历史数据集上，AIstock 执行引擎的输出与 RD-Agent 回测结果在可接受误差内一致；
  - 在此基础上，将等价实现用于模拟盘与实盘，并通过 LiveFeedback 等机制将真实表现回传给 RD-Agent 作为后续演进的输入。

---

## 4. 原则重申

1. **研发与交易解耦**：
   - RD-Agent 只通过成果 API 和因子共享包交付“可复现的研发成果”；
   - AIstock 负责一切在线执行逻辑，即使内部使用 Qlib 作为解析/验证工具，也不通过 RD-Agent 进程参与实盘链路。

2. **先可见、后可执行**：
   - Phase 2 的责任是“看得见、拿得到、结构化存得下”；
   - Phase 3 才承担“在 AIstock 执行栈中能跑起来，并与 RD-Agent 成果对齐”。

3. **版本化与可回溯**：
   - 因子共享包版本与因子元数据紧密绑定，任何回测结果都可以追溯到具体实现版本；
   - 模型与策略配置同样通过版本化与元数据记录，保证整个研发—执行链条可审计、可重放。
