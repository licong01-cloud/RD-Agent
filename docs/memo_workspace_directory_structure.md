# 备忘录：Workspace 目录结构改造方案与工作量评估

## 1. 背景与现状

### 1.1 现状：Workspace 目录无层级语义、与 loop/任务对应关系弱

当前 File-based workspace（`FBWorkspace`）在初始化时直接创建目录：

- 代码位置：`rdagent/core/experiment.py::FBWorkspace.__init__`
- 当前规则：
  - `workspace_path = RD_AGENT_SETTINGS.workspace_path / uuid.uuid4().hex`
  - 默认根目录：`git_ignore_folder/RD-Agent_workspace/`

因此在一次运行中：

- 同一 loop 内多个因子/策略会创建多个不同 uuid 的 workspace
- 同一因子在同一 loop 内反复演进时，也可能产生多个 workspace 目录（每次新实例化即新 uuid）
- 目录名无法体现：run、loop、step、task_type（factor/model/quant）、task_name、attempt 等信息

对应关系目前主要依赖日志：

- 日志根目录：`LOG_SETTINGS.trace_path`（默认 `./log/<UTC时间戳>/`）
- step tag：`Loop_{li}.{step_name}`
- coder/runner 输出对象里包含 `workspace_path`


## 2. 目标（需求拆解）

目标目录结构：

- 一次任务（一次 run）使用一个总目录
- 该总目录下按 loop 生成子目录
- 每个 loop 下按因子/策略生成子目录
- 命名规则包含更多语义字段（便于人工排查与自动化治理）

建议将需求拆解为两类：

- **分组（grouping）**：run/loop/task 目录层级
- **唯一性（uniqueness）**：避免并发/多次执行覆盖（仍需 uuid 或 attempt 序号）


## 3. 方案选型（从低风险到完整实现）

### 3.1 方案 A（低风险）：仅做“按 run 分组”

#### 设计
让一次 run 的所有 workspace 进入同一目录，例如：

```
RD-Agent_workspace/
  <trace_ts>/
    <uuid>/
    <uuid>/
```

其中：

- `<trace_ts>` 可取 `Path(LOG_SETTINGS.trace_path).name`，与日志目录一一对应
- `<uuid>` 仍保留，保证并发安全与唯一性

#### 修改点
- 修改 `FBWorkspace.__init__` 中 `self.workspace_path` 的生成逻辑（单点改动）
  - 引入 `LOG_SETTINGS.trace_path`（`rdagent/log/conf.py`）
  - 目录形如：`RD_AGENT_SETTINGS.workspace_path / trace_ts / uuid`

#### 优点
- 修改面小
- 与 log 目录自然对齐（排查成本显著下降）
- 不依赖 loop_id 的传递

#### 风险/兼容
- 极低。仍保持 uuid workspace，不改变执行与缓存机制。

#### 工作量评估
- 代码修改：1 个文件（`rdagent/core/experiment.py`）
- 回归验证：factor/model/quant 基本场景各跑一次
- 预计：0.5~1 人日


### 3.2 方案 B（中风险，推荐演进）：run 分组 + task_type/task_name 语义命名（不含 loop）

#### 设计

```
RD-Agent_workspace/
  <trace_ts>/
    factor__<factor_name>/
      <uuid>/
    model__<model_name>/
      <uuid>/
```

要点：

- 目录名加入 `task_type`（factor/model/quant）与 `task_name`
- 最终仍保留 `<uuid>`，避免同名冲突与并发覆盖

#### 修改点
- 需要让 `FBWorkspace` 拿到 `target_task`（当前已存在）并提取 name
- 增加 `sanitize`（替换非法字符、限制长度）

#### 优点
- 人可读性更强
- 仍不依赖 loop_id

#### 风险/兼容
- 低到中。需要保证跨平台路径安全、长度限制（Windows 路径长度）

#### 工作量评估
- 代码修改：1~2 个文件
- 预计：1~2 人日


### 3.3 方案 C（中等偏大）：完整实现 run/loop/task 分层（满足“每个 loop 子目录”）

#### 设计（推荐结构）

```
RD-Agent_workspace/
  <trace_ts>__<scenario>/
    loop_000/
      factor__<factor_name>/
        attempt_000__<uuid>/
      factor__<factor_name2>/
        attempt_000__<uuid>/
    loop_001/
      ...
```

关键点：

- **attempt 层必须存在**（或等价的 uuid 层）
  - 因为同因子在同 loop 内反复演进会出现多次实现
  - 并行执行时也会同时写 workspace

#### 难点与原因
当前 workspace 创建发生在 `FBWorkspace.__init__`：

- 此处默认 **不知道 loop_id**
- loop_id 只在 `LoopBase._run_step(li, ...)` 中明确

因此必须把 `loop_id` 从 workflow 层传递到 workspace 构造处。

#### 两种实现路径

- **C1（推荐）显式传参**：
  - 在创建 workspace 的逻辑处（例如 coder.develop 构造 `FactorFBWorkspace(...)`）
  - 新增参数 `loop_id` / `run_id` / `task_group_id`
  - workspace_path 由 factory 统一生成

- **C2（不推荐）隐式上下文（ContextVar）**：
  - 在 `_run_step` 进入时设置 `ContextVar(loop_id)`
  - workspace 构造时读取
  - 并行/子进程情况下需谨慎验证

#### 影响范围（需要排查）
- factor/model/quant/general_model 等多类场景都可能创建 FBWorkspace
- 恢复 session（`LoopBase.load`）对 workspace_path 的复用与兼容
- 执行锁与输出文件（`execution.lock` / `result.h5`）覆盖风险
- pickle cache（`cache_with_pickle`）是否把 workspace_path 作为缓存键的一部分（当前不是，但需要确认调用路径）

#### 工作量评估
- 代码修改：
  - 需要查清所有 workspace 的创建点（通常不止 1 处）
  - 引入统一的 workspace_path_factory 或在各创建点改签名
- 测试/回归：
  - 至少覆盖 factor/model/quant 三条主链
  - 并行模式（`step_semaphore>1`）必须测
  - session resume/checkout 必须测

预计：
- **最小可用（只支持 factor loop）**：3~5 人日
- **全链路一致（factor+model+quant+others）**：5~10 人日


## 4. 并发与“目录复用”的结论建议

### 4.1 是否建议“同因子同 loop 复用同一目录”？

不建议（风险高）：

- 会引入执行覆盖：`result.h5` 被后一次覆盖
- 会引入并发竞争：`execution.lock` 互相阻塞/死锁风险
- 回溯困难：无法保留历史 attempt 的中间产物

推荐：

- “逻辑分层目录 + attempt 子目录（uuid/序号）”
- 让每次实现都有独立目录，保留可追溯性


## 5. 推荐落地路线（分阶段）

- **阶段 1（建议立即做）**：方案 A（按 run 分组）
- **阶段 2（增强可读性）**：方案 B（加入 task_type/task_name）
- **阶段 3（满足 loop 分层）**：方案 C（打通 loop_id 传递 + attempt 层）

每阶段都应保持：

- 旧目录结构可兼容读取（不要求迁移历史 workspace）
- 不改变 factor 执行输出规范（`result.h5`）


## 6. 相关代码索引

- Workspace 根目录配置：`rdagent/core/conf.py::RDAgentSettings.workspace_path`
- Workspace 目录创建：`rdagent/core/experiment.py::FBWorkspace.__init__`
- Loop/step 执行与 session dump：`rdagent/utils/workflow/loop.py::LoopBase._run_step` / `LoopBase.dump`
- 日志目录结构：`rdagent/log/storage.py::FileStorage.log` / `rdagent/log/logger.py::RDAgentLog.log_object`
