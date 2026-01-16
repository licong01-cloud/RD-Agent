# 从Workspace恢复任务执行方案分析

## 实施日期
2026-01-14

## 1. 问题提出

用户提出了一个关键问题：

> "目前的Rdagent是否可以从上一个任务继续执行？第一个任务选择factor类型，第二个选择模型，从第一个任务最后的workspace开始执行是否可行？"

### 1.1 核心问题

**问题1：是否可以从上一个任务继续执行？**
- 第一个任务选择factor类型
- 第二个任务选择模型
- 从第一个任务最后的workspace开始执行是否可行？

**问题2：workspace中保存了什么信息？**
- workspace中是否包含了SOTA因子？
- workspace中是否包含了实验历史？

**问题3：是否可以从workspace恢复任务？**
- 第二个任务能否从第一个任务的workspace继续执行？
- 是否需要修改代码？

## 2. RD-Agent Workspace机制分析

### 2.1 Workspace的基本结构

从代码分析来看，RD-Agent的Workspace机制如下：

**core/experiment.py (第139-160行)**：
```python
class FBWorkspace(Workspace):
    """
    File-based task workspace

    The implemented task will be a folder which contains related elements.
    - Data
    - Code Workspace
    - Output
        - After execution, it will generate the final output as file.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_dict: dict[str, Any] = (
            {}
        )  # The code injected into the folder, store them in the variable to reproduce the former result
        self.workspace_path: Path = RD_AGENT_SETTINGS.workspace_path / uuid.uuid4().hex
        self.ws_ckp: bytes | None = None  # In-memory checkpoint data created by ``create_ws_ckp``.
        self.change_summary: str | None = None  # The change from the previous version of workspace
```

**关键发现**：
- Workspace是文件系统上的一个文件夹
- `workspace_path`是workspace的路径
- `file_dict`保存了注入到workspace的代码文件
- 每个Workspace都有一个唯一的UUID

### 2.2 Workspace中保存的信息

从代码分析来看，Workspace中保存了以下信息：

**1. 代码文件**：
- `model.py`：模型代码
- `factor.py`：因子代码
- `read_exp_res.py`：读取实验结果的脚本
- `custom_strategy.py`：自定义策略
- 配置文件（YAML格式）

**2. 数据文件**：
- `combined_factors_df.parquet`：合并后的因子数据（包含SOTA因子和新因子）
- `ret.pkl`：回测结果
- `qlib_res.csv`：Qlib结果
- `result.h5`：实验结果

**3. 输出文件**：
- 日志文件
- 执行结果
- 性能指标

**关键发现**：
- ✅ Workspace中保存了`combined_factors_df.parquet`文件
- ✅ 这个文件包含了SOTA因子和新因子的合并结果
- ❌ Workspace中没有保存实验历史（`trace.hist`）
- ❌ Workspace中没有保存`based_experiments`

### 2.3 Workspace的恢复机制

从代码分析来看，RD-Agent提供了以下Workspace恢复机制：

**1. 从文件夹恢复**：
```python
def inject_code_from_folder(self, folder_path: Path) -> None:
    """
    Load the workspace from the folder
    """
    for file_path in folder_path.rglob("*"):
        if file_path.suffix in (".py", ".yaml", ".md"):
            relative_path = file_path.relative_to(folder_path)
            self.inject_files(**{str(relative_path): file_path.read_text()})
```

**2. 从file_dict恢复**：
```python
def inject_code_from_file_dict(self, workspace: FBWorkspace) -> None:
    """
    Load the workspace from the file_dict
    """
    for name, code in workspace.file_dict.items():
        self.inject_files(**{name: code})
```

**3. 创建和恢复checkpoint**：
```python
def create_ws_ckp(self) -> None:
    """
    Zip the contents of ``workspace_path`` and persist the archive on
    ``self.ws_ckp`` for later restoration via :py:meth:`recover_ws_ckp``.
    """
    # ... 实现细节 ...

def recover_ws_ckp(self) -> None:
    """
    Restore the workspace directory from the in-memory checkpoint created by
    :py:meth:`create_ws_ckp``.
    """
    # ... 实现细节 ...
```

**关键发现**：
- ✅ Workspace支持从文件夹恢复代码文件
- ✅ Workspace支持创建和恢复checkpoint
- ❌ Workspace恢复机制只恢复代码文件，不恢复实验历史

## 3. 从Workspace恢复任务的可行性分析

### 3.1 RD-Agent的任务执行机制

从代码分析来看，RD-Agent的任务执行机制如下：

**1. 基于Trace的执行**：
- RD-Agent使用`Trace`对象来维护实验历史
- `trace.hist`包含了所有的实验历史
- 每个实验执行后，会被添加到`trace.hist`中

**2. based_experiments机制**：
- `based_experiments`是从`trace.hist`中提取的
- `based_experiments`包含了之前的所有实验
- SOTA因子是从`based_experiments`中提取的

**3. 关键依赖**：
- 任务执行依赖于`trace.hist`
- SOTA因子的使用依赖于`based_experiments`
- `based_experiments`依赖于`trace.hist`

### 3.2 从Workspace恢复的问题

**问题1：无法恢复trace.hist**

如果从第一个任务的workspace继续执行：
- Workspace中只保存了代码文件和数据文件
- Workspace中没有保存`trace.hist`
- 第二个任务无法访问第一个任务的实验历史

**问题2：无法恢复based_experiments**

如果从第一个任务的workspace继续执行：
- `based_experiments`是从`trace.hist`中提取的
- 如果`trace.hist`是空的，则`based_experiments`也是空的
- 第二个任务无法使用第一个任务的SOTA因子

**问题3：无法恢复实验链**

如果从第一个任务的workspace继续执行：
- 实验链是基于`trace.hist`构建的
- 如果`trace.hist`是空的，则实验链断裂
- 第二个任务无法继续第一个任务的实验

### 3.3 从Workspace恢复的解决方案

#### 方案1：修改代码，支持从workspace恢复trace.hist

**实现步骤**：
1. 在第一个任务执行完后，保存`trace.hist`到workspace
2. 在第二个任务执行前，从workspace加载`trace.hist`
3. 使用加载的`trace.hist`执行第二个任务

**优点**：
- ✅ 可以实现从workspace恢复任务
- ✅ 第二个任务可以访问第一个任务的实验历史
- ✅ 第二个任务可以访问第一个任务的SOTA因子

**缺点**：
- ❌ 需要修改代码
- ❌ 需要实现`trace.hist`的序列化和反序列化
- ❌ 需要修改任务执行流程

**工作量**：
- 修改代码量：约300-400行
- 实施时间：4-6天

#### 方案2：修改代码，支持从workspace加载SOTA因子

**实现步骤**：
1. 在第一个任务执行完后，保存SOTA因子到workspace（已有）
2. 在第二个任务执行前，从workspace加载SOTA因子
3. 手动设置`based_experiments`

**优点**：
- ✅ 可以实现从workspace恢复任务
- ✅ 第二个任务可以访问第一个任务的SOTA因子
- ✅ 只需修改少量代码

**缺点**：
- ❌ 需要修改代码
- ❌ 需要实现SOTA因子的加载逻辑

**工作量**：
- 修改代码量：约100-150行
- 实施时间：2-3天

#### 方案3：不修改代码，无法从workspace恢复

**结论**：
- ❌ 无法直接从workspace恢复任务
- ❌ 需要修改代码才能实现

## 4. 推荐方案

### 4.1 方案对比

| 方案 | 是否需要修改代码 | 工作量 | 是否可以从workspace恢复 | 推荐度 |
|-----|----------------|-------|----------------------|-------|
| 方案1：从workspace恢复trace.hist | ✅ 需要 | 300-400行，4-6天 | ✅ 是 | ⭐⭐⭐ |
| 方案2：从workspace加载SOTA因子 | ✅ 需要 | 100-150行，2-3天 | ✅ 是 | ⭐⭐⭐⭐ |
| 方案3：不修改代码 | ❌ 不需要 | 0行 | ❌ 否 | ⭐ |

### 4.2 推荐方案：方案2 - 从workspace加载SOTA因子

**推荐理由**：
1. ✅ 可以实现从workspace恢复任务
2. ✅ 第二个任务可以访问第一个任务的SOTA因子
3. ✅ 只需修改少量代码
4. ✅ 实施简单，易于回退

**实施步骤**：

**第一步：修改model_runner.py，从workspace加载SOTA因子**
```python
# 在model_runner.py的develop方法中，添加从workspace加载SOTA因子的逻辑
def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
    # 从workspace加载SOTA因子
    load_sota_from_workspace = os.getenv("LOAD_SOTA_FROM_WORKSPACE", "false") == "true"
    if load_sota_from_workspace:
        sota_workspace_path = Path(os.getenv("SOTA_WORKSPACE_PATH", ""))
        if sota_workspace_path.exists():
            sota_factors_path = sota_workspace_path / "combined_factors_df.parquet"
            if sota_factors_path.exists():
                SOTA_factor = pd.read_parquet(sota_factors_path, engine="pyarrow")
                logger.info(f"SOTA factors loaded from workspace: {sota_factors_path}")
                
                # 创建based_experiments
                sota_factor_exp = QlibFactorExperiment(sub_tasks=[])
                sota_factor_exp.result = {"IC": 0.0}
                exp.based_experiments = [sota_factor_exp]
    
    # ... 现有代码 ...
```

**第二步：环境变量配置**
```env
# 第一个任务：因子演进
LOAD_SOTA_FROM_WORKSPACE=false

# 第二个任务：模型演进
LOAD_SOTA_FROM_WORKSPACE=true
SOTA_WORKSPACE_PATH=./path/to/first/task/workspace
USE_ALPHA_FACTORS=true
```

**第三步：执行两个任务**
```bash
# 第一个任务：因子演进
export LOAD_SOTA_FROM_WORKSPACE=false
python run_factor_evolution.py

# 第二个任务：模型演进
export LOAD_SOTA_FROM_WORKSPACE=true
export SOTA_WORKSPACE_PATH=./path/to/first/task/workspace
export USE_ALPHA_FACTORS=true
python run_model_evolution.py
```

## 5. 结论

### 5.1 回答用户的问题

**问题1：是否可以从上一个任务继续执行？**
- ❌ 无法直接从workspace继续执行
- ✅ 需要修改代码才能实现

**问题2：workspace中保存了什么信息？**
- ✅ Workspace中保存了代码文件和数据文件
- ✅ Workspace中保存了`combined_factors_df.parquet`（包含SOTA因子）
- ❌ Workspace中没有保存实验历史（`trace.hist`）
- ❌ Workspace中没有保存`based_experiments`

**问题3：是否可以从workspace恢复任务？**
- ❌ 无法直接从workspace恢复任务
- ✅ 需要修改代码才能实现

### 5.2 最终建议

**推荐采用方案2：从workspace加载SOTA因子**

**理由**：
1. ✅ 可以实现从workspace恢复任务
2. ✅ 第二个任务可以访问第一个任务的SOTA因子
3. ✅ 只需修改少量代码（约100-150行）
4. ✅ 实施简单，易于回退

**实施时间**：
- 修改代码：2-3天
- 测试验证：1-2天
- 总计：3-5天

**预期效果**：
- ✅ 可以从workspace恢复任务
- ✅ 第二个任务可以访问第一个任务的SOTA因子
- ✅ 可以基于SOTA因子和Alpha22因子进行模型演进

## 6. 附录

### 6.1 环境变量配置示例

```env
# 第一个任务：因子演进
LOAD_SOTA_FROM_WORKSPACE=false

# 第二个任务：模型演进
LOAD_SOTA_FROM_WORKSPACE=true
SOTA_WORKSPACE_PATH=./path/to/first/task/workspace
USE_ALPHA_FACTORS=true
```

### 6.2 回滚方法

如果需要回滚到原始行为，只需设置环境变量：

```bash
# 回滚到原始行为
export LOAD_SOTA_FROM_WORKSPACE=false
```

### 6.3 监控指标

**第一个任务监控指标**：
- Workspace路径
- SOTA因子数量
- SOTA因子IC

**第二个任务监控指标**：
- 加载的SOTA因子数量
- 模型IC
- 模型类型

## 7. 版本历史

| 版本 | 日期 | 说明 |
|-----|------|------|
| v1.0 | 2026-01-14 | 初始版本，分析从workspace恢复任务的可行性和实施方案 |
