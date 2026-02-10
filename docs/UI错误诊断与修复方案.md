# RD-Agent UI错误诊断与修复方案

## 问题描述

执行自定义模型演进任务后，无法在UI界面查看任务信息，UI报错：

```
ModuleNotFoundError: No module named 'model'
ValueError: time data 'ret' does not match format '%Y-%m-%d_%H-%M-%S-%f'
```

## 问题分析

### 根本原因

RD-Agent的UI日志读取机制存在设计缺陷：

1. **`rdagent/log/storage.py:94-120` 的 `iter_msg` 方法**
   - 使用 `self.path.glob("**/*.pkl")` 遍历所有pkl文件
   - 期望所有pkl文件名符合时间戳格式：`%Y-%m-%d_%H-%M-%S-%f.pkl`
   - 对每个pkl文件调用 `datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f")`

2. **Workspace中的实验结果文件不符合此格式**
   - Qlib实验结果文件：`ret.pkl`, `label.pkl`, `params.pkl`, `pred.pkl` 等
   - MLflow artifacts文件：`ic.pkl`, `ric.pkl`, `port_analysis_1day.pkl` 等
   - 这些文件是实验数据，不是日志消息

### 错误堆栈解析

```python
# storage.py:112
timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f")
# 当file.stem = "ret"时，抛出ValueError

# storage.py:110
content = _compat_pickle_load(f)
# 反序列化workspace中的pkl文件时，缺少对应的模块定义，抛出ModuleNotFoundError
```

### 受影响的log目录结构

```
log/model_comparison_20260117_193508/
├── log/                          # 真正的日志文件（符合时间戳格式）
│   └── 2026-01-17_11-35-08-830195/
│       └── *.pkl
├── git_ignore_folder/
│   └── RD-Agent_workspace/       # 实验workspace
│       ├── 176cc47ecc7b4ea4.../
│       │   ├── ret.pkl           # ❌ 不符合时间戳格式
│       │   ├── mlruns/
│       │   │   └── .../artifacts/
│       │   │       ├── label.pkl # ❌ 不符合时间戳格式
│       │   │       └── pred.pkl  # ❌ 不符合时间戳格式
│       └── ...
└── xgboost/                      # 模型专用目录（空）
```

## 修复方案

### 方案一：数据修复脚本（推荐）

**优点**：
- 不修改RD-Agent核心代码
- 不影响runtime行为
- 可逆（可恢复）
- 保持workspace完整性

**实施步骤**：

1. 使用提供的 `debug_tools/fix_ui_log_structure.py` 脚本
2. 检查模式：识别所有无效pkl文件
   ```bash
   python debug_tools/fix_ui_log_structure.py \
     --log_dir F:/Dev/RD-Agent-main/log/model_comparison_20260117_193508 \
     --mode check
   ```

3. 修复模式：将无效pkl文件移动到 `ui_excluded/` 目录
   ```bash
   python debug_tools/fix_ui_log_structure.py \
     --log_dir F:/Dev/RD-Agent-main/log/model_comparison_20260117_193508 \
     --mode fix
   ```

4. 在UI中刷新查看结果

5. 如需恢复：
   ```bash
   python debug_tools/fix_ui_log_structure.py \
     --log_dir F:/Dev/RD-Agent-main/log/model_comparison_20260117_193508 \
     --mode restore
   ```

**工作原理**：
- 将workspace中的实验结果pkl文件移动到 `ui_excluded/` 目录
- UI的 `glob("**/*.pkl")` 仍会遍历，但不会读取 `ui_excluded/` 中的文件
- Workspace目录结构保持完整（通过符号链接或保留目录）
- 实验结果文件仍可手动访问

### 方案二：改进任务执行脚本

**目标**：让future tasks的log目录结构符合UI预期

**修改文件**：`debug_tools/run_model_comparison_with_sota.py`

**修改建议**：

```python
def run_model_evolution(model_name, config, workspace_dir):
    # 修改前：workspace_dir可能包含workspace目录
    # workspace_path = Path(workspace_dir)
    
    # 修改后：将workspace隔离到git_ignore_folder
    workspace_path = Path(workspace_dir)
    
    # 确保log和workspace分离
    log_base = workspace_path.parent  # log/model_comparison_xxx/
    model_log_dir = log_base / "log" / model_name.lower()
    model_workspace_dir = log_base / "git_ignore_folder" / model_name.lower()
    
    model_log_dir.mkdir(parents=True, exist_ok=True)
    model_workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量，分离log和workspace
    env["QLIB_LOG_DIR"] = str(model_log_dir)
    env["QLIB_WORKSPACE_DIR"] = str(model_workspace_dir)
    
    # 运行命令
    cmd = [
        sys.executable,
        "-m",
        "rdagent.app.qlib_rd_loop.model",
        "--loop_n", str(config["loop_n"]),
        "--log_dir", str(model_log_dir),      # 新增：明确指定log目录
        "--workspace", str(model_workspace_dir) # 新增：明确指定workspace目录
    ]
```

**注意**：此方案需要验证RD-Agent是否支持 `--log_dir` 和 `--workspace` 参数。如不支持，需通过环境变量或配置文件设置。

### 方案三：长期改进建议（需提交PR到RD-Agent）

**修改文件**：`rdagent/log/storage.py`

**改进点**：

1. **增强 `iter_msg` 的过滤逻辑**

```python
def iter_msg(self, tag: str | None = None, pattern: str | None = None) -> Generator[Message, None, None]:
    msg_l = []
    
    if pattern:
        pkl_files = pattern
    elif tag:
        pkl_files = f"**/{tag.replace('.','/')}/**/*.pkl"
    else:
        pkl_files = "**/*.pkl"
    
    for file in self.path.glob(pkl_files):
        # 新增：跳过不符合日志格式的文件
        if not self._is_log_file(file):
            continue
        
        if file.name == "debug_llm.pkl":
            continue
        
        # ... 原有逻辑 ...

def _is_log_file(self, file: Path) -> bool:
    """检查是否是有效的日志文件"""
    import re
    
    # 检查文件名是否符合时间戳格式
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+\.pkl$')
    if not timestamp_pattern.match(file.name):
        return False
    
    # 检查是否在workspace目录下（可选）
    if "workspace" in str(file).lower():
        return False
    
    return True
```

2. **改进错误处理**

```python
for file in self.path.glob(pkl_files):
    try:
        # 原有解析逻辑
        timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f")
        with file.open("rb") as f:
            content = _compat_pickle_load(f)
        # ...
    except (ValueError, ModuleNotFoundError) as e:
        # 跳过无法解析的文件，记录警告
        import logging
        logging.warning(f"Skipping invalid log file {file}: {e}")
        continue
```

**此方案需要**：
- 向RD-Agent项目提交PR
- 等待社区审核和合并
- 不适合立即解决当前问题

## 推荐执行方案

### 立即执行（解决当前问题）

使用 **方案一** 修复现有log目录：

```bash
cd F:/Dev/RD-Agent-main

# 1. 检查无效文件
python debug_tools/fix_ui_log_structure.py \
  --log_dir log/model_comparison_20260117_193508 \
  --mode check

# 2. 修复（移动无效文件）
python debug_tools/fix_ui_log_structure.py \
  --log_dir log/model_comparison_20260117_193508 \
  --mode fix

# 3. 启动UI查看结果
streamlit run rdagent/log/ui/app.py
```

### 未来任务预防

考虑 **方案二** 的改进建议，但需要先验证：

1. 检查RD-Agent的命令行参数支持情况
2. 查看是否有环境变量可以分离log和workspace
3. 测试修改后的脚本是否正常工作

**验证命令**：
```bash
python -m rdagent.app.qlib_rd_loop.model --help
```

查看是否支持 `--log_dir` 或 `--workspace` 参数。

### 长期改进

提交 **方案三** 的PR到RD-Agent项目，改进log存储的健壮性。

## 其他发现

### UI的Streamlit警告

```
For `use_container_width=True`, use `width='stretch'`.
```

这是Streamlit API变更警告，不影响功能，但建议RD-Agent团队更新代码以适配新版本Streamlit。

## 总结

| 方案 | 优先级 | 难度 | 风险 | 时效性 |
|------|--------|------|------|--------|
| 方案一：数据修复脚本 | ⭐⭐⭐⭐⭐ | 低 | 低 | 立即可用 |
| 方案二：改进执行脚本 | ⭐⭐⭐ | 中 | 中 | 需验证 |
| 方案三：改进RD-Agent核心 | ⭐⭐ | 高 | 低 | 长期 |

**结论**：先使用方案一解决当前问题，再考虑方案二预防future issues，最后可选择性地为社区贡献方案三。
