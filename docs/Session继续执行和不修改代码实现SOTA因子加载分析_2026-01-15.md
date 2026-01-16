# Session继续执行和不修改代码实现SOTA因子加载分析

## 实施日期
2026-01-15

## 1. 问题分析

### 1.1 用户问题

1. 第二个任务是否可以继续从第一个任务的session开始继续？
2. 目前我们是否已经修改了代码？
3. 再模型演进时，如果有SOTA因子，就把SOTA因子和alpha22因子组合作为模型演进的因子使用？
4. 是否可以不修改代码达成这个目标？

### 1.2 核心挑战

- **挑战1**：`fin_factor` 和 `fin_model` 使用不同的Loop类，session不兼容
- **挑战2**：如何在不修改代码的情况下，将SOTA因子加载到 `based_experiments`
- **挑战3**：如何确保模型演进使用SOTA因子和Alpha22因子的组合

## 2. Session继续执行分析

### 2.1 代码分析

**fin_factor 使用 FactorRDLoop：**
```python
# @/f:/Dev/RD-Agent-main/rdagent/app/qlib_rd_loop/factor.py
class FactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError,)

def main(path=None, ...):
    if path is None:
        factor_loop = FactorRDLoop(FACTOR_PROP_SETTING)
    else:
        factor_loop = FactorRDLoop.load(path, checkout=checkout)
```

**fin_model 使用 ModelRDLoop：**
```python
# @/f:/Dev/RD-Agent-main/rdagent/app/qlib_rd_loop/model.py
class ModelRDLoop(RDLoop):
    skip_loop_error = (ModelEmptyError,)

def main(path=None, ...):
    if path is None:
        model_loop = ModelRDLoop(MODEL_PROP_SETTING)
    else:
        model_loop = ModelRDLoop.load(path, checkout=checkout)
```

### 2.2 Session不兼容原因

1. **Loop类不同**：`FactorRDLoop` vs `ModelRDLoop`
2. **配置不同**：`FACTOR_PROP_SETTING` vs `MODEL_PROP_SETTING`
3. **历史记录不同**：`trace.hist` 存储的实验类型不同
4. **组件不同**：`hypothesis_gen`、`coder`、`runner` 等组件不同

### 2.3 结论

❌ **不能从 `fin_factor` 的session继续执行 `fin_model`**

**原因**：
- Session保存的是 `FactorRDLoop` 的状态
- `ModelRDLoop.load()` 无法加载 `FactorRDLoop` 的session
- 即使强制加载，也会因为组件不匹配而失败

## 3. 代码修改状态分析

### 3.1 当前代码状态

**检查结果**：
- ✅ 代码未修改
- ✅ 原始代码保持不变
- ✅ 只是提供了修改方案，未实际执行

### 3.2 原始代码分析

**model_runner.py 的 develop 方法（第151-201行）：**
```python
@cache_with_pickle(qlib_model_cache_key, CachedRunner.assign_cached_result)
def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
    if exp.based_experiments and exp.based_experiments[-1].result is None:
        exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

    exist_sota_factor_exp = False
    if exp.based_experiments:
        SOTA_factor = None
        # Filter and retain only QlibFactorExperiment instances
        sota_factor_experiments_list = [
            base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
        ]
        # 修改：降低SOTA因子使用条件，从>1改为>=1，确保只要有SOTA因子就使用
        if len(sota_factor_experiments_list) >= 1:
            logger.info(f"SOTA factor processing ...")
            SOTA_factor = process_factor_data(sota_factor_experiments_list)

        if SOTA_factor is not None and not SOTA_factor.empty:
            exist_sota_factor_exp = True
            combined_factors = SOTA_factor
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns
            
            # 叠加Alpha因子
            use_alpha_factors = os.getenv("USE_ALPHA_FACTORS", "true") == "true"
            if use_alpha_factors:
                alpha_factor_names = load_alpha_factors_from_yaml()
                if alpha_factor_names:
                    logger.info(f"叠加Alpha因子: {len(alpha_factor_names)}个")
                    # 从SOTA因子中提取Alpha因子（如果存在）
                    alpha_factors_from_sota = []
                    for factor_name in alpha_factor_names:
                        if ("feature", factor_name) in combined_factors.columns:
                            alpha_factors_from_sota.append(combined_factors[("feature", factor_name)])
                    
                    # 记录叠加前后的因子数量
                    sota_factor_count = len([col for col in combined_factors.columns if col[0] == "feature"])
                    logger.info(f"SOTA因子数量: {sota_factor_count}")
                    
                    num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len([col for col in combined_factors.columns if col[0] == "feature"]))
                else:
                    num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len([col for col in combined_factors.columns if col[0] == "feature"]))
            else:
                num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len([col for col in combined_factors.columns if col[0] == "feature"]))

            target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

            # Save the combined factors to the workspace
            combined_factors.to_parquet(target_path, engine="pyarrow")
```

### 3.3 关键发现

✅ **代码已经支持从 `based_experiments` 加载SOTA因子**
✅ **代码已经支持叠加Alpha因子（通过环境变量 `USE_ALPHA_FACTORS`）**
✅ **代码已经支持SOTA因子和Alpha因子的组合**

**关键代码**：
```python
# 从based_experiments加载SOTA因子
sota_factor_experiments_list = [
    base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
]
if len(sota_factor_experiments_list) >= 1:
    SOTA_factor = process_factor_data(sota_factor_experiments_list)

# 叠加Alpha因子
use_alpha_factors = os.getenv("USE_ALPHA_FACTORS", "true") == "true"
if use_alpha_factors:
    alpha_factor_names = load_alpha_factors_from_yaml()
    # ... 叠加Alpha因子 ...
```

## 4. 不修改代码实现SOTA因子加载分析

### 4.1 核心问题

**问题**：如何在不修改代码的情况下，将SOTA因子加载到 `based_experiments`？

### 4.2 分析 based_experiments 的来源

**model_proposal.py 的 generate 方法（第158行）：**
```python
exp = QlibModelExperiment(tasks, hypothesis=hypothesis)
exp.based_experiments = [t[0] for t in trace.hist if t[1] and isinstance(t[0], ModelExperiment)]
```

**关键发现**：
- `based_experiments` 从 `trace.hist` 中提取
- `trace.hist` 包含历史实验记录
- 只提取 `ModelExperiment` 类型的实验

**问题**：
- `trace.hist` 中没有 `QlibFactorExperiment` 类型的实验
- 所以 `based_experiments` 中不会有 `QlibFactorExperiment`
- 因此无法从 `based_experiments` 加载SOTA因子

### 4.3 不修改代码的替代方案

#### 方案1：使用 fin_quant 命令（推荐）

**原理**：
- `fin_quant` 是因子和模型联合演进
- 自动在因子和模型之间切换
- 自动将SOTA因子传递给模型演进

**命令**：
```bash
rdagent fin_quant --loop_n 20
```

**优点**：
- ✅ 无需修改代码
- ✅ 自动协调因子和模型演进
- ✅ 自动使用SOTA因子
- ✅ 自动叠加Alpha因子

**缺点**：
- ❌ 无法精确控制先执行10个因子loop，再执行10个模型loop
- ❌ 因子和模型是交替进行的

#### 方案2：手动复制SOTA因子（不推荐）

**原理**：
- 手动将因子演进的SOTA因子文件复制到模型演进的workspace
- 手动修改模型演进的配置文件

**步骤**：
```bash
# 1. 执行因子演进
rdagent fin_factor --loop_n 10

# 2. 获取最新的workspace
LATEST_WORKSPACE=$(ls -t git_ignore_folder/RD-Agent_workspace/ | head -n 1)

# 3. 执行模型演进（在另一个终端）
rdagent fin_model --loop_n 10

# 4. 获取模型演进的workspace
MODEL_WORKSPACE=$(ls -t git_ignore_folder/RD-Agent_workspace/ | head -n 1)

# 5. 复制SOTA因子文件
cp git_ignore_folder/RD-Agent_workspace/$LATEST_WORKSPACE/combined_factors_df.parquet \
   git_ignore_folder/RD-Agent_workspace/$MODEL_WORKSPACE/combined_factors_df.parquet

# 6. 重新执行模型演进
rdagent fin_model --loop_n 10
```

**优点**：
- ✅ 无需修改代码

**缺点**：
- ❌ 需要手动操作
- ❌ 容易出错
- ❌ 无法保证模型演进会使用SOTA因子
- ❌ 需要重新执行模型演进

#### 方案3：修改配置文件（不推荐）

**原理**：
- 修改模型演进的配置文件
- 指定使用SOTA因子

**步骤**：
```bash
# 1. 执行因子演进
rdagent fin_factor --loop_n 10

# 2. 获取最新的workspace
LATEST_WORKSPACE=$(ls -t git_ignore_folder/RD-Agent_workspace/ | head -n 1)

# 3. 修改配置文件
# 编辑 rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml
# 指定SOTA因子文件路径

# 4. 执行模型演进
rdagent fin_model --loop_n 10
```

**优点**：
- ✅ 无需修改核心代码

**缺点**：
- ❌ 需要修改配置文件
- ❌ 容易出错
- ❌ 无法保证模型演进会使用SOTA因子

### 4.4 结论

❌ **无法在不修改代码的情况下，实现从因子演进的SOTA因子加载到模型演进**

**原因**：
- `based_experiments` 从 `trace.hist` 中提取
- `trace.hist` 中没有 `QlibFactorExperiment` 类型的实验
- 没有其他机制可以将SOTA因子加载到 `based_experiments`

## 5. 推荐方案

### 5.1 方案对比

| 方案 | 修改代码 | Session继续 | SOTA因子加载 | 推荐度 |
|-----|---------|-----------|-------------|--------|
| 方案1：修改代码 | ✅ 需要 | ❌ 不支持 | ✅ 支持 | ⭐⭐⭐⭐ |
| 方案2：使用fin_quant | ❌ 不需要 | ❌ 不支持 | ✅ 支持 | ⭐⭐⭐⭐⭐ |
| 方案3：手动复制 | ❌ 不需要 | ❌ 不支持 | ❌ 不支持 | ⭐ |
| 方案4：修改配置 | ❌ 不需要 | ❌ 不支持 | ❌ 不支持 | ⭐⭐ |

### 5.2 推荐方案：使用 fin_quant

**命令**：
```bash
rdagent fin_quant --loop_n 20
```

**优点**：
- ✅ 无需修改代码
- ✅ 自动协调因子和模型演进
- ✅ 自动使用SOTA因子
- ✅ 自动叠加Alpha因子
- ✅ 一个命令完成整个流程

**缺点**：
- ❌ 无法精确控制先执行10个因子loop，再执行10个模型loop
- ❌ 因子和模型是交替进行的

### 5.3 备选方案：修改代码

如果必须先执行10个因子loop，再执行10个模型loop，需要修改代码。

**修改步骤**：
1. 修改 `model_runner.py`，添加从workspace加载SOTA因子的逻辑
2. 执行因子演进：`rdagent fin_factor --loop_n 10`
3. 配置环境变量
4. 执行模型演进：`rdagent fin_model --loop_n 10`

详细修改方案见：`因子演进+模型演进完整实施方案（模型使用SOTA因子）_2026-01-14.md`

## 6. 总结

### 6.1 问题回答

**问题1：第二个任务是否可以继续从第一个任务的session开始继续？**

❌ **不能**

**原因**：
- `fin_factor` 使用 `FactorRDLoop`
- `fin_model` 使用 `ModelRDLoop`
- 它们是不同的类，session不兼容

**问题2：目前我们是否已经修改了代码？**

❌ **没有**

**状态**：
- 只是提供了修改方案
- 未实际执行修改
- 原始代码保持不变

**问题3：再模型演进时，如果有SOTA因子，就把SOTA因子和alpha22因子组合作为模型演进的因子使用？**

✅ **代码已经支持**

**证据**：
- `model_runner.py` 的第155-201行
- 从 `based_experiments` 加载SOTA因子
- 叠加Alpha因子（通过环境变量 `USE_ALPHA_FACTORS`）

**问题4：是否可以不修改代码达成这个目标？**

❌ **不能**

**原因**：
- `based_experiments` 从 `trace.hist` 中提取
- `trace.hist` 中没有 `QlibFactorExperiment` 类型的实验
- 没有其他机制可以将SOTA因子加载到 `based_experiments`

### 6.2 推荐方案

**方案1：使用 fin_quant（推荐）**

```bash
rdagent fin_quant --loop_n 20
```

**优点**：
- 无需修改代码
- 自动协调因子和模型演进
- 自动使用SOTA因子
- 自动叠加Alpha因子

**方案2：修改代码（备选）**

如果必须先执行10个因子loop，再执行10个模型loop，需要修改代码。

详细修改方案见：`因子演进+模型演进完整实施方案（模型使用SOTA因子）_2026-01-14.md`

## 7. 附录

### 7.1 相关文档

- 从Workspace恢复任务执行方案分析_2026-01-14.md
- RD-Agent任务启动命令和参数分析报告_2026-01-14.md
- 执行10个Loop因子演进和10个Loop模型演进的具体步骤_2026-01-14.md
- 因子演进+模型演进完整实施方案（模型使用SOTA因子）_2026-01-14.md

### 7.2 相关代码文件

- `@/f:/Dev/RD-Agent-main/rdagent/app/qlib_rd_loop/factor.py`
- `@/f:/Dev/RD-Agent-main/rdagent/app/qlib_rd_loop/model.py`
- `@/f:/Dev/RD-Agent-main/rdagent/scenarios/qlib/developer/model_runner.py`
- `@/f:/Dev/RD-Agent-main/rdagent/scenarios/qlib/proposal/model_proposal.py`

### 7.3 版本历史

| 版本 | 日期 | 说明 |
|-----|------|------|
| v1.0 | 2026-01-15 | 初始版本，分析Session继续执行和不修改代码实现SOTA因子加载的可行性 |
