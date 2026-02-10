# Qlib与RD-Agent模型演进方案分析

**文档版本**: v1.0  
**生成时间**: 2026-01-16  
**适用场景**: 基于SOTA因子和Alpha因子的模型演进

---

## 一、Qlib功能分析

### 1.1 核心功能

Qlib是微软开源的量化投资框架，主要功能包括：

1. **因子挖掘与计算**
   - 支持自定义因子表达式
   - 提供丰富的Alpha因子库（如Alpha158）
   - 支持因子预处理（归一化、填充缺失值等）

2. **模型训练**
   - 支持多种模型类型（LightGBM、PyTorch、XGBoost等）
   - 支持时间序列模型和表格模型
   - 支持GPU/CPU训练
   - 支持早停机制

3. **回测评估**
   - 提供完整的回测框架
   - 支持多种评估指标（IC、IR、回撤、夏普比率等）
   - 支持组合分析
   - 支持多种交易策略（TopkDropoutStrategy等）

4. **实验管理**
   - 集成MLflow进行实验跟踪
   - 支持模型版本管理
   - 支持超参数优化

### 1.2 架构特点

```
Qlib架构
├── 数据层
│   ├── 数据提供者（provider_uri）
│   ├── 数据处理器（processors）
│   └── 数据集类（DatasetH, TSDatasetH）
├── 模型层
│   ├── 模型类（GeneralPTNN, LGBModel等）
│   ├── 模型配置（kwargs）
│   └── 训练参数（n_epochs, lr, batch_size等）
├── 回测层
│   ├── 策略类（TopkDropoutStrategy）
│   ├── 回测配置（start_time, end_time等）
│   └── 交易成本配置
└── 记录层
    ├── 信号记录（SignalRecord）
    ├── 信号分析（SigAnaRecord）
    └── 组合分析（PortAnaRecord）
```

### 1.3 配置文件驱动

Qlib通过YAML配置文件驱动整个训练和回测流程：

```yaml
task:
    model:
        class: GeneralPTNN
        kwargs:
            n_epochs: {{ n_epochs }}
            lr: {{ lr }}
            early_stop: {{ early_stop }}
            batch_size: {{ batch_size }}
            weight_decay: {{ weight_decay }}
    dataset:
        class: {{ dataset_cls | default("DatasetH") }}
        kwargs:
            segments:
                train: [2010-01-07, 2018-12-31]
                valid: [2019-01-01, 2020-12-31]
                test: [2021-01-01, 2025-12-01]
```

---

## 二、RD-Agent功能分析

### 2.1 核心功能

RD-Agent是基于LLM的量化投资自动化框架，主要功能包括：

1. **自动化因子开发**
   - 通过LLM生成因子代码
   - 自动执行因子回测
   - 自动评估因子性能
   - 自动更新SOTA因子列表

2. **自动化模型训练**
   - 基于SOTA因子和Alpha因子组合训练模型
   - 支持多种模型类型（TimeSeries, Tabular）
   - 自动调优训练超参数
   - 支持缓存机制避免重复训练

3. **实验管理**
   - 基于session的实验跟踪
   - 支持实验历史回溯
   - 支持基于实验的模型演进

### 2.2 架构特点

```
RD-Agent架构
├── 实验层
│   ├── 因子实验（QlibFactorExperiment）
│   ├── 模型实验（QlibModelExperiment）
│   └── 实验工作空间（QlibFBWorkspace）
├── 开发层
│   ├── 因子开发器（FactorDeveloper）
│   ├── 模型开发器（ModelDeveloper）
│   └── 运行器（QlibModelRunner）
├── 数据层
│   ├── SOTA因子处理（process_factor_data）
│   ├── Alpha因子加载（load_alpha_factors_from_yaml）
│   └── 因子合并（combined_factors）
└── 反馈层
    ├── 因子反馈（FactorFeedback）
    ├── 模型反馈（ModelFeedback）
    └── 综合反馈（CoSTEERMultiFeedback）
```

### 2.3 关键组件

#### 2.3.1 QlibModelRunner

负责模型训练和回测的核心组件：

```python
class QlibModelRunner(CachedRunner[QlibModelExperiment]):
    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        # 1. 处理SOTA因子
        if exp.based_experiments:
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments 
                if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) >= 1:
                SOTA_factor = process_factor_data(sota_factor_experiments_list)
        
        # 2. 叠加Alpha因子
        use_alpha_factors = os.getenv("USE_ALPHA_FACTORS", "true") == "true"
        if use_alpha_factors:
            alpha_factor_names = load_alpha_factors_from_yaml()
            # 从SOTA因子中提取Alpha因子
        
        # 3. 保存合并后的因子
        combined_factors.to_parquet(target_path, engine="pyarrow")
        
        # 4. 注入模型代码
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})
        
        # 5. 配置训练参数
        env_to_use = {
            "n_epochs": str(training_hyperparameters.get("n_epochs", "20")),
            "lr": str(training_hyperparameters.get("lr", "1e-3")),
            "early_stop": str(training_hyperparameters.get("early_stop", 5)),
            "batch_size": str(training_hyperparameters.get("batch_size", 256)),
            "weight_decay": str(training_hyperparameters.get("weight_decay", 1e-4)),
        }
        
        # 6. 根据模型类型选择配置
        model_type = getattr(exp.sub_tasks[0], "model_type", None)
        if model_type == "TimeSeries":
            qlib_config_name = "conf_sota_factors_model.yaml"
            attempts = [
                {"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20},
                {"dataset_cls": "DatasetH", "step_len": 20, "num_timesteps": 20},
                {"dataset_cls": "TSDatasetH", "step_len": 60, "num_timesteps": 60},
            ]
        elif model_type == "Tabular":
            qlib_config_name = "conf_sota_factors_model.yaml"
            attempts = [
                {"dataset_cls": "DatasetH"},
                {"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20},
                {"dataset_cls": "TSDatasetH", "step_len": 60, "num_timesteps": 60},
            ]
        
        # 7. 执行训练和回测
        result, stdout = self._execute_with_retry(exp, qlib_config_name, env_to_use, attempts)
```

#### 2.3.2 process_factor_data

处理SOTA因子数据的核心函数：

```python
def process_factor_data(exp_or_list: List[QlibFactorExperiment]) -> pd.DataFrame:
    """处理和合并因子数据"""
    factor_dfs = []
    
    for exp in exp_or_list:
        # 执行因子实现
        message_and_df_list = multiprocessing_wrapper(
            [(implementation.execute, ("All",)) 
             for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
             if implementation and fb],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )
        
        # 检查因子生成是否成功
        for message, df in message_and_df_list:
            if df is not None and "datetime" in df.index.names:
                time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                if pd.Timedelta(minutes=1) not in time_diff:
                    factor_dfs.append(df)
    
    # 合并所有成功的因子数据
    if factor_dfs:
        return pd.concat(factor_dfs, axis=1)
    else:
        raise FactorEmptyError("No valid factor data found")
```

#### 2.3.3 load_alpha_factors_from_yaml

从配置文件加载Alpha因子：

```python
def load_alpha_factors_from_yaml() -> list[str] | None:
    """从配置文件加载Alpha因子列表"""
    alpha_config_path = Path(__file__).parent.parent / "experiment" / "model_template" / "conf_baseline_factors_model.yaml"
    
    with open(alpha_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取col_list
    match = re.search(r'col_list:\s*\[(.*?)\]', content, re.DOTALL)
    if match:
        factors_str = match.group(1)
        alpha_factors = [f.strip().strip('"').strip("'") for f in factors_str.split(',')]
        alpha_factors = [f for f in alpha_factors if f]
        return alpha_factors
```

---

## 三、模型演进方案设计

### 3.1 方案概述

基于SOTA因子和Alpha因子的组合作为模型演进的基础因子，使用相同的因子尝试不同类型的模型，执行训练和回测最终给出对比结果。

### 3.2 实现流程

```
模型演进流程
├── 1. 因子准备
│   ├── 1.1 提取SOTA因子
│   │   ├── 从QlibFactorExperiment中提取因子代码
│   │   ├── 执行因子代码生成因子数据
│   │   └── 合并所有SOTA因子数据
│   ├── 1.2 加载Alpha因子
│   │   ├── 从配置文件中读取Alpha因子列表
│   │   └── 从SOTA因子中提取Alpha因子数据
│   └── 1.3 因子合并
│       ├── 合并SOTA因子和Alpha因子
│       ├── 去除重复因子
│       └── 保存为combined_factors_df.parquet
├── 2. 模型训练
│   ├── 2.1 模型类型选择
│   │   ├── TimeSeries模型（时间序列模型）
│   │   └── Tabular模型（表格模型）
│   ├── 2.2 模型架构设计
│   │   ├── 设计不同的模型架构
│   │   ├── 生成model.py代码
│   │   └── 注入到实验工作空间
│   ├── 2.3 训练参数配置
│   │   ├── 配置训练超参数
│   │   ├── 配置数据集参数
│   │   └── 配置回测参数
│   └── 2.4 执行训练和回测
│       ├── 选择配置文件
│       ├── 执行qrun命令
│       └── 读取回测结果
└── 3. 结果对比
    ├── 3.1 收集结果
    │   ├── 收集训练指标
    │   ├── 收集回测指标
    │   └── 收集组合分析结果
    ├── 3.2 结果分析
    │   ├── 对比不同模型的性能
    │   ├── 分析模型的优缺点
    │   └── 选择最优模型
    └── 3.3 生成报告
        ├── 生成对比表格
        ├── 生成可视化图表
        └── 生成分析报告
```

### 3.3 详细实现

#### 3.3.1 因子准备

**步骤1：提取SOTA因子**

```python
# 从session中提取SOTA因子
sota_factor_experiments = [
    exp for exp, feedback in session.trace.hist
    if feedback and feedback.decision and isinstance(exp, QlibFactorExperiment)
]

# 处理SOTA因子数据
sota_factors = process_factor_data(sota_factor_experiments)
```

**步骤2：加载Alpha因子**

```python
# 从配置文件加载Alpha因子列表
alpha_factor_names = load_alpha_factors_from_yaml()

# 从SOTA因子中提取Alpha因子
alpha_factors = []
for factor_name in alpha_factor_names:
    if ("feature", factor_name) in sota_factors.columns:
        alpha_factors.append(sota_factors[("feature", factor_name)])
```

**步骤3：因子合并**

```python
# 合并SOTA因子和Alpha因子
combined_factors = sota_factors.copy()
combined_factors = combined_factors.sort_index()
combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]

# 保存为parquet格式
target_path = workspace_path / "combined_factors_df.parquet"
combined_factors.to_parquet(target_path, engine="pyarrow")
```

#### 3.3.2 模型训练

**方案1：TimeSeries模型**

```python
# 创建TimeSeries模型实验
model_exp = QlibModelExperiment(
    based_experiments=sota_factor_experiments,
    sub_tasks=[
        ModelTask(
            name="TimeSeriesModel",
            model_type="TimeSeries",
            training_hyperparameters={
                "n_epochs": 100,
                "lr": 2e-4,
                "early_stop": 10,
                "batch_size": 256,
                "weight_decay": 0.0001,
            }
        )
    ],
    sub_workspace_list=[
        ModelFBWorkspace(file_dict={
            "model.py": time_series_model_code  # 时间序列模型代码
        })
    ]
)

# 执行训练
model_runner = QlibModelRunner()
result = model_runner.develop(model_exp)
```

**方案2：Tabular模型**

```python
# 创建Tabular模型实验
model_exp = QlibModelExperiment(
    based_experiments=sota_factor_experiments,
    sub_tasks=[
        ModelTask(
            name="TabularModel",
            model_type="Tabular",
            training_hyperparameters={
                "n_epochs": 100,
                "lr": 2e-4,
                "early_stop": 10,
                "batch_size": 256,
                "weight_decay": 0.0001,
            }
        )
    ],
    sub_workspace_list=[
        ModelFBWorkspace(file_dict={
            "model.py": tabular_model_code  # 表格模型代码
        })
    ]
)

# 执行训练
model_runner = QlibModelRunner()
result = model_runner.develop(model_exp)
```

**方案3：混合模型（LightGBM）**

```python
# 创建LightGBM模型实验
model_exp = QlibModelExperiment(
    based_experiments=sota_factor_experiments,
    sub_tasks=[
        ModelTask(
            name="LightGBMModel",
            model_type="Tabular",
            training_hyperparameters={
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
            }
        )
    ],
    sub_workspace_list=[
        ModelFBWorkspace(file_dict={
            "model.py": lightgbm_model_code  # LightGBM模型代码
        })
    ]
)

# 执行训练
model_runner = QlibModelRunner()
result = model_runner.develop(model_exp)
```

#### 3.3.3 结果对比

**步骤1：收集结果**

```python
# 收集不同模型的训练指标
results = {
    "TimeSeries": {
        "train_loss": time_series_result["train_loss"],
        "valid_loss": time_series_result["valid_loss"],
        "test_loss": time_series_result["test_loss"],
        "ic": time_series_result["ic"],
        "ir": time_series_result["ir"],
    },
    "Tabular": {
        "train_loss": tabular_result["train_loss"],
        "valid_loss": tabular_result["valid_loss"],
        "test_loss": tabular_result["test_loss"],
        "ic": tabular_result["ic"],
        "ir": tabular_result["ir"],
    },
    "LightGBM": {
        "train_loss": lightgbm_result["train_loss"],
        "valid_loss": lightgbm_result["valid_loss"],
        "test_loss": lightgbm_result["test_loss"],
        "ic": lightgbm_result["ic"],
        "ir": lightgbm_result["ir"],
    },
}
```

**步骤2：结果分析**

```python
# 对比不同模型的性能
comparison = pd.DataFrame(results).T

# 计算综合评分
comparison["score"] = (
    comparison["ic"] * 0.4 + 
    comparison["ir"] * 0.3 + 
    (1 - comparison["test_loss"]) * 0.3
)

# 选择最优模型
best_model = comparison["score"].idxmax()
```

**步骤3：生成报告**

```python
# 生成对比表格
comparison_table = comparison.to_markdown()

# 生成可视化图表
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# IC对比
comparison["ic"].plot(kind="bar", ax=axes[0, 0], title="IC对比")

# IR对比
comparison["ir"].plot(kind="bar", ax=axes[0, 1], title="IR对比")

# 测试损失对比
comparison["test_loss"].plot(kind="bar", ax=axes[1, 0], title="测试损失对比")

# 综合评分对比
comparison["score"].plot(kind="bar", ax=axes[1, 1], title="综合评分对比")

plt.tight_layout()
plt.savefig("model_comparison.png")
```

---

## 四、实现方案总结

### 4.1 技术优势

1. **因子复用**
   - SOTA因子和Alpha因子可以复用，避免重复计算
   - 因子数据以parquet格式存储，读取效率高

2. **模型多样性**
   - 支持多种模型类型（TimeSeries, Tabular, LightGBM）
   - 可以快速尝试不同的模型架构

3. **自动化程度高**
   - 自动化因子处理、模型训练、结果对比
   - 支持缓存机制，避免重复训练

4. **可扩展性强**
   - 可以轻松添加新的模型类型
   - 可以自定义训练参数和回测参数

### 4.2 实现建议

1. **分阶段实施**
   - 第一阶段：实现因子准备和单一模型训练
   - 第二阶段：实现多模型训练和结果对比
   - 第三阶段：实现自动化报告生成

2. **参数调优**
   - 使用网格搜索或贝叶斯优化进行超参数调优
   - 使用交叉验证评估模型性能

3. **结果验证**
   - 使用样本外数据验证模型性能
   - 使用滚动窗口回测验证模型稳定性

### 4.3 注意事项

1. **数据一致性**
   - 确保所有模型使用相同的因子数据
   - 确保训练、验证、测试数据集划分一致

2. **计算资源**
   - TimeSeries模型训练时间较长，需要充足的计算资源
   - 建议使用GPU加速训练

3. **过拟合风险**
   - 使用早停机制防止过拟合
   - 使用正则化方法提高模型泛化能力

---

## 五、结论

通过Qlib和RD-Agent的结合，可以实现基于SOTA因子和Alpha因子的模型演进方案。该方案具有以下特点：

1. **因子复用**：SOTA因子和Alpha因子可以复用，提高效率
2. **模型多样性**：支持多种模型类型，可以快速尝试不同的模型架构
3. **自动化程度高**：自动化因子处理、模型训练、结果对比
4. **可扩展性强**：可以轻松添加新的模型类型和评估指标

该方案为量化投资研究提供了一个强大的工具，可以帮助研究人员快速探索不同的模型架构，找到最优的模型配置。

---

## 六、不修改源代码的实现方案分析

### 6.1 分析过程

#### 6.1.1 RD-Agent扩展机制分析

通过分析RD-Agent的源代码，发现以下扩展机制：

1. **环境变量控制**
   - `USE_ALPHA_FACTORS`：控制是否使用Alpha因子
   - `QLIB_QUANT_DISABLE_CACHE`：控制是否禁用缓存

2. **配置文件支持**
   - `load_alpha_factors_from_yaml()`：从配置文件加载Alpha因子列表
   - Alpha因子配置文件：`conf_baseline_factors_model.yaml`

3. **模型任务配置**
   - `ModelTask`：支持配置training_hyperparameters和model_type
   - `model_type`：支持"TimeSeries"和"Tabular"两种类型
   - `training_hyperparameters`：支持配置n_epochs、lr、early_stop、batch_size、weight_decay等参数

4. **实验依赖机制**
   - `based_experiments`：支持传递SOTA因子实验
   - `QlibModelRunner.develop()`：自动处理SOTA因子和Alpha因子的组合

#### 6.1.2 Qlib扩展机制分析

通过分析Qlib的配置文件，发现以下扩展机制：

1. **YAML配置文件驱动**
   - 支持Jinja2模板变量（如`{{ n_epochs }}`）
   - 支持通过环境变量传递参数
   - 支持多种模型类型和数据集类型

2. **模型代码注入**
   - 支持通过`model.py`文件注入自定义模型代码
   - 支持通过`pt_model_uri`指定模型类

3. **数据集配置**
   - 支持多种数据集类型（TSDatasetH、DatasetH）
   - 支持时间序列参数（step_len、num_timesteps）

### 6.2 实现方案

#### 6.2.1 方案概述

**结论：可以通过配置文件和外部脚本实现，不需要修改RD-Agent和Qlib的源代码。**

实现方案包括以下三个部分：

1. **外部脚本**：控制实验流程，加载session，提取SOTA因子，创建模型任务
2. **配置文件**：配置环境变量、Alpha因子、模型参数、回测参数等
3. **模型代码**：提供不同类型的模型代码模板

#### 6.2.2 具体实现

**1. 外部脚本（model_evolution_comparison.py）**

```python
class ModelEvolutionComparison:
    """模型演进对比实验"""
    
    def __init__(self, log_dir: str, output_dir: str = None):
        # 初始化参数
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir) if output_dir else self.log_dir / "model_comparison"
        
        # 模型运行器
        self.model_runner = QlibModelRunner()
        
    def load_session(self) -> Any:
        """加载session文件"""
        session_file = self.log_dir / "__session__" / "1_coding"
        with open(session_file, 'rb') as f:
            session = pickle.load(f)
        return session
    
    def extract_sota_factors(self, session: Any) -> List[QlibFactorExperiment]:
        """提取SOTA因子实验"""
        sota_factor_experiments = []
        for exp, feedback in session.trace.hist:
            if feedback and feedback.decision and isinstance(exp, QlibFactorExperiment):
                sota_factor_experiments.append(exp)
        return sota_factor_experiments
    
    def create_model_tasks(self) -> List[ModelTask]:
        """创建多个模型任务"""
        model_tasks = []
        
        # 1. TimeSeries模型
        model_tasks.append(
            ModelTask(
                name="TimeSeriesModel",
                model_type="TimeSeries",
                training_hyperparameters={
                    "n_epochs": 100,
                    "lr": 2e-4,
                    "early_stop": 10,
                    "batch_size": 256,
                    "weight_decay": 0.0001,
                },
            )
        )
        
        # 2. Tabular模型
        model_tasks.append(
            ModelTask(
                name="TabularModel",
                model_type="Tabular",
                training_hyperparameters={
                    "n_epochs": 100,
                    "lr": 2e-4,
                    "early_stop": 10,
                    "batch_size": 256,
                    "weight_decay": 0.0001,
                },
            )
        )
        
        # 3. LightGBM模型
        model_tasks.append(
            ModelTask(
                name="LightGBMModel",
                model_type="Tabular",
                training_hyperparameters={
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "early_stopping_rounds": 50,
                },
            )
        )
        
        return model_tasks
    
    def run_model_experiment(
        self,
        model_task: ModelTask,
        sota_factor_experiments: List[QlibFactorExperiment]
    ) -> Dict[str, Any]:
        """运行单个模型实验"""
        # 创建模型实验
        model_exp = QlibModelExperiment(
            sub_tasks=[model_task],
            based_experiments=sota_factor_experiments,
        )
        
        # 创建模型工作空间
        model_workspace = self.create_model_workspace(model_task.model_type)
        model_exp.sub_workspace_list = [model_workspace]
        
        # 设置环境变量
        os.environ["USE_ALPHA_FACTORS"] = "true"
        
        # 运行模型训练和回测
        result_exp = self.model_runner.develop(model_exp)
        
        return {
            "model_name": model_task.name,
            "model_type": model_task.model_type,
            "result": result_exp.result,
            "stdout": result_exp.stdout,
        }
```

**2. 配置文件（model_evolution_config.yaml）**

```yaml
# 环境变量配置
environment:
  USE_ALPHA_FACTORS: "true"
  QLIB_QUANT_DISABLE_CACHE: "0"

# Alpha因子配置
alpha_factors:
  config_path: "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml"
  factor_list:
    - "RESI5"
    - "WVMA5"
    - "RSQR5"
    # ... 更多Alpha因子

# 模型配置
models:
  - name: "TimeSeriesModel"
    model_type: "TimeSeries"
    training_hyperparameters:
      n_epochs: 100
      lr: 2e-4
      early_stop: 10
      batch_size: 256
      weight_decay: 0.0001
  
  - name: "TabularModel"
    model_type: "Tabular"
    training_hyperparameters:
      n_epochs: 100
      lr: 2e-4
      early_stop: 10
      batch_size: 256
      weight_decay: 0.0001
  
  - name: "LightGBMModel"
    model_type: "Tabular"
    training_hyperparameters:
      n_estimators: 1000
      learning_rate: 0.05
      early_stopping_rounds: 50
```

**3. 模型代码模板**

- **TimeSeries模型代码**：LSTM神经网络
- **Tabular模型代码**：MLP神经网络
- **LightGBM模型代码**：梯度提升树

### 6.3 实施步骤

#### 步骤1：准备环境

```bash
# 设置环境变量
export USE_ALPHA_FACTORS="true"
export QLIB_QUANT_DISABLE_CACHE="0"

# 确保RD-Agent和Qlib已安装
pip install -r requirements.txt
```

#### 步骤2：准备数据

```bash
# 确保数据目录存在
ls /mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209

# 确保Alpha因子配置文件存在
ls rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml
```

#### 步骤3：运行模型演进对比实验

```bash
# 运行脚本
python debug_tools/model_evolution_comparison.py --log_dir log/2026-01-13_06-56-49-446055 --output_dir model_comparison_results
```

#### 步骤4：查看结果

```bash
# 查看结果文件
ls model_comparison_results/

# 查看对比报告
cat model_comparison_results/model_comparison_report.md

# 查看详细结果
cat model_comparison_results/model_comparison_results.json
```

### 6.4 分析结论

#### 6.4.1 可行性分析

**✅ 可以通过配置文件和外部脚本实现，不需要修改RD-Agent和Qlib的源代码。**

**原因：**

1. **RD-Agent已支持的功能**
   - SOTA因子和Alpha因子的组合
   - 通过环境变量控制是否使用Alpha因子
   - 通过配置文件加载Alpha因子列表
   - 通过ModelTask配置训练参数和模型类型
   - 通过based_experiments传递SOTA因子实验

2. **Qlib已支持的功能**
   - 通过YAML配置文件驱动训练和回测
   - 支持Jinja2模板变量
   - 支持通过环境变量传递参数
   - 支持多种模型类型和数据集类型

3. **外部脚本的作用**
   - 加载RD-Agent的session文件
   - 提取SOTA因子实验
   - 创建多个模型任务
   - 调用RD-Agent的QlibModelRunner执行训练和回测
   - 收集结果并生成对比报告

#### 6.4.2 优势分析

1. **无需修改源代码**
   - 利用RD-Agent和Qlib的现有功能
   - 通过配置文件和外部脚本实现
   - 不影响RD-Agent和Qlib的稳定性

2. **灵活性高**
   - 可以通过配置文件调整参数
   - 可以轻松添加新的模型类型
   - 可以自定义模型代码

3. **可扩展性强**
   - 支持多种模型类型
   - 支持多种训练参数
   - 支持多种回测配置

4. **自动化程度高**
   - 自动化因子处理
   - 自动化模型训练
   - 自动化结果对比

#### 6.4.3 注意事项

1. **环境变量设置**
   - 确保USE_ALPHA_FACTORS设置为"true"
   - 确保数据路径正确

2. **配置文件路径**
   - 确保Alpha因子配置文件存在
   - 确保数据目录存在

3. **模型代码兼容性**
   - 确保模型代码符合RD-Agent的要求
   - 确保模型代码与Qlib兼容

4. **计算资源**
   - TimeSeries模型训练时间较长
   - 建议使用GPU加速训练

### 6.5 总结

通过分析，我们得出以下结论：

1. **可以实现**：可以通过配置文件和外部脚本实现基于SOTA因子和Alpha因子的模型演进方案，不需要修改RD-Agent和Qlib的源代码。

2. **实现方案**：
   - 外部脚本：控制实验流程
   - 配置文件：配置参数
   - 模型代码：提供模型模板

3. **优势**：
   - 无需修改源代码
   - 灵活性高
   - 可扩展性强
   - 自动化程度高

4. **注意事项**：
   - 环境变量设置
   - 配置文件路径
   - 模型代码兼容性
   - 计算资源

该方案为量化投资研究提供了一个强大的工具，可以帮助研究人员快速探索不同的模型架构，找到最优的模型配置，同时不需要修改RD-Agent和Qlib的源代码。

---

## 七、文件说明与模型支持列表

### 7.1 文件说明

#### 7.1.1 新建文件列表

本方案创建的所有文件都是新建的，**不会影响现有的RD-Agent运行**：

1. **外部脚本**：`debug_tools/model_evolution_comparison.py`
   - 新建文件
   - 用于控制模型演进对比实验流程
   - 不修改RD-Agent源代码

2. **配置文件**：`debug_tools/model_evolution_config.yaml`
   - 新建文件
   - 用于配置实验参数
   - 不修改RD-Agent源代码

3. **文档文件**：`docs/Qlib与RD-Agent模型演进方案分析.md`
   - 新建文件
   - 用于记录方案分析
   - 不修改RD-Agent源代码

**结论**：所有文件都是新建的，不会影响现有的RD-Agent运行。

### 7.2 模型控制方式

#### 7.2.1 通过配置文件控制

可以在`debug_tools/model_evolution_config.yaml`中的`models`部分控制每次运行使用的模型：

```yaml
# 模型配置
models:
  # 只运行TimeSeries模型
  - name: "TimeSeriesModel"
    model_type: "TimeSeries"
    training_hyperparameters:
      n_epochs: 100
      lr: 2e-4
      early_stop: 10
      batch_size: 256
      weight_decay: 0.0001
```

#### 7.2.2 通过脚本方法控制

可以在`debug_tools/model_evolution_comparison.py`中的`create_model_tasks()`方法中控制：

```python
def create_model_tasks(self) -> List[ModelTask]:
    """创建多个模型任务"""
    model_tasks = []
    
    # 只添加需要运行的模型
    model_tasks.append(
        ModelTask(
            name="TimeSeriesModel",
            model_type="TimeSeries",
            training_hyperparameters={...},
        )
    )
    
    return model_tasks
```

### 7.3 支持的模型详细列表

#### 7.3.1 RD-Agent支持的模型类型

根据代码分析，RD-Agent的`ModelTask`支持以下模型类型：

| 模型类型 | 说明 | 数据集类型 | 支持状态 |
|---------|------|-----------|---------|
| **TimeSeries** | 时间序列模型 | TSDatasetH | ✅ 完全支持 |
| **Tabular** | 表格模型 | DatasetH | ✅ 完全支持 |
| **Graph** | 图神经网络模型 | - | ⚠️ 代码支持但未验证 |
| **XGBoost** | XGBoost模型 | DatasetH | ⚠️ 代码支持但未验证 |

**注意**：
- 在`QlibModelRunner`中，只验证了`TimeSeries`和`Tabular`两种模型类型
- `Graph`和`XGBoost`在`ModelTask`的注释中有提及，但在`QlibModelRunner`中未实现

#### 7.3.2 Qlib支持的模型类型

Qlib框架支持以下模型类型：

| 模型类型 | 说明 | 模型类 | 数据集类型 |
|---------|------|--------|-----------|
| **LightGBM** | 梯度提升树 | LGBModel | DatasetH |
| **MLP** | 多层感知机 | GeneralPTNN | DatasetH / TSDatasetH |
| **LSTM** | 长短期记忆网络 | GeneralPTNN | TSDatasetH |
| **GRU** | 门控循环单元 | GeneralPTNN | TSDatasetH |
| **Transformer** | Transformer模型 | GeneralPTNN | TSDatasetH |
| **XGBoost** | XGBoost模型 | XGBModel | DatasetH |
| **Linear** | 线性模型 | LinearModel | DatasetH |

#### 7.3.3 推荐的模型架构

基于RD-Agent和Qlib的支持情况，推荐以下模型架构：

**1. TimeSeries模型（时间序列模型）**

- **适用场景**：需要考虑时间序列特征的场景
- **数据集类型**：TSDatasetH
- **推荐架构**：
  - LSTM（长短期记忆网络）
  - GRU（门控循环单元）
  - Transformer（Transformer模型）
- **训练参数**：
  ```python
  training_hyperparameters = {
      "n_epochs": 100,
      "lr": 2e-4,
      "early_stop": 10,
      "batch_size": 256,
      "weight_decay": 0.0001,
  }
  ```

**2. Tabular模型（表格模型）**

- **适用场景**：不需要考虑时间序列特征的场景
- **数据集类型**：DatasetH
- **推荐架构**：
  - MLP（多层感知机）
  - LightGBM（梯度提升树）
  - XGBoost（XGBoost模型）
- **训练参数**：
  ```python
  training_hyperparameters = {
      "n_epochs": 100,
      "lr": 2e-4,
      "early_stop": 10,
      "batch_size": 256,
      "weight_decay": 0.0001,
  }
  ```

**3. LightGBM模型（梯度提升树）**

- **适用场景**：表格数据，需要快速训练的场景
- **数据集类型**：DatasetH
- **推荐架构**：LightGBM
- **训练参数**：
  ```python
  training_hyperparameters = {
      "n_estimators": 1000,
      "learning_rate": 0.05,
      "early_stopping_rounds": 50,
  }
  ```

### 7.4 模型选择建议

#### 7.4.1 根据数据特征选择

| 数据特征 | 推荐模型 | 原因 |
|---------|---------|------|
| 时间序列数据 | TimeSeries（LSTM/GRU/Transformer） | 能够捕捉时间依赖关系 |
| 表格数据 | Tabular（MLP/LightGBM/XGBoost） | 处理表格数据效率高 |
| 大规模数据 | LightGBM/XGBoost | 训练速度快，内存占用低 |
| 小规模数据 | MLP | 模型简单，不易过拟合 |

#### 7.4.2 根据计算资源选择

| 计算资源 | 推荐模型 | 原因 |
|---------|---------|------|
| GPU可用 | TimeSeries（LSTM/GRU/Transformer） | GPU加速训练效果好 |
| 仅CPU | LightGBM/XGBoost | CPU训练速度快 |
| 内存有限 | LightGBM | 内存占用低 |
| 内存充足 | MLP/TimeSeries | 可以使用更大的模型 |

#### 7.4.3 根据性能要求选择

| 性能要求 | 推荐模型 | 原因 |
|---------|---------|------|
| 高精度 | TimeSeries（Transformer） | 模型能力强，精度高 |
| 快速训练 | LightGBM/XGBoost | 训练速度快 |
| 快速推理 | LightGBM/XGBoost | 推理速度快 |
| 可解释性 | LightGBM/XGBoost | 特征重要性可解释 |

### 7.5 使用示例

#### 7.5.1 只运行TimeSeries模型

**配置文件**：
```yaml
models:
  - name: "TimeSeriesModel"
    model_type: "TimeSeries"
    architecture: "LSTM"
    training_hyperparameters:
      n_epochs: 100
      lr: 2e-4
      early_stop: 10
      batch_size: 256
      weight_decay: 0.0001
```

**运行命令**：
```bash
python debug_tools/model_evolution_comparison.py --log_dir log/2026-01-13_06-56-49-446055
```

#### 7.5.2 运行多个模型对比

**配置文件**：
```yaml
models:
  - name: "TimeSeriesModel"
    model_type: "TimeSeries"
    architecture: "LSTM"
    training_hyperparameters:
      n_epochs: 100
      lr: 2e-4
      early_stop: 10
      batch_size: 256
      weight_decay: 0.0001
  
  - name: "TabularModel"
    model_type: "Tabular"
    architecture: "MLP"
    training_hyperparameters:
      n_epochs: 100
      lr: 2e-4
      early_stop: 10
      batch_size: 256
      weight_decay: 0.0001
  
  - name: "LightGBMModel"
    model_type: "Tabular"
    architecture: "LightGBM"
    training_hyperparameters:
      n_estimators: 1000
      learning_rate: 0.05
      early_stopping_rounds: 50
```

**运行命令**：
```bash
python debug_tools/model_evolution_comparison.py --log_dir log/2026-01-13_06-56-49-446055
```

### 7.6 总结

1. **文件说明**：所有文件都是新建的，不会影响现有的RD-Agent运行

2. **模型控制**：可以通过配置文件或脚本方法控制每次运行使用的模型

3. **支持的模型**：
   - RD-Agent完全支持：TimeSeries、Tabular
   - RD-Agent代码支持但未验证：Graph、XGBoost
   - Qlib支持：LightGBM、MLP、LSTM、GRU、Transformer、XGBoost、Linear

4. **推荐模型**：
   - TimeSeries模型：LSTM、GRU、Transformer
   - Tabular模型：MLP、LightGBM、XGBoost

5. **使用建议**：根据数据特征、计算资源和性能要求选择合适的模型

---

## 八、RD-Agent的SOTA评分标准分析

### 8.1 当前SOTA评分标准

#### 8.1.1 评分机制

RD-Agent的SOTA评分标准基于`feedback.decision`来确定，决策由`FactorFinalDecisionEvaluator`通过LLM评估得出。

**评估流程**：

1. **执行反馈（execution_feedback）**
   - 因子执行结果
   - 错误信息
   - 执行状态

2. **代码反馈（code_feedback）**
   - 代码质量
   - 可维护性
   - 代码规范性

3. **价值反馈（value_feedback）**
   - 因子预测能力（IC值）
   - 因子收益能力（年化收益率）
   - 因子风险控制（最大回撤）

4. **最终决策（final_decision）**
   - 由LLM综合评估
   - 返回True/False
   - 提供决策理由

#### 8.1.2 代码实现

**FactorFinalDecisionEvaluator** (`rdagent/components/coder/factor_coder/eva_utils.py`):

```python
class FactorFinalDecisionEvaluator(FactorEvaluator):
    def evaluate(
        self,
        target_task: FactorTask,
        execution_feedback: str,
        value_feedback: str,
        code_feedback: str,
        **kwargs,
    ) -> Tuple:
        # 使用LLM评估最终决策
        final_evaluation_dict = json.loads(
            api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            ),
        )
        final_decision = final_evaluation_dict["final_decision"]
        final_feedback = final_evaluation_dict["final_feedback"]

        final_decision = str(final_decision).lower() in ["true", "1"]
        return final_decision, final_feedback
```

**SOTA获取** (`rdagent/core/proposal.py`):

```python
def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
    """获取最后一个SOTA实验"""
    for experiment, feedback in self.hist[::-1]:
        if feedback.decision:
            return experiment.hypothesis, experiment

    return None, None
```

### 8.2 SOTA评分标准扩展建议

#### 8.2.1 问题背景

用户观察到：
- IC值从0.033873提升到0.051569（提升约52%）
- 最大回撤从-0.340559改善至-0.264722（改善约22%）
- 年化收益率从0.679081下降至0.502718（下降约26%）

**问题**：IC值和最大回撤有显著改善，但年化收益率下降，是否可以将IC值和最大回撤也加入到SOTA评分标准中？

#### 8.2.2 建议的评分标准

**方案1：多维度评分标准**

建议添加以下具体阈值标准：

| 指标 | 阈值 | 说明 |
|------|------|------|
| IC值提升 | > 50% | 相比SOTA提升超过50% |
| 最大回撤降低 | > 20% | 相比SOTA降低超过20% |
| 年化收益率 | ≥ 90% | 不低于SOTA的90% |
| IC值绝对值 | > 0.05 | IC值绝对值大于0.05 |
| 最大回撤绝对值 | < 0.3 | 最大回撤绝对值小于0.3 |

**评分逻辑**：

```python
def evaluate_sota_criteria(current_result, sota_result):
    """
    评估是否满足SOTA标准

    Args:
        current_result: 当前实验结果
        sota_result: SOTA实验结果

    Returns:
        is_sota: 是否为SOTA
        reasons: 评分理由
    """
    reasons = []
    is_sota = False

    # 提取关键指标
    current_ic = current_result.get("IC", 0)
    sota_ic = sota_result.get("IC", 0)
    current_return = current_result.get("annualized_return", 0)
    sota_return = sota_result.get("annualized_return", 0)
    current_drawdown = current_result.get("max_drawdown", 0)
    sota_drawdown = sota_result.get("max_drawdown", 0)

    # 1. IC值提升超过50%
    if sota_ic != 0:
        ic_improvement = (current_ic - sota_ic) / abs(sota_ic)
        if ic_improvement > 0.5:
            reasons.append(f"IC值提升{ic_improvement*100:.1f}%")
            is_sota = True

    # 2. 最大回撤降低超过20%
    if sota_drawdown != 0:
        drawdown_improvement = (sota_drawdown - current_drawdown) / abs(sota_drawdown)
        if drawdown_improvement > 0.2:
            reasons.append(f"最大回撤降低{drawdown_improvement*100:.1f}%")
            is_sota = True

    # 3. 年化收益率不低于SOTA的90%
    if sota_return != 0:
        return_ratio = current_return / sota_return
        if return_ratio >= 0.9:
            reasons.append(f"年化收益率为SOTA的{return_ratio*100:.1f}%")
        else:
            reasons.append(f"年化收益率仅为SOTA的{return_ratio*100:.1f}%，未达到90%标准")

    # 4. IC值绝对值大于0.05
    if abs(current_ic) > 0.05:
        reasons.append(f"IC值绝对值{abs(current_ic):.4f}大于0.05")
    else:
        reasons.append(f"IC值绝对值{abs(current_ic):.4f}未达到0.05标准")

    # 5. 最大回撤绝对值小于0.3
    if abs(current_drawdown) < 0.3:
        reasons.append(f"最大回撤绝对值{abs(current_drawdown):.4f}小于0.3")
    else:
        reasons.append(f"最大回撤绝对值{abs(current_drawdown):.4f}未达到0.3标准")

    return is_sota, reasons
```

**方案2：加权评分标准**

建议使用加权评分标准：

```python
def calculate_sota_score(current_result, sota_result):
    """
    计算SOTA评分（加权）

    Args:
        current_result: 当前实验结果
        sota_result: SOTA实验结果

    Returns:
        score: 加权评分
        details: 详细评分
    """
    # 提取关键指标
    current_ic = current_result.get("IC", 0)
    sota_ic = sota_result.get("IC", 0)
    current_return = current_result.get("annualized_return", 0)
    sota_return = sota_result.get("annualized_return", 0)
    current_drawdown = current_result.get("max_drawdown", 0)
    sota_drawdown = sota_result.get("max_drawdown", 0)

    # 计算各项得分
    ic_score = min(current_ic / max(abs(sota_ic), 0.05), 2.0)  # IC得分，最高2倍
    return_score = min(current_return / max(abs(sota_return), 0.1), 1.2)  # 收益得分，最高1.2倍
    drawdown_score = min(abs(sota_drawdown) / max(abs(current_drawdown), 0.1), 2.0)  # 回撤得分，最高2倍

    # 加权评分
    score = (
        ic_score * 0.4 +  # IC权重40%
        return_score * 0.3 +  # 收益权重30%
        drawdown_score * 0.3  # 回撤权重30%
    )

    details = {
        "ic_score": ic_score,
        "return_score": return_score,
        "drawdown_score": drawdown_score,
        "total_score": score,
    }

    return score, details
```

#### 8.2.3 实现方式

**方式1：修改FactorFinalDecisionEvaluator**

在`rdagent/components/coder/factor_coder/eva_utils.py`中添加阈值检查：

```python
class FactorFinalDecisionEvaluator(FactorEvaluator):
    def evaluate(
        self,
        target_task: FactorTask,
        execution_feedback: str,
        value_feedback: str,
        code_feedback: str,
        **kwargs,
    ) -> Tuple:
        # 1. 提取当前实验结果
        current_result = self._extract_result(value_feedback)

        # 2. 获取SOTA实验结果
        sota_result = self._get_sota_result()

        # 3. 评估SOTA标准
        if sota_result is not None:
            is_sota, reasons = evaluate_sota_criteria(current_result, sota_result)
            if is_sota:
                return True, f"满足SOTA标准: {'; '.join(reasons)}"

        # 4. 使用LLM评估最终决策
        final_evaluation_dict = json.loads(
            api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            ),
        )
        final_decision = final_evaluation_dict["final_decision"]
        final_feedback = final_evaluation_dict["final_feedback"]

        final_decision = str(final_decision).lower() in ["true", "1"]
        return final_decision, final_feedback
```

**方式2：修改prompts.yaml**

在`rdagent/components/coder/factor_coder/prompts.yaml`中添加评分标准说明：

```yaml
evaluator_final_decision_v1_user: |-
  --------------SOTA评分标准:---------------
  请按照以下标准评估因子是否为SOTA：

  1. IC值提升超过50%
  2. 最大回撤降低超过20%
  3. 年化收益率不低于SOTA的90%
  4. IC值绝对值大于0.05
  5. 最大回撤绝对值小于0.3

  如果满足以上任意一项，可以认为因子为SOTA。

  --------------Factor information:---------------
  {{ factor_information }}
  --------------Execution feedback:---------------
  {{ execution_feedback }}
  --------------Code feedback:---------------
  {{ code_feedback }}
  --------------Value feedback:---------------
  {{ value_feedback }}

  Please response the critic in the json format. Here is an example structure for the JSON output, please strictly follow the format:
  {
      "final_decision": True,
      "final_feedback": "The final feedback message",
  }
```

### 8.3 早期Loop的SOTA因子分析

#### 8.3.1 分析工具

创建了分析脚本`debug_tools/analyze_early_loops.py`，用于分析：

1. 最早的SOTA因子
2. 第一个回测成功的loop
3. 该loop对比的SOTA因子

**使用方法**：

```bash
python debug_tools/analyze_early_loops.py --log_dir <log_directory>
```

#### 8.3.2 分析结果

**注意**：需要用户提供具体的log目录才能进行分析。

**分析内容包括**：

1. **第一个SOTA因子**
   - Loop编号
   - Hypothesis
   - 因子名称
   - 关键指标（IC、年化收益率、最大回撤）

2. **第一个回测成功的loop**
   - Loop编号
   - Hypothesis
   - 因子名称
   - 关键指标（IC、年化收益率、最大回撤）

3. **第一个回测成功的loop对比的SOTA因子**
   - 该loop之前的所有SOTA因子
   - SOTA因子列表
   - 因子总数

### 8.4 总结

#### 8.4.1 当前SOTA评分标准

- 基于LLM综合评估
- 考虑执行反馈、代码反馈、价值反馈
- 没有明确的阈值标准

#### 8.4.2 建议的扩展方案

**方案1：多维度评分标准**
- IC值提升 > 50%
- 最大回撤降低 > 20%
- 年化收益率 ≥ 90%
- IC值绝对值 > 0.05
- 最大回撤绝对值 < 0.3

**方案2：加权评分标准**
- IC权重40%
- 收益权重30%
- 回撤权重30%

#### 8.4.3 实现方式

1. 修改`FactorFinalDecisionEvaluator`
2. 修改`prompts.yaml`
3. 添加阈值检查逻辑

#### 8.4.4 分析工具

创建了`debug_tools/analyze_early_loops.py`脚本，用于分析早期loop的SOTA因子信息。

---

## 九、RD-Agent的SOTA管理机制详解

### 9.1 SOTA因子的获取逻辑

#### 9.1.1 based_experiments的构建

RD-Agent通过`based_experiments`来维护SOTA因子清单，这个列表在每个实验创建时动态构建。

**因子实验的based_experiments构建** (`rdagent/scenarios/qlib/proposal/factor_proposal.py`):

```python
exp = QlibFactorExperiment(tasks, hypothesis=hypothesis)
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
]
```

**关键点**：
1. `based_experiments`的第一个元素是一个空的`QlibFactorExperiment(sub_tasks=[])`，这是**baseline**
2. 后面的元素是从`trace.hist`中获取的所有`decision=True`的`FactorExperiment`
3. `trace.hist`是RD-Agent维护的实验历史记录

**模型实验的based_experiments构建** (`rdagent/scenarios/qlib/proposal/model_proposal.py`):

```python
exp = QlibModelExperiment(tasks, hypothesis=hypothesis)
exp.based_experiments = [t[0] for t in trace.hist if t[1] and isinstance(t[0], ModelExperiment)]
```

**关键点**：
1. 模型实验的`based_experiments`不包含baseline
2. 只包含从`trace.hist`中获取的所有`decision=True`的`ModelExperiment`

#### 9.1.2 SOTA因子的提取

**因子实验的SOTA因子提取** (`rdagent/scenarios/qlib/developer/factor_runner.py`):

```python
if exp.based_experiments:
    SOTA_factor = None
    # Filter and retain only QlibFactorExperiment instances
    sota_factor_experiments_list = [
        base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
    ]
    if len(sota_factor_experiments_list) > 1:
        logger.info(f"SOTA factor processing ...")
        SOTA_factor = process_factor_data(sota_factor_experiments_list)
```

**关键点**：
1. 如果`sota_factor_experiments_list`的长度**大于1**，才会处理SOTA因子
2. 第一个元素是baseline，后面的才是真正的SOTA因子
3. `process_factor_data`函数会合并所有SOTA因子的数据

**模型实验的SOTA因子提取** (`rdagent/scenarios/qlib/developer/model_runner.py`):

```python
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
```

**关键点**：
1. 如果`sota_factor_experiments_list`的长度**大于等于1**，就会处理SOTA因子
2. 模型实验只要有SOTA因子就会使用

### 9.2 第一轮Loop的SOTA获取

#### 9.2.1 第一轮Loop的based_experiments

**第一轮因子实验**：
```python
# trace.hist为空
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])]
```

**第一轮模型实验**：
```python
# trace.hist为空
exp.based_experiments = []
```

#### 9.2.2 第一轮Loop的SOTA处理

**第一轮因子实验**：
- `based_experiments`长度为1（只有baseline）
- `sota_factor_experiments_list`长度为1（只有baseline）
- 条件`len(sota_factor_experiments_list) > 1`为False
- **不处理SOTA因子**，只使用新因子

**第一轮模型实验**：
- `based_experiments`长度为0
- `sota_factor_experiments_list`长度为0
- 条件`len(sota_factor_experiments_list) >= 1`为False
- **不处理SOTA因子**，只使用Alpha因子

#### 9.2.3 第一轮Loop的回测配置

**第一轮因子实验的回测配置** (`rdagent/scenarios/qlib/developer/factor_runner.py`):

```python
result, stdout = exp.experiment_workspace.execute(
    qlib_config_name=(
        f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors_dynamic.yaml"
    )
)
```

**关键点**：
- 如果`len(exp.based_experiments) == 0`，使用`conf_baseline.yaml`
- 否则使用`conf_combined_factors_dynamic.yaml`

**第一轮因子实验**：
- `len(exp.based_experiments) == 1`（只有baseline）
- 使用`conf_combined_factors_dynamic.yaml`
- 回测结果只包含新因子

### 9.3 跨任务的SOTA因子清单

#### 9.3.1 SOTA因子的维护机制

RD-Agent通过`trace.hist`维护跨任务的SOTA因子清单：

```python
# trace.hist的结构
trace.hist = [
    (experiment_1, feedback_1),  # Loop 0
    (experiment_2, feedback_2),  # Loop 1
    (experiment_3, feedback_3),  # Loop 2
    ...
]
```

**关键点**：
1. `trace.hist`是RD-Agent维护的实验历史记录
2. 每个元素是一个元组`(experiment, feedback)`
3. `feedback.decision`为True的实验会被认为是SOTA
4. 每个新实验创建时，会从`trace.hist`中提取所有SOTA实验

#### 9.3.2 SOTA因子的传递

**因子实验的SOTA因子传递**：

```python
# Loop 0: 第一个因子实验
exp0.based_experiments = [QlibFactorExperiment(sub_tasks=[])]
# trace.hist = [(exp0, feedback0)]

# Loop 1: 第二个因子实验
exp1.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [exp0]  # 如果feedback0.decision为True
# trace.hist = [(exp0, feedback0), (exp1, feedback1)]

# Loop 2: 第三个因子实验
exp2.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [exp0, exp1]  # 如果feedback0.decision和feedback1.decision都为True
# trace.hist = [(exp0, feedback0), (exp1, feedback1), (exp2, feedback2)]
```

**模型实验的SOTA因子传递**：

```python
# Loop 0: 第一个模型实验
exp0.based_experiments = []
# trace.hist = [(exp0, feedback0)]

# Loop 1: 第二个模型实验
exp1.based_experiments = [exp0]  # 如果feedback0.decision为True
# trace.hist = [(exp0, feedback0), (exp1, feedback1)]
```

#### 9.3.3 SOTA因子的去重

**因子实验的SOTA因子去重** (`rdagent/scenarios/qlib/proposal/factor_proposal.py`):

```python
unique_tasks = []
for task in tasks:
    duplicate = False
    for based_exp in exp.based_experiments:
        if isinstance(based_exp, QlibModelExperiment):
            continue
        for sub_task in based_exp.sub_tasks:
            if task.factor_name == sub_task.factor_name:
                duplicate = True
                break
        if duplicate:
            break
    if not duplicate:
        unique_tasks.append(task)

exp.tasks = unique_tasks
```

**关键点**：
1. 新因子会与`based_experiments`中的所有因子进行对比
2. 如果因子名称重复，会被认为是重复因子
3. 重复因子会被过滤掉

### 9.4 回测指标数据的维护

#### 9.4.1 回测指标数据的存储

RD-Agent通过`experiment.result`存储回测指标数据：

```python
# experiment.result的结构
experiment.result = pd.DataFrame({
    "IC": [0.051569],
    "1day.excess_return_with_cost.annualized_return": [0.502718],
    "1day.excess_return_with_cost.max_drawdown": [-0.264722],
    ...
})
```

**关键点**：
1. `experiment.result`是一个DataFrame
2. 包含IC、年化收益率、最大回撤等指标
3. 每个实验的回测结果都存储在`experiment.result`中

#### 9.4.2 回测指标数据的获取

**SOTA因子的回测指标数据获取**：

```python
# 获取SOTA因子的回测指标数据
sota_results = []
for exp in sota_factor_experiments_list:
    if exp.result is not None:
        result_df = pd.DataFrame(exp.result)
        sota_results.append(result_df)

# 合并所有SOTA因子的回测指标数据
if sota_results:
    combined_sota_results = pd.concat(sota_results, axis=1)
```

**关键点**：
1. 每个SOTA因子的回测指标数据存储在`exp.result`中
2. 可以通过`based_experiments`获取所有SOTA因子的回测指标数据
3. 可以合并所有SOTA因子的回测指标数据进行对比

### 9.5 总结

#### 9.5.1 SOTA因子的获取逻辑

1. **based_experiments的构建**
   - 因子实验：`[baseline] + [所有decision=True的FactorExperiment]`
   - 模型实验：`[所有decision=True的ModelExperiment]`

2. **SOTA因子的提取**
   - 因子实验：`len(sota_factor_experiments_list) > 1`
   - 模型实验：`len(sota_factor_experiments_list) >= 1`

3. **第一轮Loop的SOTA处理**
   - 因子实验：不处理SOTA因子，只使用新因子
   - 模型实验：不处理SOTA因子，只使用Alpha因子

#### 9.5.2 跨任务的SOTA因子清单

1. **维护机制**
   - 通过`trace.hist`维护实验历史记录
   - 每个新实验创建时，从`trace.hist`中提取所有SOTA实验
   - 自动去重，避免重复因子

2. **传递机制**
   - 因子实验：`based_experiments`包含baseline和所有SOTA因子
   - 模型实验：`based_experiments`包含所有SOTA模型

#### 9.5.3 回测指标数据的维护

1. **存储机制**
   - 每个实验的回测指标数据存储在`experiment.result`中
   - 可以通过`based_experiments`获取所有SOTA因子的回测指标数据

2. **获取机制**
   - 遍历`based_experiments`获取每个SOTA因子的回测指标数据
   - 合并所有SOTA因子的回测指标数据进行对比

#### 9.5.4 关键发现

1. **RD-Agent维护跨任务的SOTA因子清单**
   - 通过`trace.hist`维护实验历史记录
   - 每个新实验创建时，从`trace.hist`中提取所有SOTA实验
   - 不需要额外的存储机制，完全基于内存中的`trace.hist`

2. **第一轮Loop没有SOTA因子**
   - `trace.hist`为空
   - `based_experiments`只包含baseline或为空
   - 不处理SOTA因子，只使用新因子或Alpha因子

3. **SOTA因子的去重**
   - 新因子会与`based_experiments`中的所有因子进行对比
   - 如果因子名称重复，会被认为是重复因子
   - 重复因子会被过滤掉

4. **回测指标数据的获取**
   - 每个实验的回测指标数据存储在`experiment.result`中
   - 可以通过`based_experiments`获取所有SOTA因子的回测指标数据
   - 可以合并所有SOTA因子的回测指标数据进行对比

---

## 十、RD-Agent的SOTA因子数量与性能分析

### 10.1 问题背景

用户提出了一个关键的性能问题：

**假设场景**：
- 已经运行过20个RD-Agent的任务
- 每个任务中都新增2个SOTA因子
- 第21次任务时，初始清单里面就已经有40个SOTA因子了？

**关键问题**：
1. 分析目前trace.hist中一共有多少个因子？
2. 如果是这样为何每次任务执行时间没有快速增长？
3. 如果SOTA因子数量一直增长，训练时间应该一直线性增长？

### 10.2 trace.hist的实际存储内容

#### 10.2.1 trace.hist的结构

```python
# trace.hist的结构
trace.hist = [
    (experiment_1, feedback_1),  # Loop 0
    (experiment_2, feedback_2),  # Loop 1
    (experiment_3, feedback_3),  # Loop 2
    ...
]
```

**关键点**：
1. **trace.hist存储的是实验对象的引用，不是实际的因子数据**
2. 每个元素是一个元组`(experiment, feedback)`
3. `experiment`包含：
   - `hypothesis`: 实验假设
   - `sub_tasks`: 任务列表（包含因子信息）
   - `sub_workspace_list`: 工作空间列表（包含因子代码）
   - `result`: 回测结果
4. `feedback`包含：
   - `decision`: 决策（True/False）
   - `reason`: 决策理由

#### 10.2.2 因子数据的存储位置

**因子数据的存储位置**：
- `experiment.sub_workspace_list`: 存储因子代码
- `experiment.result`: 存储回测结果
- **因子数据通过execute()方法动态生成，不存储在trace.hist中**

**关键发现**：
- trace.hist**不存储实际的因子数据**
- 因子数据通过`implementation.execute()`方法动态生成
- 每次处理SOTA因子时，都需要重新计算因子数据

### 10.3 SOTA因子的实际处理方式

#### 10.3.1 SOTA因子的处理流程

**SOTA因子的处理流程**：

```python
# 1. based_experiments构建
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
]

# 2. SOTA因子提取
sota_factor_experiments_list = [
    base_exp for base_exp in exp.based_experiments 
    if isinstance(base_exp, QlibFactorExperiment)
]

# 3. SOTA因子处理
if len(sota_factor_experiments_list) > 1:
    SOTA_factor = process_factor_data(sota_factor_experiments_list)

# 4. 新因子处理
new_factors = process_factor_data(exp)

# 5. 因子合并
combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
```

**关键点**：
1. `based_experiments`包含所有`decision=True`的实验
2. `process_factor_data()`会遍历所有SOTA因子实验
3. 对每个因子调用`execute()`方法生成因子数据
4. 合并所有因子数据

#### 10.3.2 process_factor_data()函数

**process_factor_data()函数** (`rdagent/scenarios/qlib/developer/utils.py`):

```python
def process_factor_data(exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
    """
    Process and combine factor data from experiment implementations.

    Args:
        exp (ASpecificExp): The experiment containing factor data.

    Returns:
        pd.DataFrame: Combined factor data without NaN values.
    """
    if isinstance(exp_or_list, QlibFactorExperiment):
        exp_or_list = [exp_or_list]
    factor_dfs = []

    # Collect all exp's dataframes
    for exp in exp_or_list:
        if isinstance(exp, QlibFactorExperiment):
            if len(exp.sub_tasks) > 0:
                # Iterate over sub-implementations and execute them to get each factor data
                message_and_df_list = multiprocessing_wrapper(
                    [
                        (implementation.execute, ("All",))
                        for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
                        if implementation and fb
                    ],  # only execute successfully feedback
                    n=RD_AGENT_SETTINGS.multi_proc_n,
                )
                for message, df in message_and_df_list:
                    if df is not None and "datetime" in df.index.names:
                        factor_dfs.append(df)

    # Combine all successful factor data
    if factor_dfs:
        return pd.concat(factor_dfs, axis=1)
    else:
        raise FactorEmptyError("No valid factor data found to merge")
```

**关键点**：
1. 遍历所有SOTA因子实验
2. 对每个因子调用`execute()`方法生成因子数据
3. 使用`multiprocessing_wrapper()`多进程并行处理
4. 合并所有因子数据

### 10.4 为何执行时间没有线性增长

#### 10.4.1 可能的原因

**原因1：多进程并行处理**
- `process_factor_data()`使用`multiprocessing_wrapper()`多进程并行处理
- 多个因子可以同时计算，减少总处理时间

**原因2：因子数据缓存机制**
- RD-Agent可能使用`cache_with_pickle`缓存因子数据
- 缓存机制可以避免重复计算

**原因3：因子去重机制**
- `deduplicate_new_factors()`函数会过滤掉重复因子
- 重复因子不会被重复计算

**原因4：SOTA因子数量限制机制**
- Data Science场景有`max_sota_retrieved_num`限制
- Qlib场景可能没有明确的限制，但可能有隐含限制

**原因5：因子数据预处理**
- 因子数据可能已经被预处理并存储
- 避免重复计算

#### 10.4.2 性能瓶颈分析

**可能的性能瓶颈**：

1. **process_factor_data()函数**
   - 遍历所有SOTA因子实验
   - 对每个因子调用`execute()`方法
   - 合并所有因子数据

2. **因子数据生成**
   - 每次都需要重新计算因子数据
   - 可能没有缓存机制

3. **因子数据合并**
   - 合并大量因子数据可能耗时
   - 可能需要优化合并算法

### 10.5 SOTA因子数量限制机制

#### 10.5.1 Data Science场景的限制

**Data Science场景的SOTA因子数量限制**：

```python
# rdagent/app/data_science/conf.py
max_sota_retrieved_num: int = 10
"""The maximum number of SOTA experiments to retrieve in a LLM call"""
```

**关键点**：
- Data Science场景有明确的SOTA数量限制
- 默认值为10
- 用于限制LLM调用的上下文窗口

#### 10.5.2 Qlib场景的限制

**Qlib场景的SOTA因子数量限制**：

**关键发现**：
- Qlib场景**没有明确的SOTA因子数量限制**
- `based_experiments`包含所有`decision=True`的实验
- SOTA因子数量会随着实验次数线性增长

**潜在问题**：
- 如果运行20个任务，每个任务新增2个SOTA因子
- 第21次任务时，`based_experiments`中可能有40个SOTA因子
- `process_factor_data()`需要处理40个SOTA因子

### 10.6 性能影响分析

#### 10.6.1 SOTA因子数量对性能的影响

**SOTA因子数量对性能的影响**：

1. **因子数据生成时间**
   - 每个因子需要调用`execute()`方法
   - 生成时间与因子数量线性相关
   - 多进程并行处理可以缓解

2. **因子数据合并时间**
   - 合并时间与因子数量相关
   - 可能是O(n)或O(n log n)复杂度

3. **内存占用**
   - 内存占用与因子数量相关
   - 大量因子可能导致内存不足

#### 10.6.2 实际性能表现

**实际性能表现**：

**假设场景**：
- 运行20个任务，每个任务新增2个SOTA因子
- 第21次任务时，`based_experiments`中可能有40个SOTA因子

**预期性能**：
- 如果没有优化，执行时间应该线性增长
- 实际上执行时间没有线性增长

**可能的原因**：
1. **多进程并行处理**：`multiprocessing_wrapper()`使用多进程并行处理
2. **因子数据缓存**：`cache_with_pickle`缓存因子数据
3. **因子去重机制**：`deduplicate_new_factors()`过滤重复因子
4. **因子数量隐含限制**：可能有隐含的SOTA因子数量限制

### 10.7 建议优化方案

#### 10.7.1 添加SOTA因子数量限制

**建议**：
- 在Qlib场景中添加SOTA因子数量限制
- 例如：只保留最近N个SOTA因子实验

**实现方式**：

```python
# rdagent/scenarios/qlib/proposal/factor_proposal.py
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
][-MAX_SOTA_FACTORS:]  # 只保留最近N个SOTA因子实验
```

#### 10.7.2 添加因子数据缓存

**建议**：
- 缓存已计算的因子数据
- 避免重复计算

**实现方式**：

```python
# rdagent/scenarios/qlib/developer/utils.py
@cache_with_pickle(qlib_factor_cache_key, CachedRunner.assign_cached_result)
def process_factor_data(exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
    # ... 现有代码 ...
```

#### 10.7.3 优化因子数据合并算法

**建议**：
- 使用更高效的合并算法
- 减少内存占用

**实现方式**：

```python
# 使用更高效的合并算法
combined_factors = pd.concat(factor_dfs, axis=1, copy=False)
```

#### 10.7.4 添加因子去重机制

**建议**：
- 在`based_experiments`构建时就进行去重
- 避免重复计算相同因子

**实现方式**：

```python
# rdagent/scenarios/qlib/proposal/factor_proposal.py
unique_tasks = []
for task in tasks:
    duplicate = False
    for based_exp in exp.based_experiments:
        if isinstance(based_exp, QlibModelExperiment):
            continue
        for sub_task in based_exp.sub_tasks:
            if task.factor_name == sub_task.factor_name:
                duplicate = True
                break
        if duplicate:
            break
    if not duplicate:
        unique_tasks.append(task)

exp.tasks = unique_tasks
```

#### 10.7.5 添加性能监控

**建议**：
- 记录每个步骤的执行时间
- 识别性能瓶颈

**实现方式**：

```python
import time

# 记录process_factor_data()的执行时间
start_time = time.time()
SOTA_factor = process_factor_data(sota_factor_experiments_list)
end_time = time.time()
logger.info(f"process_factor_data()执行时间: {end_time - start_time:.2f}秒")
```

### 10.8 总结

#### 10.8.1 关键发现

1. **trace.hist存储的是实验对象的引用，不是实际的因子数据**
   - 因子数据通过`execute()`方法动态生成
   - 每次处理SOTA因子时，都需要重新计算因子数据

2. **Qlib场景没有明确的SOTA因子数量限制**
   - SOTA因子数量会随着实验次数线性增长
   - 可能导致性能问题

3. **执行时间没有线性增长的可能原因**
   - 多进程并行处理
   - 因子数据缓存机制
   - 因子去重机制
   - 因子数量隐含限制

4. **性能瓶颈**
   - `process_factor_data()`函数
   - 因子数据生成
   - 因子数据合并

#### 10.8.2 建议优化方案

1. **添加SOTA因子数量限制**
   - 限制`based_experiments`中的SOTA因子数量
   - 例如：只保留最近N个SOTA因子实验

2. **添加因子数据缓存**
   - 缓存已计算的因子数据
   - 避免重复计算

3. **优化因子数据合并算法**
   - 使用更高效的合并算法
   - 减少内存占用

4. **添加因子去重机制**
   - 在`based_experiments`构建时就进行去重
   - 避免重复计算相同因子

5. **添加性能监控**
   - 记录每个步骤的执行时间
   - 识别性能瓶颈

#### 10.8.3 分析工具

创建了`debug_tools/analyze_sota_factor_performance.py`脚本，用于分析SOTA因子数量和性能问题。

**使用方法**：

```bash
python debug_tools/analyze_sota_factor_performance.py --log_dir <log_directory>
```

**分析内容**：
1. 统计SOTA因子数量
2. 分析第21次任务的based_experiments
3. 分析SOTA因子的实际处理方式
4. 分析为何执行时间没有线性增长
5. 查找SOTA因子数量限制机制
6. 分析性能瓶颈
7. 建议优化方案

---

## 十一、RD-Agent的SOTA因子库与回测机制深入分析

### 11.1 关键问题

用户提出了几个关键问题：

1. 跨loop的SOTA因子库是否存在？
2. 是否记录本机运行过的所有SOTA因子？
3. 分析的这个task没有产生SOTA因子，是否说明每次loop都是loop中研发的因子，与alpha因子组合做回测，但回测指标数据没有超过SOTA因子中的指标，所以没有加入SOTA？
4. task中只是跟过去所有历史中SOTA的回测记录做对比，并未与之前的SOTA因子做组合回测？

### 11.2 跨loop的SOTA因子库是否存在？

✅ **是的，`trace.hist`就是跨loop的SOTA因子库**

**证据**：
```python
# rdagent/scenarios/qlib/proposal/factor_proposal.py
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
]
```

**关键点**：
- `trace.hist`存储了所有实验的历史记录
- `based_experiments`是从`trace.hist`中提取所有`decision=True`的实验
- 这确实是一个跨loop的SOTA因子库

**实际统计结果**：
```
总实验数量: 6
总SOTA因子数量: 0
SOTA实验数量: 0
```

这说明当前task没有SOTA因子，因为所有6个实验的Decision都是False。

### 11.2 是否记录本机运行过的所有SOTA因子？

✅ **是的，`trace.hist`记录了所有实验的历史记录**

**但是，SOTA因子的作用域仅限于当前task，不跨task累积**

**证据**：

```python
# rdagent/core/proposal.py:139-141
def __init__(self, scen: ASpecificScen, knowledge_base: ASpecificKB | None = None) -> None:
    self.scen: ASpecificScen = scen
    # BEGIN: graph structure -------------------------
    self.hist: list[Trace.NodeType] = (
        []
    )  # List of tuples containing experiments and their feedback, organized over time.
```

```python
# rdagent/components/workflow/rd_loop.py:28-42
class RDLoop(LoopBase, metaclass=LoopMeta):
    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()
        # ...
        self.trace = Trace(scen=scen)  # 每次task启动都创建新的Trace对象
        super().__init__()
```

```python
# rdagent/app/qlib_rd_loop/quant.py:34-66
class QuantRDLoop(RDLoop):
    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()
        # ...
        self.trace = QuantTrace(scen=scen)  # 每次task启动都创建新的QuantTrace对象
        super(RDLoop, self).__init__()
```

**关键点**：

1. **每个task启动时都会创建新的Trace对象**
   - `trace.hist` 初始化为空列表 `[]`
   - 不加载任何历史数据

2. **SOTA因子的作用域**
   - ✅ **跨loop累积**：同一个task内的不同loop会共享SOTA因子
   - ❌ **不跨task累积**：不同task之间不共享SOTA因子

3. **trace.hist的结构**
   - `trace.hist`记录了所有实验的历史记录
   - 但它存储的是实验对象的引用，不是实际的因子数据
   - 因子数据通过`execute()`方法动态生成

**实际统计结果**：

```
Loop 0: Decision: False
Loop 1: Decision: False
Loop 2: Decision: False
Loop 3: Decision: False
Loop 4: Decision: False
Loop 5: Decision: False
```

这说明所有实验的Decision都是False，因此没有实验被加入SOTA。

**重要结论**：

**假设运行20个RD-Agent任务，每个任务都新增2个SOTA因子**：

- **Task 1**：产生2个SOTA因子 → `trace.hist` 包含这2个SOTA实验
- **Task 2**：产生2个SOTA因子 → `trace.hist` 只包含Task 2的2个SOTA实验（不包含Task 1的）
- **Task 3**：产生2个SOTA因子 → `trace.hist` 只包含Task 3的2个SOTA实验（不包含Task 1和2的）
- ...
- **Task 20**：产生2个SOTA因子 → `trace.hist` 只包含Task 20的2个SOTA实验（不包含Task 1-19的）
- **Task 21**：启动时 `trace.hist` 为空 → 第一个loop只使用新因子+alpha因子

**第21次任务时，初始清单里面不会有40个SOTA因子，只会从空开始！**

这解释了为什么每次任务执行时间没有快速增长：
- SOTA因子数量不会跨task累积
- 每个task都是从空开始
- 没有跨task的持久化机制

### 11.4 每次loop的因子组合回测逻辑

**每次loop的因子组合回测逻辑**：

```python
# rdagent/scenarios/qlib/developer/factor_runner.py
if exp.based_experiments:
    SOTA_factor = None
    # Filter and retain only QlibFactorExperiment instances
    sota_factor_experiments_list = [
        base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
    ]
    if len(sota_factor_experiments_list) > 1:
        logger.info(f"SOTA factor processing ...")
        SOTA_factor = process_factor_data(sota_factor_experiments_list)

    logger.info(f"New factor processing ...")
    # Process the new factors data
    try:
        new_factors = process_factor_data(exp)
    except FactorEmptyError as e:
        # ...

    # Combine the SOTA factor and new factors if SOTA factor exists
    if SOTA_factor is not None and not SOTA_factor.empty:
        new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
        if new_factors.empty:
            raise FactorEmptyError(
                "The factors generated in this round are highly similar to the previous factors. Please change the direction for creating new factors."
            )
        combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
    else:
        combined_factors = new_factors
```

**关键点**：
1. **SOTA因子处理**：如果`sota_factor_experiments_list`长度大于1，处理SOTA因子
2. **新因子处理**：总是处理新因子
3. **因子合并**：如果SOTA因子存在，合并SOTA因子和新因子；否则只使用新因子
4. **回测配置选择**：根据`based_experiments`长度选择回测配置

**关键条件：`len(sota_factor_experiments_list) > 1`**

这个条件决定了是否加载历史SOTA因子进行组合回测。

#### 11.4.1 第一个loop的因子组合（没有历史SOTA因子）

**场景**：假设一个task中的第一个loop，研发成功1个因子

**代码执行流程**：

```python
# based_experiments的构建
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
]
# 第一个loop时，trace.hist为空
# based_experiments = [QlibFactorExperiment(sub_tasks=[])]
```

```python
# SOTA因子处理
sota_factor_experiments_list = [
    base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
]
# sota_factor_experiments_list = [QlibFactorExperiment(sub_tasks=[])]

if len(sota_factor_experiments_list) > 1:
    # 条件：len(sota_factor_experiments_list) = 1
    # 结果：False，不执行SOTA因子处理
    SOTA_factor = process_factor_data(sota_factor_experiments_list)
```

```python
# 因子合并
if SOTA_factor is not None and not SOTA_factor.empty:
    # 条件：SOTA_factor = None
    # 结果：False，不合并
    combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
else:
    # 执行这里
    combined_factors = new_factors
```

**结果**：
- ✅ **只使用新因子和alpha因子进行回测**
- ❌ **不加载历史SOTA因子**
- ❌ **不与历史SOTA因子组合**

**原因**：
- 第一个loop时，`based_experiments`只包含baseline（长度=1）
- 条件 `len(sota_factor_experiments_list) > 1` 不满足
- 因此不会加载历史SOTA因子

#### 11.4.2 后续loop的因子组合（有历史SOTA因子）

**场景**：假设之前有2个SOTA因子实验，当前loop研发成功1个因子

**代码执行流程**：

```python
# based_experiments的构建
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
]
# 假设trace.hist中有2个SOTA实验
# based_experiments = [baseline, SOTA_exp1, SOTA_exp2]
```

```python
# SOTA因子处理
sota_factor_experiments_list = [
    base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
]
# sota_factor_experiments_list = [baseline, SOTA_exp1, SOTA_exp2]

if len(sota_factor_experiments_list) > 1:
    # 条件：len(sota_factor_experiments_list) = 3
    # 结果：True，执行SOTA因子处理
    SOTA_factor = process_factor_data(sota_factor_experiments_list)
```

```python
# 因子合并
if SOTA_factor is not None and not SOTA_factor.empty:
    # 条件：SOTA_factor 不为空
    # 结果：True，合并SOTA因子和新因子
    new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
    combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
else:
    # 不执行这里
    combined_factors = new_factors
```

**结果**：
- ✅ **使用所有SOTA因子 + 新因子 + alpha因子进行回测**
- ✅ **加载历史SOTA因子**
- ✅ **与历史SOTA因子组合**

**原因**：
- 后续loop时，`based_experiments`包含baseline+所有历史SOTA实验
- 条件 `len(sota_factor_experiments_list) > 1` 满足
- 会加载所有历史SOTA因子

#### 11.4.3 回测配置选择

**回测配置选择**：
```python
result, stdout = exp.experiment_workspace.execute(
    qlib_config_name=(
        f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors_dynamic.yaml"
    )
)
```

**关键点**：
- 如果`based_experiments`为空，使用`conf_baseline.yaml`
- 如果`based_experiments`不为空，使用`conf_combined_factors_dynamic.yaml`

**注意**：
- 第一个loop时，`based_experiments`不为空（包含baseline），所以使用`conf_combined_factors_dynamic.yaml`
- 但由于没有SOTA因子，`combined_factors`只包含新因子和alpha因子

#### 11.4.4 核心机制总结

**第一个loop**：
- **回测因子**：新因子 + alpha因子
- **对比对象**：baseline
- **目的**：建立基准，判断是否成为第一个SOTA因子

**后续loop（有历史SOTA因子）**：
- **回测因子**：所有SOTA因子 + 新因子 + alpha因子
- **对比对象**：历史SOTA因子的回测指标
- **目的**：判断新因子是否优于现有SOTA，是否需要更新SOTA

**这个设计确保了**：
- 第一个loop有基准参考（baseline）
- 后续loop能利用历史最佳因子
- 避免第一个loop就没有任何参考导致评估困难
- SOTA因子是累积的，每次loop都会利用所有历史SOTA因子

### 11.5 回测指标对比机制

**回测指标对比机制**：

从统计结果来看：
```
Loop 0: Decision: False
Loop 1: Decision: False
Loop 2: Decision: False
Loop 3: Decision: False
Loop 4: Decision: False
Loop 5: Decision: False
```

**关键点**：
- 每次loop的回测指标数据没有超过SOTA因子中的指标
- 因此没有加入SOTA
- 这说明每次loop都是loop中研发的因子，与alpha因子组合做回测

**SOTA因子的判断**：
```python
# rdagent/scenarios/qlib/proposal/factor_proposal.py
exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
    t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
]
```

**关键点**：
- 只有`feedback.decision`为True的实验才会被加入SOTA
- SOTA因子的判断是基于回测指标数据的对比

### 11.6 task中是否与之前的SOTA因子做组合回测？

✅ **是的，task中会与之前的SOTA因子做组合回测**

**证据**：
```python
# rdagent/scenarios/qlib/developer/factor_runner.py
if SOTA_factor is not None and not SOTA_factor.empty:
    new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
    combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
else:
    combined_factors = new_factors
```

**关键点**：
- 如果SOTA因子存在，会将SOTA因子和新因子合并
- 然后使用合并后的因子进行回测
- 这说明task中会与之前的SOTA因子做组合回测

**但是，从统计结果来看，当前task没有SOTA因子**：
- 所有6个实验的Decision都是False
- `sota_factor_experiments_list`长度为0
- 因此不会处理SOTA因子，只使用新因子和alpha因子

### 11.7 实际统计结果

**实际统计结果**：

```
总实验数量: 6
总SOTA因子数量: 0
SOTA实验数量: 0
```

**实验详情**：

**Loop 0**：
- 实验类型: QlibFactorExperiment
- 因子数量: 3
- Decision: False

**Loop 1**：
- 实验类型: QlibFactorExperiment
- 因子数量: 3
- Decision: False

**Loop 2**：
- 实验类型: QlibFactorExperiment
- 因子数量: 3
- Decision: False

**Loop 3**：
- 实验类型: QlibFactorExperiment
- 因子数量: 3
- Decision: False

**Loop 4**：
- 实验类型: QlibFactorExperiment
- 因子数量: 3
- Decision: False

**Loop 5**：
- 实验类型: QlibFactorExperiment
- 因子数量: 3
- Decision: False

**下一轮task的SOTA因子使用情况**：

```
下一轮task的based_experiments中SOTA实验数量: 0
based_experiments中总因子数量: 0
下一轮task第一个因子研发成功的loop会使用 0 个SOTA因子组合加上新增加的因子来做回测
```

### 11.8 总结

**问题回答**：

1. **跨loop的SOTA因子库是否存在？**
   - ✅ 是的，`trace.hist`就是跨loop的SOTA因子库

2. **是否记录本机运行过的所有SOTA因子？**
   - ✅ 是的，`trace.hist`记录了所有实验的历史记录

3. **分析的这个task没有产生SOTA因子，是否说明每次loop都是loop中研发的因子，与alpha因子组合做回测，但回测指标数据没有超过SOTA因子中的指标，所以没有加入SOTA？**
   - ✅ 是的，从统计结果来看，所有6个实验的Decision都是False

4. **task中只是跟过去所有历史中SOTA的回测记录做对比，并未与之前的SOTA因子做组合回测？**
   - ❌ 不是的，task中会与之前的SOTA因子做组合回测

**关键发现**：
- 当前task没有SOTA因子，因此不会与之前的SOTA因子做组合回测
- 只使用新因子和alpha因子做回测
- 这是因为所有6个实验的Decision都是False，没有实验被加入SOTA

**SOTA因子的处理逻辑**：
1. 如果`sota_factor_experiments_list`长度大于1，处理SOTA因子
2. 合并SOTA因子和新因子
3. 使用合并后的因子进行回测
4. 如果没有SOTA因子，只使用新因子和alpha因子

**SOTA因子的判断机制**：
1. 基于回测指标数据的对比
2. 只有`feedback.decision`为True的实验才会被加入SOTA
3. SOTA因子的判断是通过LLM评估的

---

## 十二、跨TASK SOTA因子收集与组合的价值分析

### 12.1 技术可行性分析

**✅ 完全可行**

**证据**：

1. **RD-Agent已有完善的因子去重机制**

```python
# rdagent/scenarios/qlib/developer/factor_runner.py:63-84
def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
    # calculate the IC between each column of SOTA_feature and new_feature
    # if the IC is larger than a threshold, remove the new_feature column
    concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
    IC_max = (
        concat_feature.groupby("datetime")
        .parallel_apply(
            lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
        )
        .mean()
    )
    IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
    IC_max = IC_max.unstack().max(axis=0)
    # Relax de-duplication: only drop nearly-identical new factors.
    keep_idx = IC_max[IC_max < 0.995].index
```

2. **RD-Agent已有LLM去重机制**

```python
# rdagent/scenarios/qlib/factor_experiment_loader/pdf_loader.py:453-512
def __deduplicate_factor_dict(factor_dict: dict[str, dict[str, str]]) -> list[list[str]]:
    # 使用embedding和kmeans聚类来检测重复因子
    embeddings = APIBackend.create_embedding(full_str_list)
    # K-means聚类分组
    for k in range(...):
        kmeans_index_group = __kmeans_embeddings(embeddings=embeddings, k=k)
    # LLM判断因子是否重复
    result_list = multiprocessing_wrapper([
        (__check_factor_duplication_simulate_json_mode, (factor_df.loc[factor_name_group, :],))
        for factor_name_group in factor_name_groups
    ])
```

3. **可以编写脚本收集所有历史TASK的SOTA因子**

```python
# 伪代码示例
def collect_all_sota_factors(log_dir):
    all_sota_factors = []
    for task_dir in os.listdir(log_dir):
        session_file = os.path.join(task_dir, "session.pkl")
        with open(session_file, "rb") as f:
            session = Unpickler(f).load()

        # 提取所有SOTA因子实验
        sota_experiments = [
            exp for exp, feedback in session.trace.hist
            if feedback.decision and isinstance(exp, QlibFactorExperiment)
        ]

        # 处理因子数据
        if sota_experiments:
            sota_factors = process_factor_data(sota_experiments)
            all_sota_factors.append(sota_factors)

    # 合并所有SOTA因子
    combined_sota_factors = pd.concat(all_sota_factors, axis=1)

    # 去重
    deduplicated_factors = deduplicate_factors_by_llm(combined_sota_factors)

    return deduplicated_factors
```

### 12.2 量化投资理论分析

#### 12.2.1 因子数量过多的问题

**❌ 因子数量过多不是最佳实践**

**量化投资中的经典问题**：

1. **过度拟合（Overfitting）**
   - 因子数量过多会导致模型过度拟合历史数据
   - 样本外表现会显著下降
   - 量化投资中，IC（信息系数）会虚高

2. **多重共线性（Multicollinearity）**
   - 因子间高度相关性会降低模型稳定性
   - 回归系数不稳定，难以解释
   - 模型对噪声敏感

3. **维度灾难（Curse of Dimensionality）**
   - 高维空间中数据稀疏性问题
   - 需要更多的样本来训练模型
   - 计算复杂度呈指数增长

4. **计算成本**
   - 训练时间显著增加
   - 存储需求增加
   - 推理速度变慢

#### 12.2.2 量化投资最佳实践

**✅ 因子选择和组合策略**

1. **因子筛选（Factor Selection）**
   - 使用IC、IR等指标评估因子质量
   - 只保留表现良好的因子
   - 定期评估因子有效性

2. **因子正交化（Factor Orthogonalization）**
   - 使用PCA、PLS等方法降低因子相关性
   - 提取主成分因子
   - 减少冗余信息

3. **因子组合（Factor Combination）**
   - 选择互补性强的因子组合
   - 避免因子间高度相关
   - 控制因子数量在合理范围

4. **动态调整（Dynamic Adjustment）**
   - 根据市场环境调整因子权重
   - 定期重新评估因子有效性
   - 淘汰表现不佳的因子

#### 12.2.3 因子数量与效果的关系

**非线性关系**：

```
因子数量 vs. 模型效果

效果
  ^
  |      /\
  |     /  \
  |    /    \
  |   /      \
  |  /        \
  | /          \
  |/            \
  +-----------------> 因子数量
    0   10   20   30   40
```

**关键发现**：
- 因子数量与模型效果呈非线性关系
- 存在最优因子数量区间（通常10-30个因子）
- 超过最优区间后，效果反而下降

### 12.3 实际价值分析

#### 12.3.1 潜在收益

**✅ 有价值，但需要谨慎**

**潜在收益**：

1. **扩大因子搜索空间**
   - 跨TASK收集可以获取更多样化的因子
   - 不同TASK可能发现不同类型的因子
   - 增加因子多样性

2. **提高模型鲁棒性**
   - 更多因子可以提高模型的适应性
   - 降低对单一因子的依赖
   - 提高样本外表现

3. **发现隐藏模式**
   - 不同TASK的因子可能捕捉不同市场特征
   - 组合可能发现新的投资机会
   - 提高策略的多样性

#### 12.3.2 潜在风险

**❌ 需要谨慎处理**

**潜在风险**：

1. **因子质量参差不齐**
   - 不同TASK的因子质量可能差异很大
   - 低质量因子会降低整体效果
   - 需要严格的因子筛选

2. **因子间高度相关**
   - 不同TASK可能产生相似的因子
   - 需要严格的去重机制
   - 避免冗余因子

3. **过度拟合风险**
   - 因子数量过多会导致过度拟合
   - 需要严格的样本外测试
   - 控制因子数量

#### 12.3.3 资源消耗

**❌ 资源消耗显著增加**

**计算成本**：

1. **训练时间**
   - 因子数量增加会导致训练时间显著增加
   - 模型复杂度增加
   - 超参数调优时间增加

2. **存储需求**
   - 因子数据存储需求增加
   - 模型文件大小增加
   - 历史数据存储需求增加

3. **推理速度**
   - 因子数量增加会导致推理速度变慢
   - 实时交易可能受影响
   - 需要优化推理性能

### 12.4 实施建议

#### 12.4.1 推荐方案

**✅ 推荐采用渐进式策略**

**Phase 1：因子收集和去重**
```python
# 收集所有历史TASK的SOTA因子
all_sota_factors = collect_all_sota_factors(log_dir)

# 使用RD-Agent的LLM去重机制
deduplicated_factors = deduplicate_factors_by_llm(all_sota_factors)

# 使用IC去重机制
final_factors = deduplicate_by_ic(deduplicated_factors)
```

**Phase 2：因子筛选**
```python
# 计算每个因子的IC
factor_ic = calculate_ic(final_factors)

# 只保留IC > 0.02的因子
selected_factors = final_factors[:, factor_ic > 0.02]

# 控制因子数量在20-30个
if len(selected_factors.columns) > 30:
    selected_factors = select_top_n_factors(selected_factors, n=30)
```

**Phase 3：因子正交化**
```python
# 使用PCA降低因子相关性
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # 保留95%的方差
orthogonal_factors = pca.fit_transform(selected_factors)
```

**Phase 4：模型训练和评估**
```python
# 训练模型
model = train_model(orthogonal_factors, alpha_factors)

# 样本外测试
out_of_sample_performance = evaluate_model(model, test_data)

# 与baseline比较
baseline_performance = evaluate_model(baseline_model, test_data)
```

#### 12.4.2 最佳实践

**✅ 量化投资最佳实践**

1. **因子数量控制**
   - 控制因子数量在20-30个
   - 避免因子数量过多
   - 定期评估因子有效性

2. **因子质量评估**
   - 使用IC、IR等指标评估因子质量
   - 只保留表现良好的因子
   - 定期重新评估因子有效性

3. **因子去重**
   - 使用IC去重机制
   - 使用LLM去重机制
   - 避免因子间高度相关

4. **样本外测试**
   - 严格进行样本外测试
   - 避免过度拟合
   - 评估模型的泛化能力

5. **动态调整**
   - 根据市场环境调整因子权重
   - 定期重新评估因子有效性
   - 淘汰表现不佳的因子

### 12.5 总结

**问题1：是否可以通过脚本收集所有历史TASK的SOTA因子，与alpha因子组合后进行模型的演进是否有价值？**

✅ **有价值，但需要谨慎**

- 技术上完全可行
- 可以扩大因子搜索空间
- 可以提高模型鲁棒性
- 但需要严格的因子筛选和去重

**问题2：因子数量过多是否在量化场景中是最佳实践？**

❌ **不是最佳实践**

- 因子数量过多会导致过度拟合
- 会导致多重共线性问题
- 会导致维度灾难
- 会显著增加计算成本

**问题3：前提是确保没有重复因子，除了资源消耗外，对选股效果是否可能会有提升？**

✅ **可能会有提升，但效果有限**

- 前提是严格的因子筛选和去重
- 控制因子数量在合理范围（20-30个）
- 进行严格的样本外测试
- 效果提升可能有限，但值得尝试

**推荐策略**：

1. **渐进式实施**：先收集少量TASK的SOTA因子，逐步增加
2. **严格筛选**：只保留高质量的因子
3. **控制数量**：控制因子数量在20-30个
4. **样本外测试**：严格进行样本外测试
5. **动态调整**：根据市场环境动态调整因子组合

**最终建议**：

✅ **值得尝试，但需要谨慎实施**

- 技术上完全可行
- 潜在收益存在，但需要严格控制风险
- 推荐采用渐进式策略
- 严格进行样本外测试
- 控制因子数量在合理范围

---

**文档结束**
