# Phase 1.1: ML模型适配器详细设计

## 设计目标
支持XGBoost、LightGBM、CatBoost等ML模型，通过适配器模式统一接口。

## 文件结构
```
rdagent/components/coder/model_coder/
├── adapters/
│   ├── __init__.py           # 新增
│   ├── base.py              # 新增，150行
│   ├── ml_adapter.py        # 新增，400行
│   └── pytorch_adapter.py   # 重构，300行
├── registry.py              # 新增，100行
└── evolving_strategy.py     # 修改，+50行
```

## 详细设计

### 1. base.py - 适配器基类

**路径**: `rdagent/components/coder/model_coder/adapters/base.py`  
**新增文件，约150行**

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class ModelAdapter(ABC):
    """模型适配器基类"""
    
    @abstractmethod
    def create_model(self, model_config: Dict[str, Any]) -> Any:
        """创建模型实例"""
    
    @abstractmethod
    def train(self, model, X_train, y_train, X_valid, y_valid, **kwargs) -> Any:
        """训练模型"""
    
    @abstractmethod
    def predict(self, model, X_test: np.ndarray) -> np.ndarray:
        """预测"""
    
    @abstractmethod
    def save_model(self, model, path: str) -> None:
        """保存模型"""
    
    @abstractmethod
    def load_model(self, path: str) -> Any:
        """加载模型"""
```

**集成点**: 被MLAdapter、PyTorchAdapter继承  
**风险**: 低  
**工作量**: 1人日

### 2. ml_adapter.py - ML模型适配器

**路径**: `rdagent/components/coder/model_coder/adapters/ml_adapter.py`  
**新增文件，约400行**

**支持模型**:
- XGBoost: `xgb.XGBRegressor`
- LightGBM: `lgb.LGBMRegressor`
- CatBoost: `CatBoostRegressor`
- Linear: `LinearRegression`

**核心方法**:
```python
class MLAdapter(ModelAdapter):
    def create_model(self, model_config):
        model_type = model_config["model_type"]
        if model_type == "XGBoost":
            return self._create_xgboost(model_config)
        elif model_type == "LightGBM":
            return self._create_lightgbm(model_config)
        # ...
    
    def train(self, model, X_train, y_train, X_valid, y_valid):
        early_stop = getattr(model, "_early_stop", None)
        if early_stop:
            model.fit(X_train, y_train, 
                     eval_set=[(X_valid, y_valid)],
                     early_stopping_rounds=early_stop)
        else:
            model.fit(X_train, y_train)
        return model
```

**依赖**:
- `xgboost>=1.7.0`
- `lightgbm>=3.3.0`
- `catboost>=1.1.0`
- `scikit-learn>=1.2.0`

**集成点**: 被ModelRegistry注册，被ModelCoder/Runner调用  
**风险**: 中（依赖库版本兼容）  
**工作量**: 3人日

### 3. registry.py - 模型注册表

**路径**: `rdagent/components/coder/model_coder/registry.py`  
**新增文件，约100行**

```python
class ModelRegistry:
    """单例模式，管理所有适配器"""
    
    _instance = None
    _adapters = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_adapters()
        return cls._instance
    
    def _init_adapters(self):
        self.register("PyTorch", PyTorchAdapter)
        self.register("ML", MLAdapter)
    
    def get_adapter(self, model_type: str) -> ModelAdapter:
        if model_type not in self._adapters:
            raise ValueError(f"Unknown model type: {model_type}")
        return self._adapters[model_type]()
```

**集成点**: 被ModelCoder、ModelRunner调用  
**风险**: 低  
**工作量**: 0.5人日

## 提示词修改

### 修改1: rdagent/scenarios/qlib/prompts.yaml

**位置**: `model_hypothesis_specification`（约第85-93行）

**原内容**:
```yaml
5. Focus exclusively on the architecture of PyTorch models...
```

**修改为**:
```yaml
5. Focus on model architecture. Support multiple model types:
   - PyTorch models: GRU, LSTM, Transformer, MLP (use torch.nn.Module)
   - ML models: XGBoost, LightGBM, CatBoost (do NOT wrap in torch.nn.Module)
   CRITICAL: Never mix frameworks!
```

**工作量**: 0.5人日

### 修改2: rdagent/components/coder/model_coder/prompts.yaml

**位置**: `evolving_strategy_model_coder.system`

**追加内容**:
```yaml
## FRAMEWORK SELECTION RULES
If model_type == "PyTorch": Use torch.nn.Module
If model_type == "ML": Import xgboost/lightgbm directly, NO torch.nn.Module
```

**工作量**: 0.5人日

## 集成到现有代码

### 修改ModelCoder

**文件**: `rdagent/components/coder/model_coder/model.py`（或类似）

**修改点**: 在develop()方法中，根据model_type选择适配器

```python
def develop(self, exp: ModelExperiment):
    # 1. 生成模型代码（现有逻辑）
    model_code = self.generate_code(exp)
    
    # 2. 获取适配器
    model_type = exp.sub_tasks[0].get("model_type", "PyTorch")
    from rdagent.components.coder.model_coder.registry import ModelRegistry
    adapter = ModelRegistry().get_adapter(model_type)
    
    # 3. 附加到实验
    exp._model_adapter = adapter
    return exp
```

**风险**: 低  
**工作量**: 1人日

### 修改ModelRunner

**文件**: `rdagent/scenarios/qlib/developer/model_runner.py`

**修改点**: 使用适配器训练

```python
def develop(self, exp: ModelExperiment):
    # 获取适配器
    adapter = getattr(exp, "_model_adapter", None)
    if not adapter:
        adapter = ModelRegistry().get_adapter("PyTorch")
    
    # 使用适配器
    model = adapter.create_model(exp.model_config)
    trained = adapter.train(model, X_train, y_train, X_valid, y_valid)
    predictions = adapter.predict(trained, X_test)
    adapter.save_model(trained, exp.model_path)
    
    return exp
```

**风险**: 低  
**工作量**: 1人日

## 工作量汇总

| 任务 | 工作量 | 风险 |
|------|--------|------|
| base.py | 1人日 | 低 |
| ml_adapter.py | 3人日 | 中 |
| pytorch_adapter.py | 2人日 | 低 |
| registry.py | 0.5人日 | 低 |
| 提示词修改 | 1人日 | 低 |
| ModelCoder集成 | 1人日 | 低 |
| ModelRunner集成 | 1人日 | 低 |
| **总计** | **9.5人日** | - |

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 依赖库版本冲突 | 中 | 中 | 使用conda环境隔离，锁定版本 |
| 适配器接口设计不完善 | 低 | 低 | 充分测试，预留扩展点 |
| 破坏现有PyTorch逻辑 | 高 | 低 | 保持向后兼容，充分测试 |

## 测试计划

1. **单元测试** (2人日)
   - 测试MLAdapter各模型创建
   - 测试训练/预测/保存/加载
   
2. **集成测试** (1人日)
   - 端到端测试XGBoost模型演进
   - 验证与PyTorch模型共存

3. **回归测试** (1人日)
   - 确保现有PyTorch模型演进不受影响
