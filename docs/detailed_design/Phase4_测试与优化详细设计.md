# Phase 4 详细设计：测试与优化

## Phase 4 总览

**预计总工作量**: 15人日（约3周，单人）  
**优先级**: P0（必须完成）  
**前置依赖**: Phase 1-3全部完成  
**核心目标**: 全面测试、性能优化、文档完善

---

## 4.1 测试体系设计

### 4.1.1 单元测试

**目录结构**:
```
tests/
├── unit/
│   ├── test_ml_adapter.py           # 新增，200行
│   ├── test_factor_library.py       # 新增，250行
│   ├── test_ensemble.py             # 新增，150行
│   ├── test_hmm_analyzer.py         # 新增，150行
│   ├── test_strategy_optimizer.py   # 新增，180行
│   └── test_graph_adapter.py        # 新增，200行
```

**核心测试用例设计**:

**test_ml_adapter.py**:
```python
import pytest
from rdagent.components.coder.model_coder.adapters.ml_adapter import MLAdapter

class TestMLAdapter:
    def setup_method(self):
        """每个测试前的设置"""
        self.adapter = MLAdapter()
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randn(100)
        self.X_valid = np.random.randn(20, 10)
        self.y_valid = np.random.randn(20)
    
    def test_create_xgboost(self):
        """测试XGBoost模型创建"""
        config = {
            "model_name": "XGBoost",
            "n_estimators": 100,
            "max_depth": 6,
        }
        model = self.adapter.create_model(config)
        assert model is not None
        assert hasattr(model, "fit")
    
    def test_train_xgboost(self):
        """测试XGBoost训练"""
        model = self.adapter.create_model({"model_name": "XGBoost"})
        trained_model = self.adapter.train(
            model, self.X_train, self.y_train,
            self.X_valid, self.y_valid
        )
        assert trained_model is not None
    
    def test_predict_xgboost(self):
        """测试XGBoost预测"""
        model = self.adapter.create_model({"model_name": "XGBoost"})
        trained_model = self.adapter.train(
            model, self.X_train, self.y_train,
            self.X_valid, self.y_valid
        )
        X_test = np.random.randn(10, 10)
        predictions = self.adapter.predict(trained_model, X_test)
        assert len(predictions) == 10
    
    def test_save_load_model(self):
        """测试模型保存和加载"""
        model = self.adapter.create_model({"model_name": "XGBoost"})
        trained_model = self.adapter.train(
            model, self.X_train, self.y_train,
            self.X_valid, self.y_valid
        )
        
        # 保存
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.pkl"
            self.adapter.save_model(trained_model, path)
            
            # 加载
            loaded_model = self.adapter.load_model(path)
            
            # 验证预测一致性
            X_test = np.random.randn(10, 10)
            pred1 = self.adapter.predict(trained_model, X_test)
            pred2 = self.adapter.predict(loaded_model, X_test)
            np.testing.assert_allclose(pred1, pred2)
    
    @pytest.mark.parametrize("model_name", ["XGBoost", "LightGBM", "CatBoost"])
    def test_all_ml_models(self, model_name):
        """测试所有ML模型"""
        model = self.adapter.create_model({"model_name": model_name})
        trained_model = self.adapter.train(
            model, self.X_train, self.y_train,
            self.X_valid, self.y_valid
        )
        X_test = np.random.randn(10, 10)
        predictions = self.adapter.predict(trained_model, X_test)
        assert len(predictions) == 10
```

**test_factor_library.py**:
```python
class TestFactorLibrary:
    def test_add_factor(self):
        """测试添加因子"""
        from rdagent.scenarios.qlib.factor_library import FactorLibrary, FactorMeta
        
        lib = FactorLibrary()
        factor = FactorMeta(
            name="test_factor",
            code="($close - $open) / $open",
            factor_type=FactorType.CROSS_SECTIONAL,
            source="test",
        )
        
        factor_id = lib.add_factor(factor)
        assert factor_id is not None
        
        # 验证可以查询到
        retrieved = lib.get_factor(factor_id)
        assert retrieved.name == "test_factor"
    
    def test_list_by_type(self):
        """测试按类型查询因子"""
        # ... 测试实现
    
    def test_mark_as_sota(self):
        """测试标记SOTA因子"""
        # ... 测试实现
```

**工作量**: 5人日

---

### 4.1.2 集成测试

**目录结构**:
```
tests/
├── integration/
│   ├── test_ml_model_evolution.py      # 新增，300行
│   ├── test_factor_optimization.py     # 新增，250行
│   ├── test_multi_model_ensemble.py    # 新增，200行
│   ├── test_strategy_evolution.py      # 新增，250行
│   └── test_api_endpoints.py           # 新增，300行
```

**核心测试用例设计**:

**test_ml_model_evolution.py**:
```python
class TestMLModelEvolution:
    def test_end_to_end_xgboost_evolution(self):
        """端到端测试XGBoost模型演进"""
        # 1. 设置
        from rdagent.app.qlib_rd_loop.quant import QuantRDLoop
        
        config = PropSetting(
            scen="rdagent.scenarios.qlib.experiment.model_experiment:QlibModelScenario",
            model_types=["XGBoost"],
            loop_n=1,
        )
        
        rd_loop = QuantRDLoop(config)
        
        # 2. 运行一个演进循环
        result = rd_loop.run()
        
        # 3. 验证结果
        assert result is not None
        assert len(rd_loop.trace.hist) > 0
        
        # 4. 验证XGBoost模型被正确使用
        last_exp = rd_loop.trace.hist[-1].exp
        assert "XGBoost" in last_exp.model_type
    
    def test_pytorch_xgboost_coexistence(self):
        """测试PyTorch和XGBoost共存"""
        # 配置同时使用两种模型
        config = PropSetting(
            model_types=["PyTorch", "XGBoost"],
            loop_n=2,
        )
        
        rd_loop = QuantRDLoop(config)
        result = rd_loop.run()
        
        # 验证两种模型都被使用
        model_types_used = set()
        for step in rd_loop.trace.hist:
            model_types_used.add(step.exp.model_type)
        
        assert "PyTorch" in model_types_used
        assert "XGBoost" in model_types_used
```

**test_api_endpoints.py**:
```python
from fastapi.testclient import TestClient

class TestAPIEndpoints:
    def setup_method(self):
        """设置测试客户端"""
        from rdagent.app.scheduler.server import create_app
        app = create_app()
        self.client = TestClient(app)
    
    def test_list_factors(self):
        """测试查询因子列表"""
        response = self.client.get("/api/v1/factors")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_optimize_factors(self):
        """测试因子优化"""
        payload = {
            "max_factors": 10,
            "include_sota": True,
            "include_alpha158": True,
        }
        response = self.client.post("/api/v1/factors/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 10
    
    def test_get_market_regime(self):
        """测试获取市场状态"""
        response = self.client.get("/api/v1/market/regime")
        assert response.status_code == 200
        data = response.json()
        assert "regime" in data
        assert data["regime"] in ["牛市", "熊市", "震荡"]
```

**工作量**: 4人日

---

### 4.1.3 回归测试

**目标**: 确保新功能不破坏现有功能

**测试用例**:
```python
class TestRegression:
    def test_original_pytorch_evolution_still_works(self):
        """回归测试：原有PyTorch模型演进不受影响"""
        # 使用原始配置运行
        config = PropSetting(
            scen="rdagent.scenarios.qlib.experiment.model_experiment:QlibModelScenario",
            model_types=["PyTorch"],  # 只使用PyTorch
            loop_n=1,
        )
        
        rd_loop = QuantRDLoop(config)
        result = rd_loop.run()
        
        # 验证结果格式与之前一致
        assert result is not None
        assert len(rd_loop.trace.hist) > 0
    
    def test_factor_evolution_unchanged(self):
        """回归测试：因子演进逻辑不变"""
        # ... 测试实现
```

**工作量**: 2人日

---

## 4.2 性能优化

### 4.2.1 数据库查询优化

**优化点**:
1. 为常用查询添加索引
2. 使用查询缓存
3. 批量操作优化

**实现**:

**文件**: `rdagent/scenarios/qlib/factor_library/database.py`

**优化前**:
```python
def list_by_type(self, factor_type: FactorType) -> List[FactorMeta]:
    cursor = self.conn.cursor()
    cursor.execute(
        "SELECT * FROM factors WHERE factor_type = ?",
        (factor_type.value,)
    )
    # ... 处理结果
```

**优化后**:
```python
from functools import lru_cache

class FactorDatabase:
    @lru_cache(maxsize=128)
    def list_by_type(self, factor_type: FactorType) -> List[FactorMeta]:
        """添加缓存的查询"""
        # 原有查询逻辑
        pass
    
    def batch_insert(self, factors: List[FactorMeta]) -> List[str]:
        """批量插入优化"""
        cursor = self.conn.cursor()
        
        # 使用executemany批量插入
        values = [
            (f.id, f.name, f.code, f.factor_type.value, ...)
            for f in factors
        ]
        
        cursor.executemany(
            "INSERT INTO factors VALUES (?, ?, ?, ?, ...)",
            values
        )
        
        self.conn.commit()
        return [f.id for f in factors]
```

**工作量**: 1人日

---

### 4.2.2 模型训练并行化

**优化点**: 多模型训练时并行执行

**实现**:

**文件**: `rdagent/scenarios/qlib/ensemble/model_selector.py`

**优化前**（串行训练）:
```python
def train_multiple_models(self, model_assignments):
    trained_models = {}
    for model_name, factors in model_assignments.items():
        model = self.train_model(model_name, factors)  # 串行
        trained_models[model_name] = model
    return trained_models
```

**优化后**（并行训练）:
```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def train_multiple_models(self, model_assignments):
    trained_models = {}
    
    # 使用进程池并行训练
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(self.train_model, model_name, factors): model_name
            for model_name, factors in model_assignments.items()
        }
        
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                model = future.result()
                trained_models[model_name] = model
            except Exception as e:
                logging.error(f"Model {model_name} training failed: {e}")
    
    return trained_models
```

**工作量**: 1.5人日

---

### 4.2.3 因子计算缓存

**优化点**: 缓存已计算的因子值

**实现**:

**文件**: `rdagent/scenarios/qlib/developer/factor_runner.py`

```python
import hashlib
import pickle
from pathlib import Path

class FactorRunner(Developer):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.cache_dir = Path(".cache/factors")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, factor_code: str, data_config: Dict) -> str:
        """生成缓存键"""
        content = f"{factor_code}_{str(data_config)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def develop(self, exp: FactorExperiment):
        cache_key = self._get_cache_key(exp.factor_code, exp.data_config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # 检查缓存
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached_results = pickle.load(f)
            exp.results = cached_results
            return exp
        
        # 正常计算
        results = self.run_factor_extraction(exp)
        exp.results = results
        
        # 保存缓存
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
        
        return exp
```

**工作量**: 1人日

---

## 4.3 文档完善

### 4.3.1 API文档

**工具**: FastAPI自动生成Swagger文档

**补充内容**:
- 每个端点的详细说明
- 请求/响应示例
- 错误代码说明
- 认证方式（如果需要）

**工作量**: 1人日

---

### 4.3.2 用户手册

**文件**: `docs/user_guide/`

**内容**:
1. 快速开始指南
2. ML模型使用教程
3. 因子库管理教程
4. 多模型协同选股教程
5. 策略优化教程
6. API调用示例

**工作量**: 2人日

---

### 4.3.3 开发者文档

**文件**: `docs/developer_guide/`

**内容**:
1. 架构设计文档
2. 扩展指南（如何添加新模型类型）
3. 代码规范
4. 贡献指南

**工作量**: 1.5人日

---

## 4.4 部署与运维

### 4.4.1 Docker化

**文件**: `Dockerfile`（新增）

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露API端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "rdagent.app.scheduler.server"]
```

**docker-compose.yml**（新增）:
```yaml
version: '3.8'

services:
  rdagent-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./workspace:/app/workspace
      - ./data:/app/data
    environment:
      - QLIB_DATA_PATH=/app/data/qlib
      - LOG_LEVEL=INFO
```

**工作量**: 1人日

---

### 4.4.2 监控与日志

**实现日志结构化**:

**文件**: `rdagent/utils/logging_config.py`（新增）

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """结构化日志"""
    
    @staticmethod
    def log_experiment(exp_id: str, exp_type: str, metrics: Dict):
        """记录实验日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "experiment_completed",
            "exp_id": exp_id,
            "exp_type": exp_type,
            "metrics": metrics,
        }
        logging.info(json.dumps(log_entry))
    
    @staticmethod
    def log_model_training(model_type: str, duration: float, success: bool):
        """记录模型训练日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "model_training",
            "model_type": model_type,
            "duration_seconds": duration,
            "success": success,
        }
        logging.info(json.dumps(log_entry))
```

**工作量**: 1人日

---

## Phase 4 工作量汇总

| 模块 | 工作量 | 风险 |
|------|--------|------|
| 单元测试 | 5人日 | 低 |
| 集成测试 | 4人日 | 低 |
| 回归测试 | 2人日 | 低 |
| 性能优化 | 3.5人日 | 中 |
| - 数据库查询优化 | 1人日 | 低 |
| - 模型训练并行化 | 1.5人日 | 中 |
| - 因子计算缓存 | 1人日 | 低 |
| 文档完善 | 4.5人日 | 低 |
| - API文档 | 1人日 | 低 |
| - 用户手册 | 2人日 | 低 |
| - 开发者文档 | 1.5人日 | 低 |
| 部署与运维 | 2人日 | 低 |
| **总计** | **21人日** | - |

**调整后总计**（考虑10%缓冲）：**23人日 ≈ 4.5周**

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 测试覆盖不全 | 中 | 中 | 代码审查，增加测试用例 |
| 性能优化效果不明显 | 低 | 中 | 性能基准测试，持续优化 |
| 文档滞后 | 低 | 高 | 在开发阶段同步编写文档 |

## 验收标准

✅ 单元测试覆盖率 > 80%  
✅ 集成测试通过率 100%  
✅ 回归测试通过率 100%  
✅ 关键路径性能提升 > 30%  
✅ API文档完整（Swagger）  
✅ 用户手册完整  
✅ Docker镜像可正常构建和运行
