# Phase 1.3: API扩展详细设计

## 设计目标
扩展现有FastAPI接口，支持任务、因子、模型、实验管理，为AIStock集成提供API基础。

## 文件结构
```
rdagent/app/api/
├── __init__.py
├── main.py              # 扩展现有server.py，+200行
├── models.py            # 新增，Pydantic模型，300行
├── routes/
│   ├── __init__.py      # 新增
│   ├── tasks.py         # 扩展现有，+100行
│   ├── factors.py       # 新增，200行
│   ├── models.py        # 新增，150行
│   └── experiments.py   # 新增，150行
└── dependencies.py      # 新增，100行
```

## 详细设计

### 1. 扩展现有API入口

**文件**: `rdagent/app/scheduler/server.py`  
**修改方式**: 扩展现有函数，+200行

**新增路由注册**:
```python
def create_app() -> FastAPI:
    app = FastAPI(title="RD-Agent Scheduler")
    
    # 现有路由（保持不变）
    @app.get("/tasks")
    def list_tasks(): ...
    
    @app.post("/tasks")
    def create_task(): ...
    
    # 新增路由组
    from rdagent.app.api.routes import factors, models_routes, experiments
    app.include_router(factors.router, prefix="/api/v1", tags=["factors"])
    app.include_router(models_routes.router, prefix="/api/v1", tags=["models"])
    app.include_router(experiments.router, prefix="/api/v1", tags=["experiments"])
    
    return app
```

**风险**: 低（向后兼容）  
**工作量**: 1人日

### 2. Pydantic数据模型

**文件**: `rdagent/app/api/models.py`  
**新增文件，约300行**

**核心模型**:
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# 任务相关
class TaskCreate(BaseModel):
    name: str
    dataset_ids: List[str] = []
    loop_n: int = 1
    all_duration: str = "1:00:00"
    evolving_mode: str = "llm"
    model_types: List[str] = ["PyTorch"]  # 新增：支持多模型
    factor_sources: List[str] = ["rdagent"]  # 新增：因子来源

class TaskResponse(BaseModel):
    id: str
    name: str
    status: str  # pending/running/completed/failed
    progress: float  # 0.0-1.0
    created_at: datetime
    updated_at: datetime
    current_loop: int
    total_loops: int

# 因子相关
class FactorResponse(BaseModel):
    id: str
    name: str
    code: str
    factor_type: str  # cross_sectional/time_series/hybrid
    source: str
    ic: float
    ir: float
    icir: float
    is_sota: bool

class FactorOptimizeRequest(BaseModel):
    max_factors: int = 10
    include_sota: bool = True
    include_alpha158: bool = True
    include_alpha360: bool = True

# 模型相关
class ModelResponse(BaseModel):
    id: str
    name: str
    model_type: str  # PyTorch/ML
    model_name: str  # XGBoost/LSTM/...
    ic: float
    ir: float
    created_at: datetime

# 实验相关
class ExperimentResponse(BaseModel):
    id: str
    exp_type: str  # factor/model
    status: str
    ic: float
    ir: float
    created_at: datetime
    workspace_path: str
```

**集成点**: 被所有API路由使用  
**风险**: 低  
**工作量**: 1.5人日

### 3. 因子管理路由

**文件**: `rdagent/app/api/routes/factors.py`  
**新增文件，约200行**

**核心端点**:
```python
from fastapi import APIRouter, HTTPException
from rdagent.scenarios.qlib.factor_library import FactorLibrary

router = APIRouter()

@router.get("/factors", response_model=List[FactorResponse])
def list_factors(
    source: Optional[str] = None,
    factor_type: Optional[str] = None,
    is_sota: Optional[bool] = None,
):
    """查询因子列表"""
    factor_lib = FactorLibrary()
    
    if is_sota:
        factors = factor_lib.get_sota_factors()
    elif source:
        factors = factor_lib.get_alpha_factors(source)
    elif factor_type:
        factors = factor_lib.get_factors_by_type(FactorType(factor_type))
    else:
        factors = factor_lib.list_all_factors()
    
    return [FactorResponse(**f.to_dict()) for f in factors]

@router.get("/factors/{factor_id}", response_model=FactorResponse)
def get_factor(factor_id: str):
    """获取单个因子详情"""
    factor_lib = FactorLibrary()
    factor = factor_lib.get_factor(factor_id)
    if not factor:
        raise HTTPException(status_code=404, detail="Factor not found")
    return FactorResponse(**factor.to_dict())

@router.post("/factors/optimize", response_model=List[FactorResponse])
def optimize_factors(request: FactorOptimizeRequest):
    """触发因子组合优化"""
    from rdagent.scenarios.qlib.factor_library.optimizer import FactorCombinationOptimizer
    
    factor_lib = FactorLibrary()
    optimizer = FactorCombinationOptimizer(factor_lib)
    
    optimal_factors = optimizer.optimize_greedy(
        max_factors=request.max_factors,
        include_sota=request.include_sota,
        include_alpha158=request.include_alpha158,
        include_alpha360=request.include_alpha360,
    )
    
    return [FactorResponse(**f.to_dict()) for f in optimal_factors]

@router.get("/factors/sota", response_model=List[FactorResponse])
def list_sota_factors():
    """查询所有SOTA因子"""
    factor_lib = FactorLibrary()
    factors = factor_lib.get_sota_factors()
    return [FactorResponse(**f.to_dict()) for f in factors]

@router.post("/factors/import/alpha158")
def import_alpha158():
    """导入Alpha158因子"""
    from rdagent.scenarios.qlib.factor_library.importers.alpha158 import Alpha158Importer
    
    factor_lib = FactorLibrary()
    importer = Alpha158Importer(factor_lib)
    importer.import_all()
    
    return {"message": "Alpha158 factors imported successfully"}
```

**集成点**: 被AIStock前端/后端调用  
**风险**: 低  
**工作量**: 2人日

### 4. 模型管理路由

**文件**: `rdagent/app/api/routes/models.py`  
**新增文件，约150行**

**核心端点**:
```python
router = APIRouter()

@router.get("/models", response_model=List[ModelResponse])
def list_models(model_type: Optional[str] = None):
    """查询模型列表"""
    # 从Trace中读取模型实验结果
    # 简化实现：读取workspace下的模型元数据

@router.get("/models/sota", response_model=List[ModelResponse])
def list_sota_models():
    """查询SOTA模型"""

@router.get("/models/types")
def list_model_types():
    """列出支持的模型类型"""
    from rdagent.components.coder.model_coder.registry import ModelRegistry
    registry = ModelRegistry()
    return {
        "model_types": registry.list_types(),
        "ml_models": ["XGBoost", "LightGBM", "CatBoost", "Linear"],
        "pytorch_models": ["LSTM", "GRU", "Transformer", "MLP"],
    }
```

**集成点**: 被AIStock调用  
**风险**: 低  
**工作量**: 1.5人日

### 5. 实验管理路由

**文件**: `rdagent/app/api/routes/experiments.py`  
**新增文件，约150行**

**核心端点**:
```python
router = APIRouter()

@router.get("/experiments", response_model=List[ExperimentResponse])
def list_experiments(
    exp_type: Optional[str] = None,
    status: Optional[str] = None,
):
    """查询实验列表"""
    # 从Trace读取实验历史

@router.get("/experiments/{exp_id}", response_model=ExperimentResponse)
def get_experiment(exp_id: str):
    """获取实验详情"""

@router.get("/experiments/{exp_id}/export")
def export_experiment(exp_id: str):
    """导出实验结果（JSON格式）"""
    # 读取workspace下的实验结果
    # 返回统一格式的JSON
    return {
        "exp_id": exp_id,
        "exp_type": "factor",
        "metrics": {"ic": 0.05, "ir": 0.8},
        "code": "...",
        "workspace_path": "/path/to/workspace",
    }
```

**集成点**: 被AIStock调用获取实验结果  
**风险**: 低  
**工作量**: 1.5人日

### 6. 扩展任务管理

**文件**: `rdagent/app/scheduler/api_stub.py`  
**修改方式**: 扩展现有函数，+100行

**新增功能**:
```python
def api_get_task_progress(task_id: str) -> Dict:
    """获取任务实时进度"""
    task_rec = get_task(task_id)
    
    # 从Trace中读取当前循环进度
    current_loop = ...
    total_loops = task_rec.loop_n
    
    return {
        "task_id": task_id,
        "status": task_rec.status,
        "progress": current_loop / total_loops,
        "current_loop": current_loop,
        "total_loops": total_loops,
        "current_action": "running_experiment",  # 当前执行的步骤
    }

@app.get("/tasks/{task_id}/progress")
def get_task_progress(task_id: str):
    return api_get_task_progress(task_id)
```

**集成点**: 被AIStock轮询获取进度  
**风险**: 低  
**工作量**: 1人日

## API端点汇总

### 因子管理
```
GET    /api/v1/factors                # 查询因子列表
GET    /api/v1/factors/{id}           # 获取因子详情
GET    /api/v1/factors/sota           # 查询SOTA因子
POST   /api/v1/factors/optimize       # 触发因子组合优化
POST   /api/v1/factors/import/alpha158  # 导入Alpha158
POST   /api/v1/factors/import/alpha360  # 导入Alpha360
```

### 模型管理
```
GET    /api/v1/models                 # 查询模型列表
GET    /api/v1/models/sota            # 查询SOTA模型
GET    /api/v1/models/types           # 列出支持的模型类型
```

### 实验管理
```
GET    /api/v1/experiments            # 查询实验列表
GET    /api/v1/experiments/{id}       # 获取实验详情
GET    /api/v1/experiments/{id}/export  # 导出实验结果
```

### 任务管理（扩展现有）
```
GET    /tasks                         # 查询任务列表（现有）
POST   /tasks                         # 创建任务（现有，扩展model_types参数）
GET    /tasks/{id}                    # 获取任务详情（现有）
PATCH  /tasks/{id}/status             # 更新任务状态（现有）
GET    /tasks/{id}/progress           # 获取任务进度（新增）
```

## 工作量汇总

| 任务 | 工作量 | 风险 |
|------|--------|------|
| 扩展server.py | 1人日 | 低 |
| models.py（Pydantic） | 1.5人日 | 低 |
| routes/factors.py | 2人日 | 低 |
| routes/models.py | 1.5人日 | 低 |
| routes/experiments.py | 1.5人日 | 低 |
| 扩展tasks进度查询 | 1人日 | 低 |
| API文档生成 | 0.5人日 | 低 |
| **总计** | **9人日** | - |

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| API接口设计不合理 | 中 | 低 | 与AIStock侧对接确认需求 |
| 响应速度慢 | 中 | 低 | 优化数据库查询，添加缓存 |
| 破坏现有API兼容性 | 高 | 低 | 使用/api/v1前缀隔离新接口 |

## 测试计划

1. **API测试** (2人日)
   - 使用Postman/curl测试所有端点
   - 验证请求/响应格式
   
2. **集成测试** (1人日)
   - 测试与因子库、模型注册表集成
   - 测试任务进度查询准确性

3. **文档生成** (0.5人日)
   - FastAPI自动生成Swagger文档
   - 编写API使用示例
