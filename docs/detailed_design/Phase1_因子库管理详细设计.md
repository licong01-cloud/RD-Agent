# Phase 1.2: 因子库管理系统详细设计

## 设计目标
建立因子库持久化管理系统，支持SOTA因子自动入库、Alpha因子导入、因子分类和组合优化。

## 文件结构
```
rdagent/scenarios/qlib/factor_library/
├── __init__.py              # 新增
├── models.py                # 新增，150行
├── database.py              # 新增，300行
├── factor_lib.py            # 新增，200行
├── classifier.py            # 新增，150行
├── optimizer.py             # 新增，400行
└── importers/
    ├── __init__.py          # 新增
    ├── alpha158.py          # 新增，200行
    └── alpha360.py          # 新增，200行
```

## 详细设计

### 1. models.py - 数据模型

**路径**: `rdagent/scenarios/qlib/factor_library/models.py`  
**新增文件，约150行**

**核心类**:
```python
@dataclass
class FactorMeta:
    """因子元数据"""
    id: Optional[str] = None
    name: str = ""
    code: str = ""  # Qlib表达式
    factor_type: FactorType = FactorType.HYBRID  # 截面/时序/混合
    source: str = "rdagent"  # rdagent/alpha158/alpha360
    ic: float = 0.0
    ir: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    is_sota: bool = False
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**集成点**: 被FactorLibrary、FactorDatabase使用  
**风险**: 低  
**工作量**: 0.5人日

### 2. database.py - SQLite数据库

**路径**: `rdagent/scenarios/qlib/factor_library/database.py`  
**新增文件，约300行**

**核心方法**:
```python
class FactorDatabase:
    def _init_database(self):
        """创建factors表及索引"""
    
    def insert(self, factor: FactorMeta) -> str:
        """插入因子，返回ID"""
    
    def update(self, factor: FactorMeta):
        """更新因子"""
    
    def get_by_id(self, factor_id: str) -> Optional[FactorMeta]:
        """根据ID查询"""
    
    def list_all(self) -> List[FactorMeta]:
        """列出所有因子"""
    
    def list_by_type(self, factor_type: FactorType) -> List[FactorMeta]:
        """按类型查询"""
    
    def list_by_source(self, source: str) -> List[FactorMeta]:
        """按来源查询"""
    
    def list_sota(self) -> List[FactorMeta]:
        """查询SOTA因子"""
```

**表结构**:
```sql
CREATE TABLE factors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    code TEXT NOT NULL,
    factor_type TEXT NOT NULL,  -- cross_sectional/time_series/hybrid
    source TEXT NOT NULL,        -- rdagent/alpha158/alpha360
    ic REAL DEFAULT 0.0,
    ir REAL DEFAULT 0.0,
    icir REAL DEFAULT 0.0,
    rank_ic REAL DEFAULT 0.0,
    is_sota INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT               -- JSON字符串
);
CREATE INDEX idx_factor_type ON factors(factor_type);
CREATE INDEX idx_source ON factors(source);
CREATE INDEX idx_is_sota ON factors(is_sota);
```

**集成点**: 被FactorLibrary调用  
**风险**: 低  
**工作量**: 2人日

### 3. factor_lib.py - 高层接口

**路径**: `rdagent/scenarios/qlib/factor_library/factor_lib.py`  
**新增文件，约200行**

**核心方法**:
```python
class FactorLibrary:
    def add_factor(self, factor: FactorMeta) -> str
    def update_factor(self, factor: FactorMeta)
    def get_factor(self, factor_id: str) -> Optional[FactorMeta]
    def delete_factor(self, factor_id: str)
    def list_all_factors(self) -> List[FactorMeta]
    def get_sota_factors(self) -> List[FactorMeta]
    def get_factors_by_type(self, factor_type: FactorType) -> List[FactorMeta]
    def get_alpha_factors(self, alpha_version: str) -> List[FactorMeta]
    def mark_as_sota(self, factor_id: str)
    def search_by_name(self, name_pattern: str) -> List[FactorMeta]
    def get_top_factors(self, top_n: int, metric: str) -> List[FactorMeta]
```

**集成点**: 被演进循环、API、优化器调用  
**风险**: 低  
**工作量**: 1.5人日

### 4. classifier.py - 因子分类器

**路径**: `rdagent/scenarios/qlib/factor_library/classifier.py`  
**新增文件，约150行**

**分类规则**:
```python
class FactorClassifier:
    # 截面算子
    CROSS_SECTIONAL_OPS = ["Rank", "CSRank", "Zscore", "CSZscore", "Normalize"]
    
    # 时序算子
    TIME_SERIES_OPS = ["Ref", "Ts", "Mean", "Std", "Corr", "Delta", "EMA", "RSI"]
    
    @classmethod
    def classify(cls, factor_code: str) -> FactorType:
        """根据因子代码自动分类"""
        has_cs = any(op in factor_code for op in cls.CROSS_SECTIONAL_OPS)
        has_ts = any(op in factor_code for op in cls.TIME_SERIES_OPS)
        
        if has_cs and has_ts:
            return FactorType.HYBRID
        elif has_cs:
            return FactorType.CROSS_SECTIONAL
        elif has_ts:
            return FactorType.TIME_SERIES
        else:
            return FactorType.HYBRID
```

**集成点**: 在添加因子时自动分类  
**风险**: 低（启发式规则可能不准确）  
**工作量**: 1人日

### 5. optimizer.py - 因子组合优化

**路径**: `rdagent/scenarios/qlib/factor_library/optimizer.py`  
**新增文件，约400行**

**核心方法**:
```python
class FactorCombinationOptimizer:
    def optimize_greedy(
        self,
        max_factors: int = 10,
        include_sota: bool = True,
        include_alpha158: bool = True,
        include_alpha360: bool = True,
    ) -> List[FactorMeta]:
        """贪心算法优化因子组合"""
        # 1. 收集候选因子（SOTA + Alpha158前20 + Alpha360前20）
        # 2. 贪心选择：每次选择能使组合IR最大的因子
        # 3. 返回最优组合
    
    def optimize_genetic(
        self,
        population_size: int = 50,
        generations: int = 100,
        max_factors: int = 10,
    ) -> List[FactorMeta]:
        """遗传算法优化（Phase 1暂不实现）"""
    
    def _evaluate_combination(self, factors: List[FactorMeta]) -> float:
        """评估因子组合性能（简化版本使用平均IR）"""
```

**优化策略**:
- 贪心算法：每次添加IR增益最大的因子
- 遗传算法：Phase 2实现（可选）

**集成点**: 被API调用（手动触发）、被演进循环调用（定期优化）  
**风险**: 中（优化效果依赖数据质量）  
**工作量**: 3人日

### 6. Alpha因子导入器

**路径**: `rdagent/scenarios/qlib/factor_library/importers/alpha158.py`  
**新增文件，约200行**

**核心方法**:
```python
class Alpha158Importer:
    ALPHA158_FACTORS = [
        {"name": "KMID", "code": "($close - $open) / ($high - $low + 1e-8)"},
        {"name": "KLEN", "code": "($high - $low) / ($open + 1e-8)"},
        # ... 共158个因子
    ]
    
    def import_all(self, calculate_ic: bool = False):
        """导入所有Alpha158因子到因子库"""
        for factor_def in self.ALPHA158_FACTORS:
            factor_type = FactorClassifier.classify(factor_def["code"])
            factor = FactorMeta(
                name=f"alpha158_{factor_def['name']}",
                code=factor_def["code"],
                factor_type=factor_type,
                source="alpha158",
            )
            self.factor_lib.add_factor(factor)
```

**工作内容**:
- 整理完整的Alpha158因子列表（158个）
- 整理完整的Alpha360因子列表（360个）
- 实现批量导入逻辑

**集成点**: 被初始化脚本调用  
**风险**: 低  
**工作量**: 2人日（需整理因子列表）

## 集成到演进循环

### 修改FactorRunner

**文件**: `rdagent/scenarios/qlib/developer/factor_runner.py`

**修改点**: 在因子实验成功后自动添加到因子库

```python
def develop(self, exp: FactorExperiment):
    # 现有逻辑：运行因子提取
    results = self.run_factor_extraction(exp)
    
    # 新增：如果IC > 阈值，添加到因子库
    if results.ic > 0.02:
        from rdagent.scenarios.qlib.factor_library import FactorLibrary, FactorMeta
        from rdagent.scenarios.qlib.factor_library.classifier import FactorClassifier
        
        factor_lib = FactorLibrary()
        factor = FactorMeta(
            name=exp.factor_name,
            code=exp.factor_code,
            factor_type=FactorClassifier.classify(exp.factor_code),
            source="rdagent",
            ic=results.ic,
            ir=results.ir,
        )
        factor_lib.add_factor(factor)
    
    return exp
```

**风险**: 低  
**工作量**: 1人日

### 修改Summarizer（标记SOTA）

**文件**: `rdagent/scenarios/qlib/feedback/summarizer.py`

**修改点**: 当feedback.decision=True时，标记为SOTA

```python
def summarize(self, exp: Experiment, feedback: Feedback):
    # 现有逻辑...
    
    # 新增：如果决策为接受，标记为SOTA
    if feedback.decision:
        from rdagent.scenarios.qlib.factor_library import FactorLibrary
        
        factor_lib = FactorLibrary()
        # 根据实验名称或code查找因子
        factors = factor_lib.search_by_name(exp.name)
        if factors:
            factor_lib.mark_as_sota(factors[0].id)
    
    return summary
```

**风险**: 低  
**工作量**: 0.5人日

## 工作量汇总

| 任务 | 工作量 | 风险 |
|------|--------|------|
| models.py | 0.5人日 | 低 |
| database.py | 2人日 | 低 |
| factor_lib.py | 1.5人日 | 低 |
| classifier.py | 1人日 | 低 |
| optimizer.py | 3人日 | 中 |
| alpha158.py + alpha360.py | 2人日 | 低 |
| FactorRunner集成 | 1人日 | 低 |
| Summarizer集成 | 0.5人日 | 低 |
| **总计** | **11.5人日** | - |

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 因子分类不准确 | 低 | 中 | 使用启发式规则+人工校验 |
| 组合优化效果不佳 | 中 | 中 | 先用简单贪心，后续优化 |
| Alpha因子列表不全 | 低 | 低 | 参考Qlib官方文档 |
| SQLite性能问题 | 低 | 低 | 因子数量有限（<1000），足够 |

## 测试计划

1. **单元测试** (1.5人日)
   - 测试FactorDatabase CRUD操作
   - 测试FactorClassifier分类准确性
   - 测试Optimizer贪心算法
   
2. **集成测试** (1人日)
   - 测试因子演进自动入库
   - 测试SOTA因子标记
   - 测试Alpha因子批量导入

3. **性能测试** (0.5人日)
   - 测试1000+因子查询性能
