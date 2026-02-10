# Phase 2 详细设计：多模型协同选股 + HMM大盘分析

## Phase 2 总览

**预计总工作量**: 21人日（约4周，单人）  
**优先级**: P1（重要但非必须）  
**前置依赖**: Phase 1完成  
**核心目标**: 实现多模型协同选股和HMM市场状态识别

---

## 2.1 多模型协同选股设计

### 设计目标

支持不同因子类型使用不同模型训练，并在选股时集成多个模型的预测结果。

### 架构设计

```
rdagent/scenarios/qlib/ensemble/
├── __init__.py
├── model_selector.py       # 模型选择器（新增，200行）
├── predictor_ensemble.py   # 预测集成器（新增，250行）
└── weight_optimizer.py     # 权重优化器（新增，300行）
```

### 2.1.1 model_selector.py - 模型选择器

**路径**: `rdagent/scenarios/qlib/ensemble/model_selector.py`  
**新增文件，约200行**

**职责**: 根据因子类型自动选择合适的模型

**核心类设计**:
```python
class ModelSelector:
    """模型选择器
    
    根据因子类型推荐合适的模型：
    - 截面因子 -> ML模型（XGBoost/LightGBM）
    - 时序因子 -> PyTorch时序模型（LSTM/GRU）
    - 混合因子 -> 根据主导类型选择
    """
    
    # 推荐规则
    RECOMMENDED_MODELS = {
        FactorType.CROSS_SECTIONAL: ["XGBoost", "LightGBM", "CatBoost"],
        FactorType.TIME_SERIES: ["LSTM", "GRU", "Transformer"],
        FactorType.HYBRID: ["XGBoost", "LSTM"],
    }
    
    def select_model_for_factors(
        self, 
        factors: List[FactorMeta]
    ) -> Dict[str, List[FactorMeta]]:
        """为因子列表分配模型
        
        Args:
            factors: 因子列表
        
        Returns:
            {model_name: [factors]}，按模型分组的因子
        """
        # 1. 统计因子类型分布
        type_counts = self._count_factor_types(factors)
        
        # 2. 根据分布选择模型
        if type_counts[FactorType.CROSS_SECTIONAL] > len(factors) * 0.7:
            # 70%以上截面因子，使用ML模型
            return {"XGBoost": factors}
        elif type_counts[FactorType.TIME_SERIES] > len(factors) * 0.7:
            # 70%以上时序因子，使用时序模型
            return {"LSTM": factors}
        else:
            # 混合情况，分别训练
            return self._split_factors_by_type(factors)
    
    def _split_factors_by_type(
        self, 
        factors: List[FactorMeta]
    ) -> Dict[str, List[FactorMeta]]:
        """按类型分组因子"""
        cs_factors = [f for f in factors if f.factor_type == FactorType.CROSS_SECTIONAL]
        ts_factors = [f for f in factors if f.factor_type == FactorType.TIME_SERIES]
        
        result = {}
        if cs_factors:
            result["XGBoost"] = cs_factors
        if ts_factors:
            result["LSTM"] = ts_factors
        
        return result
```

**集成点**: 被演进循环调用，在模型训练前  
**风险**: 低  
**工作量**: 1.5人日

---

### 2.1.2 predictor_ensemble.py - 预测集成器

**路径**: `rdagent/scenarios/qlib/ensemble/predictor_ensemble.py`  
**新增文件，约250行**

**职责**: 集成多个模型的预测结果

**核心类设计**:
```python
class PredictorEnsemble:
    """预测集成器
    
    支持多种集成策略：
    - 加权平均
    - 投票
    - Stacking
    """
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """初始化集成器
        
        Args:
            models: {model_name: trained_model}
            weights: {model_name: weight}，如果为None则均权
        """
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models}
    
    def predict(self, X: np.ndarray, method: str = "weighted_average") -> np.ndarray:
        """集成预测
        
        Args:
            X: 特征矩阵
            method: 集成方法
        
        Returns:
            集成后的预测结果
        """
        if method == "weighted_average":
            return self._weighted_average(X)
        elif method == "voting":
            return self._voting(X)
        elif method == "stacking":
            return self._stacking(X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _weighted_average(self, X: np.ndarray) -> np.ndarray:
        """加权平均"""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights[name]
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def _voting(self, X: np.ndarray) -> np.ndarray:
        """投票法（用于分类）"""
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # 多数投票
        from scipy.stats import mode
        return mode(predictions, axis=0)[0]
    
    def _stacking(self, X: np.ndarray) -> np.ndarray:
        """Stacking集成（需要meta-learner）"""
        # Phase 2简化实现，Phase 3可优化
        return self._weighted_average(X)
```

**集成点**: 被选股策略调用  
**风险**: 低  
**工作量**: 2人日

---

### 2.1.3 weight_optimizer.py - 权重优化器

**路径**: `rdagent/scenarios/qlib/ensemble/weight_optimizer.py`  
**新增文件，约300行**

**职责**: 优化集成权重

**核心类设计**:
```python
class WeightOptimizer:
    """权重优化器
    
    基于验证集优化集成权重，目标是最大化IC/IR。
    ""1"
    
    def optimize(
        self,
        models: Dict[str, Any],
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        method: str = "grid_search",
    ) -> Dict[str, float]:
        """优化权重
        
        Args:
            models: {model_name: trained_model}
            X_valid: 验证集特征
            y_valid: 验证集标签
            method: 优化方法（grid_search/bayesian）
        
        Returns:
            最优权重 {model_name: weight}
        """
        if method == "grid_search":
            return self._grid_search(models, X_valid, y_valid)
        elif method == "bayesian":
            return self._bayesian_optimization(models, X_valid, y_valid)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _grid_search(
        self,
        models: Dict[str, Any],
        X_valid: np.ndarray,
        y_valid: np.ndarray,
    ) -> Dict[str, float]:
        """网格搜索最优权重"""
        import itertools
        
        # 生成权重候选（简化版，3个模型 -> 5x5x5种组合）
        n_models = len(models)
        weight_steps = 5
        weight_grid = np.linspace(0, 1, weight_steps)
        
        best_weights = None
        best_ic = -float("inf")
        
        # 遍历所有权重组合
        for weights in itertools.product(weight_grid, repeat=n_models):
            # 归一化
            weights = np.array(weights)
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            
            # 评估IC
            weight_dict = dict(zip(models.keys(), weights))
            ensemble = PredictorEnsemble(models, weight_dict)
            y_pred = ensemble.predict(X_valid, method="weighted_average")
            ic = self._calculate_ic(y_valid, y_pred)
            
            if ic > best_ic:
                best_ic = ic
                best_weights = weight_dict
        
        return best_weights
    
    def _calculate_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算IC值"""
        from scipy.stats import spearmanr
        return spearmanr(y_true, y_pred)[0]
```

**集成点**: 在模型训练后调用  
**风险**: 中（计算密集）  
**工作量**: 2.5人日

---

### 2.1.4 集成到演进循环

**修改文件**: `rdagent/app/qlib_rd_loop/quant.py`

**修改点**: 在模型训练阶段，根据因子类型选择多个模型训练

```python
def running(self, prev_out: dict):
    # 现有逻辑：训练单个模型
    
    # 新增：多模型训练
    if self.config.enable_multi_model:
        from rdagent.scenarios.qlib.ensemble import ModelSelector, WeightOptimizer
        
        # 1. 获取当前因子列表
        factors = self.get_current_factors()
        
        # 2. 根据因子类型分配模型
        selector = ModelSelector()
        model_assignments = selector.select_model_for_factors(factors)
        
        # 3. 训练多个模型
        trained_models = {}
        for model_name, factor_subset in model_assignments.items():
            model = self.train_model(model_name, factor_subset)
            trained_models[model_name] = model
        
        # 4. 优化集成权重
        optimizer = WeightOptimizer()
        optimal_weights = optimizer.optimize(trained_models, X_valid, y_valid)
        
        # 5. 保存集成配置
        ensemble_config = {
            "models": list(trained_models.keys()),
            "weights": optimal_weights,
        }
        self.save_ensemble_config(ensemble_config)
    
    return prev_out
```

**风险**: 中（需要仔细处理多模型训练流程）  
**工作量**: 3人日

---

## 2.2 HMM大盘分析设计

### 设计目标

使用隐马尔可夫模型识别市场状态（牛市/熊市/震荡），为策略参数调整提供依据。

### 架构设计

```
rdagent/scenarios/qlib/market_regime/
├── __init__.py
├── hmm_analyzer.py         # HMM分析器（新增，350行）
├── feature_extractor.py    # 特征提取器（新增，200行）
└── regime_detector.py      # 市场状态检测器（新增，150行）
```

### 2.2.1 hmm_analyzer.py - HMM分析器

**路径**: `rdagent/scenarios/qlib/market_regime/hmm_analyzer.py`  
**新增文件，约350行**

**职责**: 训练HMM模型，识别市场状态

**核心类设计**:
```python
from hmmlearn import hmm
import numpy as np

class HMMAnalyzer:
    """HMM市场状态分析器
    
    使用GaussianHMM识别市场的隐藏状态。
    """
    
    def __init__(self, n_states: int = 3):
        """初始化分析器
        
        Args:
            n_states: 隐藏状态数量（默认3：牛市/熊市/震荡）
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        self.state_names = ["牛市", "震荡", "熊市"]  # 需要训练后映射
    
    def train(self, features: np.ndarray) -> None:
        """训练HMM模型
        
        Args:
            features: 市场特征矩阵，shape (n_days, n_features)
        """
        self.model.fit(features)
        
        # 根据均值映射状态名称
        means = self.model.means_[:, 0]  # 假设第一个特征是收益率
        sorted_indices = np.argsort(means)
        self.state_mapping = {
            sorted_indices[0]: "熊市",
            sorted_indices[1]: "震荡",
            sorted_indices[2]: "牛市",
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测市场状态
        
        Args:
            features: 市场特征矩阵
        
        Returns:
            状态序列，shape (n_days,)
        """
        return self.model.predict(features)
    
    def get_current_state(self, recent_features: np.ndarray) -> str:
        """获取当前市场状态
        
        Args:
            recent_features: 最近N天的特征
        
        Returns:
            状态名称（牛市/熊市/震荡）
        """
        state_id = self.model.predict(recent_features)[-1]
        return self.state_mapping.get(state_id, "未知")
    
    def get_state_probability(self, features: np.ndarray) -> np.ndarray:
        """获取状态概率分布
        
        Args:
            features: 市场特征
        
        Returns:
            概率矩阵，shape (n_days, n_states)
        """
        return self.model.predict_proba(features)
```

**依赖**: `hmmlearn>=0.3.0`  
**集成点**: 被策略演进模块调用  
**风险**: 中（HMM训练可能不稳定）  
**工作量**: 3人日

---

### 2.2.2 feature_extractor.py - 特征提取器

**路径**: `rdagent/scenarios/qlib/market_regime/feature_extractor.py`  
**新增文件，约200行**

**职责**: 提取市场特征用于HMM训练

**核心类设计**:
```python
class MarketFeatureExtractor:
    """市场特征提取器
    
    提取大盘指数的技术指标作为HMM输入特征。
    """
    
    def extract(self, index_data: pd.DataFrame) -> np.ndarray:
        """提取特征
        
        Args:
            index_data: 大盘指数数据，包含OHLCV
        
        Returns:
            特征矩阵，shape (n_days, n_features)
        """
        features = []
        
        # 1. 收益率
        returns = index_data['close'].pct_change()
        features.append(returns)
        
        # 2. 波动率（20日滚动标准差）
        volatility = returns.rolling(20).std()
        features.append(volatility)
        
        # 3. 成交量变化
        volume_change = index_data['volume'].pct_change()
        features.append(volume_change)
        
        # 4. RSI指标
        rsi = self._calculate_rsi(index_data['close'], period=14)
        features.append(rsi)
        
        # 5. MACD
        macd = self._calculate_macd(index_data['close'])
        features.append(macd)
        
        # 6. 布林带位置
        bb_position = self._calculate_bb_position(index_data['close'])
        features.append(bb_position)
        
        # 合并所有特征
        feature_matrix = pd.concat(features, axis=1).fillna(0).values
        
        return feature_matrix
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50  # 归一化到[-1, 1]
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """计算MACD指标"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        return macd / prices  # 归一化
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """计算布林带位置"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        position = (prices - lower) / (upper - lower + 1e-8)
        return (position - 0.5) * 2  # 归一化到[-1, 1]
```

**集成点**: 被HMMAnalyzer调用  
**风险**: 低  
**工作量**: 1.5人日

---

### 2.2.3 regime_detector.py - 市场状态检测器

**路径**: `rdagent/scenarios/qlib/market_regime/regime_detector.py`  
**新增文件，约150行**

**职责**: 高层接口，封装HMM分析流程

**核心类设计**:
```python
class MarketRegimeDetector:
    """市场状态检测器
    
    提供简单的API接口。
    """
    
    def __init__(self, index_code: str = "000001.SH"):
        """初始化检测器
        
        Args:
            index_code: 大盘指数代码（默认上证指数）
        """
        self.index_code = index_code
        self.feature_extractor = MarketFeatureExtractor()
        self.hmm_analyzer = HMMAnalyzer(n_states=3)
        self.is_trained = False
    
    def train(self, start_date: str, end_date: str):
        """训练HMM模型
        
        Args:
            start_date: 训练开始日期
            end_date: 训练结束日期
        """
        # 1. 加载大盘数据
        index_data = self._load_index_data(start_date, end_date)
        
        # 2. 提取特征
        features = self.feature_extractor.extract(index_data)
        
        # 3. 训练HMM
        self.hmm_analyzer.train(features)
        self.is_trained = True
    
    def get_current_regime(self, lookback_days: int = 60) -> Dict[str, Any]:
        """获取当前市场状态
        
        Args:
            lookback_days: 回看天数
        
        Returns:
            {
                "regime": "牛市",
                "probability": 0.85,
                "state_probs": [0.05, 0.10, 0.85],
            }
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        # 1. 加载最近数据
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        index_data = self._load_index_data(start_date, end_date)
        
        # 2. 提取特征
        features = self.feature_extractor.extract(index_data)
        
        # 3. 预测状态
        current_state = self.hmm_analyzer.get_current_state(features)
        state_probs = self.hmm_analyzer.get_state_probability(features)[-1]
        
        return {
            "regime": current_state,
            "probability": float(state_probs.max()),
            "state_probs": state_probs.tolist(),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _load_index_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载大盘指数数据"""
        # 从Qlib加载数据
        import qlib
        from qlib.data import D
        
        df = D.features(
            [self.index_code],
            ["$open", "$high", "$low", "$close", "$volume"],
            start_time=start_date,
            end_time=end_date,
        )
        
        return df.droplevel(0)  # 移除instrument层级
```

**集成点**: 被API调用、被策略演进模块调用  
**风险**: 低  
**工作量**: 1.5人日

---

### 2.2.4 API集成

**文件**: `rdagent/app/api/routes/market.py`（新增）

**新增端点**:
```python
@router.post("/market/train")
def train_hmm(
    start_date: str,
    end_date: str,
    index_code: str = "000001.SH",
):
    """训练HMM模型"""
    detector = MarketRegimeDetector(index_code)
    detector.train(start_date, end_date)
    return {"message": "HMM model trained successfully"}

@router.get("/market/regime")
def get_market_regime(lookback_days: int = 60):
    """获取当前市场状态"""
    detector = MarketRegimeDetector()
    # 加载预训练模型
    detector.load_model("hmm_model.pkl")
    regime = detector.get_current_regime(lookback_days)
    return regime
```

**工作量**: 1人日

---

## Phase 2 工作量汇总

| 模块 | 工作量 | 风险 |
|------|--------|------|
| 多模型协同选股 | 9人日 | 中 |
| - model_selector.py | 1.5人日 | 低 |
| - predictor_ensemble.py | 2人日 | 低 |
| - weight_optimizer.py | 2.5人日 | 中 |
| - 集成到演进循环 | 3人日 | 中 |
| HMM大盘分析 | 7人日 | 中 |
| - hmm_analyzer.py | 3人日 | 中 |
| - feature_extractor.py | 1.5人日 | 低 |
| - regime_detector.py | 1.5人日 | 低 |
| - API集成 | 1人日 | 低 |
| 测试与文档 | 5人日 | 低 |
| **总计** | **21人日** | - |

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 多模型训练耗时长 | 中 | 高 | 并行训练，优化数据加载 |
| 集成权重优化效果不佳 | 中 | 中 | 尝试多种优化算法 |
| HMM训练不稳定 | 高 | 中 | 多次训练取最优，调整参数 |
| 市场状态识别不准 | 中 | 中 | 增加特征，调整状态数量 |

## 开发顺序

```
Week 1: 多模型协同基础
  ├─ ModelSelector开发
  └─ PredictorEnsemble开发

Week 2: 多模型协同集成
  ├─ WeightOptimizer开发
  └─ 集成到演进循环

Week 3: HMM分析开发
  ├─ FeatureExtractor开发
  ├─ HMMAnalyzer开发
  └─ RegimeDetector开发

Week 4: API集成与测试
  ├─ API端点开发
  ├─ 单元测试
  └─ 集成测试
```
