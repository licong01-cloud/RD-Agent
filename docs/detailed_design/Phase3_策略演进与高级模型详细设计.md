# Phase 3 详细设计：策略演进 + 高级模型

## Phase 3 总览

**预计总工作量**: 25人日（约5周，单人）  
**优先级**: P2（可选增强）  
**前置依赖**: Phase 1完成，Phase 2部分完成  
**核心目标**: 实现策略参数演进、集成GraphNN和RL模型

---

## 3.1 策略演进框架设计

### 设计目标

支持量化策略参数的自动演进，根据市场状态动态调整策略参数。

### 架构设计

```
rdagent/scenarios/qlib/strategy/
├── __init__.py
├── strategy_template.py      # 策略模板（新增，200行）
├── strategy_coder.py          # 策略代码生成器（新增，300行）
├── strategy_runner.py         # 策略运行器（新增，250行）
├── strategy_optimizer.py      # 策略参数优化器（新增，350行）
└── backtest_evaluator.py      # 回测评估器（新增，200行）
```

### 3.1.1 strategy_template.py - 策略模板

**路径**: `rdagent/scenarios/qlib/strategy/strategy_template.py`  
**新增文件，约200行**

**职责**: 定义可演进的策略模板

**核心类设计**:
```python
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class StrategyParameter:
    """策略参数定义"""
    name: str
    param_type: str  # int/float/bool/str
    default_value: Any
    search_space: Dict[str, Any]  # 搜索空间
    description: str = ""
    
    def sample(self) -> Any:
        """从搜索空间采样"""
        if self.param_type == "int":
            return random.randint(
                self.search_space["min"],
                self.search_space["max"]
            )
        elif self.param_type == "float":
            return random.uniform(
                self.search_space["min"],
                self.search_space["max"]
            )
        elif self.param_type == "bool":
            return random.choice([True, False])
        elif self.param_type == "str":
            return random.choice(self.search_space["choices"])

@dataclass
class StrategyTemplate:
    """策略模板
    
    定义一个可演进的策略，包括：
    - 策略名称
    - 可调参数列表
    - 策略代码模板
    """
    name: str
    parameters: List[StrategyParameter]
    code_template: str
    description: str = ""
    
    def instantiate(self, param_values: Dict[str, Any]) -> str:
        """实例化策略代码
        
        Args:
            param_values: {param_name: value}
        
        Returns:
            实例化后的策略代码
        """
        code = self.code_template
        for param in self.parameters:
            placeholder = f"{{{{ {param.name} }}}}"
            value = param_values.get(param.name, param.default_value)
            code = code.replace(placeholder, str(value))
        return code

# 预定义策略模板
TOPK_DROPOUT_TEMPLATE = StrategyTemplate(
    name="TopkDropoutStrategy",
    parameters=[
        StrategyParameter(
            name="topk",
            param_type="int",
            default_value=50,
            search_space={"min": 20, "max": 100},
            description="选择排名前topk的股票",
        ),
        StrategyParameter(
            name="n_drop",
            param_type="int",
            default_value=5,
            search_space={"min": 1, "max": 20},
            description="每次调仓时剔除的股票数量",
        ),
    ],
    code_template="""
class TopkDropoutStrategy(BaseSignalStrategy):
    def __init__(self, topk={{ topk }}, n_drop={{ n_drop }}, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
    """,
    description="Topk Dropout策略模板",
)
```

**集成点**: 被StrategyCoder使用  
**风险**: 低  
**工作量**: 1.5人日

---

### 3.1.2 strategy_coder.py - 策略代码生成器

**路径**: `rdagent/scenarios/qlib/strategy/strategy_coder.py`  
**新增文件，约300行**

**职责**: 基于LLM生成策略代码

**核心类设计**:
```python
class StrategyCoder(Developer):
    """策略代码生成器
    
    类似于ModelCoder和FactorCoder，负责生成策略代码。
    """
    
    def develop(self, exp: StrategyExperiment) -> StrategyExperiment:
        """生成策略代码
        
        Args:
            exp: 策略实验对象
        
        Returns:
            包含生成代码的实验对象
        """
        # 1. 构造提示词
        context = self._prepare_context(exp)
        
        # 2. 调用LLM生成代码
        prompt = self._build_prompt(context)
        response = self.llm.chat(prompt)
        
        # 3. 解析代码
        strategy_code = self._parse_code(response)
        
        # 4. 写入实验工作区
        exp.strategy_code = strategy_code
        self._write_to_workspace(exp, strategy_code)
        
        return exp
    
    def _prepare_context(self, exp: StrategyExperiment) -> Dict[str, Any]:
        """准备生成上下文"""
        return {
            "hypothesis": exp.hypothesis,
            "market_regime": exp.market_regime,  # 当前市场状态
            "available_strategies": self._list_strategy_templates(),
            "historical_performance": self._get_historical_performance(),
        }
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """构造LLM提示词"""
        # 从prompts.yaml加载
        template = T("scenarios.qlib.prompts:strategy_generation").r()
        return template.format(**context)
```

**提示词修改**:

**文件**: `rdagent/scenarios/qlib/prompts.yaml`

**新增提示词**:
```yaml
strategy_generation: |
  你是一个量化策略专家。请基于以下信息生成一个量化交易策略：
  
  # 假设
  {{ hypothesis }}
  
  # 当前市场状态
  {{ market_regime }}
  
  # 可用策略模板
  {{ available_strategies }}
  
  # 历史表现
  {{ historical_performance }}
  
  请生成一个Python策略类，继承自BaseSignalStrategy，包括：
  1. __init__方法定义可调参数
  2. generate_trade_decision方法实现交易逻辑
  3. 参数说明
  
  输出格式：
  ```python
  # 策略代码
  ```

strategy_hypothesis_specification: |
  策略假设应该包括：
  1. 策略类型（趋势跟踪/均值回归/统计套利等）
  2. 关键参数（如topk、n_drop、持仓周期等）
  3. 适用市场环境（牛市/熊市/震荡）
  4. 风险控制措施
```

**集成点**: 被演进循环调用  
**风险**: 中（LLM生成代码质量）  
**工作量**: 2.5人日

---

### 3.1.3 strategy_runner.py - 策略运行器

**路径**: `rdagent/scenarios/qlib/strategy/strategy_runner.py`  
**新增文件，约250行**

**职责**: 运行策略回测

**核心类设计**:
```python
class StrategyRunner(Developer):
    """策略运行器
    
    执行策略回测，评估策略性能。
    """
    
    def develop(self, exp: StrategyExperiment) -> StrategyExperiment:
        """运行策略回测
        
        Args:
            exp: 策略实验对象
        
        Returns:
            包含回测结果的实验对象
        """
        # 1. 加载策略代码
        strategy_cls = self._load_strategy(exp)
        
        # 2. 配置Qlib回测
        backtest_config = self._build_backtest_config(exp)
        
        # 3. 运行回测
        portfolio_metrics = qrun(backtest_config)
        
        # 4. 提取关键指标
        exp.results = self._extract_metrics(portfolio_metrics)
        
        return exp
    
    def _build_backtest_config(self, exp: StrategyExperiment) -> Dict:
        """构造Qlib回测配置"""
        return {
            "task": {
                "market": "csi300",
                "model": exp.model_path,  # 使用演进得到的模型
            },
            "strategy": {
                "class": exp.strategy_class_name,
                "module_path": exp.workspace / "strategy.py",
                "kwargs": exp.strategy_params,
            },
            "backtest": {
                "start_time": exp.backtest_start,
                "end_time": exp.backtest_end,
                "account": 100000000,
                "benchmark": "SH000300",
            },
        }
    
    def _extract_metrics(self, portfolio_metrics: pd.DataFrame) -> Dict[str, float]:
        """提取关键指标"""
        return {
            "annualized_return": portfolio_metrics["return"].mean() * 252,
            "sharpe_ratio": portfolio_metrics["return"].mean() / portfolio_metrics["return"].std() * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown(portfolio_metrics),
            "win_rate": (portfolio_metrics["return"] > 0).sum() / len(portfolio_metrics),
            "turnover": portfolio_metrics["turnover"].mean(),
        }
    
    def _calculate_max_drawdown(self, portfolio_metrics: pd.DataFrame) -> float:
        """计算最大回撤"""
        cumulative_return = (1 + portfolio_metrics["return"]).cumprod()
        running_max = cumulative_return.expanding().max()
        drawdown = (cumulative_return - running_max) / running_max
        return drawdown.min()
```

**集成点**: 被演进循环调用  
**风险**: 低  
**工作量**: 2人日

---

### 3.1.4 strategy_optimizer.py - 策略参数优化器

**路径**: `rdagent/scenarios/qlib/strategy/strategy_optimizer.py`  
**新增文件，约350行**

**职责**: 优化策略参数

**核心类设计**:
```python
class StrategyOptimizer:
    """策略参数优化器
    
    使用贝叶斯优化或网格搜索优化策略参数。
    """
    
    def __init__(self, strategy_template: StrategyTemplate):
        self.strategy_template = strategy_template
        self.strategy_runner = StrategyRunner()
    
    def optimize(
        self,
        method: str = "bayesian",
        n_trials: int = 50,
        objective: str = "sharpe_ratio",
    ) -> Dict[str, Any]:
        """优化策略参数
        
        Args:
            method: 优化方法（bayesian/grid_search/random）
            n_trials: 试验次数
            objective: 优化目标（sharpe_ratio/annualized_return）
        
        Returns:
            最优参数 {param_name: value}
        """
        if method == "bayesian":
            return self._bayesian_optimization(n_trials, objective)
        elif method == "grid_search":
            return self._grid_search(objective)
        elif method == "random":
            return self._random_search(n_trials, objective)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _bayesian_optimization(
        self, 
        n_trials: int, 
        objective: str
    ) -> Dict[str, Any]:
        """贝叶斯优化"""
        import optuna
        
        def objective_func(trial: optuna.Trial) -> float:
            # 1. 采样参数
            param_values = {}
            for param in self.strategy_template.parameters:
                if param.param_type == "int":
                    param_values[param.name] = trial.suggest_int(
                        param.name,
                        param.search_space["min"],
                        param.search_space["max"],
                    )
                elif param.param_type == "float":
                    param_values[param.name] = trial.suggest_float(
                        param.name,
                        param.search_space["min"],
                        param.search_space["max"],
                    )
            
            # 2. 实例化策略
            strategy_code = self.strategy_template.instantiate(param_values)
            
            # 3. 运行回测
            exp = StrategyExperiment(strategy_code=strategy_code)
            exp = self.strategy_runner.develop(exp)
            
            # 4. 返回目标值
            return exp.results[objective]
        
        # 创建优化任务
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=n_trials)
        
        return study.best_params
    
    def _grid_search(self, objective: str) -> Dict[str, Any]:
        """网格搜索（简化版）"""
        # 生成参数网格
        param_grid = self._generate_param_grid()
        
        best_params = None
        best_score = -float("inf")
        
        for param_values in param_grid:
            # 运行回测
            strategy_code = self.strategy_template.instantiate(param_values)
            exp = StrategyExperiment(strategy_code=strategy_code)
            exp = self.strategy_runner.develop(exp)
            
            # 更新最优参数
            score = exp.results[objective]
            if score > best_score:
                best_score = score
                best_params = param_values
        
        return best_params
```

**依赖**: `optuna>=3.0.0`  
**集成点**: 被API调用、被演进循环调用  
**风险**: 中（优化耗时长）  
**工作量**: 3人日

---

### 3.1.5 集成到演进循环

**修改文件**: `rdagent/app/qlib_rd_loop/quant.py`

**新增策略演进分支**:
```python
class QuantRDLoop(RDLoop):
    def __init__(self, PROP_SETTING: BasePropSetting):
        # 现有初始化...
        
        # 新增：策略演进组件
        self.strategy_coder: Developer = import_class(PROP_SETTING.strategy_coder)(scen)
        self.strategy_runner: Developer = import_class(PROP_SETTING.strategy_runner)(scen)
    
    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        # 现有逻辑：生成因子/模型假设
        
        # 新增：决策是否演进策略
        if self._should_evolve_strategy():
            # 生成策略假设
            strategy_hypothesis = self.hypothesis_gen.gen_strategy_hypothesis(self.trace)
            
            # 转换为实验
            strategy_exp = self._hypothesis_to_strategy_experiment(strategy_hypothesis)
            
            prev_out["strategy_exp"] = strategy_exp
        
        return prev_out
    
    def _should_evolve_strategy(self) -> bool:
        """决策是否演进策略
        
        规则：
        1. 每N个循环演进一次策略
        2. 当市场状态发生变化时
        3. 当模型性能提升但策略性能未提升时
        """
        # 简化实现：每5个循环演进一次
        return len(self.trace.hist) % 5 == 0
```

**工作量**: 2人日

---

## 3.2 高级模型支持设计

### 3.2.1 GraphNN模型适配器

**路径**: `rdagent/components/coder/model_coder/adapters/graph_adapter.py`  
**新增文件，约400行**

**职责**: 支持图神经网络模型（用于股票关系建模）

**核心类设计**:
```python
class GraphAdapter(ModelAdapter):
    """图神经网络适配器
    
    支持的模型：
    - GCN (Graph Convolutional Network)
    - GAT (Graph Attention Network)
    - GraphSAGE
    """
    
    def create_model(self, model_config: Dict[str, Any]) -> Any:
        """创建图模型"""
        model_name = model_config.get("model_name", "GCN")
        
        if model_name == "GCN":
            return self._create_gcn(model_config)
        elif model_name == "GAT":
            return self._create_gat(model_config)
        elif model_name == "GraphSAGE":
            return self._create_graphsage(model_config)
        else:
            raise ValueError(f"Unknown graph model: {model_name}")
    
    def _create_gcn(self, config: Dict[str, Any]) -> Any:
        """创建GCN模型"""
        import torch
        import torch_geometric.nn as pyg_nn
        
        class GCNModel(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
                self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels)
            
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x
        
        return GCNModel(
            in_channels=config.get("in_channels", 64),
            hidden_channels=config.get("hidden_channels", 128),
            out_channels=config.get("out_channels", 1),
        )
    
    def train(self, model, X_train, y_train, X_valid, y_valid, **kwargs):
        """训练图模型
        
        注意：X_train应该包含图结构信息（edge_index）
        """
        # 构造图数据
        graph_data = self._build_graph_data(X_train, y_train)
        
        # 训练循环
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 0.001))
        criterion = torch.nn.MSELoss()
        
        for epoch in range(kwargs.get("n_epochs", 100)):
            model.train()
            optimizer.zero_grad()
            
            out = model(graph_data.x, graph_data.edge_index)
            loss = criterion(out, graph_data.y)
            
            loss.backward()
            optimizer.step()
        
        return model
    
    def _build_graph_data(self, X, y):
        """构造图数据
        
        将股票之间的关系转换为图结构。
        关系可以基于：
        - 行业相似度
        - 因子相关性
        - 供应链关系
        """
        import torch
        from torch_geometric.data import Data
        
        # 简化实现：基于因子相关性构造边
        correlation_matrix = np.corrcoef(X)
        edge_index = []
        
        threshold = 0.5
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                if correlation_matrix[i, j] > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # 无向图
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        x = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
        
        return Data(x=x, edge_index=edge_index, y=y)
```

**依赖**: `torch-geometric>=2.3.0`  
**集成点**: 注册到ModelRegistry  
**风险**: 高（图结构构造复杂）  
**工作量**: 4人日

---

### 3.2.2 RL模型适配器

**路径**: `rdagent/components/coder/model_coder/adapters/rl_adapter.py`  
**新增文件，约450行**

**职责**: 支持强化学习模型（用于交易决策）

**核心类设计**:
```python
class RLAdapter(ModelAdapter):
    """强化学习适配器
    
    支持的算法：
    - DQN (Deep Q-Network)
    - PPO (Proximal Policy Optimization)
    - A3C (Asynchronous Advantage Actor-Critic)
    """
    
    def create_model(self, model_config: Dict[str, Any]) -> Any:
        """创建RL模型"""
        model_name = model_config.get("model_name", "PPO")
        
        if model_name == "DQN":
            return self._create_dqn(model_config)
        elif model_name == "PPO":
            return self._create_ppo(model_config)
        else:
            raise ValueError(f"Unknown RL model: {model_name}")
    
    def _create_ppo(self, config: Dict[str, Any]) -> Any:
        """创建PPO模型"""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # 创建交易环境
        env = self._create_trading_env(config)
        env = DummyVecEnv([lambda: env])
        
        # 创建PPO agent
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            verbose=1,
        )
        
        return model
    
    def _create_trading_env(self, config: Dict[str, Any]):
        """创建交易环境
        
        定义状态空间、动作空间和奖励函数。
        """
        import gym
        from gym import spaces
        
        class TradingEnv(gym.Env):
            def __init__(self, data, initial_balance=100000):
                super().__init__()
                self.data = data
                self.initial_balance = initial_balance
                
                # 定义动作空间：[持仓比例]，范围[-1, 1]
                # -1表示做空100%，1表示做多100%
                self.action_space = spaces.Box(
                    low=-1, high=1, shape=(1,), dtype=np.float32
                )
                
                # 定义状态空间：[价格, 技术指标, 持仓]
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
            
            def step(self, action):
                # 执行交易动作
                position = action[0]
                
                # 计算收益
                self.current_step += 1
                price_change = self.data[self.current_step] / self.data[self.current_step - 1] - 1
                reward = position * price_change
                
                # 更新状态
                done = self.current_step >= len(self.data) - 1
                obs = self._get_observation()
                
                return obs, reward, done, {}
            
            def reset(self):
                self.current_step = 0
                self.balance = self.initial_balance
                return self._get_observation()
            
            def _get_observation(self):
                # 构造观测状态
                return np.zeros(10, dtype=np.float32)  # 简化实现
        
        return TradingEnv(config.get("data"))
    
    def train(self, model, X_train, y_train, X_valid, y_valid, **kwargs):
        """训练RL模型"""
        # RL训练不需要X_train, y_train
        # 通过环境交互学习
        total_timesteps = kwargs.get("total_timesteps", 100000)
        model.learn(total_timesteps=total_timesteps)
        return model
    
    def predict(self, model, X_test: np.ndarray) -> np.ndarray:
        """RL预测"""
        predictions = []
        for obs in X_test:
            action, _ = model.predict(obs, deterministic=True)
            predictions.append(action[0])
        return np.array(predictions)
```

**依赖**: `stable-baselines3>=2.0.0`, `gym>=0.26.0`  
**集成点**: 注册到ModelRegistry  
**风险**: 高（RL训练不稳定、环境设计复杂）  
**工作量**: 5人日

---

## Phase 3 工作量汇总

| 模块 | 工作量 | 风险 |
|------|--------|------|
| 策略演进框架 | 11人日 | 中 |
| - strategy_template.py | 1.5人日 | 低 |
| - strategy_coder.py | 2.5人日 | 中 |
| - strategy_runner.py | 2人日 | 低 |
| - strategy_optimizer.py | 3人日 | 中 |
| - 集成到演进循环 | 2人日 | 中 |
| GraphNN适配器 | 4人日 | 高 |
| RL适配器 | 5人日 | 高 |
| 提示词工程 | 2人日 | 中 |
| 测试与文档 | 3人日 | 低 |
| **总计** | **25人日** | - |

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 策略参数优化耗时长 | 高 | 高 | 限制优化次数，使用缓存 |
| LLM生成策略代码质量差 | 中 | 中 | 人工审核，模板约束 |
| GraphNN图结构构造复杂 | 高 | 高 | 先用简单相关性，逐步优化 |
| RL训练不稳定 | 高 | 高 | 调参，使用预训练模型 |
| RL环境设计不合理 | 高 | 中 | 参考现有FinRL实现 |

## 开发顺序

```
Week 1-2: 策略演进基础
  ├─ strategy_template.py
  ├─ strategy_coder.py
  └─ strategy_runner.py

Week 3: 策略参数优化
  ├─ strategy_optimizer.py
  └─ 集成到演进循环

Week 4: GraphNN适配器
  ├─ graph_adapter.py开发
  ├─ 图结构构造
  └─ 测试

Week 5: RL适配器（可选）
  ├─ rl_adapter.py开发
  ├─ 交易环境设计
  └─ 测试

缓冲时间：视情况调整
```

## 验收标准

✅ 策略参数可自动演进  
✅ 贝叶斯优化可正常运行  
✅ GraphNN模型可训练（至少GCN）  
✅ RL模型可训练（可选，PPO）  
✅ API支持策略优化接口  
✅ 单元测试通过率 > 80%
