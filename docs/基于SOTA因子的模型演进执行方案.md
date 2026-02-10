# 基于SOTA因子的模型演进执行方案

## 概述

本方案描述如何从最近有SOTA因子的TASK中提取因子组合，然后进行所有支持的模型的模型演进和回测。

**关键特点**：
- 不修改RD-Agent代码和qlib代码
- 使用SOTA因子作为固定输入
- 覆盖所有支持的模型
- 每个模型进行多次迭代优化

## 执行步骤

### Step 1: 收集最近有SOTA因子的TASK

**目的**：查找最近产生SOTA因子的TASK

**命令**：
```bash
cd F:/Dev/RD-Agent-main
python debug_tools/collect_latest_sota_task.py
```

**输出**：
- `debug_tools/latest_sota_task.txt`：包含TASK目录和SOTA因子信息

**说明**：
- 脚本会遍历`log`目录下的所有TASK
- 按时间排序，找到最新的有SOTA因子的TASK
- 提取SOTA因子的数量和名称

### Step 2: 提取SOTA因子列表

**目的**：从TASK中提取SOTA因子列表和代码

**命令**：
```bash
python debug_tools/extract_sota_factors.py
```

**输出**：
- `debug_tools/sota_factors_config.py`：因子配置文件
- `debug_tools/sota_factors_code.py`：因子代码文件

**说明**：
- 脚本会读取`latest_sota_task.txt`
- 提取所有SOTA因子的名称和代码
- 生成Python配置文件供后续使用

### Step 3: 生成模型演进配置文件

**目的**：生成模型演进所需的配置文件

**命令**：
```bash
python debug_tools/generate_model_config.py
```

**输出**：
- `debug_tools/model_config.json`：模型配置文件
- `debug_tools/loop_plan.json`：Loop计划文件
- `debug_tools/rdagent_model_evolution_config.json`：RD-Agent配置文件

**说明**：
- 脚本会生成所有支持的模型配置
- 生成详细的Loop计划
- 生成RD-Agent配置文件

### Step 4: 准备SOTA因子数据

**目的**：将SOTA因子数据准备好供模型训练使用

**命令**：
```bash
# 进入SOTA TASK目录
cd F:/Dev/RD-Agent-main/log/<TASK_DIR>

# 复制SOTA因子数据到新的工作目录
mkdir -p ../model_evolution_workspace/factors
cp -r */workspace/factor_data ../model_evolution_workspace/factors/
```

**说明**：
- 需要将SOTA因子的数据复制到新的工作目录
- 确保因子数据格式正确
- 包含训练集、验证集、测试集

### Step 5: 启动模型演进任务

**目的**：使用SOTA因子进行模型演进和回测

**命令**：
```bash
cd F:/Dev/RD-Agent-main

# 使用RD-Agent启动模型演进任务
python -m rdagent.app.qlib_rd_loop.quant \
    --config debug_tools/rdagent_model_evolution_config.json \
    --workspace log/model_evolution_$(date +%Y%m%d_%H%M%S)
```

**说明**：
- 使用生成的配置文件启动RD-Agent
- RD-Agent会按照Loop计划进行模型演进
- 每个Loop会使用不同的模型

## Loop数量建议

### 支持的模型列表

RD-Agent支持以下6种模型：

1. **LSTM**（时间序列模型）
   - 适合处理时间序列数据
   - 能够捕捉长期依赖关系

2. **GRU**（时间序列模型）
   - 类似LSTM，但参数更少
   - 训练速度更快

3. **LightGBM**（表格模型）
   - 梯度提升框架
   - 训练速度快，性能优秀

4. **XGBoost**（表格模型）
   - 梯度提升框架
   - 在表格数据上表现优异

5. **MLP**（深度学习模型）
   - 多层感知机
   - 适合处理表格数据

6. **Transformer**（深度学习模型）
   - 注意力机制
   - 适合处理序列数据

### Loop数量计算

**基础Loop数量**：
- 每个模型至少需要2次迭代
- 总Loop数量 = 6个模型 × 2次迭代 = 12个Loop

**详细Loop计划**：

| Loop ID | 模型名称 | 迭代次数 | 超参数组合 | 预计时间 |
|---------|---------|---------|-----------|---------|
| 0 | LSTM | 1 | hidden_size=64, num_layers=2 | ~30分钟 |
| 1 | LSTM | 2 | hidden_size=128, num_layers=3 | ~45分钟 |
| 2 | GRU | 1 | hidden_size=64, num_layers=2 | ~30分钟 |
| 3 | GRU | 2 | hidden_size=128, num_layers=3 | ~45分钟 |
| 4 | LightGBM | 1 | num_leaves=31, max_depth=5 | ~20分钟 |
| 5 | LightGBM | 2 | num_leaves=63, max_depth=7 | ~25分钟 |
| 6 | XGBoost | 1 | max_depth=5, learning_rate=0.05 | ~20分钟 |
| 7 | XGBoost | 2 | max_depth=7, learning_rate=0.1 | ~25分钟 |
| 8 | MLP | 1 | hidden_size=128, dropout=0.1 | ~25分钟 |
| 9 | MLP | 2 | hidden_size=256, dropout=0.2 | ~35分钟 |
| 10 | Transformer | 1 | d_model=64, nhead=4 | ~40分钟 |
| 11 | Transformer | 2 | d_model=128, nhead=8 | ~60分钟 |

**总预计时间**：~400分钟（约6.7小时）

### Loop数量优化建议

**方案A：快速验证（推荐）**
- 每个模型1次迭代
- 总Loop数量：6个
- 预计时间：~3小时
- 适合快速验证SOTA因子的效果

**方案B：标准配置**
- 每个模型2次迭代
- 总Loop数量：12个
- 预计时间：~7小时
- 平衡了效果和时间

**方案C：深度优化**
- 每个模型3次迭代
- 总Loop数量：18个
- 预计时间：~10小时
- 适合深度优化模型性能

## 执行命令汇总

### 完整执行流程

```bash
# Step 1: 收集最近有SOTA因子的TASK
cd F:/Dev/RD-Agent-main
python debug_tools/collect_latest_sota_task.py

# Step 2: 提取SOTA因子列表
python debug_tools/extract_sota_factors.py

# Step 3: 生成模型演进配置文件
python debug_tools/generate_model_config.py

# Step 4: 准备SOTA因子数据
# （需要根据实际情况调整）
cd F:/Dev/RD-Agent-main/log/<TASK_DIR>
mkdir -p ../model_evolution_workspace/factors
cp -r */workspace/factor_data ../model_evolution_workspace/factors/

# Step 5: 启动模型演进任务
cd F:/Dev/RD-Agent-main
python -m rdagent.app.qlib_rd_loop.quant \
    --config debug_tools/rdagent_model_evolution_config.json \
    --workspace log/model_evolution_$(date +%Y%m%d_%H%M%S)
```

### 一键执行脚本

创建一个一键执行脚本`run_model_evolution.sh`：

```bash
#!/bin/bash

# 设置工作目录
WORK_DIR="F:/Dev/RD-Agent-main"
cd $WORK_DIR

# Step 1: 收集最近有SOTA因子的TASK
echo "Step 1: 收集最近有SOTA因子的TASK..."
python debug_tools/collect_latest_sota_task.py

# 检查是否找到SOTA因子
if [ ! -f debug_tools/latest_sota_task.txt ]; then
    echo "错误：未找到有SOTA因子的TASK"
    exit 1
fi

# Step 2: 提取SOTA因子列表
echo "Step 2: 提取SOTA因子列表..."
python debug_tools/extract_sota_factors.py

# Step 3: 生成模型演进配置文件
echo "Step 3: 生成模型演进配置文件..."
python debug_tools/generate_model_config.py

# Step 4: 准备SOTA因子数据
echo "Step 4: 准备SOTA因子数据..."
TASK_DIR=$(head -1 debug_tools/latest_sota_task.txt | cut -d' ' -f2)
mkdir -p log/model_evolution_workspace/factors
cp -r log/$TASK_DIR/*/workspace/factor_data log/model_evolution_workspace/factors/

# Step 5: 启动模型演进任务
echo "Step 5: 启动模型演进任务..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python -m rdagent.app.qlib_rd_loop.quant \
    --config debug_tools/rdagent_model_evolution_config.json \
    --workspace log/model_evolution_$TIMESTAMP

echo "模型演进任务已启动！"
echo "工作目录: log/model_evolution_$TIMESTAMP"
```

使用方法：
```bash
chmod +x run_model_evolution.sh
./run_model_evolution.sh
```

## 注意事项

### 1. SOTA因子数据准备

**重要**：确保SOTA因子数据正确复制

- 检查因子数据格式是否正确
- 确保包含训练集、验证集、测试集
- 验证因子数据的时间范围

### 2. 模型演进配置

**重要**：根据实际情况调整配置

- 根据SOTA因子数量调整模型输入维度
- 根据数据集大小调整训练参数
- 根据计算资源调整并行度

### 3. 回测配置

**重要**：确保回测配置正确

- 检查市场代码是否正确
- 检查时间范围是否合理
- 检查基准指数是否正确

### 4. 资源监控

**重要**：监控计算资源使用情况

- 监控CPU使用率
- 监控内存使用情况
- 监控磁盘空间

### 5. 结果分析

**重要**：及时分析模型演进结果

- 查看每个Loop的回测结果
- 比较不同模型的性能
- 选择最优模型进行部署

## 预期结果

### 1. 模型性能对比

| 模型名称 | IC | RankIC | ICIR | RankICIR | Sharpe |
|---------|----|---------|------|----------|-------|
| LSTM | ? | ? | ? | ? | ? |
| GRU | ? | ? | ? | ? | ? |
| LightGBM | ? | ? | ? | ? | ? |
| XGBoost | ? | ? | ? | ? | ? |
| MLP | ? | ? | ? | ? | ? |
| Transformer | ? | ? | ? | ? | ? |

### 2. 最优模型选择

根据回测结果选择最优模型：
- 主要指标：IC、RankIC
- 次要指标：ICIR、RankICIR、Sharpe
- 综合考虑模型复杂度和推理速度

### 3. 模型部署

将最优模型部署到生产环境：
- 导出模型权重
- 编写推理脚本
- 集成到交易系统

## 故障排查

### 问题1：未找到有SOTA因子的TASK

**原因**：log目录下没有产生SOTA因子的TASK

**解决方案**：
1. 检查log目录是否存在
2. 检查是否有TASK运行完成
3. 检查TASK是否有SOTA因子

### 问题2：因子数据复制失败

**原因**：因子数据路径不正确

**解决方案**：
1. 检查TASK目录结构
2. 检查因子数据是否存在
3. 手动复制因子数据

### 问题3：模型训练失败

**原因**：模型配置或数据格式不正确

**解决方案**：
1. 检查模型配置文件
2. 检查因子数据格式
3. 查看错误日志

### 问题4：回测失败

**原因**：回测配置不正确

**解决方案**：
1. 检查回测配置文件
2. 检查市场代码和时间范围
3. 查看回测日志

## 总结

本方案提供了一个完整的流程，用于从SOTA因子进行模型演进和回测：

1. **收集SOTA因子**：从最近的TASK中提取SOTA因子
2. **生成配置文件**：生成模型演进所需的配置文件
3. **准备数据**：准备SOTA因子数据
4. **启动任务**：使用RD-Agent启动模型演进任务
5. **分析结果**：分析模型演进结果，选择最优模型

**推荐配置**：
- 每个模型2次迭代
- 总Loop数量：12个
- 预计时间：~7小时

**预期收益**：
- 全面评估SOTA因子的效果
- 找到最适合SOTA因子的模型
- 为后续优化提供参考
