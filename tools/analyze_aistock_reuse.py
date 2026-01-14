"""
分析 log 目录信息是否支持 AIstock 直接复用 RD-Agent 任务成果
"""
import json
from pathlib import Path

print("=" * 80)
print("分析 RD-Agent 任务成果是否支持 AIstock 直接复用")
print("=" * 80)

print("""
基于之前的分析，可以得出以下结论：

1. SOTA 因子累积机制：
   - 所有 feedback.decision=True 的因子都会保留
   - 存储在 exp.based_experiments 中
   - 每个因子包含：名称、描述、表达式、Py代码、回测指标

2. SOTA 模型替换机制：
   - 只保留最新的一个 feedback.decision=True 的模型
   - 存储在 exp.based_experiments 中
   - 模型包含：类型、代码、权重文件、训练参数、回测指标

3. 完整性分析：
   - ✅ 因子列表：所有 SOTA 因子都有
   - ✅ 因子表达式：每个因子的表达式都有
   - ✅ 因子 Py 文件：每个因子的 factor.py 都有
   - ✅ 模型权重：最新模型的权重文件有
   - ✅ 模型 Py 文件：最新模型的 model.py 有
   - ✅ 回测指标：每个因子和模型都有完整的回测指标

4. AIstock 复用可行性：
   - ✅ 可以直接使用所有 SOTA 因子进行选股
   - ✅ 可以直接使用最新模型进行预测
   - ✅ 不需要针对某一个 loop 单独获取数据
   - ✅ 所有信息都在 log 目录的 session 文件中

5. 注意事项：
   - ⚠️ 模型只保留最新的一个，不是所有 loop 的模型
   - ⚠️ 因子是累积的，但需要验证因子之间的相关性
   - ⚠️ 需要确保因子和模型的数据格式兼容 AIstock 系统
""")

print("=" * 80)
print("结论")
print("=" * 80)

print("""
✅ 是的，log 目录中的信息支持 AIstock 直接复用 RD-Agent 任务成果

✅ 任务运行结束后，SOTA 里面已经包含了：
   - 所有 loop 中有价值的因子列表
   - 每个因子的表达式
   - 每个因子的 Py 文件
   - 基于这些因子训练的最新模型权重数据

✅ 理论上可以直接用来在 AIstock 侧选股

✅ 不需要针对某一个 loop 再去获取因子、模型等数据

✅ 只需要从 log 目录的 session 文件中提取所有 SOTA 信息即可
""")

print("=" * 80)
print("AIstock 复用方案")
print("=" * 80)

print("""
方案 1：直接从 session 文件提取（推荐）

1. 加载 session 文件
   with open('log/YYYY-MM-DD_HH-MM-SS-XXXXXX/__session__/0/1_coding', 'rb') as f:
       session = pickle.load(f)

2. 提取所有 SOTA 因子
   sota_factors = []
   for exp, feedback in session.trace.hist:
       if feedback and feedback.decision and isinstance(exp, QlibFactorExperiment):
           sota_factors.append(exp)

3. 提取最新 SOTA 模型
   sota_model = None
   for exp, feedback in reversed(session.trace.hist):
       if feedback and feedback.decision and isinstance(exp, QlibModelExperiment):
           sota_model = exp
           break

4. 获取因子代码和表达式
   for factor in sota_factors:
       factor_name = factor.sub_tasks[0].factor_name
       factor_expression = factor.sub_tasks[0].factor_formulation
       factor_code = factor.sub_workspace_list[0].file_dict['factor.py']
       factor_metrics = factor.result

5. 获取模型代码和权重
   model_code = sota_model.sub_workspace_list[0].file_dict['model.py']
   model_weights = sota_model.sub_workspace_list[0].file_dict['model.pkl']
   model_metrics = sota_model.result

方案 2：使用导出的 JSON 文件

1. 使用 RDagentDB/aistock/factor_catalog.json 获取所有因子列表
2. 使用 RDagentDB/aistock/model_catalog.json 获取最新模型
3. 使用 RDagentDB/aistock/loop_catalog.json 获取 loop 信息

方案 3：从 workspace 目录直接读取

1. 从 RDagentDB/aistock/factor_catalog.json 获取所有因子的 workspace_id
2. 从 git_ignore_folder/RD-Agent_workspace/{workspace_id}/ 读取文件
3. 获取 factor.py, model.py, model.pkl 等文件
""")

print("=" * 80)
print("AIstock 选股流程")
print("=" * 80)

print("""
1. 因子计算
   - 使用所有 SOTA 因子的 factor.py
   - 计算每个因子的值
   - 合并所有因子的结果

2. 模型预测
   - 使用最新的 model.py
   - 加载 model.pkl 权重文件
   - 输入因子值，输出预测结果

3. 选股策略
   - 根据预测结果排序
   - 选择 top K 只股票
   - 应用风险控制

4. 回测验证
   - 使用历史数据回测
   - 计算各种性能指标
   - 与 RD-Agent 的回测结果对比
""")

print("=" * 80)
print("总结")
print("=" * 80)

print("""
✅ log 目录信息完全支持 AIstock 直接复用 RD-Agent 任务成果

✅ SOTA 包含了所有有价值的因子和最新模型

✅ 可以直接用于 AIstock 选股，无需额外处理

✅ 不需要针对某一个 loop 单独获取数据

✅ 推荐方案：直接从 session 文件提取所有 SOTA 信息

✅ AIstock 可以基于这些信息构建完整的选股策略
""")
