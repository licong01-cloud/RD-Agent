"""
统计 RD-Agent 演进因子的准确数量
"""
import pickle
import os
from pathlib import Path

# 设置路径
log_folder = Path(r"F:\Dev\RD-Agent-main\log\2026-01-13_06-56-49-446055")
session_folder = log_folder / "__session__"

# 查找最新的 session 文件
session_files = list(session_folder.glob("*/1_coding"))
if not session_files:
    print("未找到 session 文件")
    exit(1)

# 使用最新的 session
latest_session = max(session_files, key=lambda p: p.stat().st_mtime)
print(f"加载 session 文件: {latest_session}")
print(f"文件大小: {latest_session.stat().st_size / 1024 / 1024:.2f} MB")

# 加载 session
with open(latest_session, "rb") as f:
    session = pickle.load(f)

# 获取 trace
trace = session.trace
print(f"\nTrace 类型: {type(trace)}")
print(f"Trace.hist 长度: {len(trace.hist)}")

# 统计演进因子
evolved_factors = []
factor_experiments = []
model_experiments = []

for idx, (exp, feedback) in enumerate(trace.hist):
    exp_type = type(exp).__name__
    
    if exp_type == "QlibFactorExperiment":
        factor_experiments.append({
            "index": idx,
            "decision": feedback.decision,
            "hypothesis": exp.hypothesis.hypothesis[:50] if hasattr(exp, 'hypothesis') and hasattr(exp.hypothesis, 'hypothesis') else "N/A",
        })
        
        if feedback.decision:
            evolved_factors.append({
                "index": idx,
                "hypothesis": exp.hypothesis.hypothesis[:100] if hasattr(exp, 'hypothesis') and hasattr(exp.hypothesis, 'hypothesis') else "N/A",
            })
    
    elif exp_type == "QlibModelExperiment":
        model_experiments.append({
            "index": idx,
            "decision": feedback.decision,
            "hypothesis": exp.hypothesis.hypothesis[:50] if hasattr(exp, 'hypothesis') and hasattr(exp.hypothesis, 'hypothesis') else "N/A",
        })

# 输出统计结果
print("\n" + "=" * 80)
print("统计结果")
print("=" * 80)
print(f"\n总实验数量: {len(trace.hist)}")
print(f"因子实验数量: {len(factor_experiments)}")
print(f"模型实验数量: {len(model_experiments)}")
print(f"\n演进因子数量 (decision=True): {len(evolved_factors)}")
print(f"演进模型数量 (decision=True): {sum(1 for m in model_experiments if m['decision'])}")

# 输出演进因子详情
print("\n" + "=" * 80)
print("演进因子详情")
print("=" * 80)
for i, factor in enumerate(evolved_factors, 1):
    print(f"\n{i}. Loop {factor['index']}")
    print(f"   假设: {factor['hypothesis']}")

# 输出因子实验分布
print("\n" + "=" * 80)
print("因子实验分布")
print("=" * 80)
accepted_factors = [f for f in factor_experiments if f['decision']]
rejected_factors = [f for f in factor_experiments if not f['decision']]

print(f"\n接受的因子 (decision=True): {len(accepted_factors)}")
print(f"拒绝的因子 (decision=False): {len(rejected_factors)}")
print(f"接受率: {len(accepted_factors) / len(factor_experiments) * 100:.2f}%")

# 输出模型实验分布
print("\n" + "=" * 80)
print("模型实验分布")
print("=" * 80)
accepted_models = [m for m in model_experiments if m['decision']]
rejected_models = [m for m in model_experiments if not m['decision']]

print(f"\n接受的模型 (decision=True): {len(accepted_models)}")
print(f"拒绝的模型 (decision=False): {len(rejected_models)}")
if len(model_experiments) > 0:
    print(f"接受率: {len(accepted_models) / len(model_experiments) * 100:.2f}%")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"\nRD-Agent 演进出来的新因子准确数量: {len(evolved_factors)}")
print(f"Alpha158 内置因子数量: 158")
print(f"总因子数: {158 + len(evolved_factors)}")
