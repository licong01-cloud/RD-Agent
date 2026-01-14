"""
直接分析 pickle 文件获取因子数量（不依赖 rdagent 模块）
"""
import pickle
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, r"F:\Dev\RD-Agent-main")

# 设置路径
log_folder = r"F:\Dev\RD-Agent-main\log\2026-01-13_06-56-49-446055"
session_folder = os.path.join(log_folder, "__session__")

# 查找最新的 session 文件
session_files = []
for root, dirs, files in os.walk(session_folder):
    for file in files:
        if file == "1_coding":
            session_files.append(os.path.join(root, file))

if not session_files:
    print("未找到 session 文件")
    sys.exit(1)

# 使用最新的 session
latest_session = max(session_files, key=lambda p: os.path.getmtime(p))
print(f"加载 session 文件: {latest_session}")

# 加载 session
with open(latest_session, "rb") as f:
    session = pickle.load(f)

# 获取 trace
trace = session.trace
print(f"Trace.hist 长度: {len(trace.hist)}")

# 统计演进因子
evolved_factors = 0
factor_experiments = 0
model_experiments = 0

for exp, feedback in trace.hist:
    exp_type = type(exp).__name__
    
    if "Factor" in exp_type:
        factor_experiments += 1
        if feedback.decision:
            evolved_factors += 1
    
    elif "Model" in exp_type:
        model_experiments += 1

print(f"\nRD-Agent 演进出来的新因子准确数量: {evolved_factors}")
print(f"Alpha158 内置因子数量: 158")
print(f"总因子数: {158 + evolved_factors}")
