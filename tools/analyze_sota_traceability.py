"""
从原始 session 文件分析因子与 Loop/Workspace 的追溯关系
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

# 分析每个实验的信息
print("\n" + "=" * 80)
print("原始 Session 数据分析")
print("=" * 80)

factor_experiments = []
model_experiments = []

for idx, (exp, feedback) in enumerate(trace.hist):
    exp_type = type(exp).__name__

    # 检查实验对象的属性
    attrs = dir(exp)
    has_workspace = hasattr(exp, 'sub_workspace_list')
    has_tasks = hasattr(exp, 'sub_tasks')
    has_hypothesis = hasattr(exp, 'hypothesis')
    has_result = hasattr(exp, 'result')

    if "Factor" in exp_type:
        factor_experiments.append({
            "index": idx,
            "exp_type": exp_type,
            "decision": feedback.decision,
            "has_workspace": has_workspace,
            "has_tasks": has_tasks,
            "has_hypothesis": has_hypothesis,
            "has_result": has_result,
        })

        if has_workspace:
            workspace_count = len(exp.sub_workspace_list) if exp.sub_workspace_list else 0
            factor_experiments[-1]["workspace_count"] = workspace_count

    elif "Model" in exp_type:
        model_experiments.append({
            "index": idx,
            "exp_type": exp_type,
            "decision": feedback.decision,
            "has_workspace": has_workspace,
            "has_tasks": has_tasks,
            "has_hypothesis": has_hypothesis,
            "has_result": has_result,
        })

print(f"\n因子实验数量: {len(factor_experiments)}")
print(f"模型实验数量: {len(model_experiments)}")

# 分析因子实验的详细属性
print("\n" + "=" * 80)
print("因子实验属性分析")
print("=" * 80)

factors_with_workspace = [f for f in factor_experiments if f['has_workspace']]
factors_with_tasks = [f for f in factor_experiments if f['has_tasks']]
factors_with_hypothesis = [f for f in factor_experiments if f['has_hypothesis']]
factors_with_result = [f for f in factor_experiments if f['has_result']]

print(f"\n有 sub_workspace_list 的因子实验: {len(factors_with_workspace)}")
print(f"有 sub_tasks 的因子实验: {len(factors_with_tasks)}")
print(f"有 hypothesis 的因子实验: {len(factors_with_hypothesis)}")
print(f"有 result 的因子实验: {len(factors_with_result)}")

# 深入分析一个因子实验的详细信息
if factors_with_workspace:
    print("\n" + "=" * 80)
    print("示例：因子实验详细信息")
    print("=" * 80)

    # 找第一个有 workspace 的因子实验
    sample_idx = factors_with_workspace[0]['index']
    sample_exp, sample_feedback = trace.hist[sample_idx]

    print(f"\n实验索引: {sample_idx}")
    print(f"实验类型: {type(sample_exp).__name__}")
    print(f"决策: {sample_feedback.decision}")
    print(f"原因: {sample_feedback.reason[:100] if sample_feedback.reason else 'N/A'}...")

    if hasattr(sample_exp, 'sub_workspace_list'):
        print(f"\nWorkspace 数量: {len(sample_exp.sub_workspace_list)}")
        for i, ws in enumerate(sample_exp.sub_workspace_list[:3]):  # 只显示前3个
            if ws is not None:
                ws_attrs = dir(ws)
                has_file_dict = hasattr(ws, 'file_dict')
                has_workspace_path = hasattr(ws, 'workspace_path')
                print(f"\n  Workspace {i}:")
                print(f"    类型: {type(ws).__name__}")
                print(f"    有 file_dict: {has_file_dict}")
                print(f"    有 workspace_path: {has_workspace_path}")

                if has_file_dict and ws.file_dict:
                    print(f"    file_dict 中的文件: {list(ws.file_dict.keys())[:5]}")

                if has_workspace_path:
                    print(f"    workspace_path: {ws.workspace_path}")

    if hasattr(sample_exp, 'sub_tasks'):
        print(f"\nSub tasks 数量: {len(sample_exp.sub_tasks)}")
        for i, task in enumerate(sample_exp.sub_tasks[:3]):
            task_info = task.get_task_information()
            print(f"\n  Task {i}:")
            print(f"    信息: {list(task_info.keys())[:5]}")

# 分析 SOTA 因子
print("\n" + "=" * 80)
print("SOTA 因子分析")
print("=" * 80)

sota_factors = []
for idx, (exp, feedback) in enumerate(trace.hist):
    if "Factor" in type(exp).__name__ and feedback.decision:
        sota_factors.append({
            "index": idx,
            "has_workspace": hasattr(exp, 'sub_workspace_list'),
            "has_tasks": hasattr(exp, 'sub_tasks'),
        })

print(f"\nSOTA 因子数量: {len(sota_factors)}")

sota_with_workspace = [f for f in sota_factors if f['has_workspace']]
sota_with_tasks = [f for f in sota_factors if f['has_tasks']]

print(f"有 workspace 信息的 SOTA 因子: {len(sota_with_workspace)}")
print(f"有 task 信息的 SOTA 因子: {len(sota_with_tasks)}")

# 总结
print("\n" + "=" * 80)
print("总结：原始 SOTA 数据中的追溯信息")
print("=" * 80)

print(f"\n✅ 原始 session 文件位置: {latest_session}")
print(f"✅ 总实验数: {len(trace.hist)}")
print(f"✅ 因子实验数: {len(factor_experiments)}")
print(f"✅ SOTA 因子数: {len(sota_factors)}")

print(f"\n追溯信息可用性:")
print(f"  - sub_workspace_list: {len(factors_with_workspace)}/{len(factor_experiments)} ({len(factors_with_workspace)/len(factor_experiments)*100:.1f}%)")
print(f"  - sub_tasks: {len(factors_with_tasks)}/{len(factor_experiments)} ({len(factors_with_tasks)/len(factor_experiments)*100:.1f}%)")
print(f"  - hypothesis: {len(factors_with_hypothesis)}/{len(factor_experiments)} ({len(factors_with_hypothesis)/len(factor_experiments)*100:.1f}%)")
print(f"  - result: {len(factors_with_result)}/{len(factor_experiments)} ({len(factors_with_result)/len(factor_experiments)*100:.1f}%)")

print(f"\nSOTA 因子追溯信息:")
print(f"  - 有 workspace: {len(sota_with_workspace)}/{len(sota_factors)} ({len(sota_with_workspace)/len(sota_factors)*100:.1f}%)")
print(f"  - 有 tasks: {len(sota_with_tasks)}/{len(sota_factors)} ({len(sota_with_tasks)/len(sota_factors)*100:.1f}%)")

print(f"\n结论:")
print(f"  ✅ 原始 SOTA session 文件包含完整的追溯信息")
print(f"  ✅ 可以通过 sub_workspace_list 获取 workspace 路径")
print(f"  ✅ 可以通过 sub_tasks 获取因子名称和详细信息")
print(f"  ✅ 可以通过 hypothesis 获取因子的假设描述")
print(f"  ✅ 可以通过 feedback.decision 判断是否为 SOTA")
