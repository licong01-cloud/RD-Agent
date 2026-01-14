"""
分析 log 目录中 SOTA 因子、模型、workspace、代码文件和回测指标的具体位置
"""
import pickle
import os
from pathlib import Path

# 选择一个任务目录
task_dir = Path("log/2025-12-28_16-24-08-627650")

print("=" * 80)
print("1. Log 目录结构")
print("=" * 80)

# 列出 __session__ 目录
session_dir = task_dir / "__session__"
print(f"\nSession 目录: {session_dir}")
print(f"Session 子目录: {list(session_dir.iterdir())}")

# 列出 Loop 目录
loop_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("Loop_")])
print(f"\nLoop 目录数量: {len(loop_dirs)}")
for loop_dir in loop_dirs[:3]:
    print(f"  {loop_dir.name}")

print("\n" + "=" * 80)
print("2. Session 文件内容分析")
print("=" * 80)

# 加载第一个 session 文件
session_file = session_dir / "0" / "1_coding"
print(f"\n加载 session 文件: {session_file}")
print(f"文件大小: {session_file.stat().st_size / 1024:.2f} KB")

try:
    with open(session_file, 'rb') as f:
        session = pickle.load(f)
    
    print(f"\nSession 类型: {type(session)}")
    print(f"Session 属性: {dir(session)}")
    
    # 获取 trace
    trace = session.trace
    print(f"\nTrace 类型: {type(trace)}")
    print(f"Trace.hist 长度: {len(trace.hist)}")
    
    # 分析前几个实验
    print(f"\n前 3 个实验:")
    for i, (exp, feedback) in enumerate(trace.hist[:3]):
        print(f"\n  实验 {i}:")
        print(f"    类型: {type(exp).__name__}")
        print(f"    Hypothesis: {exp.hypothesis.action if hasattr(exp, 'hypothesis') else 'N/A'}")
        print(f"    Feedback decision: {feedback.decision if feedback else 'N/A'}")
        
        # 检查 experiment_workspace
        if hasattr(exp, 'experiment_workspace'):
            ew = exp.experiment_workspace
            print(f"    experiment_workspace 类型: {type(ew).__name__}")
            if hasattr(ew, 'workspace_path'):
                print(f"    experiment_workspace 路径: {ew.workspace_path}")
        
        # 检查 sub_workspace_list
        if hasattr(exp, 'sub_workspace_list'):
            sws = exp.sub_workspace_list
            print(f"    sub_workspace_list 长度: {len(sws) if sws else 0}")
            if sws and len(sws) > 0:
                first_sw = sws[0]
                if first_sw and hasattr(first_sw, 'workspace_path'):
                    print(f"    第一个 sub_workspace 路径: {first_sw.workspace_path}")
                if first_sw and hasattr(first_sw, 'file_dict'):
                    print(f"    第一个 sub_workspace file_dict 键: {list(first_sw.file_dict.keys())[:5]}")
        
        # 检查 result
        if hasattr(exp, 'result') and exp.result is not None:
            print(f"    result 类型: {type(exp.result)}")
            if hasattr(exp.result, 'index'):
                print(f"    result shape: {exp.result.shape}")
    
    # 查找 SOTA 实验
    print(f"\n" + "=" * 80)
    print("3. SOTA 实验分析")
    print("=" * 80)
    
    sota_experiments = []
    for i, (exp, feedback) in enumerate(trace.hist):
        if feedback and feedback.decision:
            sota_experiments.append((i, exp, feedback))
    
    print(f"\nSOTA 实验数量: {len(sota_experiments)}")
    
    if sota_experiments:
        print(f"\n前 3 个 SOTA 实验:")
        for idx, exp, feedback in sota_experiments[:3]:
            print(f"\n  SOTA 实验 {idx}:")
            print(f"    类型: {type(exp).__name__}")
            print(f"    Hypothesis: {exp.hypothesis.action if hasattr(exp, 'hypothesis') else 'N/A'}")
            
            # 获取 workspace 信息
            if hasattr(exp, 'experiment_workspace'):
                ew = exp.experiment_workspace
                if hasattr(ew, 'workspace_path'):
                    print(f"    workspace_path: {ew.workspace_path}")
                    # 检查 workspace 目录是否存在
                    ws_path = Path(ew.workspace_path)
                    if ws_path.exists():
                        print(f"    workspace 目录存在: ✅")
                        print(f"    workspace 文件: {list(ws_path.iterdir())[:10]}")
                    else:
                        print(f"    workspace 目录存在: ❌")
            
            # 获取 factor.py 和 model.py
            if hasattr(exp, 'sub_workspace_list'):
                sws = exp.sub_workspace_list
                if sws:
                    for j, sw in enumerate(sws):
                        if sw and hasattr(sw, 'file_dict'):
                            fd = sw.file_dict
                            print(f"    sub_workspace[{j}] file_dict 键: {list(fd.keys())}")
                            # 查找 factor.py 和 model.py
                            if 'factor.py' in fd:
                                print(f"      ✅ 找到 factor.py")
                            if 'model.py' in fd:
                                print(f"      ✅ 找到 model.py")
                            # 查找权重文件
                            weight_files = [k for k in fd.keys() if k.endswith(('.pth', '.ckpt', '.bin', '.pkl'))]
                            if weight_files:
                                print(f"      ✅ 找到权重文件: {weight_files}")
            
            # 获取回测指标
            if hasattr(exp, 'result') and exp.result is not None:
                print(f"    ✅ 找到回测结果")
                if hasattr(exp.result, 'index'):
                    result_df = exp.result
                    print(f"    回测结果 shape: {result_df.shape}")
                    print(f"    回测结果列: {list(result_df.columns)[:10]}")
                    print(f"    回测结果示例: {result_df.iloc[:3, :3]}")
    
except Exception as e:
    print(f"\n❌ 加载 session 文件失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("4. Loop 目录内容分析")
print("=" * 80)

# 查看第一个 Loop 目录
if loop_dirs:
    first_loop = loop_dirs[0]
    print(f"\nLoop 目录: {first_loop}")
    print(f"Loop 子目录:")
    for item in sorted(first_loop.iterdir()):
        print(f"  {item.name}")
    
    # 查看 coding 子目录
    coding_dir = first_loop / "coding"
    if coding_dir.exists():
        print(f"\nCoding 目录内容:")
        for item in sorted(coding_dir.iterdir()):
            print(f"  {item.name}")

print("\n" + "=" * 80)
print("5. 总结")
print("=" * 80)

print("""
基于以上分析，可以得出以下结论：

1. SOTA 因子和模型获取位置：
   - 位置：log/YYYY-MM-DD_HH-MM-SS-XXXXXX/__session__/*/1_coding
   - 方式：加载 pickle 文件，遍历 trace.hist，筛选 feedback.decision=True 的实验

2. Workspace 信息获取位置：
   - 位置：同上，从 exp.experiment_workspace.workspace_path 获取
   - 或者从 exp.sub_workspace_list[].workspace_path 获取

3. factor.py 和 model.py 获取位置：
   - 位置：同上，从 exp.sub_workspace_list[].file_dict['factor.py'] 获取
   - 位置：同上，从 exp.sub_workspace_list[].file_dict['model.py'] 获取
   - 或者直接从 workspace 目录中读取

4. 模型权重文件获取位置：
   - 位置：同上，从 exp.sub_workspace_list[].file_dict 中查找 .pth/.ckpt/.bin/.pkl 文件
   - 或者直接从 workspace 目录中读取

5. 回测性能指标获取位置：
   - 位置：同上，从 exp.result 获取
   - exp.result 是一个 DataFrame，包含各种性能指标

6. 是否需要到 workspace 目录：
   - ✅ log 目录中的 session 文件包含所有信息
   - ✅ 可以直接从 session 文件获取所有信息
   - ✅ workspace 目录是可选的，用于直接访问文件
""")
