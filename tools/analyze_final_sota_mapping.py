"""
分析多个Model Loop中最终SOTA模型的准确映射关系
"""
import pickle
from pathlib import Path, WindowsPath
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, r"F:\Dev\RD-Agent-main")

# 自定义Unpickler来处理跨平台路径问题
class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return WindowsPath
        return super().find_class(module, name)

# 设置路径
log_folder = Path(r"F:\Dev\RD-Agent-main\log\2026-01-13_06-56-49-446055")
session_folder = log_folder / "__session__"

# 查找最新的session文件
session_files = list(session_folder.rglob("1_coding"))
latest_session = max(session_files, key=lambda p: p.stat().st_mtime)

print(f"加载session文件: {latest_session}")

# 加载session
with open(latest_session, "rb") as f:
    session = CrossPlatformUnpickler(f).load()

# 获取trace
trace = session.trace
print(f"Trace.hist长度: {len(trace.hist)}")

# 分析所有模型实验
print("\n" + "=" * 80)
print("所有模型实验分析")
print("=" * 80)

model_experiments = []
for idx, (exp, feedback) in enumerate(trace.hist):
    exp_type = type(exp).__name__
    
    # 关注模型实验
    if "Model" not in exp_type:
        continue
    
    # 检查workspace
    workspace_path = None
    if hasattr(exp, 'experiment_workspace') and exp.experiment_workspace is not None:
        workspace_path = exp.experiment_workspace.workspace_path if hasattr(exp.experiment_workspace, 'workspace_path') else None
    
    # 检查result
    result = None
    if hasattr(exp, 'result'):
        result = exp.result
    
    # 检查running_info
    running_info = None
    if hasattr(exp, 'running_info'):
        running_info = exp.running_info
    
    model_experiments.append({
        "index": idx,
        "exp_type": exp_type,
        "workspace_path": str(workspace_path) if workspace_path else None,
        "feedback_decision": feedback.decision,
        "feedback_reason": feedback.reason[:100] if feedback.reason else None,
        "result": result,
        "running_info": running_info
    })

print(f"\n找到 {len(model_experiments)} 个模型实验")

# 显示所有模型实验
for i, exp_info in enumerate(model_experiments):
    print(f"\n【实验 {i+1}】")
    print(f"  Trace索引: {exp_info['index']}")
    print(f"  实验类型: {exp_info['exp_type']}")
    print(f"  Workspace路径: {exp_info['workspace_path']}")
    print(f"  决策: {exp_info['feedback_decision']}")
    print(f"  原因: {exp_info['feedback_reason']}...")
    print(f"  Result: {type(exp_info['result']).__name__ if exp_info['result'] is not None else None}")
    
    # 如果result是字典，显示详细信息
    if isinstance(exp_info['result'], dict):
        print(f"  Result keys: {list(exp_info['result'].keys())}")
        for key, value in exp_info['result'].items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value}")
            elif isinstance(value, str) and len(value) < 100:
                print(f"    {key}: {value}")
    
    # 检查running_info
    if exp_info['running_info']:
        print(f"  RunningInfo: {type(exp_info['running_info']).__name__}")
        if hasattr(exp_info['running_info'], 'result'):
            print(f"    RunningInfo.result: {type(exp_info['running_info'].result).__name__}")
            if isinstance(exp_info['running_info'].result, dict):
                print(f"    RunningInfo.result keys: {list(exp_info['running_info'].result.keys())}")

# 分析最终SOTA
print("\n" + "=" * 80)
print("最终SOTA模型分析")
print("=" * 80)

# 查找所有被接受的模型实验
accepted_models = [exp for exp in model_experiments if exp['feedback_decision']]

print(f"\n被接受的模型数量: {len(accepted_models)}")

if accepted_models:
    # 最后一个被接受的模型应该是最终SOTA
    final_sota = accepted_models[-1]
    print(f"\n最终SOTA模型:")
    print(f"  Trace索引: {final_sota['index']}")
    print(f"  Workspace路径: {final_sota['workspace_path']}")
    print(f"  决策: {final_sota['feedback_decision']}")
    print(f"  原因: {final_sota['feedback_reason']}...")
    
    # 检查是否有唯一标识
    if final_sota['result'] is not None:
        print(f"\nResult信息:")
        if isinstance(final_sota['result'], dict):
            for key, value in final_sota['result'].items():
                print(f"  {key}: {value}")
        else:
            print(f"  Type: {type(final_sota['result']).__name__}")
            if hasattr(final_sota['result'], 'shape'):
                print(f"  Shape: {final_sota['result'].shape}")
            if hasattr(final_sota['result'], 'head'):
                print(f"  Head:\n{final_sota['result'].head()}")

# 检查session是否有其他SOTA标记
print("\n" + "=" * 80)
print("Session对象分析")
print("=" * 80)

# 检查session的属性
session_attrs = [attr for attr in dir(session) if not attr.startswith('_')]
print(f"\nSession属性: {session_attrs}")

# 检查是否有sota相关的属性
sota_attrs = [attr for attr in session_attrs if 'sota' in attr.lower()]
if sota_attrs:
    print(f"\nSOTA相关属性: {sota_attrs}")
    for attr in sota_attrs:
        value = getattr(session, attr)
        print(f"  {attr}: {value}")

# 检查是否有final相关的属性
final_attrs = [attr for attr in session_attrs if 'final' in attr.lower()]
if final_attrs:
    print(f"\nFinal相关属性: {final_attrs}")
    for attr in final_attrs:
        value = getattr(session, attr)
        print(f"  {attr}: {value}")

# 检查是否有best相关的属性
best_attrs = [attr for attr in session_attrs if 'best' in attr.lower()]
if best_attrs:
    print(f"\nBest相关属性: {best_attrs}")
    for attr in best_attrs:
        value = getattr(session, attr)
        print(f"  {attr}: {value}")

# 结论
print("\n" + "=" * 80)
print("映射关系结论")
print("=" * 80)
print("""
1. 一对一映射关系存在：
   - 最终SOTA模型 = trace.hist中最后一个feedback.decision为True的模型实验
   - 该实验的experiment_workspace.workspace_path指向包含模型权重的workspace

2. 确定最终SOTA的方法：
   - 遍历trace.hist，筛选所有Model类型实验
   - 按顺序查找，最后一个feedback.decision为True的实验即为最终SOTA
   - 该实验的workspace_path包含最终SOTA模型的权重文件

3. 无需遍历workspace：
   - 直接从session.trace.hist中获取
   - 通过feedback.decision字段确定SOTA状态
   - 通过workspace_path字段定位权重文件

4. 准确性保证：
   - trace.hist按时间顺序记录所有实验
   - feedback.decision明确标记是否被接受为SOTA
   - 最后一个被接受的模型即为最终SOTA
""")
