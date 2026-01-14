"""
分析只有因子演进任务（无模型演进）时的模型权重获取情况
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

# 分析所有实验
print("\n" + "=" * 80)
print("所有实验类型分析")
print("=" * 80)

all_experiments = []
for idx, (exp, feedback) in enumerate(trace.hist):
    exp_type = type(exp).__name__
    
    # 检查workspace
    workspace_path = None
    if hasattr(exp, 'experiment_workspace') and exp.experiment_workspace is not None:
        workspace_path = exp.experiment_workspace.workspace_path if hasattr(exp.experiment_workspace, 'workspace_path') else None
    
    # 检查result
    result = None
    if hasattr(exp, 'result'):
        result = exp.result
    
    all_experiments.append({
        "index": idx,
        "exp_type": exp_type,
        "workspace_path": str(workspace_path) if workspace_path else None,
        "feedback_decision": feedback.decision,
        "feedback_reason": feedback.reason[:100] if feedback.reason else None,
        "result": result
    })

# 统计实验类型
from collections import Counter
type_counts = Counter(exp['exp_type'] for exp in all_experiments)
print(f"\n实验类型统计:")
for exp_type, count in type_counts.items():
    print(f"  {exp_type}: {count}")

# 显示所有实验
print(f"\n所有实验详情:")
for i, exp_info in enumerate(all_experiments):
    print(f"\n【实验 {i+1}】")
    print(f"  Trace索引: {exp_info['index']}")
    print(f"  实验类型: {exp_info['exp_type']}")
    print(f"  Workspace路径: {exp_info['workspace_path']}")
    print(f"  决策: {exp_info['feedback_decision']}")
    print(f"  原因: {exp_info['feedback_reason']}...")
    print(f"  Result: {type(exp_info['result']).__name__ if exp_info['result'] is not None else None}")
    
    # 如果result是Series，显示详细信息
    if exp_info['result'] is not None and hasattr(exp_info['result'], 'shape'):
        print(f"  Result Shape: {exp_info['result'].shape}")

# 检查是否有模型实验
model_experiments = [exp for exp in all_experiments if "Model" in exp['exp_type']]
factor_experiments = [exp for exp in all_experiments if "Factor" in exp['exp_type']]

print("\n" + "=" * 80)
print("模型 vs 因子实验分析")
print("=" * 80)
print(f"\n模型实验数量: {len(model_experiments)}")
print(f"因子实验数量: {len(factor_experiments)}")

# 如果没有模型实验，分析因子实验的workspace
if len(model_experiments) == 0:
    print("\n" + "=" * 80)
    print("无模型演进 - 分析因子实验Workspace")
    print("=" * 80)
    
    print("\n因子实验Workspace分析:")
    for i, exp_info in enumerate(factor_experiments):
        if exp_info['workspace_path']:
            print(f"\n【因子实验 {i+1}】")
            print(f"  Workspace路径: {exp_info['workspace_path']}")
            
            # 转换路径格式
            ws_path = exp_info['workspace_path'].replace("\\mnt\\f\\", "f:/").replace("\\", "/")
            print(f"  转换后路径: {ws_path}")
            
            # 检查workspace目录结构
            import os
            if os.path.exists(ws_path):
                print(f"  ✅ Workspace目录存在")
                
                # 列出workspace内容
                try:
                    items = os.listdir(ws_path)
                    print(f"  目录内容: {items}")
                    
                    # 检查是否有mlruns目录
                    mlruns_path = os.path.join(ws_path, "mlruns")
                    if os.path.exists(mlruns_path):
                        print(f"  ✅ 存在mlruns目录")
                        
                        # 检查mlruns内容
                        mlruns_items = os.listdir(mlruns_path)
                        print(f"  mlruns内容: {mlruns_items}")
                        
                        # 检查是否有模型相关文件
                        has_model_files = False
                        for root, dirs, files in os.walk(mlruns_path):
                            for file in files:
                                if 'model' in file.lower() or 'params' in file.lower():
                                    has_model_files = True
                                    print(f"  📄 发现模型相关文件: {os.path.join(root, file)}")
                        
                        if not has_model_files:
                            print(f"  ⚠️  未发现模型相关文件")
                    else:
                        print(f"  ❌ 不存在mlruns目录")
                        
                    # 检查其他可能的模型文件
                    for item in items:
                        item_path = os.path.join(ws_path, item)
                        if os.path.isfile(item_path):
                            if 'model' in item.lower() or 'pkl' in item.lower():
                                print(f"  📄 发现可能包含模型的文件: {item}")
                except Exception as e:
                    print(f"  ❌ 检查目录时出错: {e}")
            else:
                print(f"  ❌ Workspace目录不存在")
else:
    print("\n" + "=" * 80)
    print("存在模型演进")
    print("=" * 80)
    print("\n此任务包含模型演进，可以使用之前的方案获取模型权重。")

# 结论
print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("""
情况1: 有模型演进
  - trace.hist中包含Model类型实验
  - 可以通过最后一个feedback.decision=True的Model实验获取模型权重
  - 模型权重文件位于workspace/mlruns/{experiment_id}/{run_id}/artifacts/

情况2: 无模型演进（仅因子演进）
  - trace.hist中只包含Factor类型实验
  - Factor实验的workspace通常不包含训练好的模型权重文件
  - Factor实验主要产出：
    * 因子计算结果
    * 因子IC分析
    * 因子回测结果
    * 可能包含简单的评估模型（如线性回归），但不是深度学习模型权重

情况3: 因子+模型混合演进
  - trace.hist中同时包含Factor和Model类型实验
  - 需要分别处理：
    * Factor实验: 获取因子相关结果
    * Model实验: 获取模型权重文件

如果任务只有因子演进，没有模型训练：
  - 无法通过上述方案获取模型权重文件
  - 因为根本不存在训练好的模型
  - 只能获取因子相关的分析结果
""")
