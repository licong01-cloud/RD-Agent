"""
分析Log目录中SOTA因子使用的模型权重文件的映射关系
"""
import pickle
import os
import sys
from pathlib import Path, PosixPath, WindowsPath

# 添加项目根目录到Python路径
sys.path.insert(0, r"F:\Dev\RD-Agent-main")

# 设置路径
log_folder = Path(r"F:\Dev\RD-Agent-main\log\2026-01-13_06-56-49-446055")
session_folder = log_folder / "__session__"

# 查找最新的session文件
session_files = list(session_folder.rglob("1_coding"))

if not session_files:
    print("未找到session文件")
    sys.exit(1)

# 使用最新的session
latest_session = max(session_files, key=lambda p: p.stat().st_mtime)
print(f"加载session文件: {latest_session}")

# 自定义Unpickler来处理跨平台路径问题
class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            # 在Windows上将PosixPath转换为WindowsPath
            return WindowsPath
        return super().find_class(module, name)

# 加载session
with open(latest_session, "rb") as f:
    session = CrossPlatformUnpickler(f).load()

# 获取trace
trace = session.trace
print(f"Trace.hist长度: {len(trace.hist)}")

# 分析每个实验
print("\n" + "=" * 80)
print("SOTA模型实验与Workspace映射关系")
print("=" * 80)

sota_model_experiments = []

for idx, (exp, feedback) in enumerate(trace.hist):
    exp_type = type(exp).__name__
    
    # 只关注模型实验
    if "Model" not in exp_type:
        continue
    
    # 只关注SOTA实验（feedback.decision为True）
    if not feedback.decision:
        continue
    
    # 检查是否有workspace
    if not hasattr(exp, 'experiment_workspace') or exp.experiment_workspace is None:
        continue
    
    workspace = exp.experiment_workspace
    workspace_path = workspace.workspace_path if hasattr(workspace, 'workspace_path') else None
    
    # 检查是否有sub_workspace_list
    has_sub_workspaces = hasattr(exp, 'sub_workspace_list') and exp.sub_workspace_list
    
    sota_model_experiments.append({
        "index": idx,
        "exp_type": exp_type,
        "workspace_path": str(workspace_path) if workspace_path else None,
        "has_sub_workspaces": has_sub_workspaces,
        "feedback_decision": feedback.decision,
        "feedback_reason": feedback.reason[:100] if feedback.reason else None
    })

print(f"\n找到 {len(sota_model_experiments)} 个SOTA模型实验")

# 显示映射关系
print("\n" + "=" * 80)
print("详细的Workspace路径映射")
print("=" * 80)

for i, exp_info in enumerate(sota_model_experiments):
    print(f"\n【实验 {i+1}】")
    print(f"  Trace索引: {exp_info['index']}")
    print(f"  实验类型: {exp_info['exp_type']}")
    print(f"  Workspace路径: {exp_info['workspace_path']}")
    print(f"  有子workspace: {exp_info['has_sub_workspaces']}")
    print(f"  决策: {exp_info['feedback_decision']}")
    print(f"  原因: {exp_info['feedback_reason']}...")
    
    # 检查workspace中的mlruns目录
    if exp_info['workspace_path']:
        workspace_path = Path(exp_info['workspace_path'])
        mlruns_path = workspace_path / "mlruns"
        
        if mlruns_path.exists():
            print(f"\n  ✅ Workspace存在mlruns目录")
            
            # 列出mlruns下的实验
            if mlruns_path.is_dir():
                experiment_dirs = [d for d in mlruns_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
                print(f"  实验数量: {len(experiment_dirs)}")
                
                for exp_dir in experiment_dirs[:3]:  # 只显示前3个
                    print(f"\n    📁 实验: {exp_dir.name}")
                    
                    # 列出run
                    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    for run_dir in run_dirs[:2]:  # 只显示前2个run
                        print(f"      📄 Run: {run_dir.name}")
                        
                        # 检查artifacts
                        artifacts_path = run_dir / "artifacts"
                        if artifacts_path.exists():
                            artifact_files = list(artifacts_path.iterdir())
                            print(f"        Artifacts: {len(artifact_files)} 个文件")
                            
                            # 列出关键文件
                            key_files = ['params.pkl', 'pred.pkl', 'label.pkl', 'config']
                            for key_file in key_files:
                                file_path = artifacts_path / key_file
                                if file_path.exists():
                                    size_mb = file_path.stat().st_size / (1024 * 1024)
                                    print(f"          ✓ {key_file}: {size_mb:.2f} MB")
                        else:
                            print(f"        ✗ Artifacts目录不存在")
        else:
            print(f"\n  ✗ Workspace不存在mlruns目录")

# 总结
print("\n" + "=" * 80)
print("映射关系总结")
print("=" * 80)

if sota_model_experiments:
    latest_sota = sota_model_experiments[-1]
    print(f"\n最新SOTA模型实验:")
    print(f"  Trace索引: {latest_sota['index']}")
    print(f"  Workspace路径: {latest_sota['workspace_path']}")
    
    if latest_sota['workspace_path']:
        workspace_path = Path(latest_sota['workspace_path'])
        
        # 构建模型权重文件路径
        mlruns_path = workspace_path / "mlruns"
        
        if mlruns_path.exists():
            experiment_dirs = [d for d in mlruns_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            if experiment_dirs:
                # 找最新的实验
                latest_exp_dir = max(experiment_dirs, key=lambda d: d.stat().st_mtime)
                run_dirs = [d for d in latest_exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                
                if run_dirs:
                    # 找最新的run
                    latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
                    artifacts_path = latest_run_dir / "artifacts"
                    
                    print(f"\n模型权重文件位置:")
                    print(f"  Workspace: {workspace_path}")
                    print(f"  MLflow实验ID: {latest_exp_dir.name}")
                    print(f"  MLflow Run ID: {latest_run_dir.name}")
                    print(f"  Artifacts路径: {artifacts_path}")
                    
                    print(f"\n关键文件:")
                    for file_name in ['params.pkl', 'pred.pkl', 'label.pkl', 'config']:
                        file_path = artifacts_path / file_name
                        if file_path.exists():
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            print(f"  ✓ {file_name}: {size_mb:.2f} MB")
                        else:
                            print(f"  ✗ {file_name}: 不存在")

print("\n" + "=" * 80)
print("映射关系说明")
print("=" * 80)
print("""
1. Log目录中的session文件包含trace.hist
2. trace.hist记录了所有实验历史（exp, feedback）
3. 每个实验对象都有experiment_workspace属性
4. experiment_workspace.workspace_path指向实际的workspace目录
5. 模型权重文件存储在workspace/mlruns/{experiment_id}/{run_id}/artifacts/目录下

获取SOTA模型权重文件的步骤:
1. 加载log目录中的session文件
2. 遍历trace.hist，找到feedback.decision为True的模型实验
3. 获取该实验的experiment_workspace.workspace_path
4. 在workspace/mlruns/{experiment_id}/{run_id}/artifacts/中找到模型权重文件
""")
