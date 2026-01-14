"""
分析所有log目录的任务类型（因子演进 vs 模型演进）
"""
import pickle
from pathlib import Path, WindowsPath
import sys
from collections import Counter

# 添加项目根目录到Python路径
sys.path.insert(0, r"F:\Dev\RD-Agent-main")

# 自定义Unpickler来处理跨平台路径问题
class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return WindowsPath
        return super().find_class(module, name)

# 分析单个log目录
def analyze_log_directory(log_dir):
    """分析单个log目录的任务类型"""
    session_folder = log_dir / "__session__"
    
    # 查找session文件
    session_files = list(session_folder.rglob("1_coding"))
    
    if not session_files:
        return None
    
    # 使用最新的session
    latest_session = max(session_files, key=lambda p: p.stat().st_mtime)
    
    try:
        # 加载session
        with open(latest_session, "rb") as f:
            session = CrossPlatformUnpickler(f).load()
        
        trace = session.trace
        
        # 统计实验类型
        experiment_types = []
        for exp, feedback in trace.hist:
            exp_type = type(exp).__name__
            experiment_types.append(exp_type)
        
        type_counts = Counter(experiment_types)
        
        # 判断任务类型
        has_model = any("Model" in t for t in experiment_types)
        has_factor = any("Factor" in t for t in experiment_types)
        
        if has_model and has_factor:
            task_type = "因子+模型混合演进"
        elif has_model:
            task_type = "仅模型演进"
        elif has_factor:
            task_type = "仅因子演进"
        else:
            task_type = "未知类型"
        
        return {
            "log_dir": str(log_dir),
            "task_type": task_type,
            "experiment_types": list(type_counts.keys()),
            "type_counts": dict(type_counts),
            "total_experiments": len(trace.hist),
            "has_model": has_model,
            "has_factor": has_factor
        }
    except Exception as e:
        return {
            "log_dir": str(log_dir),
            "task_type": "解析失败",
            "error": str(e)
        }

# 分析所有log目录
log_root = Path(r"F:\Dev\RD-Agent-main\log")
log_dirs = [d for d in log_root.iterdir() if d.is_dir() and d.name != "__pycache__"]

print(f"找到 {len(log_dirs)} 个log目录")
print("=" * 80)

results = []
for log_dir in sorted(log_dirs):
    result = analyze_log_directory(log_dir)
    if result:
        results.append(result)

# 统计任务类型分布
task_type_counts = Counter(r['task_type'] for r in results if 'task_type' in r)

print("\n任务类型统计:")
print("=" * 80)
for task_type, count in task_type_counts.items():
    print(f"{task_type}: {count}")

# 列出所有目录详情
print("\n所有目录详情:")
print("=" * 80)
for result in sorted(results, key=lambda x: x['log_dir']):
    print(f"\n目录: {result['log_dir']}")
    print(f"  任务类型: {result['task_type']}")
    if 'experiment_types' in result:
        print(f"  实验类型: {result['experiment_types']}")
        print(f"  实验数量: {result['type_counts']}")
        print(f"  总实验数: {result['total_experiments']}")
    if 'error' in result:
        print(f"  错误: {result['error']}")

# 列出只有因子演进的目录
print("\n" + "=" * 80)
print("只有因子演进的目录:")
print("=" * 80)

factor_only_dirs = [r for r in results if r['task_type'] == "仅因子演进"]
if factor_only_dirs:
    for result in factor_only_dirs:
        print(f"\n{result['log_dir']}")
        print(f"  实验数量: {result['total_experiments']}")
        print(f"  实验类型分布: {result['type_counts']}")
else:
    print("未找到只有因子演进的目录")

# 详细分析第一个因子演进目录的权重分配方式
if factor_only_dirs:
    print("\n" + "=" * 80)
    print("分析因子演进目录的权重分配方式")
    print("=" * 80)
    
    # 选择第一个因子演进目录
    target_log = Path(factor_only_dirs[0]['log_dir'])
    session_folder = target_log / "__session__"
    session_files = list(session_folder.rglob("1_coding"))
    latest_session = max(session_files, key=lambda p: p.stat().st_mtime)
    
    print(f"\n分析目录: {target_log}")
    print(f"Session文件: {latest_session}")
    
    with open(latest_session, "rb") as f:
        session = CrossPlatformUnpickler(f).load()
    
    trace = session.trace
    
    # 分析因子实验的workspace配置
    print(f"\n因子实验分析:")
    for idx, (exp, feedback) in enumerate(trace.hist):
        if "Factor" in type(exp).__name__:
            print(f"\n实验 {idx}:")
            print(f"  类型: {type(exp).__name__}")
            print(f"  决策: {feedback.decision}")
            
            # 检查workspace
            if hasattr(exp, 'experiment_workspace'):
                ws = exp.experiment_workspace
                print(f"  Workspace类型: {type(ws).__name__}")
                if hasattr(ws, 'workspace_path'):
                    ws_path = str(ws.workspace_path)
                    print(f"  Workspace路径: {ws_path}")
                    
                    # 转换路径并检查配置文件
                    converted_path = ws_path.replace("\\mnt\\f\\", "f:/").replace("\\", "/")
                    import os
                    
                    # 检查配置文件
                    conf_files = []
                    if os.path.exists(converted_path):
                        for file in os.listdir(converted_path):
                            if file.endswith('.yaml') or file.endswith('.yml'):
                                conf_files.append(file)
                                print(f"  配置文件: {file}")
                                
                                # 读取配置文件查找权重配置
                                conf_path = os.path.join(converted_path, file)
                                try:
                                    with open(conf_path, 'r', encoding='utf-8') as cf:
                                        content = cf.read()
                                        if 'weight' in content.lower():
                                            print(f"    包含权重配置")
                                            # 提取权重相关行
                                            for line in content.split('\n'):
                                                if 'weight' in line.lower() and not line.strip().startswith('#'):
                                                    print(f"    {line.strip()}")
                                except Exception as e:
                                    pass
