"""
分析因子任务中训练完的模型权重数据位置
"""
import pickle
from pathlib import Path, WindowsPath
import sys
import os
from collections import Counter

# 添加项目根目录到Python路径
sys.path.insert(0, r"F:\Dev\RD-Agent-main")

# 自定义Unpickler来处理跨平台路径问题
class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return WindowsPath
        return super().find_class(module, name)

# 分析因子任务的模型权重
def analyze_factor_model_weights(log_dir):
    """分析因子任务的模型权重位置"""
    session_folder = log_dir / "__session__"
    session_files = list(session_folder.rglob("1_coding"))
    
    if not session_files:
        return None
    
    latest_session = max(session_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_session, "rb") as f:
            session = CrossPlatformUnpickler(f).load()
        
        trace = session.trace
        
        # 找到SOTA因子实验
        sota_factor_exp = None
        for exp, feedback in trace.hist:
            if "Factor" in type(exp).__name__ and feedback.decision:
                sota_factor_exp = exp
        
        if not sota_factor_exp:
            return None
        
        # 获取workspace路径
        workspace_path = str(sota_factor_exp.experiment_workspace.workspace_path)
        converted_path = workspace_path.replace("\\mnt\\f\\", "f:/").replace("\\", "/")
        
        # 检查mlruns目录
        mlruns_path = os.path.join(converted_path, "mlruns")
        
        model_files = []
        if os.path.exists(mlruns_path):
            # 遍历mlruns目录查找模型文件
            for root, dirs, files in os.walk(mlruns_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 查找模型相关文件
                    if any(keyword in file.lower() for keyword in ['model', 'lgb', 'booster', 'params', 'pkl']):
                        model_files.append({
                            'file': file,
                            'path': file_path,
                            'size': os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                        })
        
        return {
            'log_dir': str(log_dir),
            'workspace_path': converted_path,
            'mlruns_path': mlruns_path,
            'mlruns_exists': os.path.exists(mlruns_path),
            'model_files': model_files,
            'model_count': len(model_files)
        }
    except Exception as e:
        return {
            'log_dir': str(log_dir),
            'error': str(e)
        }

# 分析一个具体的因子演进目录
log_root = Path(r"F:\Dev\RD-Agent-main\log")
log_dirs = [d for d in log_root.iterdir() if d.is_dir() and d.name != "__pycache__"]

# 选择第一个因子演进目录
target_log = Path(r"F:\Dev\RD-Agent-main\log\2025-12-18_10-38-22-336632")

print("=" * 80)
print("分析因子任务的模型权重数据")
print("=" * 80)
print(f"\n目标目录: {target_log}")

result = analyze_factor_model_weights(target_log)

if result:
    print(f"\nWorkspace路径: {result['workspace_path']}")
    print(f"MLruns路径: {result['mlruns_path']}")
    print(f"MLruns存在: {result['mlruns_exists']}")
    print(f"模型文件数量: {result['model_count']}")
    
    if result['model_files']:
        print(f"\n模型文件列表:")
        for i, mf in enumerate(result['model_files'], 1):
            print(f"\n  [{i}] {mf['file']}")
            print(f"      路径: {mf['path']}")
            print(f"      大小: {mf['size']} bytes")
            
            # 如果是模型文件，尝试读取详细信息
            if mf['file'].endswith('.pkl'):
                try:
                    import pickle
                    with open(mf['path'], 'rb') as f:
                        obj = pickle.load(f)
                    print(f"      类型: {type(obj).__name__}")
                    if hasattr(obj, '__dict__'):
                        print(f"      属性: {list(obj.__dict__.keys())}")
                except Exception as e:
                    print(f"      读取失败: {e}")
    else:
        print(f"\n未找到模型文件")
        
        # 检查artifacts目录结构
        if result['mlruns_exists']:
            print(f"\nMLruns目录结构:")
            for root, dirs, files in os.walk(result['mlruns_path']):
                level = root.replace(result['mlruns_path'], '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # 只显示前5个文件
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... 还有{len(files)-5}个文件")
                if level > 2:  # 限制深度
                    dirs[:] = []
else:
    print(f"分析失败: {result.get('error', '未知错误')}")

# 详细分析第一个因子实验的workspace
print("\n" + "=" * 80)
print("详细分析SOTA因子实验的Workspace")
print("=" * 80)

session_folder = target_log / "__session__"
session_files = list(session_folder.rglob("1_coding"))
latest_session = max(session_files, key=lambda p: p.stat().st_mtime)

with open(latest_session, "rb") as f:
    session = CrossPlatformUnpickler(f).load()

trace = session.trace

# 找到SOTA因子实验
for exp, feedback in trace.hist:
    if "Factor" in type(exp).__name__ and feedback.decision:
        print(f"\nSOTA因子实验:")
        print(f"  实验类型: {type(exp).__name__}")
        print(f"  决策: {feedback.decision}")
        
        workspace_path = str(exp.experiment_workspace.workspace_path)
        converted_path = workspace_path.replace("\\mnt\\f\\", "f:/").replace("\\", "/")
        
        print(f"  Workspace路径: {converted_path}")
        
        # 检查workspace内容
        if os.path.exists(converted_path):
            print(f"\n  Workspace目录内容:")
            items = os.listdir(converted_path)
            for item in sorted(items):
                item_path = os.path.join(converted_path, item)
                if os.path.isdir(item_path):
                    print(f"    📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    print(f"    📄 {item} ({size} bytes)")
            
            # 检查mlruns目录
            mlruns_path = os.path.join(converted_path, "mlruns")
            if os.path.exists(mlruns_path):
                print(f"\n  MLruns目录分析:")
                
                # 查找experiment和run
                exp_dirs = [d for d in os.listdir(mlruns_path) if os.path.isdir(os.path.join(mlruns_path, d))]
                print(f"    Experiments: {len(exp_dirs)}")
                
                for exp_dir in exp_dirs[:3]:  # 只显示前3个
                    exp_path = os.path.join(mlruns_path, exp_dir)
                    print(f"\n    Experiment: {exp_dir}")
                    
                    run_dirs = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
                    print(f"      Runs: {len(run_dirs)}")
                    
                    for run_dir in run_dirs[:2]:  # 只显示前2个
                        run_path = os.path.join(exp_path, run_dir)
                        print(f"\n      Run: {run_dir}")
                        
                        # 检查artifacts
                        artifacts_path = os.path.join(run_path, "artifacts")
                        if os.path.exists(artifacts_path):
                            print(f"        Artifacts:")
                            artifact_files = os.listdir(artifacts_path)
                            for af in artifact_files:
                                af_path = os.path.join(artifacts_path, af)
                                if os.path.isfile(af_path):
                                    size = os.path.getsize(af_path)
                                    print(f"          📄 {af} ({size} bytes)")
                                else:
                                    print(f"          📁 {af}/")
                        
                        # 检查meta文件
                        meta_path = os.path.join(run_path, "meta.yaml")
                        if os.path.exists(meta_path):
                            print(f"        Meta文件存在")
                            try:
                                import yaml
                                with open(meta_path, 'r', encoding='utf-8') as f:
                                    meta = yaml.safe_load(f)
                                if 'artifact_uri' in meta:
                                    print(f"          Artifact URI: {meta['artifact_uri']}")
                            except Exception as e:
                                pass
        break

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("""
1. 因子任务的模型权重数据位置：
   - 存储在workspace/mlruns/{experiment_id}/{run_id}/artifacts/目录
   - 主要文件包括：
     * model.pkl 或 model文件：LightGBM模型对象
     * params.pkl：模型参数
     * pred.pkl：预测结果
     * ic.pkl：IC分析结果

2. 从SOTA因子获取模型的方法：
   - 从session.trace.hist找到feedback.decision=True的因子实验
   - 获取该实验的experiment_workspace.workspace_path
   - 在workspace/mlruns目录下查找最新的run
   - 从artifacts目录获取模型文件

3. 用于实盘选股：
   - 加载模型文件（model.pkl）
   - 加载因子数据
   - 使用模型预测股票收益率
   - 根据预测分数进行TopK选股
   - 等权重买入

4. 模型复用：
   - 模型文件可以直接加载使用
   - 需要确保因子数据格式一致
   - 需要定期重新训练以适应市场变化
""")
