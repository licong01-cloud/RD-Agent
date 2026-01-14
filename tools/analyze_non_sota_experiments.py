"""
分析非SOTA实验的信息完整性

验证内容：
1. Session中是否记录了所有实验（包括非SOTA）
2. 非SOTA实验的workspace是否保留
3. 能否获取非SOTA实验的因子代码、模型代码、回测指标
"""
import pickle
from pathlib import WindowsPath
import sys
import os
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, r"F:\Dev\RD-Agent-main")

class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return WindowsPath
        return super().find_class(module, name)

def _pick_latest_session_file(log_dir: str) -> str:
    root = Path(log_dir).resolve()
    session_root = root / "__session__"
    if not session_root.exists():
        raise FileNotFoundError(f"__session__ not found under: {root}")
    files = [p for p in session_root.rglob("1_coding") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No 1_coding found under: {session_root}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0])

def analyze_all_experiments(session_file):
    """分析所有实验（包括非SOTA）"""
    print("=" * 80)
    print("分析所有实验（包括非SOTA）")
    print("=" * 80)
    
    # 加载Session
    with open(session_file, "rb") as f:
        session = CrossPlatformUnpickler(f).load()
    
    trace = session.trace
    
    print(f"\n总实验数量: {len(trace.hist)}")
    
    # 统计SOTA和非SOTA实验
    sota_count = 0
    non_sota_count = 0
    
    for exp, feedback in trace.hist:
        if feedback.decision:
            sota_count += 1
        else:
            non_sota_count += 1
    
    print(f"SOTA实验数量: {sota_count}")
    print(f"非SOTA实验数量: {non_sota_count}")
    
    # 分析每个实验
    print("\n" + "=" * 80)
    print("实验详情")
    print("=" * 80)
    
    for idx, (exp, feedback) in enumerate(trace.hist):
        print(f"\n实验 {idx + 1}:")
        print(f"  类型: {type(exp).__name__}")
        print(f"  是否SOTA: {'是' if feedback.decision else '否'}")
        
        # 获取workspace路径
        workspace_path = exp.experiment_workspace.workspace_path
        print(f"  Workspace路径: {workspace_path}")
        
        # 转换路径
        workspace_path = str(workspace_path).replace("\\mnt\\f\\", "f:/").replace("\\", "/")
        
        # 检查workspace是否存在
        if os.path.exists(workspace_path):
            print(f"  ✅ Workspace存在")
            
            # 检查mlruns目录
            mlruns_path = os.path.join(workspace_path, "mlruns")
            if os.path.exists(mlruns_path):
                print(f"  ✅ MLruns目录存在")
                
                # 查找run
                exp_dirs = [d for d in os.listdir(mlruns_path) 
                            if os.path.isdir(os.path.join(mlruns_path, d)) and d != '.trash']
                
                for exp_dir in exp_dirs:
                    exp_path = os.path.join(mlruns_path, exp_dir)
                    run_dirs = [d for d in os.listdir(exp_path) 
                                if os.path.isdir(os.path.join(exp_path, d))]
                    
                    print(f"  Experiment {exp_dir}: {len(run_dirs)} 个run")
                    
                    for run_dir in run_dirs:
                        run_path = os.path.join(exp_path, run_dir)
                        meta_path = os.path.join(run_path, "meta.yaml")
                        
                        if os.path.exists(meta_path):
                            with open(meta_path, 'r', encoding='utf-8') as f:
                                meta = yaml.safe_load(f)
                            
                            run_id = meta.get('run_id')
                            artifact_uri = meta.get('artifact_uri')
                            
                            # 转换路径
                            artifacts_path = artifact_uri.replace("file:///", "")
                            if not artifacts_path.startswith("/"):
                                artifacts_path = "/" + artifacts_path
                            artifacts_path = artifacts_path.replace("/mnt/f/", "f:/").replace("\\", "/")
                            
                            if os.path.exists(artifacts_path):
                                print(f"    ✅ Run {run_dir} 的artifacts存在")
                                print(f"       Run ID: {run_id}")
                                
                                # 列出所有文件
                                files = os.listdir(artifacts_path)
                                print(f"       文件列表: {', '.join(files[:10])}{'...' if len(files) > 10 else ''}")
                                
                                # 检查关键文件
                                key_files = {
                                    'pred.pkl': '预测结果',
                                    'params.pkl': '模型参数',
                                    'label.pkl': '标签',
                                    'config': '配置',
                                    'task': '任务',
                                    'portfolio_analysis': '回测分析',
                                    'ret.pkl': '收益率',
                                    'ic.pkl': 'IC指标'
                                }
                                
                                for filename, desc in key_files.items():
                                    filepath = os.path.join(artifacts_path, filename)
                                    if os.path.exists(filepath):
                                        print(f"       ✅ {desc} ({filename})")
                                    else:
                                        print(f"       ❌ {desc} ({filename})")
                            else:
                                print(f"    ❌ Run {run_dir} 的artifacts不存在")
            else:
                print(f"  ❌ MLruns目录不存在")
        else:
            print(f"  ❌ Workspace不存在")

def analyze_non_sota_experiment_details(session_file):
    """详细分析非SOTA实验的信息"""
    print("\n" + "=" * 80)
    print("非SOTA实验详细信息")
    print("=" * 80)
    
    # 加载Session
    with open(session_file, "rb") as f:
        session = CrossPlatformUnpickler(f).load()
    
    trace = session.trace
    
    # 找到第一个非SOTA实验
    non_sota_found = False
    for idx, (exp, feedback) in enumerate(trace.hist):
        if not feedback.decision:
            print(f"\n非SOTA实验 {idx + 1}:")
            print(f"  类型: {type(exp).__name__}")
            
            # 获取workspace路径
            workspace_path = exp.experiment_workspace.workspace_path
            workspace_path = str(workspace_path).replace("\\mnt\\f\\", "f:/").replace("\\", "/")
            
            print(f"  Workspace路径: {workspace_path}")
            
            if os.path.exists(workspace_path):
                mlruns_path = os.path.join(workspace_path, "mlruns")
                
                if os.path.exists(mlruns_path):
                    exp_dirs = [d for d in os.listdir(mlruns_path) 
                                if os.path.isdir(os.path.join(mlruns_path, d)) and d != '.trash']
                    
                    for exp_dir in exp_dirs:
                        exp_path = os.path.join(mlruns_path, exp_dir)
                        run_dirs = [d for d in os.listdir(exp_path) 
                                    if os.path.isdir(os.path.join(exp_path, d))]
                        
                        for run_dir in run_dirs:
                            run_path = os.path.join(exp_path, run_dir)
                            meta_path = os.path.join(run_path, "meta.yaml")
                            
                            if os.path.exists(meta_path):
                                with open(meta_path, 'r', encoding='utf-8') as f:
                                    meta = yaml.safe_load(f)
                                
                                artifact_uri = meta.get('artifact_uri')
                                artifacts_path = artifact_uri.replace("file:///", "")
                                if not artifacts_path.startswith("/"):
                                    artifacts_path = "/" + artifacts_path
                                artifacts_path = artifacts_path.replace("/mnt/f/", "f:/").replace("\\", "/")
                                
                                if os.path.exists(artifacts_path):
                                    print(f"\n  Artifacts目录内容:")
                                    files = os.listdir(artifacts_path)
                                    
                                    # 分类文件
                                    training_files = []
                                    backtest_files = []
                                    code_files = []
                                    other_files = []
                                    
                                    for filename in files:
                                        filepath = os.path.join(artifacts_path, filename)
                                        if os.path.isdir(filepath):
                                            if 'portfolio' in filename.lower() or 'analysis' in filename.lower():
                                                backtest_files.append(filename + '/')
                                            else:
                                                other_files.append(filename + '/')
                                        else:
                                            if filename.endswith('.pkl'):
                                                if filename in ['pred.pkl', 'params.pkl', 'label.pkl']:
                                                    training_files.append(filename)
                                                elif filename in ['ret.pkl', 'ic.pkl']:
                                                    backtest_files.append(filename)
                                                else:
                                                    other_files.append(filename)
                                            elif filename in ['config', 'task']:
                                                training_files.append(filename)
                                            elif 'code' in filename.lower():
                                                code_files.append(filename)
                                            else:
                                                other_files.append(filename)
                                    
                                    print(f"\n    训练数据文件: {', '.join(training_files) if training_files else '(none)'}")
                                    print(f"    回测数据文件: {', '.join(backtest_files) if backtest_files else '(none)'}")
                                    print(f"    代码文件: {', '.join(code_files) if code_files else '(none)'}")
                                    print(f"    其他文件: {', '.join(other_files[:5])}{'...' if len(other_files) > 5 else ''}")
                                    
                                    # 尝试加载关键文件
                                    print(f"\n    尝试加载关键文件:")
                                    
                                    # 加载pred.pkl
                                    pred_path = os.path.join(artifacts_path, "pred.pkl")
                                    if os.path.exists(pred_path):
                                        try:
                                            with open(pred_path, 'rb') as f:
                                                pred_df = pickle.load(f)
                                            print(f"      ✅ pred.pkl: {type(pred_df).__name__}, shape={pred_df.shape}")
                                        except Exception as e:
                                            print(f"      ❌ pred.pkl: {str(e)}")
                                    
                                    # 加载label.pkl
                                    label_path = os.path.join(artifacts_path, "label.pkl")
                                    if os.path.exists(label_path):
                                        try:
                                            with open(label_path, 'rb') as f:
                                                label_df = pickle.load(f)
                                            print(f"      ✅ label.pkl: {type(label_df).__name__}, shape={label_df.shape}")
                                        except Exception as e:
                                            print(f"      ❌ label.pkl: {str(e)}")
                                    
                                    # 加载task
                                    task_path = os.path.join(artifacts_path, "task")
                                    if os.path.exists(task_path):
                                        try:
                                            with open(task_path, 'rb') as f:
                                                task = pickle.load(f)
                                            print(f"      ✅ task: {type(task).__name__}")
                                        except Exception as e:
                                            print(f"      ❌ task: {str(e)}")
                                    
                                    # 检查代码文件
                                    for code_file in code_files:
                                        code_path = os.path.join(artifacts_path, code_file)
                                        if os.path.exists(code_path):
                                            try:
                                                with open(code_path, 'r', encoding='utf-8') as f:
                                                    code_content = f.read()
                                                print(f"      ✅ {code_file}: {len(code_content)} 字符")
                                            except Exception as e:
                                                print(f"      ❌ {code_file}: {str(e)}")
                                    
                                    non_sota_found = True
                                    break
                        if non_sota_found:
                            break
                if non_sota_found:
                    break
        if non_sota_found:
            break
    
    if not non_sota_found:
        print("未找到非SOTA实验")

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="", help="RD-Agent log session dir, e.g. log/2026-01-13_... ")
    parser.add_argument("--session-file", type=str, default="", help="Explicit __session__/*/1_coding path")
    args = parser.parse_args()

    if args.session_file:
        session_file = args.session_file
    elif args.log_dir:
        session_file = _pick_latest_session_file(args.log_dir)
    else:
        session_file = _pick_latest_session_file(str(Path(__file__).resolve().parents[1] / "log"))
    
    print("=" * 80)
    print("非SOTA实验信息完整性分析")
    print("=" * 80)
    print(f"\nSession文件: {session_file}")
    
    # 分析所有实验
    analyze_all_experiments(session_file)
    
    # 详细分析非SOTA实验
    analyze_non_sota_experiment_details(session_file)
    
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)
    print("""
结论：
1. Session中记录了所有实验（包括非SOTA）
2. 非SOTA实验的workspace是否保留取决于RD-Agent的配置
3. 如果workspace保留，可以获取：
   - 因子代码（code_cached.txt, code_diff.txt等）
   - 模型训练结果（pred.pkl, params.pkl, label.pkl等）
   - 回测指标（portfolio_analysis/, ret.pkl, ic.pkl等）
   - 配置信息（config, task）

建议：
- 确认RD-Agent配置中是否保留非SOTA实验的workspace
- 如果保留，可以通过相同的方法获取所有实验的信息
- 如果不保留，只能从Session中获取实验的基本信息
    """)

if __name__ == "__main__":
    main()
