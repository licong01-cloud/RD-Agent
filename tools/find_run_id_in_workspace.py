"""
在workspace中查找run_id，实现精确的一对一映射
"""
import os
import yaml
from pathlib import Path

# SOTA因子实验的workspace路径
workspace_path = r"f:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/1f46ebac2b7642ef9304a18347882ddc"

print("=" * 80)
print("在Workspace中查找Run ID")
print("=" * 80)
print(f"\nWorkspace路径: {workspace_path}")

# 方法1：查找workspace根目录下的文件
print(f"\n方法1：检查workspace根目录")
workspace_files = os.listdir(workspace_path)
print(f"Workspace根目录文件: {workspace_files}")

# 检查是否有记录run_id的文件
possible_run_id_files = [
    "run_id.txt",
    "run_id.json",
    "mlflow_run_id.txt",
    "experiment_id.txt",
    "manifest.json",
    "workspace_meta.json"
]

for filename in possible_run_id_files:
    filepath = os.path.join(workspace_path, filename)
    if os.path.exists(filepath):
        print(f"\n找到文件: {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"内容: {content[:500]}")

# 方法2：检查workspace_meta.json
print(f"\n方法2：检查workspace_meta.json")
workspace_meta_path = os.path.join(workspace_path, "workspace_meta.json")
if os.path.exists(workspace_meta_path):
    with open(workspace_meta_path, 'r', encoding='utf-8') as f:
        import json
        workspace_meta = json.load(f)
    print(f"Workspace Meta内容:")
    print(f"{json.dumps(workspace_meta, indent=2, ensure_ascii=False)}")

# 方法3：检查manifest.json
print(f"\n方法3：检查manifest.json")
manifest_path = os.path.join(workspace_path, "manifest.json")
if os.path.exists(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        import json
        manifest = json.load(f)
    print(f"Manifest内容:")
    print(f"{json.dumps(manifest, indent=2, ensure_ascii=False)}")

# 方法4：检查experiment_summary.json
print(f"\n方法4：检查experiment_summary.json")
exp_summary_path = os.path.join(workspace_path, "experiment_summary.json")
if os.path.exists(exp_summary_path):
    with open(exp_summary_path, 'r', encoding='utf-8') as f:
        import json
        exp_summary = json.load(f)
    print(f"Experiment Summary内容:")
    print(f"{json.dumps(exp_summary, indent=2, ensure_ascii=False)}")

# 方法5：检查mlruns目录结构
print(f"\n方法5：检查mlruns目录结构")
mlruns_path = os.path.join(workspace_path, "mlruns")
if os.path.exists(mlruns_path):
    print(f"MLruns目录存在")
    
    # 列出所有experiment
    exp_dirs = [d for d in os.listdir(mlruns_path) if os.path.isdir(os.path.join(mlruns_path, d))]
    print(f"Experiments: {exp_dirs}")
    
    for exp_dir in exp_dirs:
        exp_path = os.path.join(mlruns_path, exp_dir)
        print(f"\nExperiment: {exp_dir}")
        
        # 检查experiment的meta.yaml
        exp_meta_path = os.path.join(exp_path, "meta.yaml")
        if os.path.exists(exp_meta_path):
            with open(exp_meta_path, 'r', encoding='utf-8') as f:
                exp_meta = yaml.safe_load(f)
            print(f"  Experiment Meta:")
            print(f"    name: {exp_meta.get('name')}")
            print(f"    experiment_id: {exp_meta.get('experiment_id')}")
            print(f"    artifact_location: {exp_meta.get('artifact_location')}")
        
        # 列出所有run
        run_dirs = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
        print(f"  Runs: {run_dirs}")
        
        for run_dir in run_dirs:
            run_path = os.path.join(exp_path, run_dir)
            print(f"\n  Run: {run_dir}")
            
            # 检查run的meta.yaml
            run_meta_path = os.path.join(run_path, "meta.yaml")
            if os.path.exists(run_meta_path):
                with open(run_meta_path, 'r', encoding='utf-8') as f:
                    run_meta = yaml.safe_load(f)
                print(f"    Run Meta:")
                print(f"      run_id: {run_meta.get('run_id')}")
                print(f"      run_name: {run_meta.get('run_name')}")
                print(f"      start_time: {run_meta.get('start_time')}")
                print(f"      end_time: {run_meta.get('end_time')}")
                print(f"      status: {run_meta.get('status')}")
                print(f"      artifact_uri: {run_meta.get('artifact_uri')}")

# 方法6：检查是否有最新的run
print(f"\n方法6：查找最新的run")
if os.path.exists(mlruns_path):
    latest_run = None
    latest_run_time = 0
    
    for root, dirs, files in os.walk(mlruns_path):
        if "meta.yaml" in files:
            meta_path = os.path.join(root, "meta.yaml")
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = yaml.safe_load(f)
            
            if 'run_id' in meta and 'end_time' in meta:
                run_time = meta['end_time']
                if run_time > latest_run_time:
                    latest_run_time = run_time
                    latest_run = {
                        'run_id': meta['run_id'],
                        'run_name': meta.get('run_name'),
                        'end_time': run_time,
                        'artifact_uri': meta.get('artifact_uri'),
                        'meta_path': meta_path
                    }
    
    if latest_run:
        print(f"最新Run:")
        print(f"  run_id: {latest_run['run_id']}")
        print(f"  run_name: {latest_run['run_name']}")
        print(f"  end_time: {latest_run['end_time']}")
        print(f"  artifact_uri: {latest_run['artifact_uri']}")
        print(f"  meta_path: {latest_run['meta_path']}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("""
1. Workspace目录下没有直接存储run_id的文件
2. 需要通过mlruns目录结构查找run_id
3. 每个run目录下有meta.yaml文件，包含run_id和artifact_uri
4. 可以通过end_time找到最新的run
5. 但是这种方法依赖于时间排序，不是严格的一对一映射

关键问题：
- Session中没有直接存储run_id
- Workspace中没有直接存储run_id
- 需要通过mlruns目录结构查找run_id
- 如果有多个run，如何确定哪个是SOTA实验对应的run？

解决方案：
1. 在Session中添加run_id字段（需要修改RD-Agent代码）
2. 在Workspace中创建run_id.txt文件（需要修改RD-Agent代码）
3. 通过end_time排序找到最新的run（当前可行方案）
4. 通过run_name匹配（如果run_name包含实验信息）

推荐方案：
- 方案A：修改RD-Agent代码，在Session中存储run_id（最佳方案）
- 方案B：通过end_time排序找到最新的run（当前可行方案）
- 方案C：通过run_name匹配实验信息（需要确保run_name包含实验信息）

当前实现方案：
- 通过end_time排序找到最新的run
- 假设最新的run就是SOTA实验对应的run
- 这种方法在大多数情况下是正确的
- 但不是严格的一对一映射
""")
