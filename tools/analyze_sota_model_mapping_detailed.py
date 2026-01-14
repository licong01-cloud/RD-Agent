"""
详细分析Session结构，找出SOTA实验与模型文件的精确映射关系
目标：不遍历workspace，实现一对一映射
"""
import pickle
from pathlib import Path, WindowsPath
import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, r"F:\Dev\RD-Agent-main")

# 自定义Unpickler来处理跨平台路径问题
class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return WindowsPath
        return super().find_class(module, name)

def analyze_session_structure(session_file):
    """详细分析Session结构"""
    with open(session_file, "rb") as f:
        session = CrossPlatformUnpickler(f).load()
    
    trace = session.trace
    
    print("=" * 80)
    print("Session结构分析")
    print("=" * 80)
    print(f"\nSession文件: {session_file}")
    print(f"Trace历史长度: {len(trace.hist)}")
    
    # 分析每个实验
    for idx, (exp, feedback) in enumerate(trace.hist):
        print(f"\n{'='*80}")
        print(f"实验 {idx}: {type(exp).__name__}")
        print(f"{'='*80}")
        
        print(f"\n1. 实验基本信息:")
        print(f"   类型: {type(exp).__name__}")
        print(f"   决策: {feedback.decision}")
        
        # 检查实验的所有属性
        print(f"\n2. 实验属性:")
        for attr in dir(exp):
            if not attr.startswith('_'):
                try:
                    value = getattr(exp, attr)
                    if not callable(value):
                        print(f"   {attr}: {type(value).__name__}")
                        if 'workspace' in attr.lower():
                            print(f"      值: {value}")
                        elif 'result' in attr.lower():
                            print(f"      值: {value}")
                except Exception as e:
                    pass
        
        # 详细分析workspace
        print(f"\n3. Workspace分析:")
        workspace = exp.experiment_workspace
        print(f"   Workspace类型: {type(workspace).__name__}")
        
        # 检查workspace的属性
        for attr in dir(workspace):
            if not attr.startswith('_'):
                try:
                    value = getattr(workspace, attr)
                    if not callable(value):
                        if 'path' in attr.lower() or 'uri' in attr.lower():
                            print(f"   {attr}: {value}")
                except Exception as e:
                    pass
        
        # 检查result
        print(f"\n4. Result分析:")
        result = exp.result
        if result is not None:
            print(f"   Result类型: {type(result).__name__}")
            print(f"   Result属性:")
            for attr in dir(result):
                if not attr.startswith('_'):
                    try:
                        value = getattr(result, attr)
                        if not callable(value):
                            print(f"     {attr}: {type(value).__name__}")
                            if 'path' in attr.lower() or 'uri' in attr.lower() or 'id' in attr.lower():
                                print(f"       值: {value}")
                    except Exception as e:
                        pass
        else:
            print(f"   Result为None")
        
        # 检查sub_workspace_list
        print(f"\n5. Sub Workspace List分析:")
        if hasattr(exp, 'sub_workspace_list') and exp.sub_workspace_list:
            print(f"   子Workspace数量: {len(exp.sub_workspace_list)}")
            for sub_idx, sub_ws in enumerate(exp.sub_workspace_list):
                print(f"\n   子Workspace {sub_idx}:")
                print(f"     类型: {type(sub_ws).__name__}")
                for attr in dir(sub_ws):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(sub_ws, attr)
                            if not callable(value):
                                if 'path' in attr.lower() or 'uri' in attr.lower() or 'id' in attr.lower():
                                    print(f"     {attr}: {value}")
                        except Exception as e:
                            pass
        
        # 检查是否有MLflow相关的信息
        print(f"\n6. MLflow信息:")
        if hasattr(workspace, 'workspace_path'):
            workspace_path = workspace.workspace_path
            print(f"   Workspace路径: {workspace_path}")
            
            # 转换路径
            converted_path = str(workspace_path).replace("\\mnt\\f\\", "f:/").replace("\\", "/")
            print(f"   转换后路径: {converted_path}")
            
            # 检查meta.yaml
            mlruns_path = os.path.join(converted_path, "mlruns")
            if os.path.exists(mlruns_path):
                print(f"   MLruns目录存在")
                
                # 查找meta.yaml
                for root, dirs, files in os.walk(mlruns_path):
                    if "meta.yaml" in files:
                        meta_path = os.path.join(root, "meta.yaml")
                        print(f"   找到meta.yaml: {meta_path}")
                        
                        try:
                            import yaml
                            with open(meta_path, 'r', encoding='utf-8') as f:
                                meta = yaml.safe_load(f)
                            
                            print(f"   Meta信息:")
                            for key, value in meta.items():
                                print(f"     {key}: {value}")
                            
                            # 检查artifact_uri
                            if 'artifact_uri' in meta:
                                artifact_uri = meta['artifact_uri']
                                print(f"\n   Artifact URI: {artifact_uri}")
                                
                                # 提取run_id
                                if 'afdc1ca4084443dfb64c800602bbcaf7' in artifact_uri:
                                    print(f"   Run ID: afdc1ca4084443dfb64c800602bbcaf7")
                                
                                # 构建artifacts路径
                                artifacts_path = artifact_uri.replace("file://", "").replace("\\mnt\\f\\", "f:/").replace("\\", "/")
                                print(f"   Artifacts路径: {artifacts_path}")
                                
                                # 检查artifacts目录内容
                                if os.path.exists(artifacts_path):
                                    print(f"   Artifacts目录存在")
                                    artifact_files = os.listdir(artifacts_path)
                                    print(f"   Artifacts文件: {artifact_files}")
                                    
                                    # 区分训练数据和回测数据
                                    print(f"\n   数据分类:")
                                    for af in artifact_files:
                                        af_path = os.path.join(artifacts_path, af)
                                        size = os.path.getsize(af_path)
                                        if 'pred' in af.lower() or 'params' in af.lower() or 'label' in af.lower():
                                            print(f"     [训练数据] {af} ({size} bytes)")
                                        elif 'portfolio' in af.lower() or 'ret' in af.lower() or 'ic' in af.lower():
                                            print(f"     [回测数据] {af} ({size} bytes)")
                                        else:
                                            print(f"     [其他] {af} ({size} bytes)")
                        except Exception as e:
                            print(f"   读取meta.yaml失败: {e}")
                        
                        break

# 分析一个具体的session
session_file = r"F:\Dev\RD-Agent-main\log\2025-12-18_10-38-22-336632\__session__\3\1_coding"
analyze_session_structure(session_file)

print("\n" + "=" * 80)
print("关键发现")
print("=" * 80)
print("""
1. Session中包含完整的实验历史
2. 每个实验有experiment_workspace属性
3. experiment_workspace有workspace_path属性
4. workspace_path指向具体的workspace目录
5. workspace/mlruns/{exp_id}/{run_id}/meta.yaml包含artifact_uri
6. artifact_uri直接指向artifacts目录
7. artifacts目录包含训练数据和回测数据

关键问题：
- 如何区分训练数据和回测数据？
- 如何确保只获取训练数据？
- 如何实现一对一映射？

解决方案：
1. 通过meta.yaml的artifact_uri直接定位artifacts目录
2. 根据文件名区分训练数据和回测数据
3. 只读取训练相关文件（pred.pkl, params.pkl, label.pkl, config, task）
4. 不读取回测相关文件（portfolio_analysis/, ret.pkl, ic.pkl）
""")
