"""
检查因子任务中的模型参数文件
"""
import pickle
import os
from pathlib import Path

# 模型参数文件路径
params_path = r"f:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/06b7d40d499b4bfa817e981fdad6f2b0/mlruns/271519888079461818/afdc1ca4084443dfb64c800602bbcaf7/artifacts/params.pkl"

print("=" * 80)
print("分析模型参数文件")
print("=" * 80)

if os.path.exists(params_path):
    print(f"\n文件路径: {params_path}")
    print(f"文件大小: {os.path.getsize(params_path)} bytes")
    
    # 加载参数文件
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    
    print(f"\n参数类型: {type(params)}")
    
    if isinstance(params, dict):
        print(f"\n参数键: {list(params.keys())}")
        
        for key, value in params.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool)):
                        print(f"  {k}: {v}")
                    elif isinstance(v, (list, tuple)):
                        print(f"  {k}: {type(v).__name__} (length={len(v)})")
                    else:
                        print(f"  {k}: {type(v).__name__}")
            else:
                print(f"  {value}")
    
    # 检查是否包含模型对象
    if hasattr(params, '__dict__'):
        print(f"\n对象属性: {list(params.__dict__.keys())}")
    
    # 检查config文件
    config_path = os.path.dirname(params_path) + "/config"
    if os.path.exists(config_path):
        print(f"\n" + "=" * 80)
        print("Config文件内容")
        print("=" * 80)
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        print(f"\nConfig类型: {type(config)}")
        
        if isinstance(config, dict):
            print(f"\nConfig键: {list(config.keys())}")
            
            # 查找模型相关配置
            if 'task' in config:
                task = config['task']
                print(f"\nTask配置:")
                if isinstance(task, dict):
                    if 'model' in task:
                        model_config = task['model']
                        print(f"  Model配置:")
                        if isinstance(model_config, dict):
                            for k, v in model_config.items():
                                print(f"    {k}: {v}")
    
    # 检查task文件
    task_path = os.path.dirname(params_path) + "/task"
    if os.path.exists(task_path):
        print(f"\n" + "=" * 80)
        print("Task文件内容")
        print("=" * 80)
        
        with open(task_path, 'rb') as f:
            task = pickle.load(f)
        
        print(f"\nTask类型: {type(task)}")
        
        if isinstance(task, dict):
            print(f"\nTask键: {list(task.keys())}")
            
            if 'model' in task:
                model_task = task['model']
                print(f"\nModel Task:")
                if isinstance(model_task, dict):
                    for k, v in model_task.items():
                        print(f"  {k}: {v}")
else:
    print(f"文件不存在: {params_path}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("""
1. 因子任务的模型权重存储：
   - params.pkl：包含模型超参数（如learning_rate, max_depth等）
   - config：包含完整的训练配置
   - task：包含任务配置（包括模型类型）
   - LightGBM模型本身可能存储在MLflow的内部结构中

2. 用于实盘选股的方法：
   方案A：使用Qlib的Recorder API
   - 通过MLflow加载Recorder
   - 使用recorder.load_object("pred.pkl")获取预测结果
   - 或者使用recorder.load_model()加载模型

   方案B：重新训练模型
   - 从config和task获取训练配置
   - 使用相同的因子数据重新训练
   - 得到新的模型用于预测

3. 推荐方案：
   - 使用方案A（Qlib Recorder API）
   - 因为模型已经训练好，可以直接加载使用
   - 避免重复训练的时间和资源消耗
""")
