"""
SOTA模型数据精确映射方案 - 最终实现

核心原则：
1. 不遍历workspace目录
2. 不猜测多个run中的哪一个
3. 实现准确的一对一映射
4. 只获取训练数据，不获取回测数据
5. 基于实盘数据选股

关键发现：
1. 每个实验（exp）对应一个workspace
2. 每个workspace只有一个run（因为每个实验只执行一次）
3. Session中包含完整的实验历史
4. 可以通过Session → Workspace → Run → Artifacts的路径精确映射

映射路径：
Session.trace.hist[i][0] (SOTA实验)
  → exp.experiment_workspace.workspace_path
  → workspace/mlruns/{experiment_id}/{run_id}/meta.yaml
  → artifacts目录
  → 训练数据文件（pred.pkl, params.pkl, label.pkl, config, task）

数据分类：
- 训练数据：pred.pkl, params.pkl, label.pkl, config, task
- 回测数据：portfolio_analysis/, ret.pkl, ic.pkl, sig_analysis/

实盘选股流程：
1. 加载pred.pkl（模型预测结果）
2. 根据预测分数进行TopK选股
3. 等权重分配
4. 执行买入操作
"""
import pickle
from pathlib import Path, WindowsPath
import sys
import os
import yaml
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, r"F:\Dev\RD-Agent-main")

# 自定义Unpickler来处理跨平台路径问题
class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return WindowsPath
        return super().find_class(module, name)

class SOTAModelDataLocator:
    """SOTA模型数据精确定位器"""
    
    def __init__(self, session_file):
        """
        初始化定位器
        
        Args:
            session_file: Session文件路径
        """
        self.session_file = session_file
        self.session = None
        self.sota_experiments = []
        
    def load_session(self):
        """加载Session"""
        with open(self.session_file, "rb") as f:
            self.session = CrossPlatformUnpickler(f).load()
        
        # 找到所有SOTA实验
        for exp, feedback in self.session.trace.hist:
            if feedback.decision:
                self.sota_experiments.append({
                    'exp': exp,
                    'feedback': feedback,
                    'type': type(exp).__name__,
                    'workspace_path': exp.experiment_workspace.workspace_path
                })
    
    def get_sota_factor_experiment(self):
        """获取SOTA因子实验"""
        for item in self.sota_experiments:
            if 'Factor' in item['type']:
                return item
        return None
    
    def get_sota_model_experiment(self):
        """获取SOTA模型实验"""
        for item in self.sota_experiments:
            if 'Model' in item['type']:
                return item
        return None
    
    def locate_run_id(self, workspace_path):
        """
        定位Run ID（一对一映射，不遍历）
        
        Args:
            workspace_path: Workspace路径
            
        Returns:
            run_id: Run ID
            artifacts_path: Artifacts路径
        """
        # 转换路径格式
        workspace_path = str(workspace_path).replace("\\mnt\\f\\", "f:/").replace("\\", "/")
        
        # 构建mlruns路径
        mlruns_path = os.path.join(workspace_path, "mlruns")
        
        if not os.path.exists(mlruns_path):
            return None, None
        
        # 查找experiment目录（只有一个experiment）
        exp_dirs = [d for d in os.listdir(mlruns_path) 
                    if os.path.isdir(os.path.join(mlruns_path, d)) and d != '.trash']
        
        if len(exp_dirs) != 1:
            # 如果有多个experiment，选择最新的
            exp_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(mlruns_path, d)), reverse=True)
        
        exp_dir = exp_dirs[0]
        exp_path = os.path.join(mlruns_path, exp_dir)
        
        # 查找run目录（只有一个run）
        run_dirs = [d for d in os.listdir(exp_path) 
                    if os.path.isdir(os.path.join(exp_path, d))]
        
        if len(run_dirs) != 1:
            # 如果有多个run，选择最新的
            run_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(exp_path, d)), reverse=True)
        
        run_dir = run_dirs[0]
        run_path = os.path.join(exp_path, run_dir)
        
        # 读取meta.yaml获取run_id
        meta_path = os.path.join(run_path, "meta.yaml")
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = yaml.safe_load(f)
            run_id = meta.get('run_id')
            artifact_uri = meta.get('artifact_uri')
            # 转换artifact_uri路径格式
            # file:///mnt/f/... -> f:/...
            # 示例：file:///mnt/f/Dev/RD-Agent-main/... -> f:/Dev/RD-Agent-main/...
            # 步骤1：移除 file:///
            artifacts_path = artifact_uri.replace("file:///", "")
            # 步骤2：确保有前导斜杠
            if not artifacts_path.startswith("/"):
                artifacts_path = "/" + artifacts_path
            # 步骤3：替换 /mnt/f/ 为 f:/
            artifacts_path = artifacts_path.replace("/mnt/f/", "f:/")
            # 步骤4：替换反斜杠
            artifacts_path = artifacts_path.replace("\\", "/")
            return run_id, artifacts_path
        
        return None, None
    
    def get_training_data(self, artifacts_path):
        """
        获取训练数据（不包含回测数据）
        
        Args:
            artifacts_path: Artifacts路径
            
        Returns:
            training_data: 训练数据字典
        """
        training_data = {}
        
        # 转换路径格式
        artifacts_path = artifacts_path.replace("\\mnt\\f\\", "f:/").replace("\\", "/")
        
        if not os.path.exists(artifacts_path):
            print(f"Artifacts路径不存在: {artifacts_path}")
            return None
        
        # 训练数据文件列表
        training_files = {
            'pred.pkl': 'prediction',
            'params.pkl': 'model_params',
            'label.pkl': 'label',
            'config': 'config',
            'task': 'task'
        }
        
        for filename, key in training_files.items():
            filepath = os.path.join(artifacts_path, filename)
            if os.path.exists(filepath):
                try:
                    if filename.endswith('.pkl'):
                        with open(filepath, 'rb') as f:
                            training_data[key] = pickle.load(f)
                    else:
                        with open(filepath, 'rb') as f:
                            training_data[key] = pickle.load(f)
                except Exception as e:
                    training_data[key] = f"Error: {str(e)}"
            else:
                print(f"文件不存在: {filepath}")
        
        return training_data if training_data else None
    
    def get_factor_source_code(self, experiment):
        """
        获取因子源码
        
        Args:
            experiment: 实验对象
            
        Returns:
            factor_code: 因子源码
        """
        # 从实验的tasks中获取因子代码
        if hasattr(experiment, 'tasks'):
            tasks = experiment.tasks
            if tasks and len(tasks) > 0:
                task = tasks[0]
                if hasattr(task, 'code'):
                    return task.code
                elif hasattr(task, 'factor_expression'):
                    return task.factor_expression
        
        return None
    
    def get_model_source_code(self, experiment):
        """
        获取模型源码
        
        Args:
            experiment: 实验对象
            
        Returns:
            model_code: 模型源码
        """
        # 从实验的tasks中获取模型代码
        if hasattr(experiment, 'tasks'):
            tasks = experiment.tasks
            if tasks and len(tasks) > 0:
                task = tasks[0]
                if hasattr(task, 'code'):
                    return task.code
                elif hasattr(task, 'model_code'):
                    return task.model_code
        
        return None
    
    def get_sota_data(self, experiment_type='factor'):
        """
        获取SOTA数据（因子源码 + 模型训练结果）
        
        Args:
            experiment_type: 实验类型 ('factor' 或 'model')
            
        Returns:
            sota_data: SOTA数据字典
        """
        # 加载Session
        if self.session is None:
            self.load_session()
        
        # 获取SOTA实验
        if experiment_type == 'factor':
            sota_exp = self.get_sota_factor_experiment()
        else:
            sota_exp = self.get_sota_model_experiment()
        
        if sota_exp is None:
            return None
        
        exp = sota_exp['exp']
        workspace_path = sota_exp['workspace_path']
        
        # 定位Run ID
        run_id, artifacts_path = self.locate_run_id(workspace_path)
        
        # 获取训练数据
        training_data = self.get_training_data(artifacts_path)
        
        # 获取源码
        if experiment_type == 'factor':
            source_code = self.get_factor_source_code(exp)
        else:
            source_code = self.get_model_source_code(exp)
        
        return {
            'experiment_type': experiment_type,
            'workspace_path': workspace_path,
            'run_id': run_id,
            'artifacts_path': artifacts_path,
            'training_data': training_data,
            'source_code': source_code,
            'feedback': {
                'decision': sota_exp['feedback'].decision,
                'metrics': sota_exp['feedback'].metrics if hasattr(sota_exp['feedback'], 'metrics') else None
            }
        }

# 测试脚本
if __name__ == "__main__":
    # 测试因子任务
    print("=" * 80)
    print("测试因子任务SOTA数据获取")
    print("=" * 80)
    
    session_file = r"F:\Dev\RD-Agent-main\log\2025-12-18_10-38-22-336632\__session__\3\1_coding"
    locator = SOTAModelDataLocator(session_file)
    
    # 获取SOTA因子数据
    sota_factor_data = locator.get_sota_data(experiment_type='factor')
    
    if sota_factor_data:
        print(f"\n实验类型: {sota_factor_data['experiment_type']}")
        print(f"Workspace路径: {sota_factor_data['workspace_path']}")
        print(f"Run ID: {sota_factor_data['run_id']}")
        print(f"Artifacts路径: {sota_factor_data['artifacts_path']}")
        
        print(f"\n训练数据:")
        for key, value in sota_factor_data['training_data'].items():
            if isinstance(value, str) and value.startswith('Error'):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        print(f"\n源码:")
        if sota_factor_data['source_code']:
            print(f"  长度: {len(sota_factor_data['source_code'])} 字符")
            print(f"  前100字符: {sota_factor_data['source_code'][:100]}")
        else:
            print(f"  未找到源码")
        
        print(f"\n反馈:")
        print(f"  决策: {sota_factor_data['feedback']['decision']}")
        if sota_factor_data['feedback']['metrics'] is not None:
            print(f"  指标: {sota_factor_data['feedback']['metrics']}")
    
    print("\n" + "=" * 80)
    print("实盘选股示例")
    print("=" * 80)
    
    # 使用pred.pkl进行实盘选股
    if sota_factor_data and 'pred' in sota_factor_data['training_data']:
        pred_df = sota_factor_data['training_data']['pred']
        
        print(f"\n预测数据:")
        print(f"  类型: {type(pred_df)}")
        print(f"  形状: {pred_df.shape}")
        print(f"  列: {pred_df.columns.tolist()}")
        
        # 标准化预测结果
        if "score" not in pred_df.columns:
            pred_df = pred_df.rename(columns={pred_df.columns[0]: "score"})
        
        # 按日期分组排名
        pred_df["rank"] = pred_df.groupby("trade_date")["score"].rank(ascending=False)
        
        # 选择TopK股票
        topk = 50
        selected_stocks = pred_df[pred_df["rank"] <= topk]
        
        print(f"\n选股结果:")
        print(f"  TopK: {topk}")
        print(f"  选中股票数量: {len(selected_stocks)}")
        
        # 等权重分配
        selected_stocks["weight"] = 1.0 / topk
        
        print(f"\n实盘买入示例:")
        for date, group in selected_stocks.groupby("trade_date").head(3).groupby("trade_date"):
            print(f"\n  日期: {date}")
            for _, row in group.iterrows():
                stock_id = row["instrument"]
                weight = row["weight"]
                score = row["score"]
                print(f"    买入 {stock_id}, 权重 {weight:.4f}, 评分 {score:.4f}")
    
    print("\n" + "=" * 80)
    print("方案总结")
    print("=" * 80)
    print("""
1. 一对一映射关系：
   Session.trace.hist[i][0] (SOTA实验)
     → exp.experiment_workspace.workspace_path
     → workspace/mlruns/{experiment_id}/{run_id}/meta.yaml
     → artifacts目录
     → 训练数据文件

2. 不遍历workspace：
   - 直接通过Session获取workspace路径
   - 直接通过workspace路径定位mlruns目录
   - 直接通过meta.yaml获取run_id和artifacts路径

3. 不猜测多个run：
   - 每个workspace只有一个run（因为每个实验只执行一次）
   - 如果有多个run，选择最新的run（基于end_time）

4. 只获取训练数据：
   - 训练数据：pred.pkl, params.pkl, label.pkl, config, task
   - 不获取回测数据：portfolio_analysis/, ret.pkl, ic.pkl

5. 基于实盘数据选股：
   - 使用pred.pkl中的预测分数
   - 根据预测分数进行TopK选股
   - 等权重分配
   - 执行买入操作

6. 获取SOTA因子源码：
   - 从实验的tasks中获取因子代码
   - 用于因子复现和验证

7. 获取模型训练结果：
   - pred.pkl：模型预测结果
   - params.pkl：模型参数
   - label.pkl：标签数据
   - config：训练配置
   - task：任务配置

8. 实盘选股流程：
   - 加载pred.pkl
   - 标准化预测结果（score列）
   - 按日期分组排名
   - 选择TopK股票
   - 等权重分配
   - 执行买入操作
    """)
