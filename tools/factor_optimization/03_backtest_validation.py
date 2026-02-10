"""
因子组合回测验证工具
使用qlib对筛选出的因子组合进行回测验证
"""
import pandas as pd
import pickle
from pathlib import Path
from typing import List
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


class FactorBacktester:
    """因子回测验证器"""
    
    def __init__(self, selected_factors_pkl: Path):
        self.selected_df = pd.read_pickle(selected_factors_pkl)
        print(f"加载 {len(self.selected_df)} 个待验证因子")
        
    def generate_factor_combinations(self, batch_size: int = 10) -> List[List[int]]:
        """生成因子组合批次用于回测"""
        total_factors = len(self.selected_df)
        batches = []
        
        for i in range(0, total_factors, batch_size):
            batch_indices = list(range(i, min(i + batch_size, total_factors)))
            batches.append(batch_indices)
        
        print(f"生成 {len(batches)} 个批次, 每批最多 {batch_size} 个因子")
        return batches
    
    def create_combined_factor_experiment(self, factor_indices: List[int]) -> QlibFactorExperiment:
        """创建组合因子实验对象"""
        experiments = []
        
        for idx in factor_indices:
            if idx < len(self.selected_df):
                row = self.selected_df.iloc[idx]
                if 'experiment' in row and row['experiment'] is not None:
                    experiments.append(row['experiment'])
        
        if not experiments:
            print("警告: 没有有效的实验对象")
            return None
        
        print(f"组合 {len(experiments)} 个因子实验")
        return experiments
    
    def export_factors_for_qlib(self, output_dir: Path):
        """导出因子数据为qlib可用格式"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_factor_data = []
        factor_names = []
        
        for idx, row in self.selected_df.iterrows():
            if 'experiment' not in row or row['experiment'] is None:
                continue
                
            exp = row['experiment']
            factor_name = row.get('factor_name', f'factor_{idx}')
            
            try:
                if hasattr(exp, 'sub_workspace_list') and exp.sub_workspace_list:
                    for ws in exp.sub_workspace_list:
                        if ws and hasattr(ws, 'workspace_path'):
                            h5_file = ws.workspace_path / 'result.h5'
                            if h5_file.exists():
                                df = pd.read_hdf(h5_file, key='factor')
                                if not df.empty:
                                    df.columns = [factor_name]
                                    all_factor_data.append(df)
                                    factor_names.append(factor_name)
                                    break
            except Exception as e:
                print(f"导出因子 {factor_name} 失败: {e}")
                continue
        
        if all_factor_data:
            combined_factors = pd.concat(all_factor_data, axis=1)
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep='last')]
            
            parquet_file = output_dir / 'combined_factors.parquet'
            combined_factors.to_parquet(parquet_file, engine='pyarrow')
            print(f"\n已导出 {len(factor_names)} 个因子到: {parquet_file}")
            
            meta_file = output_dir / 'factor_metadata.csv'
            meta_df = pd.DataFrame({
                'factor_name': factor_names,
                'factor_index': range(len(factor_names))
            })
            meta_df.to_csv(meta_file, index=False)
            print(f"元数据已保存: {meta_file}")
            
            return parquet_file
        else:
            print("没有成功导出任何因子数据")
            return None
    
    def generate_qlib_backtest_config(self, factor_file: Path, output_config: Path):
        """生成qlib回测配置文件"""
        config_template = f"""
qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn
market: csi300
benchmark: SH000300
data_handler_config: &data_handler_config
    start_time: 2015-01-01
    end_time: 2023-12-31
    fit_start_time: 2015-01-01
    fit_end_time: 2020-12-31
    instruments: csi300
    infer_processors:
        - class: FilterCol
          kwargs:
              fields_group: feature
              col_list: ["RESI5","WVMA5","RSQR5","KLEN","RSQR10","CORR5","CORD5","CORR10", 
                        "ROC60","RESI10","VSTD5","RSQR60","CORR60","WVMA60","STD5",
                        "RSQR20","CORD60","CORD10","CORR20","KLOW"
                        ]
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSRankNorm
          kwargs:
              fields_group: label
    label: ["Ref($close, -2) / Ref($close, -1) - 1"]

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
    backtest:
        start_time: 2021-01-01
        end_time: 2023-12-31
        account: 100000000
        benchmark: SH000300
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.0421
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: DataHandlerLP
                module_path: qlib.data.dataset.handler
                kwargs: *data_handler_config
            segments:
                train: [2015-01-01, 2020-12-31]
                valid: [2021-01-01, 2021-12-31] 
                test: [2022-01-01, 2023-12-31]
    record: 
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs: {{}}
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs: 
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs: 
            config: *port_analysis_config
"""
        
        with open(output_config, 'w', encoding='utf-8') as f:
            f.write(config_template)
        
        print(f"已生成qlib配置文件: {output_config}")
        return output_config


def main():
    """主函数 - 回测验证流程"""
    
    base_dir = Path("F:/Dev/RD-Agent-main/tools/factor_optimization")
    output_dir = base_dir / "output"
    
    greedy_pkl = output_dir / "selected_factors_greedy.pkl"
    cluster_pkl = output_dir / "selected_factors_cluster.pkl"
    
    selected_pkl = greedy_pkl if greedy_pkl.exists() else cluster_pkl
    
    if not selected_pkl.exists():
        print(f"请先运行 02_factor_selection_and_combination.py 生成筛选结果")
        return
    
    print("=" * 80)
    print("因子组合回测验证")
    print("=" * 80)
    
    backtester = FactorBacktester(selected_pkl)
    
    print("\n步骤1: 导出因子数据")
    print("=" * 80)
    backtest_dir = output_dir / "backtest"
    factor_file = backtester.export_factors_for_qlib(backtest_dir)
    
    if factor_file:
        print("\n步骤2: 生成qlib回测配置")
        print("=" * 80)
        config_file = backtest_dir / "backtest_config.yaml"
        backtester.generate_qlib_backtest_config(factor_file, config_file)
        
        print("\n" + "=" * 80)
        print("回测准备完成！")
        print("=" * 80)
        print(f"\n因子数据: {factor_file}")
        print(f"配置文件: {config_file}")
        print("\n下一步：")
        print("1. 检查配置文件中的数据路径和参数")
        print("2. 使用以下命令运行回测:")
        print(f"   cd {backtest_dir}")
        print(f"   qrun backtest_config.yaml")


if __name__ == "__main__":
    main()
