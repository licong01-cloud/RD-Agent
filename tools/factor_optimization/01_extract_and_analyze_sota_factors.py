"""
SOTA因子提取和分析工具
从RD-Agent的session文件中提取所有SOTA因子，并进行详细分析
"""
import pickle
import ast
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


class SOTAFactorExtractor:
    """提取和分析SOTA因子"""
    
    def __init__(self, log_folder: Path, aggregate_all_tasks: bool = True):
        self.log_folder = Path(log_folder)
        self.aggregate_all_tasks = aggregate_all_tasks
        self.sota_factors = []
        self.factor_metadata = []
        
    def find_all_task_dirs(self) -> list:
        """查找所有task目录"""
        if not self.log_folder.exists():
            return []
        
        task_dirs = []
        for item in self.log_folder.iterdir():
            if item.is_dir() and (item / "__session__").exists():
                task_dirs.append(item)
        
        task_dirs.sort(key=lambda x: x.stat().st_mtime)
        return task_dirs
    
    def load_session_from_task(self, task_dir: Path) -> any:
        """从单个task目录加载最新session"""
        session_folder = task_dir / "__session__"
        session_files = sorted(
            session_folder.glob("*.pkl"), 
            key=lambda x: x.stat().st_mtime
        )
        if not session_files:
            return None
        
        latest_session = session_files[-1]
        try:
            with open(latest_session, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"加载session失败 {latest_session}: {e}")
            return None
    
    def extract_sota_factors_from_single_task(self, task_dir: Path, task_id: str) -> list:
        """从单个task提取SOTA因子"""
        session = self.load_session_from_task(task_dir)
        if session is None:
            return []
        
        trace = session.trace
        factor_records = []
        
        for idx, (exp, feedback) in enumerate(trace.hist):
            if isinstance(exp, QlibFactorExperiment) and feedback.decision:
                # 提取每个子任务（因子）
                for task_idx, task in enumerate(exp.sub_tasks):
                    task_info = task.get_task_information()
                    factor_name = task_info.get("factor_name", f"factor_{idx}_{task_idx}")
                    
                    # 提取因子代码
                    factor_code = ""
                    if hasattr(exp, 'sub_workspace_list') and exp.sub_workspace_list:
                        if task_idx < len(exp.sub_workspace_list):
                            ws = exp.sub_workspace_list[task_idx]
                            if ws and hasattr(ws, 'file_dict'):
                                factor_code = ws.file_dict.get("factor.py", "")
                    
                    # 提取回测指标
                    metrics = {}
                    if exp.result is not None:
                        result_df = pd.DataFrame(exp.result)
                        metric_keys = [
                            "IC", "ICIR", "Rank IC", "Rank ICIR",
                            "1day.excess_return_with_cost.annualized_return",
                            "1day.excess_return_with_cost.information_ratio",
                            "1day.excess_return_with_cost.max_drawdown"
                        ]
                        for key in metric_keys:
                            if key in result_df.index:
                                metrics[key] = float(result_df.loc[key, 0])
                    
                    # 分析因子类型
                    factor_type = self._classify_factor_type(factor_code)
                    
                    factor_records.append({
                        'task_name': task_id,
                        'task_dir': str(task_dir),
                        'loop_id': idx,
                        'task_id': task_idx,
                        'factor_name': factor_name,
                        'hypothesis': exp.hypothesis.hypothesis if hasattr(exp, 'hypothesis') else '',
                        'factor_code': factor_code,
                        'factor_type': factor_type,
                        'code_length': len(factor_code),
                        'experiment': exp,
                        **metrics
                    })
        
        return factor_records
    
    def extract_sota_factors(self) -> pd.DataFrame:
        """提取所有SOTA因子及其元数据（支持多task聚合）"""
        all_factor_records = []
        
        if self.aggregate_all_tasks:
            task_dirs = self.find_all_task_dirs()
            print(f"\n找到 {len(task_dirs)} 个task目录")
            
            for task_dir in task_dirs:
                task_name = task_dir.name
                print(f"\n处理task: {task_name}")
                
                records = self.extract_sota_factors_from_single_task(task_dir, task_name)
                print(f"  提取到 {len(records)} 个SOTA因子")
                all_factor_records.extend(records)
        else:
            task_dirs = self.find_all_task_dirs()
            if task_dirs:
                latest_task = task_dirs[-1]
                print(f"\n使用最新task: {latest_task.name}")
                all_factor_records = self.extract_sota_factors_from_single_task(latest_task, latest_task.name)
        
        df = pd.DataFrame(all_factor_records)
        print(f"\n=== 汇总 ===")
        print(f"总SOTA因子数: {len(df)}")
        
        if 'task_name' in df.columns:
            print(f"\n各task贡献:")
            print(df['task_name'].value_counts())
        
        return df
    
    def _classify_factor_type(self, code: str) -> str:
        """通过AST分析分类因子类型"""
        if not code:
            return "unknown"
        
        try:
            tree = ast.parse(code)
            features = {
                'has_rolling': 'rolling' in code.lower(),
                'has_shift': 'shift' in code,
                'has_diff': 'diff' in code,
                'has_mean': 'mean' in code,
                'has_std': 'std' in code,
                'has_corr': 'corr' in code,
                'has_rank': 'rank' in code,
                'has_volume': 'volume' in code.lower(),
                'has_return': 'return' in code.lower(),
                'has_vwap': 'vwap' in code.lower(),
            }
            
            # 简单分类逻辑
            if features['has_rolling'] and features['has_mean']:
                return "momentum"
            elif features['has_volume']:
                return "volume"
            elif features['has_corr']:
                return "correlation"
            elif features['has_rank']:
                return "rank_based"
            else:
                return "mixed"
        except:
            return "parse_error"
    
    def deduplicate_factors(self, df: pd.DataFrame, similarity_threshold: float = 0.95) -> pd.DataFrame:
        """跨task去重SOTA因子"""
        if df.empty:
            return df
        
        print(f"\n=== 因子去重 ===")
        print(f"去重前: {len(df)} 个因子")
        
        df_dedup = df.copy()
        
        if 'factor_code' in df_dedup.columns:
            df_dedup['code_hash'] = df_dedup['factor_code'].apply(
                lambda x: hash(x.strip()) if x else 0
            )
            
            before_count = len(df_dedup)
            df_dedup = df_dedup.drop_duplicates(subset=['code_hash'], keep='first')
            code_dup_count = before_count - len(df_dedup)
            print(f"完全相同代码去重: 移除 {code_dup_count} 个")
        
        if 'IC' in df_dedup.columns and len(df_dedup) > 1:
            to_remove = set()
            df_sorted = df_dedup.sort_values('IC', ascending=False)
            
            for i, row_i in df_sorted.iterrows():
                if i in to_remove:
                    continue
                
                hypothesis_i = str(row_i.get('hypothesis', ''))
                for j, row_j in df_sorted.iterrows():
                    if i >= j or j in to_remove:
                        continue
                    
                    hypothesis_j = str(row_j.get('hypothesis', ''))
                    
                    if hypothesis_i and hypothesis_j:
                        common_words = set(hypothesis_i.lower().split()) & set(hypothesis_j.lower().split())
                        total_words = set(hypothesis_i.lower().split()) | set(hypothesis_j.lower().split())
                        
                        if len(total_words) > 0:
                            similarity = len(common_words) / len(total_words)
                            if similarity > similarity_threshold:
                                to_remove.add(j)
            
            print(f"假设相似度去重: 移除 {len(to_remove)} 个")
            df_dedup = df_dedup.drop(index=list(to_remove))
        
        print(f"去重后: {len(df_dedup)} 个因子")
        return df_dedup
    
    def calculate_factor_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算因子统计信息"""
        stats = df.groupby('factor_type').agg({
            'IC': ['mean', 'std', 'min', 'max'],
            'ICIR': ['mean', 'std'],
            '1day.excess_return_with_cost.annualized_return': ['mean', 'std'],
            '1day.excess_return_with_cost.max_drawdown': ['mean', 'min', 'max'],
            'factor_name': 'count'
        })
        return stats


def main():
    """主函数"""
    log_root = Path("F:/Dev/RD-Agent-main/log")
    
    if not log_root.exists():
        print(f"未找到log根目录: {log_root}")
        return
    
    print("=" * 80)
    print("SOTA因子提取工具（多task聚合版本）")
    print("=" * 80)
    print(f"Log根目录: {log_root}")
    
    extractor = SOTAFactorExtractor(log_root, aggregate_all_tasks=True)
    sota_df = extractor.extract_sota_factors()
    
    if sota_df.empty:
        print("\n未找到任何SOTA因子")
        return
    
    print("\n=== 去重处理 ===")
    sota_df = extractor.deduplicate_factors(sota_df, similarity_threshold=0.90)
    
    # 保存结果
    output_dir = Path("F:/Dev/RD-Agent-main/tools/factor_optimization/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / "sota_factors_full.csv"
    sota_df.drop(columns=['experiment', 'factor_code'], errors='ignore').to_csv(
        csv_file, index=False, encoding='utf-8-sig'
    )
    print(f"\n因子列表已保存: {csv_file}")
    
    # 保存完整pickle（包含代码和实验对象）
    pkl_file = output_dir / "sota_factors_full.pkl"
    sota_df.to_pickle(pkl_file)
    print(f"完整数据已保存: {pkl_file}")
    
    # 统计分析
    print("\n=== 因子类型统计 ===")
    if 'factor_type' in sota_df.columns:
        print(sota_df['factor_type'].value_counts())
    
    print("\n=== IC统计 ===")
    if 'IC' in sota_df.columns:
        print(sota_df['IC'].describe())
    
    print("\n=== 年化收益统计 ===")
    if '1day.excess_return_with_cost.annualized_return' in sota_df.columns:
        print(sota_df['1day.excess_return_with_cost.annualized_return'].describe())
    
    print("\n=== Task分布统计 ===")
    if 'task_name' in sota_df.columns:
        task_stats = sota_df.groupby('task_name').agg({
            'factor_name': 'count',
            'IC': 'mean'
        }).round(4)
        task_stats.columns = ['因子数', '平均IC']
        print(task_stats.sort_values('因子数', ascending=False))


if __name__ == "__main__":
    main()
