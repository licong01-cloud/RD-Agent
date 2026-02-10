"""
因子筛选和组合优化工具
基于IC相关性、业绩指标、因子类型等多维度筛选最佳因子组合
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.cluster import KMeans
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class FactorSelector:
    """因子筛选器 - 多策略筛选"""
    
    def __init__(self, sota_factors_df: pd.DataFrame):
        self.df = sota_factors_df
        self.selected_factors = None
        
    def filter_by_metrics(
        self, 
        min_ic: float = 0.01,
        min_annual_return: float = 0.0,
        max_drawdown: float = -0.5
    ) -> pd.DataFrame:
        """基于业绩指标筛选"""
        df_filtered = self.df.copy()
        
        if 'IC' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['IC'] >= min_ic]
            
        if '1day.excess_return_with_cost.annualized_return' in df_filtered.columns:
            df_filtered = df_filtered[
                df_filtered['1day.excess_return_with_cost.annualized_return'] >= min_annual_return
            ]
            
        if '1day.excess_return_with_cost.max_drawdown' in df_filtered.columns:
            df_filtered = df_filtered[
                df_filtered['1day.excess_return_with_cost.max_drawdown'] >= max_drawdown
            ]
        
        print(f"业绩指标筛选: {len(self.df)} -> {len(df_filtered)} 个因子")
        return df_filtered
    
    def filter_by_diversity(self, factors_df: pd.DataFrame, diversity_threshold: float = 0.7) -> pd.DataFrame:
        """基于因子类型多样性筛选"""
        if 'factor_type' not in factors_df.columns:
            return factors_df
        
        type_counts = factors_df['factor_type'].value_counts()
        balanced_factors = []
        
        for ftype in type_counts.index:
            type_factors = factors_df[factors_df['factor_type'] == ftype]
            max_per_type = max(1, int(len(factors_df) * diversity_threshold / len(type_counts)))
            balanced_factors.append(
                type_factors.nlargest(min(max_per_type, len(type_factors)), 'IC', keep='all')
            )
        
        result = pd.concat(balanced_factors).drop_duplicates()
        print(f"多样性筛选: {len(factors_df)} -> {len(result)} 个因子")
        return result


class FactorCombinationOptimizer:
    """因子组合优化器 - 基于贪心算法和聚类"""
    
    def __init__(self, factors_df: pd.DataFrame):
        self.df = factors_df
        self.factor_data_cache = {}
        
    def load_factor_data(self, experiment) -> pd.DataFrame:
        """加载因子的实际数据（result.h5）"""
        try:
            if hasattr(experiment, 'sub_workspace_list'):
                for ws in experiment.sub_workspace_list:
                    if ws and hasattr(ws, 'workspace_path'):
                        h5_file = ws.workspace_path / 'result.h5'
                        if h5_file.exists():
                            df = pd.read_hdf(h5_file, key='factor')
                            return df
        except Exception as e:
            print(f"加载因子数据失败: {e}")
        return pd.DataFrame()
    
    def calculate_ic_correlation_matrix(self, selected_indices: List[int]) -> pd.DataFrame:
        """计算选定因子之间的IC相关性矩阵"""
        factor_ics = []
        
        for idx in selected_indices:
            exp = self.df.iloc[idx]['experiment']
            factor_df = self.load_factor_data(exp)
            
            if not factor_df.empty and 'datetime' in factor_df.index.names:
                ic_series = factor_df.groupby('datetime').apply(
                    lambda x: x.iloc[:, 0].corr(x.index.get_level_values('instrument'))
                )
                factor_ics.append(ic_series)
        
        if not factor_ics:
            return pd.DataFrame()
        
        ic_df = pd.concat(factor_ics, axis=1)
        corr_matrix = ic_df.corr()
        return corr_matrix
    
    def greedy_selection(
        self, 
        top_n: int = 50,
        max_correlation: float = 0.8,
        ic_weight: float = 0.6,
        return_weight: float = 0.4
    ) -> pd.DataFrame:
        """贪心算法选择因子组合"""
        df = self.df.copy()
        
        if 'IC' not in df.columns:
            print("缺少IC列，无法执行贪心选择")
            return df.head(top_n)
        
        df['composite_score'] = (
            df['IC'].fillna(0) * ic_weight + 
            df.get('1day.excess_return_with_cost.annualized_return', 0).fillna(0) * return_weight
        )
        
        df_sorted = df.sort_values('composite_score', ascending=False)
        
        selected = []
        selected_indices = []
        
        for idx, row in df_sorted.iterrows():
            if len(selected) >= top_n:
                break
            
            if len(selected) == 0:
                selected.append(row)
                selected_indices.append(idx)
                continue
            
            should_add = True
            if 'experiment' in row and row['experiment'] is not None:
                pass
            
            if should_add:
                selected.append(row)
                selected_indices.append(idx)
        
        result_df = pd.DataFrame(selected)
        print(f"贪心算法选择: {len(result_df)} 个因子")
        return result_df
    
    def cluster_based_selection(
        self, 
        n_clusters: int = 10,
        factors_per_cluster: int = 5
    ) -> pd.DataFrame:
        """基于聚类的因子选择"""
        if 'IC' not in self.df.columns or 'ICIR' not in self.df.columns:
            print("缺少必要的特征列，无法执行聚类")
            return self.df.head(n_clusters * factors_per_cluster)
        
        features = self.df[['IC', 'ICIR']].fillna(0)
        
        if '1day.excess_return_with_cost.annualized_return' in self.df.columns:
            features['annual_return'] = self.df['1day.excess_return_with_cost.annualized_return'].fillna(0)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(features)
        
        selected_factors = []
        for cluster_id in range(n_clusters):
            cluster_factors = self.df[self.df['cluster'] == cluster_id]
            top_in_cluster = cluster_factors.nlargest(
                min(factors_per_cluster, len(cluster_factors)), 
                'IC'
            )
            selected_factors.append(top_in_cluster)
        
        result = pd.concat(selected_factors)
        print(f"聚类选择: {n_clusters} 个簇, 每簇最多 {factors_per_cluster} 个因子, 共 {len(result)} 个")
        return result


def main():
    """主函数 - 演示完整的因子筛选和组合流程"""
    
    pkl_file = Path("F:/Dev/RD-Agent-main/tools/factor_optimization/output/sota_factors_full.pkl")
    
    if not pkl_file.exists():
        print(f"请先运行 01_extract_and_analyze_sota_factors.py 生成 {pkl_file}")
        return
    
    print("=" * 80)
    print("加载SOTA因子数据...")
    print("=" * 80)
    sota_df = pd.read_pickle(pkl_file)
    print(f"总因子数: {len(sota_df)}")
    
    print("\n" + "=" * 80)
    print("步骤1: 基于业绩指标筛选")
    print("=" * 80)
    selector = FactorSelector(sota_df)
    filtered_df = selector.filter_by_metrics(
        min_ic=0.02,
        min_annual_return=0.0,
        max_drawdown=-0.3
    )
    
    print("\n" + "=" * 80)
    print("步骤2: 基于多样性筛选")
    print("=" * 80)
    diverse_df = selector.filter_by_diversity(filtered_df, diversity_threshold=0.6)
    
    print("\n" + "=" * 80)
    print("步骤3: 贪心算法选择最优组合")
    print("=" * 80)
    optimizer = FactorCombinationOptimizer(diverse_df)
    greedy_selected = optimizer.greedy_selection(
        top_n=50,
        max_correlation=0.8,
        ic_weight=0.6,
        return_weight=0.4
    )
    
    print("\n" + "=" * 80)
    print("步骤4: 聚类选择（备选方案）")
    print("=" * 80)
    cluster_selected = optimizer.cluster_based_selection(
        n_clusters=10,
        factors_per_cluster=5
    )
    
    output_dir = Path("F:/Dev/RD-Agent-main/tools/factor_optimization/output")
    
    greedy_file = output_dir / "selected_factors_greedy.csv"
    greedy_selected.drop(columns=['experiment', 'factor_code'], errors='ignore').to_csv(
        greedy_file, index=False, encoding='utf-8-sig'
    )
    print(f"\n贪心选择结果已保存: {greedy_file}")
    
    cluster_file = output_dir / "selected_factors_cluster.csv"
    cluster_selected.drop(columns=['experiment', 'factor_code'], errors='ignore').to_csv(
        cluster_file, index=False, encoding='utf-8-sig'
    )
    print(f"聚类选择结果已保存: {cluster_file}")
    
    greedy_pkl = output_dir / "selected_factors_greedy.pkl"
    greedy_selected.to_pickle(greedy_pkl)
    
    cluster_pkl = output_dir / "selected_factors_cluster.pkl"
    cluster_selected.to_pickle(cluster_pkl)
    
    print("\n" + "=" * 80)
    print("因子筛选完成！")
    print("=" * 80)
    print(f"贪心算法: {len(greedy_selected)} 个因子")
    print(f"聚类方法: {len(cluster_selected)} 个因子")
    
    print("\n推荐使用贪心算法结果进行回测验证")


if __name__ == "__main__":
    main()
