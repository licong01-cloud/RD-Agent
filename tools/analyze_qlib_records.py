"""
分析 Qlib 回测交易记录
读取持仓和指标数据，进行详细分析
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

# 配置路径
workspace_path = Path("F:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f")
artifacts_path = workspace_path / "mlruns/891791629306182420/a334523060ec40d998f521fe3db0df87/artifacts"

print("=" * 80)
print("Qlib 回测交易记录分析")
print("=" * 80)

# 1. 读取回测结果汇总
print("\n【1. 回测结果汇总】")
qlib_res_file = workspace_path / "qlib_res.csv"
if qlib_res_file.exists():
    qlib_res = pd.read_csv(qlib_res_file, header=None, index_col=0)
    print(qlib_res.to_string())
else:
    print("未找到 qlib_res.csv")

# 2. 读取持仓数据
print("\n【2. 持仓数据结构】")
positions_file = artifacts_path / "portfolio_analysis/positions_normal_1day.pkl"
if positions_file.exists():
    with open(positions_file, 'rb') as f:
        positions = pickle.load(f)
    
    print(f"数据类型: {type(positions)}")
    print(f"数据形状: {positions.shape if hasattr(positions, 'shape') else 'N/A'}")
    print(f"数据列: {positions.columns.tolist() if hasattr(positions, 'columns') else 'N/A'}")
    print(f"数据索引: {positions.index.tolist()[:5] if hasattr(positions, 'index') else 'N/A'}")
    print(f"\n前5行数据:")
    print(positions.head())
    
    # 持仓统计
    if hasattr(positions, 'shape'):
        print(f"\n持仓数据统计:")
        print(f"  总交易日数: {len(positions)}")
        if 'amount' in positions.columns:
            print(f"  非零持仓天数: {(positions['amount'] > 0).sum()}")
        if 'value' in positions.columns:
            print(f"  平均持仓价值: {positions['value'].mean():.2f}")
            print(f"  最大持仓价值: {positions['value'].max():.2f}")
            print(f"  最小持仓价值: {positions['value'].min():.2f}")
else:
    print("未找到 positions_normal_1day.pkl")

# 3. 读取指标数据
print("\n【3. 指标数据结构】")
indicators_file = artifacts_path / "portfolio_analysis/indicators_normal_1day.pkl"
if indicators_file.exists():
    with open(indicators_file, 'rb') as f:
        indicators = pickle.load(f)
    
    print(f"数据类型: {type(indicators)}")
    print(f"数据形状: {indicators.shape if hasattr(indicators, 'shape') else 'N/A'}")
    print(f"数据列: {indicators.columns.tolist() if hasattr(indicators, 'columns') else 'N/A'}")
    print(f"\n前5行数据:")
    print(indicators.head())
    
    # 指标统计
    if hasattr(indicators, 'shape'):
        print(f"\n指标数据统计:")
        print(f"  总交易日数: {len(indicators)}")
        print(f"  指标数量: {len(indicators.columns)}")
        print(f"\n各指标统计:")
        print(indicators.describe())
else:
    print("未找到 indicators_normal_1day.pkl")

# 4. 读取预测数据
print("\n【4. 预测数据结构】")
pred_file = artifacts_path / "pred.pkl"
if pred_file.exists():
    with open(pred_file, 'rb') as f:
        pred = pickle.load(f)
    
    print(f"数据类型: {type(pred)}")
    print(f"数据形状: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")
    print(f"数据列: {pred.columns.tolist() if hasattr(pred, 'columns') else 'N/A'}")
    print(f"\n前5行数据:")
    print(pred.head())
    
    # 预测统计
    if hasattr(pred, 'shape'):
        print(f"\n预测数据统计:")
        print(f"  总预测记录数: {len(pred)}")
        if 'score' in pred.columns:
            print(f"  平均评分: {pred['score'].mean():.4f}")
            print(f"  最高评分: {pred['score'].max():.4f}")
            print(f"  最低评分: {pred['score'].min():.4f}")
            print(f"  评分标准差: {pred['score'].std():.4f}")
            print(f"\n评分分布:")
            print(pred['score'].describe())
else:
    print("未找到 pred.pkl")

# 5. 读取 IC 数据
print("\n【5. IC 分析数据】")
ic_file = artifacts_path / "sig_analysis/ic.pkl"
if ic_file.exists():
    with open(ic_file, 'rb') as f:
        ic = pickle.load(f)
    
    print(f"数据类型: {type(ic)}")
    print(f"数据形状: {ic.shape if hasattr(ic, 'shape') else 'N/A'}")
    print(f"\n前10行数据:")
    print(ic.head(10))
    
    if hasattr(ic, 'shape'):
        print(f"\nIC 统计:")
        print(f"  平均 IC: {ic.mean():.4f}")
        print(f"  IC 标准差: {ic.std():.4f}")
        print(f"  IC IR: {ic.mean() / ic.std() if ic.std() > 0 else 0:.4f}")
        print(f"  正 IC 比例: {(ic > 0).sum() / len(ic):.2%}")
else:
    print("未找到 ic.pkl")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
