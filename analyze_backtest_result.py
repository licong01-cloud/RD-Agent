#!/usr/bin/env python
"""分析回测结果，找出策略表现差的原因"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

workspace_path = Path("/mnt/f/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f")
run_id = "35eb0e48e7524506b6ce00c40c34d187"
artifacts_path = workspace_path / "mlruns" / "891791629306182420" / run_id / "artifacts"

print("=" * 80)
print("回测结果分析")
print("=" * 80)

# 1. 读取投资组合分析数据
port_analysis_path = artifacts_path / "portfolio_analysis" / "port_analysis_1day.pkl"
if port_analysis_path.exists():
    with open(port_analysis_path, "rb") as f:
        port_data = pickle.load(f)
    
    print("\n【投资组合分析】")
    print(f"类型: {type(port_data)}")
    if isinstance(port_data, dict):
        for key, value in port_data.items():
            print(f"  {key}: {value}")
    elif hasattr(port_data, '__dict__'):
        for key, value in port_data.__dict__.items():
            print(f"  {key}: {value}")

# 2. 读取持仓数据
positions_path = artifacts_path / "portfolio_analysis" / "positions_normal_1day.pkl"
if positions_path.exists():
    with open(positions_path, "rb") as f:
        positions_data = pickle.load(f)
    
    print("\n【持仓数据分析】")
    if isinstance(positions_data, pd.DataFrame):
        print(f"持仓数据形状: {positions_data.shape}")
        print(f"列名: {positions_data.columns.tolist()}")
        print(f"\n持仓数据统计:")
        print(positions_data.describe())
        
        # 分析持仓数量
        if "position" in positions_data.columns:
            pos_counts = positions_data.groupby("datetime")["position"].apply(lambda x: (x > 0).sum())
            print(f"\n每日持仓股票数量统计:")
            print(f"  平均持仓数: {pos_counts.mean():.2f}")
            print(f"  最大持仓数: {pos_counts.max()}")
            print(f"  最小持仓数: {pos_counts.min()}")
            print(f"  空仓天数: {(pos_counts == 0).sum()}")
            
            # 持仓数量分布
            print(f"\n持仓数量分布:")
            print(pos_counts.value_counts().sort_index())

# 3. 读取指标数据
indicators_path = artifacts_path / "portfolio_analysis" / "indicators_normal_1day.pkl"
if indicators_path.exists():
    with open(indicators_path, "rb") as f:
        indicators_data = pickle.load(f)
    
    print("\n【指标数据分析】")
    if isinstance(indicators_data, pd.DataFrame):
        print(f"指标数据形状: {indicators_data.shape}")
        print(f"列名: {indicators_data.columns.tolist()}")
        print(f"\n指标数据统计:")
        print(indicators_data.describe())

# 4. 读取报告数据
report_path = artifacts_path / "portfolio_analysis" / "report_normal_1day.pkl"
if report_path.exists():
    with open(report_path, "rb") as f:
        report_data = pickle.load(f)
    
    print("\n【报告数据分析】")
    if isinstance(report_data, pd.DataFrame):
        print(f"报告数据形状: {report_data.shape}")
        print(f"列名: {report_data.columns.tolist()}")
        print(f"\n报告数据统计:")
        print(report_data.describe())

# 5. 读取预测数据
pred_path = artifacts_path / "pred.pkl"
if pred_path.exists():
    with open(pred_path, "rb") as f:
        pred_data = pickle.load(f)
    
    print("\n【预测数据分析】")
    if isinstance(pred_data, pd.DataFrame):
        print(f"预测数据形状: {pred_data.shape}")
        print(f"列名: {pred_data.columns.tolist()}")
        
        if "score" in pred_data.columns:
            print(f"\n预测评分统计:")
            print(pred_data["score"].describe())
            
            # 评分分布
            print(f"\n评分分布:")
            print(f"  评分>0.1: {(pred_data['score'] > 0.1).sum()} ({(pred_data['score'] > 0.1).sum()/len(pred_data)*100:.2f}%)")
            print(f"  评分>0.2: {(pred_data['score'] > 0.2).sum()} ({(pred_data['score'] > 0.2).sum()/len(pred_data)*100:.2f}%)")
            print(f"  评分>0.5: {(pred_data['score'] > 0.5).sum()} ({(pred_data['score'] > 0.5).sum()/len(pred_data)*100:.2f}%)")
            print(f"  评分>1.0: {(pred_data['score'] > 1.0).sum()} ({(pred_data['score'] > 1.0).sum()/len(pred_data)*100:.2f}%)")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
