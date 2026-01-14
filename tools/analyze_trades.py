"""
分析 RD-Agent 回测交易记录
统计选股数量、止盈止损次数、评分阈值合理性
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# 读取持仓数据
workspace_path = Path("F:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f")
positions_file = workspace_path / "mlruns/891791629306182420/a334523060ec40d998f521fe3db0df87/artifacts/portfolio_analysis/positions_normal_1day.pkl"

print(f"读取持仓数据: {positions_file}")
with open(positions_file, 'rb') as f:
    positions_data = pickle.load(f)

print(f"\n持仓数据类型: {type(positions_data)}")
print(f"持仓数据形状: {positions_data.shape if hasattr(positions_data, 'shape') else 'N/A'}")
print(f"持仓数据列: {positions_data.columns.tolist() if hasattr(positions_data, 'columns') else 'N/A'}")
print(f"\n持仓数据前5行:")
print(positions_data.head())

# 读取预测数据
pred_file = workspace_path / "mlruns/891791629306182420/a334523060ec40d998f521fe3db0df87/artifacts/pred.pkl"
print(f"\n读取预测数据: {pred_file}")
with open(pred_file, 'rb') as f:
    pred_data = pickle.load(f)

print(f"预测数据类型: {type(pred_data)}")
if hasattr(pred_data, 'shape'):
    print(f"预测数据形状: {pred_data.shape}")
    print(f"预测数据前5行:")
    print(pred_data.head())
    print(f"\n预测数据统计:")
    print(pred_data.describe())
