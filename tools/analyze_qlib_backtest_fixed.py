#!/usr/bin/env python3
"""
Qlib 回测交易记录详细分析脚本（修正版）
正确读取持仓数据
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# 配置路径
WORKSPACE_PATH = Path("/mnt/f/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f")
ARTIFACTS_PATH = WORKSPACE_PATH / "mlruns/891791629306182420/a334523060ec40d998f521fe3db0df87/artifacts"
OUTPUT_PATH = WORKSPACE_PATH / "backtest_analysis_report"

OUTPUT_PATH.mkdir(exist_ok=True)

print("=" * 80)
print("Qlib 回测交易记录详细分析（修正版）")
print("=" * 80)
print(f"工作空间: {WORKSPACE_PATH}")
print(f"输出目录: {OUTPUT_PATH}")
print()

analysis_results = {
    "summary": {},
    "positions": {},
    "indicators": {},
    "predictions": {},
    "trading_behavior": {},
    "risk_analysis": {},
    "recommendations": []
}

# ============================================================================
# 1. 读取回测结果汇总
# ============================================================================
print("【1/7】读取回测结果汇总...")
qlib_res_file = WORKSPACE_PATH / "qlib_res.csv"
if qlib_res_file.exists():
    qlib_res = pd.read_csv(qlib_res_file, header=None, index_col=0)
    qlib_res_dict = qlib_res.to_dict()[1]
    analysis_results["summary"] = qlib_res_dict
    print("  ✓ 回测结果汇总读取成功")
else:
    print("  ✗ 未找到 qlib_res.csv")

# ============================================================================
# 2. 读取持仓数据（修正版）
# ============================================================================
print("【2/7】读取持仓数据...")
positions_file = ARTIFACTS_PATH / "portfolio_analysis/positions_normal_1day.pkl"
if positions_file.exists():
    with open(positions_file, 'rb') as f:
        positions_dict = pickle.load(f)
    
    # positions_dict 是 {date: Position} 的字典
    position_data = []
    
    for date, position in positions_dict.items():
        # 获取当前日期的所有持仓股票
        stock_list = position.get_stock_list()
        
        for stock_id in stock_list:
            amount = position.get_stock_amount(stock_id)
            price = position.get_stock_price(stock_id)
            value = float(amount) * float(price)  # 手动计算股票价值
            
            position_data.append({
                "date": str(date),
                "stock_id": stock_id,
                "amount": float(amount),
                "price": float(price),
                "value": float(value)
            })
    
    if position_data:
        positions_df = pd.DataFrame(position_data)
        positions_df['date'] = pd.to_datetime(positions_df['date'])
        positions_df = positions_df.sort_values('date')
        
        # 持仓统计
        analysis_results["positions"]["total_records"] = len(positions_df)
        analysis_results["positions"]["unique_stocks"] = positions_df['stock_id'].nunique()
        analysis_results["positions"]["date_range"] = {
            "start": str(positions_df['date'].min()),
            "end": str(positions_df['date'].max())
        }
        
        # 每日持仓统计
        daily_positions = positions_df.groupby('date').agg({
            'stock_id': 'count',
            'value': 'sum'
        }).rename(columns={'stock_id': 'stock_count', 'value': 'total_value'})
        
        analysis_results["positions"]["daily_stats"] = {
            "avg_stock_count": float(daily_positions['stock_count'].mean()),
            "max_stock_count": int(daily_positions['stock_count'].max()),
            "min_stock_count": int(daily_positions['stock_count'].min()),
            "avg_total_value": float(daily_positions['total_value'].mean()),
            "max_total_value": float(daily_positions['total_value'].max()),
            "min_total_value": float(daily_positions['total_value'].min())
        }
        
        # 持仓天数统计
        stock_hold_days = positions_df.groupby('stock_id').size().sort_values(ascending=False)
        analysis_results["positions"]["top_10_stocks_by_hold_days"] = {
            k: int(v) for k, v in stock_hold_days.head(10).items()
        }
        
        # 保存持仓数据
        positions_df.to_csv(OUTPUT_PATH / "positions_detail.csv", index=False)
        daily_positions.to_csv(OUTPUT_PATH / "daily_positions_summary.csv")
        
        print(f"  ✓ 持仓数据读取成功，共 {len(positions_df)} 条记录")
    else:
        print("  ✗ 持仓数据为空")
        analysis_results["positions"]["error"] = "持仓数据为空"
else:
    print("  ✗ 未找到 positions_normal_1day.pkl")

# ============================================================================
# 3. 读取指标数据
# ============================================================================
print("【3/7】读取指标数据...")
indicators_file = ARTIFACTS_PATH / "portfolio_analysis/indicators_normal_1day.pkl"
if indicators_file.exists():
    with open(indicators_file, 'rb') as f:
        indicators = pickle.load(f)
    
    if isinstance(indicators, pd.DataFrame):
        analysis_results["indicators"]["columns"] = indicators.columns.tolist()
        analysis_results["indicators"]["shape"] = indicators.shape
        analysis_results["indicators"]["date_range"] = {
            "start": str(indicators.index.min()) if hasattr(indicators, 'index') else "N/A",
            "end": str(indicators.index.max()) if hasattr(indicators, 'index') else "N/A"
        }
        
        if 'return' in indicators.columns:
            analysis_results["indicators"]["return_stats"] = {
                "mean": float(indicators['return'].mean()),
                "std": float(indicators['return'].std()),
                "max": float(indicators['return'].max()),
                "min": float(indicators['return'].min()),
                "positive_days": int((indicators['return'] > 0).sum()),
                "negative_days": int((indicators['return'] < 0).sum())
            }
        
        if 'turnover' in indicators.columns:
            analysis_results["indicators"]["turnover_stats"] = {
                "mean": float(indicators['turnover'].mean()),
                "max": float(indicators['turnover'].max()),
                "total": float(indicators['turnover'].sum())
            }
        
        indicators.to_csv(OUTPUT_PATH / "indicators_detail.csv")
        print(f"  ✓ 指标数据读取成功，共 {len(indicators)} 行")
    else:
        print("  ✗ 指标数据格式错误")
else:
    print("  ✗ 未找到 indicators_normal_1day.pkl")

# ============================================================================
# 4. 读取预测数据
# ============================================================================
print("【4/7】读取预测数据...")
pred_file = ARTIFACTS_PATH / "pred.pkl"
if pred_file.exists():
    with open(pred_file, 'rb') as f:
        pred = pickle.load(f)
    
    if isinstance(pred, pd.DataFrame):
        # 如果是 MultiIndex，展开
        if isinstance(pred.index, pd.MultiIndex):
            pred = pred.reset_index()
        
        analysis_results["predictions"]["shape"] = pred.shape
        analysis_results["predictions"]["columns"] = pred.columns.tolist()
        
        if 'score' in pred.columns:
            analysis_results["predictions"]["score_stats"] = {
                "mean": float(pred['score'].mean()),
                "std": float(pred['score'].std()),
                "max": float(pred['score'].max()),
                "min": float(pred['score'].min()),
                "median": float(pred['score'].median())
            }
            
            score_bins = pd.cut(pred['score'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
            score_distribution = score_bins.value_counts().sort_index()
            analysis_results["predictions"]["score_distribution"] = {
                str(k): int(v) for k, v in score_distribution.items()
            }
            
            low_score_ratio = (pred['score'] < 0.2).sum() / len(pred)
            analysis_results["predictions"]["low_score_ratio"] = float(low_score_ratio)
        
        pred.to_csv(OUTPUT_PATH / "predictions_detail.csv")
        print(f"  ✓ 预测数据读取成功，共 {len(pred)} 条记录")
    else:
        print("  ✗ 预测数据格式错误")
else:
    print("  ✗ 未找到 pred.pkl")

# ============================================================================
# 5. 交易行为分析
# ============================================================================
print("【5/7】分析交易行为...")
if 'positions' in analysis_results and 'error' not in analysis_results["positions"]:
    positions_df = pd.read_csv(OUTPUT_PATH / "positions_detail.csv")
    positions_df['date'] = pd.to_datetime(positions_df['date'])
    
    # 识别买入和卖出
    trading_events = []
    
    for stock_id in positions_df['stock_id'].unique():
        stock_data = positions_df[positions_df['stock_id'] == stock_id].sort_values('date')
        
        for i in range(len(stock_data)):
            current = stock_data.iloc[i]
            
            if i > 0:
                prev = stock_data.iloc[i - 1]
                
                if current['amount'] > prev['amount']:
                    trading_events.append({
                        "date": str(current['date']),
                        "stock_id": stock_id,
                        "action": "BUY",
                        "amount_change": float(current['amount'] - prev['amount']),
                        "price": float(current['price'])
                    })
                
                elif current['amount'] < prev['amount']:
                    trading_events.append({
                        "date": str(current['date']),
                        "stock_id": stock_id,
                        "action": "SELL",
                        "amount_change": float(prev['amount'] - current['amount']),
                        "price": float(current['price'])
                    })
    
    if trading_events:
        trading_df = pd.DataFrame(trading_events)
        trading_df['date'] = pd.to_datetime(trading_df['date'])
        
        buy_count = len(trading_df[trading_df['action'] == 'BUY'])
        sell_count = len(trading_df[trading_df['action'] == 'SELL'])
        
        analysis_results["trading_behavior"] = {
            "total_trades": len(trading_df),
            "buy_trades": buy_count,
            "sell_trades": sell_count,
            "unique_stocks_traded": trading_df['stock_id'].nunique(),
            "avg_daily_trades": float(len(trading_df) / trading_df['date'].nunique())
        }
        
        daily_trades = trading_df.groupby('date').size()
        analysis_results["trading_behavior"]["daily_trades_stats"] = {
            "mean": float(daily_trades.mean()),
            "max": int(daily_trades.max()),
            "min": int(daily_trades.min())
        }
        
        trading_df.to_csv(OUTPUT_PATH / "trading_events.csv", index=False)
        print(f"  ✓ 交易行为分析完成，共 {len(trading_df)} 笔交易")
    else:
        print("  ✗ 未检测到交易事件")
else:
    print("  ✗ 持仓数据不可用，跳过交易行为分析")

# ============================================================================
# 6. 风险分析
# ============================================================================
print("【6/7】分析风险指标...")
if 'indicators' in analysis_results and 'error' not in analysis_results["indicators"]:
    indicators_df = pd.read_csv(OUTPUT_PATH / "indicators_detail.csv", index_col=0)
    
    if 'account' in indicators_df.columns:
        account_values = indicators_df['account']
        running_max = account_values.expanding().max()
        drawdown = (account_values - running_max) / running_max
        
        analysis_results["risk_analysis"]["drawdown_stats"] = {
            "max_drawdown": float(drawdown.min()),
            "avg_drawdown": float(drawdown.mean()),
            "drawdown_days": int((drawdown < 0).sum()),
            "max_drawdown_date": str(drawdown.idxmin())
        }
    
    if 'return' in indicators_df.columns:
        returns = indicators_df['return']
        analysis_results["risk_analysis"]["volatility"] = {
            "daily_volatility": float(returns.std()),
            "annualized_volatility": float(returns.std() * np.sqrt(252)),
            "downside_volatility": float(returns[returns < 0].std())
        }
        
        annual_return = float(returns.mean() * 252)
        annual_volatility = float(returns.std() * np.sqrt(252))
        sharpe_ratio = (annual_return - 0.03) / annual_volatility if annual_volatility > 0 else 0
        analysis_results["risk_analysis"]["sharpe_ratio"] = sharpe_ratio
    
    print("  ✓ 风险分析完成")
else:
    print("  ✗ 指标数据不可用，跳过风险分析")

# ============================================================================
# 7. 生成分析报告
# ============================================================================
print("【7/7】生成分析报告...")

with open(OUTPUT_PATH / "analysis_report.json", 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2, ensure_ascii=False)

# 生成文本报告
report_text = []
report_text.append("=" * 80)
report_text.append("Qlib 回测交易记录详细分析报告")
report_text.append("=" * 80)
report_text.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_text.append("")

# 1. 回测结果汇总
report_text.append("【一、回测结果汇总】")
if "summary" in analysis_results:
    for key, value in analysis_results["summary"].items():
        if isinstance(value, float):
            report_text.append(f"  {key}: {value:.4f}")
        else:
            report_text.append(f"  {key}: {value}")
report_text.append("")

# 2. 持仓分析
report_text.append("【二、持仓分析】")
if "positions" in analysis_results and "error" not in analysis_results["positions"]:
    pos = analysis_results["positions"]
    report_text.append(f"  总持仓记录数: {pos.get('total_records', 'N/A')}")
    report_text.append(f"  持有股票数量: {pos.get('unique_stocks', 'N/A')}")
    if 'date_range' in pos:
        report_text.append(f"  持仓日期范围: {pos['date_range']['start']} ~ {pos['date_range']['end']}")
    
    if 'daily_stats' in pos:
        ds = pos['daily_stats']
        report_text.append(f"  平均持仓股票数: {ds.get('avg_stock_count', 'N/A'):.1f}")
        report_text.append(f"  最大持仓股票数: {ds.get('max_stock_count', 'N/A')}")
        report_text.append(f"  最小持仓股票数: {ds.get('min_stock_count', 'N/A')}")
        report_text.append(f"  平均持仓价值: {ds.get('avg_total_value', 'N/A'):.2f}")
        report_text.append(f"  最大持仓价值: {ds.get('max_total_value', 'N/A'):.2f}")
    
    if 'top_10_stocks_by_hold_days' in pos:
        report_text.append("  持仓天数前 10 股票:")
        for stock, days in list(pos['top_10_stocks_by_hold_days'].items())[:10]:
            report_text.append(f"    {stock}: {days} 天")
else:
    report_text.append("  数据不可用")
report_text.append("")

# 3. 指标分析
report_text.append("【三、指标分析】")
if "indicators" in analysis_results and "error" not in analysis_results["indicators"]:
    ind = analysis_results["indicators"]
    report_text.append(f"  数据形状: {ind.get('shape', 'N/A')}")
    
    if 'return_stats' in ind:
        rs = ind['return_stats']
        report_text.append(f"  日收益率均值: {rs.get('mean', 'N/A'):.4f}")
        report_text.append(f"  日收益率标准差: {rs.get('std', 'N/A'):.4f}")
        report_text.append(f"  最大单日收益: {rs.get('max', 'N/A'):.4f}")
        report_text.append(f"  最大单日亏损: {rs.get('min', 'N/A'):.4f}")
        report_text.append(f"  盈利天数: {rs.get('positive_days', 'N/A')}")
        report_text.append(f"  亏损天数: {rs.get('negative_days', 'N/A')}")
        win_rate = rs.get('positive_days', 0) / (rs.get('positive_days', 0) + rs.get('negative_days', 0))
        report_text.append(f"  胜率: {win_rate:.2%}")
    
    if 'turnover_stats' in ind:
        ts = ind['turnover_stats']
        report_text.append(f"  平均换手率: {ts.get('mean', 'N/A'):.4f}")
        report_text.append(f"  最大换手率: {ts.get('max', 'N/A'):.4f}")
        report_text.append(f"  总换手率: {ts.get('total', 'N/A'):.4f}")
else:
    report_text.append("  数据不可用")
report_text.append("")

# 4. 预测分析
report_text.append("【四、预测分析】")
if "predictions" in analysis_results and "error" not in analysis_results["predictions"]:
    pred = analysis_results["predictions"]
    report_text.append(f"  预测记录数: {pred.get('shape', ['N/A'])[0] if isinstance(pred.get('shape'), list) else 'N/A'}")
    
    if 'score_stats' in pred:
        ss = pred['score_stats']
        report_text.append(f"  评分均值: {ss.get('mean', 'N/A'):.4f}")
        report_text.append(f"  评分标准差: {ss.get('std', 'N/A'):.4f}")
        report_text.append(f"  最高评分: {ss.get('max', 'N/A'):.4f}")
        report_text.append(f"  最低评分: {ss.get('min', 'N/A'):.4f}")
        report_text.append(f"  评分中位数: {ss.get('median', 'N/A'):.4f}")
    
    if 'low_score_ratio' in pred:
        report_text.append(f"  低于 0.2 评分比例: {pred['low_score_ratio']:.2%}")
    
    if 'score_distribution' in pred:
        report_text.append("  评分分布:")
        for score_range, count in pred['score_distribution'].items():
            report_text.append(f"    {score_range}: {count}")
else:
    report_text.append("  数据不可用")
report_text.append("")

# 5. 交易行为分析
report_text.append("【五、交易行为分析】")
if "trading_behavior" in analysis_results and "error" not in analysis_results["trading_behavior"]:
    tb = analysis_results["trading_behavior"]
    report_text.append(f"  总交易次数: {tb.get('total_trades', 'N/A')}")
    report_text.append(f"  买入次数: {tb.get('buy_trades', 'N/A')}")
    report_text.append(f"  卖出次数: {tb.get('sell_trades', 'N/A')}")
    report_text.append(f"  交易股票数: {tb.get('unique_stocks_traded', 'N/A')}")
    report_text.append(f"  平均每日交易次数: {tb.get('avg_daily_trades', 'N/A')}")
    
    if 'daily_trades_stats' in tb:
        dts = tb['daily_trades_stats']
        report_text.append(f"  每日交易次数均值: {dts.get('mean', 'N/A'):.1f}")
        report_text.append(f"  每日交易次数最大值: {dts.get('max', 'N/A')}")
        report_text.append(f"  每日交易次数最小值: {dts.get('min', 'N/A')}")
else:
    report_text.append("  数据不可用")
report_text.append("")

# 6. 风险分析
report_text.append("【六、风险分析】")
if "risk_analysis" in analysis_results:
    ra = analysis_results["risk_analysis"]
    
    if 'drawdown_stats' in ra:
        ds = ra['drawdown_stats']
        report_text.append(f"  最大回撤: {ds.get('max_drawdown', 'N/A'):.2%}")
        report_text.append(f"  平均回撤: {ds.get('avg_drawdown', 'N/A'):.2%}")
        report_text.append(f"  回撤天数: {ds.get('drawdown_days', 'N/A')}")
        report_text.append(f"  最大回撤日期: {ds.get('max_drawdown_date', 'N/A')}")
    
    if 'volatility' in ra:
        v = ra['volatility']
        report_text.append(f"  日波动率: {v.get('daily_volatility', 'N/A'):.4f}")
        report_text.append(f"  年化波动率: {v.get('annualized_volatility', 'N/A'):.4f}")
        report_text.append(f"  下行波动率: {v.get('downside_volatility', 'N/A'):.4f}")
    
    if 'sharpe_ratio' in ra:
        report_text.append(f"  夏普比率: {ra['sharpe_ratio']:.4f}")
else:
    report_text.append("  数据不可用")
report_text.append("")

report_text.append("=" * 80)
report_text.append("分析完成")
report_text.append("=" * 80)

with open(OUTPUT_PATH / "analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_text))

print("  ✓ 分析报告生成完成")
print()
print("=" * 80)
print("分析完成！生成的文件:")
print(f"  1. {OUTPUT_PATH / 'analysis_report.json'} - JSON 格式详细数据")
print(f"  2. {OUTPUT_PATH / 'analysis_report.txt'} - 文本格式报告")
print(f"  3. {OUTPUT_PATH / 'positions_detail.csv'} - 持仓明细")
print(f"  4. {OUTPUT_PATH / 'daily_positions_summary.csv'} - 每日持仓汇总")
print(f"  5. {OUTPUT_PATH / 'indicators_detail.csv'} - 指标明细")
print(f"  6. {OUTPUT_PATH / 'predictions_detail.csv'} - 预测明细")
print(f"  7. {OUTPUT_PATH / 'trading_events.csv'} - 交易事件")
print("=" * 80)
