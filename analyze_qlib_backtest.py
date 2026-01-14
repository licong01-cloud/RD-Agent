#!/usr/bin/env python3
"""
Qlib 回测交易记录详细分析脚本
在 WSL Qlib 环境中执行，生成详细的分析报告

使用方法:
    conda activate rdagent-gpu
    python analyze_qlib_backtest.py
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import traceback

# 配置路径
WORKSPACE_PATH = Path("/mnt/f/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/dcf014c2ce2a4255bb795ee0f7eb7d9f")
EXPERIMENT_ID = "891791629306182420"


def _pick_latest_run_artifacts(workspace_path: Path, experiment_id: str) -> tuple[Path | None, str | None]:
    exp_dir = workspace_path / "mlruns" / experiment_id
    if not exp_dir.exists():
        return None, None

    best_run_id = None
    best_mtime = None
    best_artifacts = None

    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        artifacts_dir = run_dir / "artifacts"
        positions_file = artifacts_dir / "portfolio_analysis" / "positions_normal_1day.pkl"
        if not positions_file.exists():
            continue
        try:
            mtime = positions_file.stat().st_mtime
        except Exception:
            continue
        if best_mtime is None or mtime > best_mtime:
            best_mtime = mtime
            best_run_id = run_dir.name
            best_artifacts = artifacts_dir

    return best_artifacts, best_run_id


ARTIFACTS_PATH, RUN_ID = _pick_latest_run_artifacts(WORKSPACE_PATH, EXPERIMENT_ID)
if ARTIFACTS_PATH is None:
    raise FileNotFoundError(f"No valid artifacts found under {WORKSPACE_PATH}/mlruns/{EXPERIMENT_ID}")

OUTPUT_PATH = WORKSPACE_PATH / "backtest_analysis_report"

# 创建输出目录
OUTPUT_PATH.mkdir(exist_ok=True)

# 清理旧输出，避免复用旧 CSV/报告导致误判
for _p in [
    OUTPUT_PATH / "analysis_report.json",
    OUTPUT_PATH / "analysis_report.txt",
    OUTPUT_PATH / "positions_detail.csv",
    OUTPUT_PATH / "daily_positions_summary.csv",
    OUTPUT_PATH / "indicators_detail.csv",
    OUTPUT_PATH / "predictions_detail.csv",
    OUTPUT_PATH / "trading_events.csv",
    OUTPUT_PATH / "detailed_analysis.json",
    OUTPUT_PATH / "detailed_analysis.txt",
    OUTPUT_PATH / "score_distribution_detailed.csv",
    OUTPUT_PATH / "stock_buy_sell_stats.csv",
    OUTPUT_PATH / "stock_trades_detail.csv",
]:
    try:
        if _p.exists():
            _p.unlink()
    except Exception:
        pass


def _preview_obj(obj, max_keys=20):
    """打印pickle对象的结构信息，辅助定位为什么解析不到持仓。"""
    info = {"type": str(type(obj))}
    try:
        if isinstance(obj, dict):
            keys = list(obj.keys())
            info["dict_len"] = len(keys)
            info["dict_keys_preview"] = [str(k) for k in keys[:max_keys]]
        elif isinstance(obj, (list, tuple)):
            info["len"] = len(obj)
            info["first_item_type"] = str(type(obj[0])) if len(obj) > 0 else "N/A"
        elif hasattr(obj, "__dict__"):
            attrs = list(getattr(obj, "__dict__", {}).keys())
            info["attrs_preview"] = attrs[:max_keys]
    except Exception as e:
        info["preview_error"] = str(e)
    return info


def _is_date_like(v):
    try:
        pd.to_datetime(v)
        return True
    except Exception:
        return False


def _is_account_field(stock_id: object) -> bool:
    s = str(stock_id).strip().lower()
    if s in {"", "cash", "$cash", "now_account_value", "account_value", "total_value"}:
        return True
    if "cash" in s:
        return True
    if "account" in s:
        return True
    return False

print("=" * 80)
print("Qlib 回测交易记录详细分析")
print("=" * 80)
print(f"脚本路径: {Path(__file__).resolve()}")
print(f"Python: {sys.executable}")
print(f"PythonVersion: {sys.version.splitlines()[0]}")
print(f"工作空间: {WORKSPACE_PATH}")
print(f"ExperimentId: {EXPERIMENT_ID}")
print(f"RunId: {RUN_ID}")
print(f"Artifacts: {ARTIFACTS_PATH}")
print(f"输出目录: {OUTPUT_PATH}")
print()

# 存储所有分析结果
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
    sys.exit(1)

# 重要：qlib_res.csv 可能不是本次 RunId 的产物（例如来自 read_exp_res.py 的旧输出）。
# 为避免“指标/回撤与交易不一致”，这里优先从本次 run 的 artifacts 读取官方分析结果，覆盖 summary 的关键字段。
try:
    _pa_dir = ARTIFACTS_PATH / "portfolio_analysis"
    _indicator_analysis_file = _pa_dir / "indicator_analysis_1day.pkl"
    _port_analysis_file = _pa_dir / "port_analysis_1day.pkl"
    if _indicator_analysis_file.exists():
        with open(_indicator_analysis_file, "rb") as f:
            _ia = pickle.load(f)
        if isinstance(_ia, dict):
            for k, v in _ia.items():
                if isinstance(v, (int, float, np.number)):
                    analysis_results["summary"][f"1day.{k}"] = float(v)
    if _port_analysis_file.exists():
        with open(_port_analysis_file, "rb") as f:
            _pa = pickle.load(f)
        # port_analysis_1day.pkl 在不同版本 Qlib 中结构不完全一致，这里只做“尽量提取”。
        if isinstance(_pa, dict):
            # 常见结构：{"excess_return_with_cost": {"risk": pd.Series/df}, ...}
            for _k, _v in _pa.items():
                if isinstance(_v, dict) and "risk" in _v:
                    risk_obj = _v.get("risk")
                    if isinstance(risk_obj, pd.DataFrame) and "risk" in risk_obj.columns:
                        risk_s = risk_obj["risk"]
                    elif isinstance(risk_obj, pd.Series):
                        risk_s = risk_obj
                    elif isinstance(risk_obj, dict):
                        risk_s = pd.Series(risk_obj)
                    else:
                        continue
                    for rk, rv in risk_s.items():
                        if isinstance(rv, (int, float, np.number)):
                            analysis_results["summary"][f"1day.{_k}.{rk}"] = float(rv)
except Exception:
    # 这里不让分析中断，后续仍可用 positions/trading_events 做交易层分析
    pass

# ============================================================================
# 2. 读取持仓数据
# ============================================================================
print("【2/7】读取持仓数据...")
positions_file = ARTIFACTS_PATH / "portfolio_analysis/positions_normal_1day.pkl"
if not positions_file.exists():
    # 兜底：避免文件名/目录结构变动导致找不到
    matches = list((ARTIFACTS_PATH / "portfolio_analysis").glob("*positions*_1day*.pkl"))
    if len(matches) > 0:
        positions_file = matches[0]

print(f"  - positions_file: {positions_file}")
if positions_file.exists():
    try:
        with open(positions_file, "rb") as f:
            positions = pickle.load(f)
    except Exception as e:
        print(f"  ✗ positions pickle 加载失败: {e}")
        print(traceback.format_exc())
        analysis_results["positions"]["error"] = f"pickle加载失败: {e}"
        positions = None
    if positions is not None:
        print(f"  - positions_obj: {_preview_obj(positions)}")
    
    # 提取持仓信息
    position_data = []

    # 兼容多种positions对象结构
    try:
        # Case 1: qlib position object: positions.position[stock].data[date] = {...}
        if hasattr(positions, "position"):
            position_dict = positions.position
            for stock_id, stock_position in position_dict.items():
                if hasattr(stock_position, "data"):
                    data = stock_position.data
                    for date, info in data.items():
                        position_data.append(
                            {
                                "date": str(date),
                                "stock_id": stock_id,
                                "amount": info.get("amount", 0),
                                "value": info.get("value", 0),
                                "price": info.get("price", 0),
                            }
                        )

        # Case 2: DataFrame
        elif isinstance(positions, pd.DataFrame):
            df = positions.copy()
            # 常见：index为datetime，columns为instrument；或MultiIndex
            if {"date", "stock_id", "amount"}.issubset(set(df.columns)):
                for _, r in df.iterrows():
                    position_data.append(
                        {
                            "date": str(r["date"]),
                            "stock_id": str(r["stock_id"]),
                            "amount": float(r.get("amount", 0)),
                            "value": float(r.get("value", 0)),
                            "price": float(r.get("price", 0)),
                        }
                    )

        # Case 3: dict
        elif isinstance(positions, dict):
            # 可能是：{stock_id: {date: info}} 或 {date: {stock_id: info}}
            # 先打印一个样本，避免“盲猜结构”
            try:
                _first_k = next(iter(positions.keys())) if len(positions) > 0 else None
                if _first_k is not None:
                    _first_v = positions[_first_k]
                    print(f"  - positions_dict_sample_key: {_first_k}")
                    print(f"  - positions_dict_sample_value: {_preview_obj(_first_v)}")
                    if isinstance(_first_v, dict):
                        print(f"  - positions_dict_sample_value_keys: {[str(k) for k in list(_first_v.keys())[:30]]}")
            except Exception:
                pass

            # Case 3.1: dict[date] -> position_obj / dict
            # 常见：第一层 key 是日期
            if len(positions) > 0 and _is_date_like(next(iter(positions.keys()))):
                first_non_cash_dt = None
                first_non_cash_preview_printed = False
                daily_cash = {}  # 新增：存储每日现金
                for dt, pos_obj in list(positions.items()):
                    _before_cnt = len(position_data)

                    # 提取现金信息
                    cash = None
                    if hasattr(pos_obj, "get_cash"):
                        try:
                            cash = float(pos_obj.get_cash())
                        except Exception:
                            pass
                    elif hasattr(pos_obj, "cash"):
                        try:
                            cash = float(pos_obj.cash)
                        except Exception:
                            pass
                    elif hasattr(pos_obj, "position") and isinstance(pos_obj.position, dict):
                        for k in pos_obj.position.keys():
                            if _is_account_field(k):
                                try:
                                    cash = float(pos_obj.position[k])
                                    break
                                except Exception:
                                    pass
                    if cash is not None:
                        daily_cash[str(dt)] = cash

                    if (not first_non_cash_preview_printed) and hasattr(pos_obj, "position"):
                        try:
                            _pd = getattr(pos_obj, "position", None)
                            print(f"  - dt={dt} pos_obj_type={type(pos_obj)}")
                            print(f"  - dt={dt} pos_obj_attrs={_preview_obj(pos_obj)}")
                            print(f"  - dt={dt} pos_obj.position_type={type(_pd)}")
                            if isinstance(_pd, dict):
                                _keys = [str(k) for k in list(_pd.keys())[:50]]
                                print(f"  - dt={dt} pos_obj.position_keys_preview={_keys}")
                        except Exception:
                            pass

                    # 3.1.1 优先走 Position 的公开方法（不同版本字段更稳定）
                    if hasattr(pos_obj, "get_stock_list") and hasattr(pos_obj, "get_stock_amount"):
                        try:
                            _stock_list = list(pos_obj.get_stock_list())
                            if (not first_non_cash_preview_printed):
                                print(f"  - dt={dt} get_stock_list_len={len(_stock_list)}")
                            for stock_id in _stock_list:
                                amount = pos_obj.get_stock_amount(stock_id)
                                # price/value 不一定有，尽量取
                                price = pos_obj.get_stock_price(stock_id) if hasattr(pos_obj, "get_stock_price") else 0
                                value = pos_obj.get_stock_value(stock_id) if hasattr(pos_obj, "get_stock_value") else 0
                                try:
                                    amount = float(amount)
                                except Exception:
                                    amount = 0
                                try:
                                    price = float(price)
                                except Exception:
                                    price = 0
                                try:
                                    value = float(value)
                                except Exception:
                                    value = 0
                                if value == 0 and amount and price:
                                    value = amount * price
                                if amount and amount > 0:
                                    position_data.append(
                                        {
                                            "date": str(dt),
                                            "stock_id": str(stock_id),
                                            "amount": amount,
                                            "value": value,
                                            "price": price,
                                        }
                                    )
                            if len(_stock_list) == 0 and (not first_non_cash_preview_printed):
                                print(f"  - dt={dt} get_stock_list为空：可能当天只有现金或Position未记录股票")
                        except Exception:
                            pass

                    # 3.1.2 qlib.backtest.position.Position: 典型为 pos_obj.position = {inst: {amount, price, ...}, ...}
                    _added_this_dt = len(position_data) > _before_cnt
                    if (not _added_this_dt) and hasattr(pos_obj, "position"):
                        position_dict = getattr(pos_obj, "position", None)
                        if dt == next(iter(positions.keys())):
                            try:
                                print(f"  - pos_obj.position_preview: {_preview_obj(position_dict)}")
                                if isinstance(position_dict, dict):
                                    _items = list(position_dict.items())[:5]
                                    print(
                                        "  - pos_obj.position_sample_items: "
                                        + str([(str(k), _preview_obj(v)) for k, v in _items])
                                    )
                                    if len(_items) > 0 and isinstance(_items[0][1], dict):
                                        print(
                                            "  - pos_obj.position_sample_value_keys: "
                                            + str([str(k) for k in list(_items[0][1].keys())[:30]])
                                        )
                            except Exception:
                                pass
                        if isinstance(position_dict, dict):
                            for stock_id, stock_position in position_dict.items():
                                # 跳过现金/空键
                                if _is_account_field(stock_id):
                                    continue

                                amount = 0
                                price = 0
                                value = 0

                                if isinstance(stock_position, (int, float, np.number)):
                                    # 有些版本 Position.position 直接存 amount/weight 数值
                                    amount = stock_position
                                elif isinstance(stock_position, dict):
                                    amount = stock_position.get("amount", stock_position.get("count", 0))
                                    price = stock_position.get("price", stock_position.get("last_price", 0))
                                    value = stock_position.get("value", stock_position.get("market_value", 0))
                                else:
                                    # 一些实现可能是对象
                                    for _k in ["amount", "count"]:
                                        if hasattr(stock_position, _k):
                                            try:
                                                amount = getattr(stock_position, _k)
                                                break
                                            except Exception:
                                                pass
                                    if hasattr(stock_position, "price"):
                                        try:
                                            price = getattr(stock_position, "price")
                                        except Exception:
                                            pass
                                    if hasattr(stock_position, "value"):
                                        try:
                                            value = getattr(stock_position, "value")
                                        except Exception:
                                            pass

                                try:
                                    amount = float(amount)
                                except Exception:
                                    amount = 0
                                try:
                                    price = float(price)
                                except Exception:
                                    price = 0
                                try:
                                    value = float(value)
                                except Exception:
                                    value = 0

                                if value == 0 and amount and price:
                                    value = amount * price

                                # 过滤掉 0 持仓
                                if amount and amount > 0:
                                    position_data.append(
                                        {
                                            "date": str(dt),
                                            "stock_id": str(stock_id),
                                            "amount": amount,
                                            "value": value,
                                            "price": price,
                                        }
                                    )

                        # 仅在首个日期额外打印一组样本，方便快速判断结构
                        if dt == next(iter(positions.keys())) and isinstance(position_dict, dict):
                            try:
                                _non_cash_keys = [
                                    str(k)
                                    for k in position_dict.keys()
                                    if (not _is_account_field(k))
                                ]
                                print(f"  - dt={dt} position_non_cash_key_cnt={len(_non_cash_keys)}")
                                _sample = []
                                for _k in _non_cash_keys[:10]:
                                    _v = position_dict.get(_k)
                                    if isinstance(_v, (int, float, np.number)):
                                        _sample.append((_k, str(type(_v)), float(_v)))
                                    else:
                                        _sample.append((_k, str(type(_v)), _preview_obj(_v)))
                                print(f"  - dt={dt} position_non_cash_samples={_sample}")
                            except Exception:
                                pass
                    # 3.1.3 dict 结构：pos_obj 里可能有 stock_amount/stock_weight 等
                    elif isinstance(pos_obj, dict):
                        # 优先识别 stock_amount
                        stock_amount = None
                        for k in ["stock_amount", "stocks_amount", "amount", "position", "holdings"]:
                            if k in pos_obj and isinstance(pos_obj[k], dict):
                                stock_amount = pos_obj[k]
                                break
                        if stock_amount is not None:
                            for stock_id, amount in stock_amount.items():
                                position_data.append(
                                    {
                                        "date": str(dt),
                                        "stock_id": str(stock_id),
                                        "amount": amount,
                                        "value": 0,
                                        "price": 0,
                                    }
                                )
                        else:
                            # 尝试：pos_obj 直接是 stock->info
                            for stock_id, info in list(pos_obj.items())[:5000]:
                                if isinstance(info, dict) and ("amount" in info or "value" in info):
                                    position_data.append(
                                        {
                                            "date": str(dt),
                                            "stock_id": str(stock_id),
                                            "amount": info.get("amount", 0),
                                            "value": info.get("value", 0),
                                            "price": info.get("price", 0),
                                        }
                                    )

            # Case 3.2: 继续兼容旧的 {stock:{date:info}} / {date:{stock:info}} 猜测
            if not position_data:
                for k1, v1 in list(positions.items())[:2000]:
                    if isinstance(v1, dict):
                        for k2, info in list(v1.items())[:2000]:
                            if isinstance(info, dict) and ("amount" in info or "value" in info):
                                position_data.append(
                                    {
                                        "date": str(k2),
                                        "stock_id": str(k1),
                                        "amount": info.get("amount", 0),
                                        "value": info.get("value", 0),
                                        "price": info.get("price", 0),
                                    }
                                )
                            elif isinstance(info, dict) and isinstance(k2, str):
                                position_data.append(
                                    {
                                        "date": str(k1),
                                        "stock_id": str(k2),
                                        "amount": info.get("amount", 0),
                                        "value": info.get("value", 0),
                                        "price": info.get("price", 0),
                                    }
                                )
    except Exception as e:
        print(f"  ✗ 持仓解析异常: {e}")
        print(traceback.format_exc())
        analysis_results["positions"]["error"] = f"持仓解析异常: {e}"
    
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
        
        # 每日持仓统计（包含现金）
        daily_positions = positions_df.groupby('date').agg({
            'stock_id': 'count',
            'value': 'sum'
        }).rename(columns={'stock_id': 'stock_count', 'value': 'stock_value'})

        # 添加现金信息到每日统计
        daily_positions['cash'] = daily_positions.index.map(lambda x: daily_cash.get(str(x), 0))
        daily_positions['total_value'] = daily_positions['stock_value'] + daily_positions['cash']

        analysis_results["positions"]["daily_stats"] = {
            "avg_stock_count": float(daily_positions['stock_count'].mean()),
            "max_stock_count": int(daily_positions['stock_count'].max()),
            "min_stock_count": int(daily_positions['stock_count'].min()),
            "avg_total_value": float(daily_positions['total_value'].mean()),
            "max_total_value": float(daily_positions['total_value'].max()),
            "min_total_value": float(daily_positions['total_value'].min()),
            "avg_cash": float(daily_positions['cash'].mean()),
            "max_cash": float(daily_positions['cash'].max()),
            "min_cash": float(daily_positions['cash'].min())
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
        print("  ✗ 持仓数据为空(解析结果为空)")
        analysis_results["positions"]["error"] = "持仓数据为空(解析结果为空)"
else:
    print("  ✗ 未找到 positions_normal_1day.pkl")
    analysis_results["positions"]["error"] = "文件不存在"

# ============================================================================
# 3. 读取指标数据
# ============================================================================
print("【3/7】读取指标数据...")
indicators_file = ARTIFACTS_PATH / "portfolio_analysis/indicators_normal_1day.pkl"
if indicators_file.exists():
    with open(indicators_file, 'rb') as f:
        indicators = pickle.load(f)
    
    if isinstance(indicators, pd.DataFrame):
        # 指标统计
        analysis_results["indicators"]["columns"] = indicators.columns.tolist()
        analysis_results["indicators"]["shape"] = indicators.shape
        analysis_results["indicators"]["date_range"] = {
            "start": str(indicators.index.min()) if hasattr(indicators, 'index') else "N/A",
            "end": str(indicators.index.max()) if hasattr(indicators, 'index') else "N/A"
        }
        
        # 关键指标统计
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
        
        # 保存指标数据
        indicators.to_csv(OUTPUT_PATH / "indicators_detail.csv")

        # 从 indicators 直接计算 pos/pa/ffr（避免被外部 qlib_res.csv 误导）
        try:
            for _k in ["pos", "pa", "ffr"]:
                if _k in indicators.columns:
                    _s = pd.to_numeric(indicators[_k], errors="coerce")
                    if _s.notna().any():
                        analysis_results["summary"][f"1day.{_k}"] = float(_s.dropna().mean())
        except Exception:
            pass
        
        print(f"  ✓ 指标数据读取成功，共 {len(indicators)} 行")
    else:
        print("  ✗ 指标数据格式错误")
        analysis_results["indicators"]["error"] = "数据格式错误"
else:
    print("  ✗ 未找到 indicators_normal_1day.pkl")
    analysis_results["indicators"]["error"] = "文件不存在"

# ============================================================================
# 4. 读取预测数据
# ============================================================================
print("【4/7】读取预测数据...")
pred_file = ARTIFACTS_PATH / "pred.pkl"
if pred_file.exists():
    with open(pred_file, 'rb') as f:
        pred = pickle.load(f)
    
    if isinstance(pred, pd.DataFrame):
        # 预测统计
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
            
            # 评分分布
            score_bins = pd.cut(pred['score'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
            score_distribution = score_bins.value_counts().sort_index()
            analysis_results["predictions"]["score_distribution"] = {
                str(k): int(v) for k, v in score_distribution.items()
            }
            
            # 低于 0.2 的比例
            low_score_ratio = (pred['score'] < 0.2).sum() / len(pred)
            analysis_results["predictions"]["low_score_ratio"] = float(low_score_ratio)
        
        # 保存预测数据
        pred.to_csv(OUTPUT_PATH / "predictions_detail.csv")
        
        print(f"  ✓ 预测数据读取成功，共 {len(pred)} 条记录")
    else:
        print("  ✗ 预测数据格式错误")
        analysis_results["predictions"]["error"] = "数据格式错误"
else:
    print("  ✗ 未找到 pred.pkl")
    analysis_results["predictions"]["error"] = "文件不存在"

# ============================================================================
# 5. 交易行为分析
# ============================================================================
print("【5/7】分析交易行为...")
if 'positions' in analysis_results and 'error' not in analysis_results["positions"]:
    positions_df = pd.read_csv(OUTPUT_PATH / "positions_detail.csv")
    positions_df['date'] = pd.to_datetime(positions_df['date'])

    # 识别买入和卖出
    trading_events = []

    # 修复：原逻辑会漏掉“卖到 0”场景（股票在某天后不再出现在 positions_detail.csv）。
    # 这里对每只股票补齐 0 持仓的日期点，再做 diff 推断买卖。
    all_dates = pd.to_datetime(sorted(positions_df['date'].unique()))
    all_dates_idx = pd.DatetimeIndex(all_dates)

    for stock_id in positions_df['stock_id'].unique():
        stock_data = positions_df[positions_df['stock_id'] == stock_id].sort_values('date')

        s_amt = stock_data.set_index('date')['amount'].reindex(all_dates_idx).fillna(0.0)
        s_price = stock_data.set_index('date')['price'].reindex(all_dates_idx)
        # 对于补齐的 0 仓位点，价格用前值填充（卖出点价格不可靠时至少可用于粗略统计）
        s_price = pd.to_numeric(s_price, errors='coerce').ffill()

        delta = s_amt.diff().fillna(0.0)
        for dt, chg in delta.items():
            px = float(s_price.loc[dt]) if pd.notna(s_price.loc[dt]) else float('nan')
            if chg > 0:
                trading_events.append({
                    "date": str(dt),
                    "stock_id": stock_id,
                    "action": "BUY",
                    "amount_change": float(chg),  # 正数，表示买入股数
                    "price": px,
                })
            elif chg < 0:
                trading_events.append({
                    "date": str(dt),
                    "stock_id": stock_id,
                    "action": "SELL",
                    "amount_change": float(-chg),  # 正数，表示卖出股数
                    "price": px,
                })

    if trading_events:
        trading_df = pd.DataFrame(trading_events)
        trading_df['date'] = pd.to_datetime(trading_df['date'])

        # 规范数值列类型
        trading_df['price'] = pd.to_numeric(trading_df['price'], errors='coerce')
        trading_df['amount_change'] = pd.to_numeric(trading_df['amount_change'], errors='coerce').fillna(0.0)

        # 增加带符号的成交金额字段：
        # trade_value_no_cost: 成交金额（不含费用），BUY 为正（买入金额），SELL 为正（卖出金额）
        # cash_effect_no_cost: 对现金的影响，BUY 为负，SELL 为正，便于和 positions.cash 对账
        trading_df['trade_value_no_cost'] = trading_df['amount_change'] * trading_df['price']
        trading_df['cash_effect_no_cost'] = np.where(
            trading_df['action'] == 'BUY',
            -trading_df['trade_value_no_cost'],
            trading_df['trade_value_no_cost'],
        )

        # 交易统计
        buy_count = len(trading_df[trading_df['action'] == 'BUY'])
        sell_count = len(trading_df[trading_df['action'] == 'SELL'])

        analysis_results["trading_behavior"] = {
            "total_trades": len(trading_df),
            "buy_trades": buy_count,
            "sell_trades": sell_count,
            "unique_stocks_traded": trading_df['stock_id'].nunique(),
            "avg_daily_trades": float(len(trading_df) / trading_df['date'].nunique()),
        }

        # 按日期统计交易笔数
        daily_trades = trading_df.groupby('date').size()
        analysis_results["trading_behavior"]["daily_trades_stats"] = {
            "mean": float(daily_trades.mean()),
            "max": int(daily_trades.max()),
            "min": int(daily_trades.min()),
        }

        # 按日期汇总买入/卖出金额及净现金流（不含费用），方便和 positions.cash 做对账
        daily_cash = trading_df.groupby('date').apply(
            lambda df: pd.Series({
                'buy_value_no_cost': float(df.loc[df['action'] == 'BUY', 'trade_value_no_cost'].sum()),
                'sell_value_no_cost': float(df.loc[df['action'] == 'SELL', 'trade_value_no_cost'].sum()),
                'net_cash_effect_no_cost': float(df['cash_effect_no_cost'].sum()),
            })
        ).reset_index()

        # 保存交易事件与日度现金流汇总
        trading_df.to_csv(OUTPUT_PATH / "trading_events.csv", index=False)
        daily_cash.to_csv(OUTPUT_PATH / "daily_trading_cash_summary.csv", index=False)

        # 近似止盈/止损次数统计（按策略阈值）：
        # 注意：这里只能根据“推断的交易事件 + 近似成本”估算，并不能区分卖出触发原因（候选清仓/止盈/止损）。
        try:
            tp_12 = 0
            tp_22 = 0
            tp_32 = 0
            sl_10 = 0

            # 按股票做一个简单的“加权平均成本”跟踪
            cost_state: dict[str, dict[str, float]] = {}
            trading_df_sorted = trading_df.sort_values(['stock_id', 'date']).copy()

            for _, r in trading_df_sorted.iterrows():
                sid = str(r['stock_id'])
                act = str(r['action'])
                amt = float(r['amount_change'])
                px = float(r['price']) if pd.notna(r['price']) else float('nan')

                st = cost_state.setdefault(sid, {"amt": 0.0, "cost": 0.0})
                if act == 'BUY':
                    if amt > 0 and pd.notna(px) and px > 0:
                        # 更新加权平均成本
                        new_amt = st['amt'] + amt
                        st['cost'] = (st['cost'] * st['amt'] + px * amt) / new_amt if new_amt > 0 else 0.0
                        st['amt'] = new_amt
                    else:
                        st['amt'] += max(amt, 0.0)
                elif act == 'SELL':
                    if st['amt'] <= 0 or amt <= 0:
                        continue
                    # 用当前平均成本估算收益率
                    if pd.notna(px) and px > 0 and st['cost'] > 0:
                        rr = (px - st['cost']) / st['cost']
                        if rr <= -0.10:
                            sl_10 += 1
                        if rr >= 0.32:
                            tp_32 += 1
                        elif rr >= 0.22:
                            tp_22 += 1
                        elif rr >= 0.12:
                            tp_12 += 1
                    # 减少持仓
                    st['amt'] = max(st['amt'] - amt, 0.0)
                    if st['amt'] == 0:
                        st['cost'] = 0.0

            analysis_results['trading_behavior']['approx_stop_loss_cnt'] = int(sl_10)
            analysis_results['trading_behavior']['approx_take_profit_cnt_12'] = int(tp_12)
            analysis_results['trading_behavior']['approx_take_profit_cnt_22'] = int(tp_22)
            analysis_results['trading_behavior']['approx_take_profit_cnt_32'] = int(tp_32)
        except Exception:
            pass

        print(f"  ✓ 交易行为分析完成，共 {len(trading_df)} 笔交易")
    else:
        print("  ✗ 未检测到交易事件")
        analysis_results["trading_behavior"]["error"] = "未检测到交易事件"
else:
    print("  ✗ 持仓数据不可用，跳过交易行为分析")

# ============================================================================
# 6. 风险分析
# ============================================================================
print("【6/7】分析风险指标...")
if 'indicators' in analysis_results and 'error' not in analysis_results["indicators"]:
    indicators_df = pd.read_csv(OUTPUT_PATH / "indicators_detail.csv", index_col=0)
    
    # 计算回撤
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
    
    # 波动率分析
    if 'return' in indicators_df.columns:
        returns = indicators_df['return']
        analysis_results["risk_analysis"]["volatility"] = {
            "daily_volatility": float(returns.std()),
            "annualized_volatility": float(returns.std() * np.sqrt(252)),
            "downside_volatility": float(returns[returns < 0].std())
        }
        
        # 夏普比率（假设无风险利率为 3%）
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

# 保存 JSON 报告
with open(OUTPUT_PATH / "analysis_report.json", 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2, ensure_ascii=False)

# 生成文本报告
report_text = []
report_text.append("=" * 80)
report_text.append("Qlib 回测交易记录详细分析报告")
report_text.append("=" * 80)
report_text.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_text.append(f"工作空间: {WORKSPACE_PATH}")
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

# 保存文本报告
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
