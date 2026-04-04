"""分析持仓数据和成本问题，找到提升空间"""
import pickle, glob, os, sys
import numpy as np
import pandas as pd

log_root = '/mnt/f/Dev/RD-Agent-main/log'
task_id = '2026-03-24_11-11-26-390074'

# ============================================================
# 1. 从 runner result 的 experiment 对象提取 sub_results
# ============================================================
for loop_n in [0, 1, 3]:
    rpkls = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/runner result/*/*.pkl'))
    if not rpkls:
        continue
    with open(rpkls[0], 'rb') as f:
        exp = pickle.load(f)

    print(f"\n{'='*80}")
    print(f"Loop {loop_n}")
    print(f"{'='*80}")

    # Check sub_results
    if hasattr(exp, 'sub_results') and exp.sub_results:
        print(f"  sub_results: {exp.sub_results}")

    # Check running_info for report data
    if hasattr(exp, 'running_info') and exp.running_info:
        ri = exp.running_info
        print(f"  running_info keys: {list(ri.keys()) if isinstance(ri, dict) else type(ri)}")
        if isinstance(ri, dict):
            for k, v in ri.items():
                if isinstance(v, pd.DataFrame):
                    print(f"    {k}: DataFrame shape={v.shape}, columns={list(v.columns)[:10]}")
                elif isinstance(v, pd.Series):
                    print(f"    {k}: Series len={len(v)}")
                elif isinstance(v, dict):
                    print(f"    {k}: dict keys={list(v.keys())[:10]}")
                else:
                    vstr = str(v)[:100]
                    print(f"    {k}: {type(v).__name__} = {vstr}")

# ============================================================
# 2. 查找 qlib_results_enhanced.json (read_exp_res.py 的输出)
# ============================================================
print(f"\n{'='*80}")
print("2. 寻找 qlib_results_enhanced.json")
print(f"{'='*80}")
import json
for loop_n in range(4):
    ws_dirs = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/runner result/*/'))
    if not ws_dirs:
        continue
    ws = ws_dirs[0]
    json_path = os.path.join(ws, 'qlib_results_enhanced.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        print(f"  L{loop_n}: found! keys={list(data.keys())}")
        if 'trade_diagnostics' in data:
            td = data['trade_diagnostics']
            print(f"    trade_diagnostics: {json.dumps(td, indent=6)}")
        if 'summary' in data:
            print(f"    summary: {json.dumps(data['summary'], indent=6)}")
        if 'return_curves' in data:
            rc = data['return_curves']
            print(f"    return_curves keys: {list(rc.keys())}")
            if 'dates' in rc:
                print(f"    dates: {len(rc['dates'])} entries, first={rc['dates'][0]}, last={rc['dates'][-1]}")
    else:
        print(f"  L{loop_n}: not found at {json_path}")

# ============================================================
# 3. 直接从 qrun log 提取换手率和交易统计
# ============================================================
print(f"\n{'='*80}")
print("3. 从 qrun log 提取交易统计")
print(f"{'='*80}")
for loop_n in [0, 1, 3]:
    qpkls = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/Qlib_execute_log/*/*.pkl'))
    if not qpkls:
        continue
    with open(qpkls[0], 'rb') as f:
        log = pickle.load(f)
    if not isinstance(log, str):
        continue
    lines = log.strip().split('\n')
    print(f"\n  Loop {loop_n}:")
    # 提取所有 analysis results 部分
    in_analysis = False
    for line in lines:
        if 'analysis results' in line.lower() or 'excess_return' in line or 'ffr' in line or 'pa ' in line:
            in_analysis = True
        if in_analysis:
            print(f"    {line.rstrip()}")
            if line.strip() == '' and in_analysis:
                in_analysis = False
        # 也找 turnover 相关
        if 'turnover' in line.lower() or 'trade_count' in line.lower() or 'total_cost' in line.lower():
            print(f"    {line.rstrip()}")

# ============================================================
# 4. 对比不同 topk 的历史数据（日线 task 有用不同 topk 的吗？）
# ============================================================
print(f"\n{'='*80}")
print("4. 分析 FFR 与收益的非线性关系")
print(f"{'='*80}")

# 收集所有分钟线任务的 FFR vs AnnRet 数据
minute_data = []
for task in sorted(os.listdir(log_root)):
    if not task.startswith('2026-03-2'):
        continue
    for loop_n in range(20):
        rpkls = sorted(glob.glob(f'{log_root}/{task}/Loop_{loop_n}/running/runner result/*/*.pkl'))
        qpkls = sorted(glob.glob(f'{log_root}/{task}/Loop_{loop_n}/running/Qlib_execute_log/*/*.pkl'))
        if not rpkls or not qpkls:
            continue
        try:
            with open(qpkls[0], 'rb') as f:
                qlog = pickle.load(f)
            if not isinstance(qlog, str) or 'Minute memory patch' not in qlog:
                continue
            with open(rpkls[0], 'rb') as f:
                exp = pickle.load(f)
            if not hasattr(exp, 'result') or exp.result is None:
                continue
            r = exp.result
            ffr = r.get('1day.ffr')
            ann_ret = r.get('1day.excess_return_without_cost.annualized_return')
            pa = r.get('1day.pa')
            ic = r.get('IC')
            if ffr and ann_ret and not np.isnan(ffr):
                minute_data.append({
                    'task': task, 'loop': loop_n,
                    'ffr': ffr, 'ann_ret': ann_ret, 'pa': pa, 'ic': ic
                })
        except:
            pass

if minute_data:
    df = pd.DataFrame(minute_data)
    print(f"  分钟线数据点: {len(df)}")

    # FFR 分组统计
    df['ffr_bin'] = pd.cut(df['ffr'], bins=[0.90, 0.93, 0.95, 0.96, 1.0])
    grouped = df.groupby('ffr_bin').agg({
        'ann_ret': ['mean', 'std', 'count'],
        'ic': 'mean',
        'pa': 'mean'
    })
    print(f"\n  FFR 分组统计:")
    print(grouped.to_string())

    # 相关系数
    corr = df[['ffr', 'ann_ret', 'pa', 'ic']].corr()
    print(f"\n  相关系数矩阵:")
    print(corr.to_string())

# ============================================================
# 5. 检查成本是否为0的根因
# ============================================================
print(f"\n{'='*80}")
print("5. 检查成本配置是否在分钟线模式下生效")
print(f"{'='*80}")
for loop_n in [0]:
    qpkls = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/Qlib_execute_log/*/*.pkl'))
    if not qpkls:
        continue
    with open(qpkls[0], 'rb') as f:
        log = pickle.load(f)
    if not isinstance(log, str):
        continue
    # 找 exchange 创建和 cost 相关信息
    for line in log.split('\n'):
        if any(kw in line.lower() for kw in ['cost', 'open_cost', 'close_cost', 'commission', 'exchange_kwargs']):
            print(f"  {line.strip()}")

    # 找 PortAnaRecord 输出段落
    lines = log.split('\n')
    for i, line in enumerate(lines):
        if 'PortAnaRecord' in line or 'port_analysis' in line:
            for j in range(max(0, i-1), min(len(lines), i+5)):
                print(f"  {lines[j].rstrip()}")
            print()
