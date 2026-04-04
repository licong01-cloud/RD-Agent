"""深入分析提升空间：成本、持仓集中度、涨停阻止分布"""
import pickle, glob, os
import numpy as np
import pandas as pd

log_root = '/mnt/f/Dev/RD-Agent-main/log'
task_id = '2026-03-24_11-11-26-390074'

# ============================================================
# 1. 成本异常诊断：with_cost vs without_cost 完全相同
# ============================================================
print("=" * 80)
print("1. 成本异常诊断")
print("=" * 80)
for loop_n in range(4):
    base = f'{log_root}/{task_id}/Loop_{loop_n}/running'
    rpkls = sorted(glob.glob(f'{base}/runner result/*/*.pkl'))
    if not rpkls:
        continue
    with open(rpkls[0], 'rb') as f:
        exp = pickle.load(f)
    if not hasattr(exp, 'result') or exp.result is None:
        continue
    r = exp.result
    with_cost = r.get('1day.excess_return_with_cost.annualized_return', 0)
    without_cost = r.get('1day.excess_return_without_cost.annualized_return', 0)
    diff = without_cost - with_cost
    print(f"  L{loop_n}: without_cost={without_cost:.6f}  with_cost={with_cost:.6f}  diff={diff:.6f}")

# ============================================================
# 2. 对比分钟线 buggy 任务的成本差异
# ============================================================
print(f"\n{'='*80}")
print("2. 对比其他任务的成本差异")
print(f"{'='*80}")
for task in ['2026-03-23_02-28-33-806649', '2026-03-23_06-37-22-928905']:
    for loop_n in range(2):
        base = f'{log_root}/{task}/Loop_{loop_n}/running'
        rpkls = sorted(glob.glob(f'{base}/runner result/*/*.pkl'))
        if not rpkls:
            continue
        with open(rpkls[0], 'rb') as f:
            exp = pickle.load(f)
        if not hasattr(exp, 'result') or exp.result is None:
            continue
        r = exp.result
        with_cost = r.get('1day.excess_return_with_cost.annualized_return', 0)
        without_cost = r.get('1day.excess_return_without_cost.annualized_return', 0)
        diff = without_cost - with_cost
        print(f"  {task} L{loop_n}: without={without_cost:.6f}  with={with_cost:.6f}  diff={diff:.6f}")

# ============================================================
# 3. 找 mlruns 里的详细 report/positions pkl
# ============================================================
print(f"\n{'='*80}")
print("3. 寻找 MLflow artifacts (report/positions)")
print(f"{'='*80}")

# 从 qrun log 提取 mlruns 路径
for loop_n in [0]:
    qpkls = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/Qlib_execute_log/*/*.pkl'))
    if not qpkls:
        continue
    with open(qpkls[0], 'rb') as f:
        log = pickle.load(f)
    if not isinstance(log, str):
        continue
    # 找 experiment ID 和 run ID
    for line in log.split('\n'):
        if 'Experiment' in line and ('saved' in line or 'created' in line or 'ID' in line):
            print(f"  {line.strip()}")

# 找 runner result workspace 下的 mlruns
for loop_n in [0, 1]:
    ws_dirs = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/runner result/*/'))
    if not ws_dirs:
        continue
    ws = ws_dirs[0]
    mlruns = os.path.join(ws, 'mlruns')
    if not os.path.isdir(mlruns):
        print(f"  L{loop_n}: mlruns not found at {mlruns}")
        continue

    # 遍历 mlruns 找到 artifacts
    for root, dirs, files in os.walk(mlruns):
        for f in files:
            if f.endswith('.pkl'):
                fpath = os.path.join(root, f)
                fsize = os.path.getsize(fpath) / 1024
                print(f"  L{loop_n}: {f} ({fsize:.1f} KB) - {os.path.relpath(fpath, mlruns)}")

# ============================================================
# 4. 加载 report_normal_1day.pkl 分析持仓和交易
# ============================================================
print(f"\n{'='*80}")
print("4. 分析 report_normal 数据")
print(f"{'='*80}")

for loop_n in [0, 1]:
    ws_dirs = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/runner result/*/'))
    if not ws_dirs:
        continue
    ws = ws_dirs[0]

    # 找 report pkl
    report_pkls = glob.glob(f'{ws}/mlruns/**/report_normal_1day.pkl', recursive=True)
    position_pkls = glob.glob(f'{ws}/mlruns/**/positions_normal_1day.pkl', recursive=True)
    indicator_pkls = glob.glob(f'{ws}/mlruns/**/indicator_normal_1day.pkl', recursive=True)

    print(f"\n  Loop {loop_n}:")
    print(f"    report_normal: {len(report_pkls)} files")
    print(f"    positions_normal: {len(position_pkls)} files")
    print(f"    indicator_normal: {len(indicator_pkls)} files")

    # 分析 report
    if report_pkls:
        with open(report_pkls[0], 'rb') as f:
            report = pickle.load(f)
        print(f"    report type: {type(report)}")
        if isinstance(report, (pd.DataFrame, pd.Series)):
            print(f"    report shape: {report.shape}")
            print(f"    report columns: {list(report.columns) if hasattr(report, 'columns') else 'N/A'}")
            if hasattr(report, 'columns'):
                for col in report.columns:
                    vals = report[col].dropna()
                    print(f"      {col}: mean={vals.mean():.6f}, std={vals.std():.6f}, min={vals.min():.6f}, max={vals.max():.6f}")
        elif isinstance(report, tuple):
            print(f"    report is tuple of {len(report)} elements")
            for i, elem in enumerate(report):
                print(f"      [{i}] type={type(elem)}, shape={getattr(elem, 'shape', 'N/A')}")
                if isinstance(elem, pd.DataFrame) and len(elem.columns) < 20:
                    print(f"          columns: {list(elem.columns)}")

    # 分析 positions
    if position_pkls:
        with open(position_pkls[0], 'rb') as f:
            positions = pickle.load(f)
        print(f"    positions type: {type(positions)}")
        if isinstance(positions, dict):
            dates = sorted(positions.keys())
            print(f"    positions: {len(dates)} dates, {dates[0]} ~ {dates[-1]}")
            # 分析持仓数量和集中度
            n_stocks_list = []
            for d in dates:
                pos = positions[d]
                n_stocks = len([k for k in pos if k != 'cash'])
                n_stocks_list.append(n_stocks)
            print(f"    avg stocks held: {np.mean(n_stocks_list):.1f}, min={min(n_stocks_list)}, max={max(n_stocks_list)}")

    # 分析 indicators
    if indicator_pkls:
        with open(indicator_pkls[0], 'rb') as f:
            indicators = pickle.load(f)
        print(f"    indicators type: {type(indicators)}")
        if isinstance(indicators, dict):
            for k, v in indicators.items():
                print(f"      {k}: type={type(v)}")
                if isinstance(v, pd.DataFrame):
                    print(f"        shape={v.shape}, columns={list(v.columns)[:10]}")
                    # 汇总统计
                    for col in v.columns[:8]:
                        vals = v[col].dropna()
                        if len(vals) > 0:
                            print(f"        {col}: mean={vals.mean():.6f}, sum={vals.sum():.4f}")
        elif isinstance(indicators, (pd.DataFrame, pd.Series)):
            print(f"      shape: {indicators.shape}")
            if hasattr(indicators, 'columns'):
                print(f"      columns: {list(indicators.columns)}")

    if loop_n == 0:
        break  # 只分析 L0 的详细数据
