"""分析 AnnRet 19% 的提升空间：完整指标、持仓、交易细节"""
import pickle, glob, sys, os
import numpy as np

log_root = '/mnt/f/Dev/RD-Agent-main/log'
task_id = '2026-03-24_11-11-26-390074'

# 1. 打印所有 loop 的完整指标
for loop_n in range(5):
    base = f'{log_root}/{task_id}/Loop_{loop_n}/running'
    rpkls = sorted(glob.glob(f'{base}/runner result/*/*.pkl'))
    if not rpkls:
        continue

    with open(rpkls[0], 'rb') as f:
        exp = pickle.load(f)

    if not hasattr(exp, 'result') or exp.result is None:
        continue

    r = exp.result
    print(f"\n{'='*80}")
    print(f"Loop {loop_n} - 完整指标:")
    print(f"{'='*80}")
    for k, v in sorted(r.items()):
        if isinstance(v, (int, float)):
            print(f"  {k:<60} = {v:.6f}")
        else:
            print(f"  {k:<60} = {v}")

# 2. 分析持仓和交易数据 (report pkl)
print(f"\n{'='*80}")
print(f"持仓和交易分析 (Loop 0):")
print(f"{'='*80}")

# 找 report_normal pkl
import pandas as pd
for loop_n in [0, 1]:
    base = f'{log_root}/{task_id}/Loop_{loop_n}/running'
    rpkls = sorted(glob.glob(f'{base}/runner result/*/*.pkl'))
    if not rpkls:
        continue

    with open(rpkls[0], 'rb') as f:
        exp = pickle.load(f)

    # 检查 experiment 对象的属性
    print(f"\nLoop {loop_n} experiment attributes: {[a for a in dir(exp) if not a.startswith('_')]}")

    # 尝试找 recorder / artifacts
    if hasattr(exp, 'recorder'):
        rec = exp.recorder
        print(f"  recorder type: {type(rec)}")
        if hasattr(rec, 'list_artifacts'):
            arts = rec.list_artifacts()
            print(f"  artifacts: {arts}")

    # 查看 sub_workspace
    if hasattr(exp, 'sub_workspace_list'):
        for i, ws in enumerate(exp.sub_workspace_list):
            print(f"  sub_workspace[{i}] type: {type(ws)}")
            if hasattr(ws, 'file_dict'):
                print(f"    file_dict keys: {list(ws.file_dict.keys())}")

    break

# 3. 寻找 report/position pkl 文件 (可能在 mlruns 里)
print(f"\n{'='*80}")
print(f"搜索回测详细数据文件:")
print(f"{'='*80}")

# 找 qlib execute log 分析
for loop_n in [0, 1]:
    qpkls = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/Qlib_execute_log/*/*.pkl'))
    if not qpkls:
        continue

    with open(qpkls[0], 'rb') as f:
        log = pickle.load(f)

    if isinstance(log, str):
        lines = log.strip().split('\n')
        # 提取关键信息
        for line in lines:
            if any(kw in line for kw in ['topk', 'n_drop', 'account', 'exchange', 'benchmark',
                                          'ffr', 'pa ', 'turnover', 'return', 'Total', 'limit']):
                print(f"  L{loop_n}: {line.strip()}")

        # 找 mlruns 路径
        for line in lines:
            if 'mlruns' in line or 'artifact' in line:
                print(f"  L{loop_n} MLflow: {line.strip()}")
                break
    break

# 4. 直接在 task workspace 找 report pkl
print(f"\n搜索 workspace 内的 pkl 文件:")
workspace_patterns = [
    f'{log_root}/{task_id}/Loop_*/running/runner result/*/*/mlruns/**/*.pkl',
    f'{log_root}/{task_id}/Loop_*/direct_exp_gen/**/*report*.pkl',
]
for pat in workspace_patterns:
    found = glob.glob(pat, recursive=True)
    if found:
        for f in found[:5]:
            print(f"  {f}")

# 5. 找 mlruns 目录
for loop_n in [0, 1]:
    ws_dirs = glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/runner result/*/')
    if not ws_dirs:
        continue
    ws = ws_dirs[0]
    # 检查 mlruns 子目录
    mlruns_path = os.path.join(ws, 'mlruns')
    if os.path.isdir(mlruns_path):
        print(f"\nLoop {loop_n} mlruns 结构:")
        for root, dirs, files in os.walk(mlruns_path):
            depth = root.replace(mlruns_path, '').count(os.sep)
            indent = ' ' * 2 * depth
            basename = os.path.basename(root)
            if depth <= 3:
                print(f"  {indent}{basename}/")
            if depth <= 3:
                for f in files[:5]:
                    print(f"  {indent}  {f}")
    else:
        # 检查 workspace 根目录下的所有内容
        for item in os.listdir(ws):
            item_path = os.path.join(ws, item)
            if os.path.isdir(item_path):
                print(f"  L{loop_n} dir: {item}/")
                subitems = os.listdir(item_path)[:5]
                for si in subitems:
                    print(f"    {si}")
            else:
                print(f"  L{loop_n} file: {item}")
