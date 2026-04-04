import pickle, glob, re, os

log_root = '/mnt/f/Dev/RD-Agent-main/log'
all_tasks = sorted([t for t in os.listdir(log_root) if t.startswith('2026-03-')])

print(f"{'Task':<35} Loop  Script    IC      AnnRet    MaxDD     FFR     PA")
print('-' * 105)

for task_id in all_tasks:
    # Check all loops, not just L0
    for loop_n in range(20):
        rpkls = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/runner result/*/*.pkl'))
        qpkls = sorted(glob.glob(f'{log_root}/{task_id}/Loop_{loop_n}/running/Qlib_execute_log/*/*.pkl'))
        if not rpkls or not qpkls:
            continue
        try:
            with open(qpkls[0], 'rb') as f:
                log = pickle.load(f)
            if not isinstance(log, str):
                continue
            has_minute = any('Minute memory patch' in l for l in log.splitlines())
            script = 'minute' if has_minute else 'daily'

            with open(rpkls[0], 'rb') as f:
                exp = pickle.load(f)
            if not hasattr(exp, 'result') or exp.result is None:
                continue
            r = exp.result
            ic = r.get('IC', 0)
            ann_ret = r.get('1day.excess_return_without_cost.annualized_return')
            max_dd = r.get('1day.excess_return_without_cost.max_drawdown')
            ffr = r.get('1day.ffr')
            pa = r.get('1day.pa')
            if ann_ret is not None:
                print(f"{task_id:<35} L{loop_n:<3}  {script:<9} {ic:>7.4f}  {ann_ret:>8.3f}  {max_dd:>8.3f}  {ffr:>7.3f} {pa:>8.5f}")
        except:
            pass
