import pickle, glob, os

log_dir = '/mnt/f/Dev/RD-Agent-main/log/2026-03-08_07-57-15-124539'

print("{:>5} {:>25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    'Loop', 'Strategy', 'AnnRet', 'MaxDD', 'IC', 'ICIR', 'InfoRatio', 'Decision'))
print('-' * 100)

for loop_id in range(6):
    runner_files = sorted(glob.glob(os.path.join(log_dir, 'Loop_{}/running/runner result/*/*.pkl'.format(loop_id))))
    metrics = {}
    for f in runner_files:
        with open(f, 'rb') as fh:
            data = pickle.load(fh)
        if hasattr(data, 'result') and data.result is not None:
            metrics = data.result.to_dict()
        break

    fb_files = sorted(glob.glob(os.path.join(log_dir, 'Loop_{}/feedback/feedback/*/*.pkl'.format(loop_id))))
    decision = '?'
    for f in fb_files:
        with open(f, 'rb') as fh:
            data = pickle.load(fh)
        if hasattr(data, 'decision'):
            decision = 'ACC' if data.decision else 'REJ'
        break

    coder_files = sorted(glob.glob(os.path.join(log_dir, 'Loop_{}/coding/coder result/*/*.pkl'.format(loop_id))))
    strategy = '?'
    for f in coder_files:
        with open(f, 'rb') as fh:
            data = pickle.load(fh)
        if hasattr(data, 'experiment_workspace') and hasattr(data.experiment_workspace, 'workspace_path'):
            ws = data.experiment_workspace.workspace_path
            for yf in os.listdir(ws):
                if yf.endswith('.yaml') and 'sota' in yf:
                    with open(os.path.join(ws, yf), 'r') as fh2:
                        for line in fh2:
                            if 'class: SoftmaxWeighted' in line:
                                strategy = 'Softmax(T=0.5)'
                                break
                            elif 'class: TopkDropout' in line:
                                strategy = 'TopkDropout(equal)'
                                break
                    if strategy != '?':
                        break
        break

    ann_ret = metrics.get('1day.excess_return_without_cost.annualized_return', 0)
    max_dd = metrics.get('1day.excess_return_without_cost.max_drawdown', 0)
    ic = metrics.get('IC', 0)
    icir = metrics.get('ICIR', 0)
    ir = metrics.get('1day.excess_return_without_cost.information_ratio', 0)

    print("{:>5} {:>25} {:>10.1%} {:>10.1%} {:>10.4f} {:>10.4f} {:>10.2f} {:>10}".format(
        loop_id, strategy, ann_ret, max_dd, ic, icir, ir, decision))
