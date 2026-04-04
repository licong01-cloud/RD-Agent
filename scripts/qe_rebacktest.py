"""
QE 策略快速回测工具 — 跳过模型训练，只调参数重新回测

用法:
  python qe_rebacktest.py <loop_path> [--topk 50] [--n_drop 5] [--hold_thresh 2]
                                       [--risk_degree 0.95] [--output_dir ./rebacktest_results]

示例:
  # 用 Loop1 的预测, 测试 n_drop=20
  python qe_rebacktest.py /path/to/Loop1 --n_drop 20

  # 批量扫描参数
  python qe_rebacktest.py /path/to/Loop1 --sweep

原理:
  1. 从已完成 Loop 中加载 pred.pkl (模型预测) + conf.yaml (回测配置)
  2. 用新的策略参数覆盖 conf.yaml 中的策略配置
  3. 只运行 normal_backtest() (分钟线回测, 含内存 patch)
  4. 输出回测指标 + 保存结果

耗时: ~3-5分钟 (vs 模型训练 30分钟+)
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


def find_pred_pkl(loop_path: Path):
    """在 mlruns 中找到 pred.pkl"""
    for pkl in loop_path.rglob("pred.pkl"):
        return pkl
    return None


def find_conf_yaml(loop_path: Path):
    """找到 conf.yaml"""
    for name in ["conf.yaml", "conf_baseline.yaml"]:
        p = loop_path / name
        if p.exists():
            return p
    return None


def load_benchmark_series(conf):
    """加载 benchmark (复用 qrun_limit_minute.py 逻辑)"""
    benchmark_path = None
    # 尝试从 loop_path 加载
    for p in [Path("benchmark_sh000300.parquet")]:
        if p.exists():
            df = pd.read_parquet(p)
            return df["bench"]

    # fallback: 从 qlib 生成
    try:
        from qlib.data import D
        bt = _find_backtest_config(conf)
        start = str(bt.get("start_time", "2024-07-01"))
        end = str(bt.get("end_time", "2026-03-10"))
        df = D.features(["000300.sh"], ["$close/Ref($close,1)-1"],
                        start_time=start, end_time=end, freq="day")
        if df.empty:
            return None
        df.columns = ["bench"]
        sr = df["bench"].droplevel("instrument")
        sr.index.name = "datetime"
        return sr
    except Exception as e:
        print(f"[WARN] benchmark load failed: {e}")
        return None


def _find_backtest_config(config):
    if isinstance(config, dict):
        if 'backtest' in config and isinstance(config['backtest'], dict):
            return config['backtest']
        for val in config.values():
            result = _find_backtest_config(val)
            if result:
                return result
    elif isinstance(config, list):
        for item in config:
            result = _find_backtest_config(item)
            if result:
                return result
    return None


def build_backtest_config(conf, pred, args):
    """从 conf.yaml 构建回测配置, 用命令行参数覆盖策略参数"""
    pa_config = conf.get("port_analysis_config", {})

    # 策略配置
    strategy_config = pa_config.get("strategy", {})
    strategy_kwargs = strategy_config.get("kwargs", {})

    # 用命令行参数覆盖
    strategy_kwargs["signal"] = pred
    if args.topk is not None:
        strategy_kwargs["topk"] = args.topk
    if args.n_drop is not None:
        strategy_kwargs["n_drop"] = args.n_drop
    if args.hold_thresh is not None:
        strategy_kwargs["hold_thresh"] = args.hold_thresh
    if args.risk_degree is not None:
        strategy_kwargs["risk_degree"] = args.risk_degree

    strategy_config["kwargs"] = strategy_kwargs

    # 执行器配置
    executor_config = pa_config.get("executor", {})

    # 回测配置
    backtest_config = pa_config.get("backtest", {})

    # limit_threshold: list -> tuple (递归处理所有层级)
    _patch_limit_threshold(executor_config)
    _patch_limit_threshold(backtest_config)

    return {
        "strategy": strategy_config,
        "executor": executor_config,
        "backtest": backtest_config,
    }


def _patch_limit_threshold(config):
    """递归将 limit_threshold list -> tuple"""
    if isinstance(config, dict):
        for key, val in config.items():
            if key == 'exchange_kwargs' and isinstance(val, dict):
                lt = val.get('limit_threshold')
                if isinstance(lt, list):
                    val['limit_threshold'] = tuple(lt)
            else:
                _patch_limit_threshold(val)
    elif isinstance(config, list):
        for item in config:
            _patch_limit_threshold(item)


def apply_minute_memory_patch():
    """复用 qrun_limit_minute.py 的内存优化 patch"""
    if os.environ.get("QLIB_MINUTE_FULL_LOAD") == "1":
        return

    from qlib.data import D
    from qlib.backtest.executor import NestedExecutor
    from qlib.backtest.exchange import Exchange
    from qlib.backtest.high_performance_ds import NumpyQuote

    _BATCH_TRADING_DAYS = 20

    def _reload_exchange_for_day(exchange, trade_start_time, trade_end_time):
        day_key = pd.Timestamp(trade_start_time).normalize()
        loaded_start = getattr(exchange, '_loaded_start', None)
        loaded_end = getattr(exchange, '_loaded_end', None)
        if loaded_start is not None and loaded_start <= day_key <= loaded_end:
            return

        backtest_end = getattr(exchange, 'end_time', trade_end_time)
        cal = D.calendar(start_time=trade_start_time, end_time=backtest_end, freq="day")
        if len(cal) == 0:
            return
        batch_end_day = cal[min(_BATCH_TRADING_DAYS - 1, len(cal) - 1)]
        batch_end_time = pd.Timestamp(batch_end_day).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)

        if hasattr(exchange, 'quote') and exchange.quote is not None:
            if hasattr(exchange.quote, 'get_data') and hasattr(exchange.quote.get_data, 'cache_clear'):
                exchange.quote.get_data.cache_clear()

        old_quote = getattr(exchange, 'quote', None)
        old_df = getattr(exchange, 'quote_df', None)

        day_df = D.features(
            exchange.codes, exchange.all_fields,
            trade_start_time, batch_end_time,
            freq=exchange.freq, disk_cache=False,
        )
        day_df.columns = exchange.all_fields
        exchange.quote_df = day_df
        exchange._update_limit(exchange.limit_threshold)
        exchange.quote = NumpyQuote(exchange.quote_df, exchange.freq)
        exchange._loaded_start = day_key
        exchange._loaded_end = pd.Timestamp(batch_end_day).normalize()
        del old_quote, old_df, day_df
        gc.collect()

    # Patch 1
    _orig_get_quote = Exchange.get_quote_from_qlib
    def _patched_get_quote(self):
        if self.freq in ('1min', '5min'):
            cal = D.calendar(start_time=self.start_time, end_time=self.end_time, freq="day")
            if len(cal) == 0:
                _orig_get_quote(self)
                return
            batch_end_day = cal[min(_BATCH_TRADING_DAYS - 1, len(cal) - 1)]
            batch_end_time = pd.Timestamp(batch_end_day).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
            if len(self.codes) == 0:
                self.codes = D.instruments()
            self.quote_df = D.features(
                self.codes, self.all_fields,
                self.start_time, batch_end_time,
                freq=self.freq, disk_cache=False,
            )
            self.quote_df.columns = self.all_fields
            self.trade_w_adj_price = (
                (self.quote_df["$factor"].isna() & ~self.quote_df["$close"].isna()).any()
            )
            self._update_limit(self.limit_threshold)
            self._loaded_start = pd.Timestamp(self.start_time).normalize()
            self._loaded_end = pd.Timestamp(batch_end_day).normalize()
        else:
            _orig_get_quote(self)
    Exchange.get_quote_from_qlib = _patched_get_quote

    # Patch 2
    _orig_init_sub = NestedExecutor._init_sub_trading
    def _patched_init_sub(self, trade_decision):
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time()
        exchange = self.trade_exchange
        if exchange.freq in ('1min', '5min'):
            _reload_exchange_for_day(exchange, trade_start_time, trade_end_time)
        _orig_init_sub(self, trade_decision)
    NestedExecutor._init_sub_trading = _patched_init_sub

    # Patch 3
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    _orig_gen_trade = TopkDropoutStrategy.generate_trade_decision
    def _patched_gen_trade(self, execute_result=None):
        exchange = self.trade_exchange
        if exchange.freq in ('1min', '5min'):
            try:
                trade_start_time, trade_end_time = self.trade_calendar.get_step_time()
                _reload_exchange_for_day(exchange, trade_start_time, trade_end_time)
            except IndexError:
                pass
        return _orig_gen_trade(self, execute_result)
    TopkDropoutStrategy.generate_trade_decision = _patched_gen_trade

    print("[INFO] Minute memory patch applied")


def run_backtest(bt_config, benchmark_series=None):
    """执行回测"""
    from qlib.backtest import backtest as qlib_backtest

    backtest_kwargs = bt_config["backtest"].copy()
    if benchmark_series is not None:
        backtest_kwargs["benchmark"] = benchmark_series

    portfolio_metric_dict, indicator_dict = qlib_backtest(
        executor=bt_config["executor"],
        strategy=bt_config["strategy"],
        **backtest_kwargs,
    )
    return portfolio_metric_dict, indicator_dict


def compute_metrics(portfolio_metric_dict, indicator_dict):
    """计算回测指标"""
    from qlib.contrib.evaluate import risk_analysis

    results = {}
    for freq, (report, positions) in portfolio_metric_dict.items():
        r = report
        # 基本指标
        init_account = r['account'].iloc[0]
        final_account = r['account'].iloc[-1]
        days = (r.index[-1] - r.index[0]).days
        years = days / 365.25
        total_return = final_account / init_account - 1
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 最大回撤
        cum_max = r['account'].cummax()
        drawdown = (r['account'] - cum_max) / cum_max
        max_dd = drawdown.min()

        # Sharpe
        daily_ret = r['return']
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

        # 超额
        if 'bench' in r.columns:
            bench_cum = (1 + r['bench']).cumprod()
            bench_total = bench_cum.iloc[-1] - 1
            bench_ann = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0
            excess_ann = ann_return - bench_ann

            excess_daily = r['return'] - r['bench']
            excess_sharpe = excess_daily.mean() / excess_daily.std() * np.sqrt(252) if excess_daily.std() > 0 else 0
        else:
            bench_ann = 0
            excess_ann = ann_return
            excess_sharpe = sharpe

        # Calmar
        calmar = ann_return / abs(max_dd) if max_dd != 0 else float('inf')

        # 月度统计
        monthly = r['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        pos_months = (monthly > 0).sum()

        results[freq] = {
            'total_return': total_return,
            'ann_return': ann_return,
            'bench_ann': bench_ann,
            'excess_ann': excess_ann,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'excess_sharpe': excess_sharpe,
            'calmar': calmar,
            'pos_months': f"{pos_months}/{len(monthly)}",
            'final_account': final_account,
        }

        # indicators
        if freq in indicator_dict:
            ind = indicator_dict[freq][0]
            if 'ffr' in ind.columns:
                results[freq]['avg_ffr'] = ind['ffr'].mean()
            if 'pa' in ind.columns:
                results[freq]['avg_pa'] = ind['pa'].mean()

    return results


def print_results(results, params):
    """打印回测结果"""
    print("\n" + "=" * 70)
    print("策略参数: topk={}, n_drop={}, hold_thresh={}, risk_degree={}".format(
        params['topk'], params['n_drop'], params['hold_thresh'], params['risk_degree']))
    print("=" * 70)

    for freq, m in results.items():
        print("\n[{}]".format(freq))
        print("  年化收益:   {:.2f}%".format(m['ann_return'] * 100))
        print("  基准年化:   {:.2f}%".format(m['bench_ann'] * 100))
        print("  超额年化:   {:.2f}%".format(m['excess_ann'] * 100))
        print("  Sharpe:     {:.2f}".format(m['sharpe']))
        print("  超额Sharpe: {:.2f}".format(m['excess_sharpe']))
        print("  最大回撤:   {:.2f}%".format(m['max_drawdown'] * 100))
        print("  Calmar:     {:.2f}".format(m['calmar']))
        print("  正收益月:   {}".format(m['pos_months']))
        print("  最终净值:   {:,.0f}".format(m['final_account']))
        if 'avg_ffr' in m:
            print("  平均成交率: {:.2f}%".format(m['avg_ffr'] * 100))
        if 'avg_pa' in m:
            print("  价格优势:   {:.4f}".format(m['avg_pa']))


def run_sweep(loop_path, conf, pred, benchmark_series):
    """参数扫描模式"""
    import qlib
    from qlib.config import C

    results_all = []

    sweep_configs = [
        # (topk, n_drop, hold_thresh, risk_degree)
        (50, 5, 2, 0.95),    # 当前配置
        (50, 10, 2, 0.95),
        (50, 15, 2, 0.95),
        (50, 20, 2, 0.95),
        (50, 30, 1, 0.95),
        (50, 5, 1, 0.95),
        (40, 5, 2, 0.95),
        (30, 5, 2, 0.95),
        (30, 10, 1, 0.95),
        (30, 15, 1, 0.95),
    ]

    print("\n===== 参数扫描开始 =====")
    print("{:>5} {:>7} {:>6} {:>5} | {:>8} {:>8} {:>8} {:>7} {:>7}".format(
        "topk", "n_drop", "hold", "risk", "年化%", "超额%", "回撤%", "Sharpe", "FFR%"))
    print("-" * 80)

    for topk, n_drop, hold_thresh, risk_degree in sweep_configs:
        args_mock = argparse.Namespace(
            topk=topk, n_drop=n_drop,
            hold_thresh=hold_thresh, risk_degree=risk_degree)

        bt_config = build_backtest_config(conf, pred, args_mock)
        try:
            portfolio_metric_dict, indicator_dict = run_backtest(bt_config, benchmark_series)
            metrics = compute_metrics(portfolio_metric_dict, indicator_dict)
            m = metrics.get('1day', {})

            marker = " <-- baseline" if (topk, n_drop, hold_thresh) == (50, 5, 2) else ""
            print("{:>5} {:>7} {:>6} {:>5.2f} | {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f} {:>6.1f}%{}".format(
                topk, n_drop, hold_thresh, risk_degree,
                m.get('ann_return', 0) * 100,
                m.get('excess_ann', 0) * 100,
                m.get('max_drawdown', 0) * 100,
                m.get('sharpe', 0),
                m.get('avg_ffr', 0) * 100,
                marker))

            results_all.append({
                'topk': topk, 'n_drop': n_drop,
                'hold_thresh': hold_thresh, 'risk_degree': risk_degree,
                **m
            })
        except Exception as e:
            print("{:>5} {:>7} {:>6} {:>5.2f} | ERROR: {}".format(
                topk, n_drop, hold_thresh, risk_degree, str(e)[:50]))

        gc.collect()

    # 保存结果
    if results_all:
        df = pd.DataFrame(results_all)
        output_path = loop_path / "rebacktest_sweep.csv"
        df.to_csv(output_path, index=False)
        print("\n结果已保存: {}".format(output_path))

    return results_all


def main():
    parser = argparse.ArgumentParser(description="QE 策略快速回测 — 跳过模型训练")
    parser.add_argument("loop_path", type=str, help="已完成的 Loop 目录路径")
    parser.add_argument("--topk", type=int, default=None, help="持仓数 (默认: 沿用原配置)")
    parser.add_argument("--n_drop", type=int, default=None, help="每日换仓数 (默认: 沿用原配置)")
    parser.add_argument("--hold_thresh", type=int, default=None, help="最小持仓天数 (默认: 沿用原配置)")
    parser.add_argument("--risk_degree", type=float, default=None, help="仓位比例 (默认: 沿用原配置)")
    parser.add_argument("--sweep", action="store_true", help="参数扫描模式")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    loop_path = Path(args.loop_path).resolve()
    print("Loop 目录: {}".format(loop_path))

    # 1. 加载 pred.pkl
    pred_path = find_pred_pkl(loop_path)
    if pred_path is None:
        print("[ERROR] 找不到 pred.pkl")
        sys.exit(1)
    print("加载预测: {}".format(pred_path))
    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)
    print("  预测形状: {}, 日期范围: {} ~ {}".format(
        pred.shape,
        pred.index.get_level_values(0).min().date(),
        pred.index.get_level_values(0).max().date()))

    # 2. 加载 conf.yaml
    conf_path = find_conf_yaml(loop_path)
    if conf_path is None:
        print("[ERROR] 找不到 conf.yaml")
        sys.exit(1)
    print("加载配置: {}".format(conf_path))

    from jinja2 import Template, meta
    from ruamel.yaml import YAML
    with open(conf_path) as f:
        content = f.read()
    template = Template(content)
    env = template.environment
    parsed = env.parse(content)
    variables = meta.find_undeclared_variables(parsed)
    context = {var: os.environ[var] for var in variables if var in os.environ}
    rendered = template.render(context)
    yaml = YAML(typ="safe", pure=True)
    conf = yaml.load(rendered)

    # 3. 初始化 qlib
    import qlib
    from qlib.config import C
    qlib_init = conf.get("qlib_init", {})
    tracking_uri = str(loop_path / "mlruns")
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + tracking_uri
    qlib.init(**qlib_init, exp_manager=exp_manager)

    # 4. 复制 tail_twap_strategy.py 到当前路径 (strategy 需要 import)
    tail_twap_src = loop_path / "tail_twap_strategy.py"
    if tail_twap_src.exists():
        sys.path.insert(0, str(loop_path))
        print("加载执行策略: {}".format(tail_twap_src))

    # 5. 应用内存 patch
    apply_minute_memory_patch()

    # 6. 加载 benchmark
    os.chdir(loop_path)  # benchmark parquet 在 loop 目录下
    benchmark_series = load_benchmark_series(conf)
    if benchmark_series is not None:
        print("Benchmark 加载成功: {} 天".format(len(benchmark_series)))

    # 7. 执行回测
    if args.sweep:
        run_sweep(loop_path, conf, pred, benchmark_series)
    else:
        start_time = time.time()
        bt_config = build_backtest_config(conf, pred, args)

        # 显示实际使用的参数
        sk = bt_config["strategy"]["kwargs"]
        params = {
            'topk': sk.get('topk'),
            'n_drop': sk.get('n_drop'),
            'hold_thresh': sk.get('hold_thresh'),
            'risk_degree': sk.get('risk_degree'),
        }
        print("\n策略参数: {}".format(params))
        print("开始回测...")

        portfolio_metric_dict, indicator_dict = run_backtest(bt_config, benchmark_series)
        elapsed = time.time() - start_time
        print("回测完成, 耗时: {:.1f}秒".format(elapsed))

        metrics = compute_metrics(portfolio_metric_dict, indicator_dict)
        print_results(metrics, params)

        # 保存结果
        output_dir = Path(args.output_dir) if args.output_dir else loop_path / "rebacktest"
        output_dir.mkdir(parents=True, exist_ok=True)

        tag = "topk{}_ndrop{}_hold{}".format(
            params['topk'], params['n_drop'], params['hold_thresh'])

        for freq, (report, positions) in portfolio_metric_dict.items():
            report.to_pickle(output_dir / "report_{}_{}.pkl".format(freq, tag))

        with open(output_dir / "metrics_{}.json".format(tag), 'w') as f:
            # convert numpy types
            serializable = {}
            for freq, m in metrics.items():
                serializable[freq] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                                      for k, v in m.items()}
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print("\n结果已保存至: {}".format(output_dir))


if __name__ == "__main__":
    main()
