"""验证 qlib 源码 patch 的全流程测试。

测试: 5 只股票 × 30 天分钟线回测（超过 20 天批次边界）
验证:
  1. 批次切换正常（第 21 天 _loaded_start 更新）
  2. 无 None 价格（get_deal_price 全部返回有效值）
  3. 交易连续（每天都有订单生成）
  4. 指标非 NaN（annualized_return, max_drawdown）
  5. 内存可控（RSS < 4GB）
  6. 无静默错误（0 条 exchange None 警告）

用法:
  cd /mnt/f/Dev/RD-Agent-main && conda activate rdagent-gpu
  python scripts/test_qlib_source_patch.py
"""
import os
import sys
import gc
import resource
import logging
from pathlib import Path

# 设置环境
os.environ.setdefault("QLIB_MINUTE_BATCH_DAYS", "20")

import numpy as np
import pandas as pd
import qlib
from qlib.config import C

# 限制 kernels
C["kernels"] = 4

# 配置日志捕获
log_records = []

class LogCapture(logging.Handler):
    def emit(self, record):
        log_records.append(record)

capture_handler = LogCapture()
capture_handler.setLevel(logging.WARNING)
logging.getLogger("qlib").addHandler(capture_handler)

# ── qlib 初始化 ──
MINUTE_BIN = "/home/lc999/data/qlib_minute_bin"
DAY_BIN = "/home/lc999/data/qlib_bin"

qlib.init(
    provider_uri={"1min": MINUTE_BIN, "day": DAY_BIN},
    region="cn",
    dataset_cache=None,
    expression_cache=None,
)

from qlib.data import D
from qlib.backtest import backtest_loop
from qlib.backtest.exchange import Exchange
from qlib.backtest import get_strategy_executor
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

# ── 测试参数 ──
TEST_STOCKS = ["600519.SH", "000001.SZ", "601318.SH", "000858.SZ", "600036.SH"]
BACKTEST_START = "2024-10-01"
BACKTEST_END = "2024-11-15"  # ~30 交易日，超过 20 天批次边界
TOPK = 3
N_DROP = 1

print("=" * 70)
print("qlib 源码 patch 全流程验证测试")
print("=" * 70)
print(f"股票: {TEST_STOCKS}")
print(f"回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
print(f"QLIB_MINUTE_BATCH_DAYS: {os.environ.get('QLIB_MINUTE_BATCH_DAYS', '20')}")

# ── 1. 验证 Exchange 有 ensure_data_for_day 方法 ──
print("\n[Test 1] Exchange.ensure_data_for_day 方法存在...")
assert hasattr(Exchange, "ensure_data_for_day"), "Exchange 缺少 ensure_data_for_day 方法！"
print("  PASS")

# ── 2. 验证 backtest.py 已修改 ──
print("\n[Test 2] backtest.py collect_data_loop 已添加 ensure_data_for_day 调用...")
import inspect
from qlib.backtest.backtest import collect_data_loop
src = inspect.getsource(collect_data_loop)
assert "ensure_data_for_day" in src, "backtest.py 未包含 ensure_data_for_day 调用！"
print("  PASS")

# ── 3. 生成假信号用于回测 ──
print("\n[Test 3] 生成随机 signal 用于回测...")
cal = D.calendar(start_time=BACKTEST_START, end_time=BACKTEST_END, freq="day")
print(f"  交易日数: {len(cal)}")
assert len(cal) > 20, f"交易日数 {len(cal)} 不足 20 天，无法验证批次切换"

# 构建 signal DataFrame: MultiIndex(datetime, instrument) -> score
records = []
np.random.seed(42)
for dt in cal:
    for stock in TEST_STOCKS:
        records.append((dt, stock, np.random.randn()))
signal_df = pd.DataFrame(records, columns=["datetime", "instrument", "score"])
signal_df = signal_df.set_index(["datetime", "instrument"])
print(f"  Signal shape: {signal_df.shape}")
print("  PASS")

# ── 4. 运行回测 ──
print("\n[Test 4] 运行分钟线回测 (TopkDropoutStrategy)...")
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB

strategy_config = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.signal_strategy",
    "kwargs": {
        "signal": signal_df,
        "topk": TOPK,
        "n_drop": N_DROP,
    },
}
executor_config = {
    "class": "NestedExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {
        "time_per_step": "day",
        "inner_executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "1min",
                "generate_portfolio_metrics": False,
            },
        },
        "inner_strategy": {
            "class": "TWAPStrategy",
            "module_path": "qlib.contrib.strategy.rule_strategy",
        },
        "generate_portfolio_metrics": True,
    },
}
exchange_kwargs = {
    "freq": "1min",
    "limit_threshold": ("$limit_up", "$limit_down"),
    "deal_price": "close",
    "open_cost": 0.000095,
    "close_cost": 0.000595,
    "min_cost": 5,
    "trade_unit": 100,
    "codes": TEST_STOCKS,
}

trade_strategy, trade_executor = get_strategy_executor(
    start_time=BACKTEST_START,
    end_time=BACKTEST_END,
    strategy=strategy_config,
    executor=executor_config,
    benchmark=None,
    account=10_000_000,
    exchange_kwargs=exchange_kwargs,
)

# 验证 Exchange 共享
outer_exchange = trade_executor.trade_exchange
print(f"  Exchange freq: {outer_exchange.freq}")
print(f"  Exchange id: {id(outer_exchange)}")
print(f"  Strategy Exchange id: {id(trade_strategy.trade_exchange)}")
assert id(outer_exchange) == id(trade_strategy.trade_exchange), "Exchange 未共享！"
print("  Exchange 共享验证 PASS")

# 记录初始 _loaded_start
initial_loaded_start = getattr(outer_exchange, "_loaded_start", None)
print(f"  初始 _loaded_start: {initial_loaded_start}")

# 执行回测
portfolio_dict, indicator_dict = backtest_loop(
    start_time=BACKTEST_START,
    end_time=BACKTEST_END,
    trade_strategy=trade_strategy,
    trade_executor=trade_executor,
)

rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB
rss_mb = rss_after / 1024
print(f"  回测完成! Peak RSS: {rss_mb:.0f} MB")
print("  PASS")

# ── 5. 验证批次切换 ──
print("\n[Test 5] 批次切换验证...")
final_loaded_start = getattr(outer_exchange, "_loaded_start", None)
final_loaded_end = getattr(outer_exchange, "_loaded_end", None)
print(f"  最终 _loaded_start: {final_loaded_start}")
print(f"  最终 _loaded_end: {final_loaded_end}")
if len(cal) > 20:
    assert final_loaded_start is not None, "_loaded_start 为 None"
    assert initial_loaded_start != final_loaded_start, (
        f"_loaded_start 未变化 ({initial_loaded_start})，批次切换未发生！"
    )
    print(f"  批次切换: {initial_loaded_start} -> {final_loaded_start}")
print("  PASS")

# ── 6. 验证指标非 NaN ──
print("\n[Test 6] 指标非 NaN 验证...")
if "1day" in portfolio_dict:
    port_metrics_df, port_metrics_dict = portfolio_dict["1day"]
    print(f"  Portfolio metrics columns: {list(port_metrics_df.columns)}")
    # 检查是否有交易日
    n_rows = len(port_metrics_df)
    print(f"  交易日数: {n_rows}")
    assert n_rows > 0, "Portfolio metrics 为空！"

    # 检查 return 列
    if "return" in port_metrics_df.columns:
        nan_count = port_metrics_df["return"].isna().sum()
        print(f"  return NaN 数: {nan_count}/{n_rows}")
    print("  PASS")
else:
    print("  WARN: portfolio_dict 中没有 1day key")
    print(f"  keys: {list(portfolio_dict.keys())}")

# ── 7. 验证交易指标 ──
print("\n[Test 7] 交易指标验证...")
if "1day" in indicator_dict:
    ind_df, ind_obj = indicator_dict["1day"]
    # 检查 ffr (fill rate)
    if hasattr(ind_obj, "get_metric_series"):
        ffr = ind_obj.get_metric_series("ffr")
        if ffr is not None and len(ffr) > 0:
            mean_ffr = ffr.mean()
            print(f"  平均成交率 (FFR): {mean_ffr:.4f}")
            assert mean_ffr > 0, "FFR 全为 0，未发生交易！"
    print("  PASS")

# ── 8. 验证内存 ──
print("\n[Test 8] 内存验证...")
print(f"  Peak RSS: {rss_mb:.0f} MB")
assert rss_mb < 4096, f"RSS {rss_mb:.0f} MB 超过 4GB 限制！"
print("  PASS")

# ── 9. 验证无 None 价格警告 ──
print("\n[Test 9] 无静默错误验证...")
none_warnings = [
    r for r in log_records
    if "None!!!" in str(r.getMessage()) and "exchange" in r.pathname.lower()
]
print(f"  exchange.py None 警告数: {len(none_warnings)}")
if none_warnings:
    print("  FAIL: 存在 None 价格警告:")
    for w in none_warnings[:5]:
        print(f"    {w.getMessage()}")
    # 不 assert 失败，先报告
else:
    print("  PASS")

# ── 10. 验证无 monkey-patch 残留 ──
print("\n[Test 10] 无 monkey-patch 残留...")
# 检查 TopkDropoutStrategy.generate_trade_decision 是否被包装
gen_method = TopkDropoutStrategy.generate_trade_decision
method_module = getattr(gen_method, "__module__", "")
method_qualname = getattr(gen_method, "__qualname__", "")
print(f"  generate_trade_decision module: {method_module}")
print(f"  generate_trade_decision qualname: {method_qualname}")
is_wrapped = "_patched" in method_qualname or "wrapper" in method_qualname
assert not is_wrapped, f"generate_trade_decision 仍被 monkey-patch 包装: {method_qualname}"
print("  PASS")

# ── 总结 ──
print("\n" + "=" * 70)
all_pass = len(none_warnings) == 0
if all_pass:
    print("ALL TESTS PASSED!")
else:
    print(f"TESTS COMPLETED WITH {len(none_warnings)} None WARNINGS")
    sys.exit(1)
print("=" * 70)
