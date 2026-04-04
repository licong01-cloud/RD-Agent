"""监控分钟线回测内存占用，验证内存修复效果。

模拟 _reload_exchange_for_day 批次切换，监控每批次前后内存变化。
"""
import gc
import os
import sys
import weakref
import tracemalloc

sys.path.insert(0, '/mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/experiment/factor_template')

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("[WARN] psutil not available, using tracemalloc only")

import qlib
from qlib.data import D
from qlib.backtest.high_performance_ds import NumpyQuote
import pandas as pd

def get_rss_mb():
    if HAVE_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return 0

print("Initializing qlib...")
qlib.init(
    provider_uri={'day': '/home/lc999/data/qlib_bin', '1min': '/home/lc999/data/qlib_minute_bin'},
    region='cn',
    dataset_cache=None,
    expression_cache=None,
)
print(f"qlib initialized. RSS: {get_rss_mb():.0f} MB")

fields = ["$close", "$open", "$high", "$low", "$volume", "$factor"]
# 用前50只股票模拟真实负载
from qlib.data import D as QD
stocks = QD.instruments(market='csi300')
if hasattr(stocks, '__iter__') and not isinstance(stocks, list):
    stocks = list(stocks)[:50]
else:
    stocks = stocks[:50]
print(f"Testing with {len(stocks)} stocks")

tracemalloc.start()

test_days = [
    '2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05',
]

print("\n{'Batch':>6} {'Day':>12} {'RSS_MB':>10} {'Delta_MB':>10} {'GC_ok':>6}")
print("-" * 55)

prev_rss = get_rss_mb()
old_quote = None
old_ref = None

for i, day in enumerate(test_days):
    day_ts = pd.Timestamp(day)
    # 跳过非交易日
    try:
        cal = D.calendar(start_time=day, end_time=day, freq='day')
        if len(cal) == 0:
            print(f"  {i+1:>5}  {day:>12}  (non-trading day, skip)")
            continue
    except Exception:
        pass

    day_end = day_ts + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # 清理旧实例
    gc_ok = 'N/A'
    if old_quote is not None:
        old_quote.data.clear()
        if old_ref is not None:
            del old_quote
            gc.collect()
            gc_ok = 'YES' if old_ref() is None else 'NO'
        else:
            del old_quote
            gc.collect()

    # 加载新批次
    df = D.features(stocks, fields,
                    start_time=day_ts, end_time=day_end,
                    freq='1min', disk_cache=False)
    df.columns = fields
    new_quote = NumpyQuote(df, '1min')
    del df
    gc.collect()

    old_ref = weakref.ref(new_quote)
    old_quote = new_quote

    rss = get_rss_mb()
    delta = rss - prev_rss
    prev_rss = rss
    print(f"  {i+1:>5}  {day:>12}  {rss:>10.0f}  {delta:>+10.0f}  {gc_ok:>6}")

# 清理最后一个
if old_quote is not None:
    old_quote.data.clear()
    del old_quote
    gc.collect()

final_rss = get_rss_mb()
print("-" * 55)
print(f"Final RSS: {final_rss:.0f} MB")

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"TraceMalloc peak: {peak/1024/1024:.0f} MB")
print("\nMemory check complete.")
if HAVE_PSUTIL and final_rss < 4000:
    print(f"OK: RSS {final_rss:.0f} MB is within acceptable range (< 4GB for 50 stocks)")
elif HAVE_PSUTIL:
    print(f"ALERT: RSS {final_rss:.0f} MB exceeds 4GB threshold!")
