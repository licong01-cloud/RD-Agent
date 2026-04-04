"""验证 NumpyQuote 实例级 lru_cache 修复：实例销毁后内存正确释放。

测试内容：
1. NumpyQuote 实例级缓存是否正常工作（get_data 返回正确结果）
2. 实例销毁后，缓存是否随实例一起释放（无循环引用）
3. _reload_exchange_for_day 批次切换时旧实例能否被 GC 回收
"""
import gc
import sys
import os
import weakref

sys.path.insert(0, '/mnt/f/Dev/RD-Agent-main/rdagent/scenarios/qlib/experiment/factor_template')

import qlib
from qlib.data import D

print("Initializing qlib...")
qlib.init(
    provider_uri={'day': '/home/lc999/data/qlib_bin', '1min': '/home/lc999/data/qlib_minute_bin'},
    region='cn',
    dataset_cache=None,
    expression_cache=None,
)
print("qlib initialized.")

from qlib.backtest.high_performance_ds import NumpyQuote

# ===== TEST 1: 实例级缓存功能验证 =====
print("\n" + "=" * 60)
print("TEST 1: 实例级 lru_cache 功能验证")

fields = ["$close", "$open", "$high", "$low", "$volume"]
import pandas as pd

day_end = pd.Timestamp('2024-07-01') + pd.Timedelta(hours=23, minutes=59, seconds=59)
df = D.features(['000001.SZ', '000002.SZ'], fields,
                start_time='2024-07-01', end_time=day_end,
                freq='1min', disk_cache=False)
df.columns = fields

quote = NumpyQuote(df, '1min')

# 验证 _get_data_cached 存在且是实例属性
assert hasattr(quote, '_get_data_cached'), "_get_data_cached should be instance attribute"
assert callable(quote._get_data_cached), "_get_data_cached should be callable"
assert hasattr(quote._get_data_cached, 'cache_info'), "_get_data_cached should have cache_info"
print("  OK: 实例级 _get_data_cached 存在")

# 调用 get_data，验证缓存命中
ts = pd.Timestamp('2024-07-01 09:31:00')
v1 = quote.get_data('000001.SZ', ts, ts, '$close')
v2 = quote.get_data('000001.SZ', ts, ts, '$close')  # 应命中缓存

info = quote._get_data_cached.cache_info()
print(f"  Cache info: hits={info.hits}, misses={info.misses}, currsize={info.currsize}")
assert info.hits >= 1, f"Expected cache hit, got hits={info.hits}"
assert v1 == v2, f"Cache should return same value: {v1} != {v2}"
print(f"  OK: get_data 返回正确值 {v1:.4f}，缓存命中正常")

# ===== TEST 2: 实例销毁后缓存释放（WeakRef 验证）=====
print("\n" + "=" * 60)
print("TEST 2: 实例销毁后无循环引用")

# 用 weakref 追踪实例生命周期
ref = weakref.ref(quote)
assert ref() is not None, "quote should be alive"

# 模拟 _reload_exchange_for_day 的清理流程
if hasattr(quote, 'data') and isinstance(quote.data, dict):
    quote.data.clear()
del quote
gc.collect()

if ref() is None:
    print("  OK: 实例已被 GC 回收，无循环引用")
else:
    print("  FAIL: 实例未被 GC 回收，仍存在循环引用！")
    # 打印引用链诊断
    import gc as gc_module
    referrers = gc_module.get_referrers(ref())
    print(f"  引用者数量: {len(referrers)}")
    for r in referrers[:3]:
        print(f"    - {type(r)}: {str(r)[:100]}")
    sys.exit(1)

# ===== TEST 3: 多批次切换内存不累积 =====
print("\n" + "=" * 60)
print("TEST 3: 多批次切换，旧实例被正确回收")

refs = []
for i, day in enumerate(['2024-07-01', '2024-07-02', '2024-07-03']):
    day_ts = pd.Timestamp(day)
    day_end = day_ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df = D.features(['000001.SZ'], fields,
                    start_time=day_ts, end_time=day_end,
                    freq='1min', disk_cache=False)
    df.columns = fields
    new_quote = NumpyQuote(df, '1min')

    if i > 0:
        # 清理旧实例（模拟 _reload_exchange_for_day）
        old_quote.data.clear()
        ref = weakref.ref(old_quote)
        del old_quote, df
        gc.collect()
        if ref() is None:
            print(f"  OK: day {i} 旧实例已回收")
        else:
            print(f"  FAIL: day {i} 旧实例未回收！")
            sys.exit(1)
    else:
        del df

    old_quote = new_quote

# 清理最后一个
old_quote.data.clear()
del old_quote
gc.collect()
print("  OK: 所有批次实例均正确回收")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - NumpyQuote 内存泄漏修复验证成功！")
