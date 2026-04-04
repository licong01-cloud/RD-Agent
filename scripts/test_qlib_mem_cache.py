"""Test if Qlib's D.features() with disk_cache=False still caches data in memory"""
import gc
import os
import sys
sys.path.insert(0, '/mnt/f/Dev/RD-Agent-main/qlib-main')
sys.path.insert(0, '/mnt/f/Dev/RD-Agent-main')

import qlib
from qlib.config import C

def get_rss_mb():
    with open(f'/proc/{os.getpid()}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0

# Init qlib
qlib.init(
    provider_uri={'day': '/home/lc999/data/qlib_bin', '1min': '/home/lc999/data/qlib_minute_bin'},
    region='cn',
    dataset_cache=None,
    expression_cache=None,
)

print(f"mem_cache_size_limit = {C.mem_cache_size_limit}")
print(f"mem_cache_limit_type = {C.mem_cache_limit_type}")
print(f"mem_cache_expire = {C.mem_cache_expire}")

from qlib.data import D
from qlib.data.cache import MemCache

# Find the MemCache instance
# It's stored in the expression/feature providers
from qlib.data.data import ExpressionD, FeatureD, DatasetD
print(f"\nExpressionD type: {type(ExpressionD)}")
print(f"DatasetD type: {type(DatasetD)}")

# Check if there's a mem_cache on the providers
for name, provider in [("ExpressionD", ExpressionD), ("FeatureD", FeatureD)]:
    if hasattr(provider, 'mem_cache'):
        mc = provider.mem_cache
        print(f"\n{name}.mem_cache: {type(mc)}")
        if hasattr(mc, 'f'):
            fc = mc.f
            print(f"  feature_cache type: {type(fc)}, len: {len(fc)}")

rss_before = get_rss_mb()
print(f"\nRSS before D.features: {rss_before:.0f} MB")

# Load 20 days of minute data for a small set of stocks
stocks = ['000001.SZ', '000002.SZ', '600000.SH']
fields = ['$close', '$open', '$high', '$low', '$volume', '$factor', '$change']

df1 = D.features(stocks, fields, start_time='2024-07-01', end_time='2024-07-26', freq='1min', disk_cache=False)
rss_after1 = get_rss_mb()
print(f"After 1st D.features (20 days, 3 stocks): {rss_after1:.0f} MB (+{rss_after1-rss_before:.0f}), shape={df1.shape}")

# Check mem_cache after first call
for name, provider in [("ExpressionD", ExpressionD), ("FeatureD", FeatureD)]:
    if hasattr(provider, 'mem_cache'):
        mc = provider.mem_cache
        if hasattr(mc, 'f'):
            print(f"  {name} feature_cache len after 1st call: {len(mc.f)}")

# Delete the DataFrame
del df1
gc.collect()
rss_after_del1 = get_rss_mb()
print(f"After del+gc: {rss_after_del1:.0f} MB (released: {rss_after1-rss_after_del1:.0f})")

# Load next 20 days
df2 = D.features(stocks, fields, start_time='2024-07-27', end_time='2024-08-23', freq='1min', disk_cache=False)
rss_after2 = get_rss_mb()
print(f"\nAfter 2nd D.features (next 20 days): {rss_after2:.0f} MB (+{rss_after2-rss_before:.0f}), shape={df2.shape}")

for name, provider in [("ExpressionD", ExpressionD), ("FeatureD", FeatureD)]:
    if hasattr(provider, 'mem_cache'):
        mc = provider.mem_cache
        if hasattr(mc, 'f'):
            print(f"  {name} feature_cache len after 2nd call: {len(mc.f)}")

del df2
gc.collect()
rss_after_del2 = get_rss_mb()
print(f"After del+gc: {rss_after_del2:.0f} MB (delta from baseline: +{rss_after_del2-rss_before:.0f})")

# Load 3rd batch
df3 = D.features(stocks, fields, start_time='2024-08-24', end_time='2024-09-20', freq='1min', disk_cache=False)
rss_after3 = get_rss_mb()
print(f"\nAfter 3rd D.features: {rss_after3:.0f} MB (+{rss_after3-rss_before:.0f})")

for name, provider in [("ExpressionD", ExpressionD), ("FeatureD", FeatureD)]:
    if hasattr(provider, 'mem_cache'):
        mc = provider.mem_cache
        if hasattr(mc, 'f'):
            fc = mc.f
            print(f"  {name} feature_cache len: {len(fc)}")
            # Estimate cache memory
            total_cache_bytes = 0
            for k, v in list(fc.items())[:3]:
                val, ts = v
                if hasattr(val, 'nbytes'):
                    total_cache_bytes += val.nbytes
                    print(f"    sample key: {str(k)[:80]}, val type: {type(val).__name__}, nbytes: {val.nbytes}")
                elif hasattr(val, '__len__'):
                    print(f"    sample key: {str(k)[:80]}, val type: {type(val).__name__}, len: {len(val)}")

del df3
gc.collect()
rss_final = get_rss_mb()
print(f"\nFinal RSS: {rss_final:.0f} MB (total leaked from baseline: +{rss_final-rss_before:.0f} MB)")
