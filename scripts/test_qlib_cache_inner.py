"""Test Qlib D.features() memory cache behavior - run from /tmp to avoid qlib source import"""
import gc
import os

import qlib
from qlib.config import C

def get_rss_mb():
    with open(f'/proc/{os.getpid()}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0

qlib.init(
    provider_uri={'day': '/home/lc999/data/qlib_bin', '1min': '/home/lc999/data/qlib_minute_bin'},
    region='cn',
    dataset_cache=None,
    expression_cache=None,
)
print(f"mem_cache_size_limit = {C.mem_cache_size_limit}")

from qlib.data import D
from qlib.data.data import ExpressionD

stocks = ['000001.SZ', '000002.SZ', '600000.SH']
fields = ['$close', '$open', '$high', '$low', '$volume', '$factor', '$change']

rss0 = get_rss_mb()
print(f"Baseline RSS: {rss0:.0f} MB")

for i, (start, end) in enumerate([
    ('2024-07-01', '2024-07-26'),
    ('2024-07-27', '2024-08-23'),
    ('2024-08-24', '2024-09-20'),
    ('2024-09-21', '2024-10-25'),
    ('2024-10-26', '2024-11-22'),
], 1):
    df = D.features(stocks, fields, start_time=start, end_time=end, freq='1min', disk_cache=False)
    rss_load = get_rss_mb()
    cache_len = len(ExpressionD.mem_cache.f) if hasattr(ExpressionD, 'mem_cache') else -1
    print(f"Batch {i} loaded: RSS={rss_load:.0f} MB (+{rss_load-rss0:.0f}), shape={df.shape}, cache_len={cache_len}")
    
    del df
    gc.collect()
    rss_del = get_rss_mb()
    print(f"Batch {i} deleted: RSS={rss_del:.0f} MB (cumulative from baseline: +{rss_del-rss0:.0f})")

# Now try clearing the mem_cache
if hasattr(ExpressionD, 'mem_cache'):
    ExpressionD.mem_cache.clear()
    gc.collect()
    rss_cleared = get_rss_mb()
    print(f"\nAfter ExpressionD.mem_cache.clear() + gc: RSS={rss_cleared:.0f} MB (from baseline: +{rss_cleared-rss0:.0f})")
