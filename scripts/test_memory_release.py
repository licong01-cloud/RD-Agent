"""Test actual memory release behavior with pandas DataFrame + NumpyQuote pattern"""
import gc
import os
import sys
sys.path.insert(0, '/mnt/f/Dev/RD-Agent-main/qlib-main')

import numpy as np
import pandas as pd

def get_rss_mb():
    with open(f'/proc/{os.getpid()}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0

print(f"numpy: {np.__version__}, pandas: {pd.__version__}")

# Simulate what _reload_exchange_for_day does
# Create a large DataFrame similar to D.features() output for 20 days of minute data
# ~5000 stocks * 240 minutes * 20 days * 7 fields

def create_fake_quote_df(n_stocks=100, n_minutes=240, n_days=20, n_fields=7):
    """Create a DataFrame mimicking D.features() output for minute data"""
    n_rows = n_stocks * n_minutes * n_days
    idx = pd.MultiIndex.from_arrays([
        np.repeat(np.arange(n_stocks), n_minutes * n_days).astype(str),
        np.tile(pd.date_range('2024-01-01', periods=n_minutes * n_days, freq='1min'), n_stocks)
    ], names=['instrument', 'datetime'])
    data = np.random.randn(n_rows, n_fields).astype(np.float32)
    df = pd.DataFrame(data, index=idx, columns=[f'field_{i}' for i in range(n_fields)])
    return df

print(f"\nRSS baseline: {get_rss_mb():.0f} MB")

# Round 1: Create and release
print("\n=== Round 1: Create large DataFrame ===")
df1 = create_fake_quote_df()
gc.collect()
rss1 = get_rss_mb()
print(f"  After create: {rss1:.0f} MB, shape={df1.shape}, memory={df1.memory_usage(deep=True).sum()/1e6:.1f} MB")

# Delete and collect
del df1
gc.collect()
rss1_after = get_rss_mb()
print(f"  After del+gc: {rss1_after:.0f} MB (released: {rss1 - rss1_after:.0f} MB)")

# Round 2: Simulate the patch pattern - create, replace, delete old
print("\n=== Round 2: Simulate _reload_exchange_for_day pattern ===")
rss_base = get_rss_mb()
print(f"  Baseline: {rss_base:.0f} MB")

# Simulate 5 reload cycles (like 5 batches of 20 days)
for i in range(5):
    new_df = create_fake_quote_df()
    if i == 0:
        current_df = new_df
    else:
        old_df = current_df
        current_df = new_df
        del old_df, new_df
        gc.collect()
    rss_now = get_rss_mb()
    print(f"  After cycle {i+1}: {rss_now:.0f} MB (delta from base: +{rss_now - rss_base:.0f} MB)")

del current_df
gc.collect()
rss_final = get_rss_mb()
print(f"  After final cleanup: {rss_final:.0f} MB (delta from base: +{rss_final - rss_base:.0f} MB)")

# Round 3: Check if DataFrame.columns assignment creates hidden references
print("\n=== Round 3: Test df.columns reassignment (patch does this) ===")
rss_base3 = get_rss_mb()
df3 = create_fake_quote_df()
df3.columns = [f'$field_{i}' for i in range(7)]  # This is what the patch does
rss3 = get_rss_mb()
print(f"  After create+columns: {rss3:.0f} MB")
del df3
gc.collect()
rss3_after = get_rss_mb()
print(f"  After del: {rss3_after:.0f} MB (leaked: {rss3_after - rss_base3:.0f} MB)")
