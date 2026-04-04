"""Test if lru_cache.cache_clear() works correctly on NumpyQuote.get_data"""
import sys
sys.path.insert(0, '/mnt/f/Dev/RD-Agent-main/qlib-main')

from functools import lru_cache

# Simulate the NumpyQuote pattern
class FakeQuote:
    def __init__(self, data_size_mb):
        import numpy as np
        # Allocate data_size_mb of memory
        self.data = np.zeros(int(data_size_mb * 1024 * 1024 / 8))  # float64 = 8 bytes
        print(f"  Created FakeQuote with {data_size_mb}MB data, id={id(self)}")

    @lru_cache(maxsize=512)
    def get_data(self, key):
        return self.data[0]

# Test 1: Does cache_clear release the cache entries?
print("=== Test 1: cache_clear on instance method ===")
q1 = FakeQuote(100)
q1.get_data("a")
q1.get_data("b")
print(f"  Cache info after 2 calls: {q1.get_data.cache_info()}")
q1.get_data.cache_clear()
print(f"  Cache info after clear: {q1.get_data.cache_info()}")

# Test 2: Does del + gc.collect release the object after cache_clear?
print("\n=== Test 2: Object release after cache_clear + del ===")
import gc
import os

def get_rss_mb():
    with open(f'/proc/{os.getpid()}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return 0

rss_before = get_rss_mb()
print(f"  RSS before: {rss_before:.0f} MB")

q2 = FakeQuote(500)
q2.get_data("x")
rss_after_create = get_rss_mb()
print(f"  RSS after create+call: {rss_after_create:.0f} MB (+{rss_after_create-rss_before:.0f})")

# This is what the patch does:
q2.get_data.cache_clear()
old_q2 = q2
q2 = FakeQuote(10)  # Replace with small object (simulates new NumpyQuote)
del old_q2
gc.collect()
rss_after_del = get_rss_mb()
print(f"  RSS after cache_clear+del+gc: {rss_after_del:.0f} MB (delta from before: +{rss_after_del-rss_before:.0f})")

if rss_after_del - rss_before < 100:
    print("  PASS: Memory was released")
else:
    print("  FAIL: Memory was NOT released - lru_cache is holding reference")

# Test 3: What if we DON'T call cache_clear before del?
print("\n=== Test 3: del WITHOUT cache_clear ===")
rss_before3 = get_rss_mb()
q3 = FakeQuote(500)
q3.get_data("y")
rss_after_create3 = get_rss_mb()
print(f"  RSS after create+call: {rss_after_create3:.0f} MB (+{rss_after_create3-rss_before3:.0f})")

old_q3 = q3
q3 = FakeQuote(10)
del old_q3
gc.collect()
rss_after_del3 = get_rss_mb()
print(f"  RSS after del+gc (no cache_clear): {rss_after_del3:.0f} MB (delta: +{rss_after_del3-rss_before3:.0f})")

if rss_after_del3 - rss_before3 < 100:
    print("  PASS: Memory was released even without cache_clear")
else:
    print("  FAIL: Memory NOT released - lru_cache prevents GC")
