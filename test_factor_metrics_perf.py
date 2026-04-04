"""直接测试因子指标计算性能，不经过API服务器。"""
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

task_ids = [
    '2026-02-02_14-58-13-793902',
    '2026-02-22_14-51-37-035827',
    '2026-02-15_04-55-14-120473',
    '2025-12-23_05-59-43-369830',
]

print("=" * 60)
print("因子指标计算性能测试")
print("=" * 60)

# Step 1: import
t0 = time.time()
from rdagent.app.api_endpoints.sota_factors_api import _resolve_parquet_path
from rdagent.app.factor_metrics.engine import compute_all_factors_metrics
t1 = time.time()
print(f"\n[import] 耗时: {t1-t0:.1f}s\n")

for tid in task_ids:
    print(f"--- task: {tid} ---")
    try:
        t2 = time.time()
        pq = _resolve_parquet_path(tid)
        t3 = time.time()
        print(f"  定位parquet: {t3-t2:.1f}s  path=.../{pq.name}")

        t4 = time.time()
        results = compute_all_factors_metrics(pq)
        t5 = time.time()
        n_factors = len(set(r["factor_name"] for r in results))
        print(f"  计算指标:    {t5-t4:.1f}s  因子={n_factors} 记录={len(results)}")
        print(f"  总耗时:      {t5-t2:.1f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
