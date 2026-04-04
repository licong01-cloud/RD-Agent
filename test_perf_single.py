"""单task指标计算性能测试（无缓冲输出）"""
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("Step 1: importing...", flush=True)
t0 = time.time()
from rdagent.app.api_endpoints.sota_factors_api import _resolve_parquet_path
from rdagent.app.factor_metrics.engine import compute_all_factors_metrics
print(f"  import done: {time.time()-t0:.1f}s", flush=True)

tid = '2026-02-02_14-58-13-793902'
print(f"\nStep 2: resolve parquet for {tid}...", flush=True)
t1 = time.time()
pq = _resolve_parquet_path(tid)
print(f"  resolve done: {time.time()-t1:.1f}s  -> {pq}", flush=True)

print(f"\nStep 3: compute metrics...", flush=True)
t2 = time.time()
results = compute_all_factors_metrics(pq)
t3 = time.time()
n_factors = len(set(r["factor_name"] for r in results))
print(f"  compute done: {t3-t2:.1f}s", flush=True)
print(f"  factors={n_factors} records={len(results)}", flush=True)
print(f"\nTotal: {t3-t0:.1f}s", flush=True)
