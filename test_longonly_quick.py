"""快速验证纯多头引擎 — 只算1个因子。"""
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("Step 1: importing...", flush=True)
t0 = time.time()
from rdagent.app.api_endpoints.sota_factors_api import _resolve_parquet_path
from rdagent.app.factor_metrics.engine import compute_all_factors_metrics
import pandas as pd
print(f"  import done: {time.time()-t0:.1f}s", flush=True)

tid = '2026-02-02_14-58-13-793902'
print(f"\nStep 2: resolve parquet for {tid}...", flush=True)
t1 = time.time()
pq = _resolve_parquet_path(tid)
print(f"  parquet: {pq}", flush=True)
print(f"  resolve done: {time.time()-t1:.1f}s", flush=True)

# 读取少量行获取展平后的列名
print("\nStep 3: peek factor columns...", flush=True)
df_peek = pd.read_parquet(pq, columns=None)
if isinstance(df_peek.columns, pd.MultiIndex):
    cols = [col[-1] if isinstance(col, tuple) else col for col in df_peek.columns]
else:
    cols = list(df_peek.columns)
del df_peek
print(f"  flattened columns ({len(cols)}): {cols[:5]}", flush=True)

# 只算第一个因子
target = cols[:1]
print(f"\nStep 4: compute metrics for {target}...", flush=True)
t2 = time.time()
results = compute_all_factors_metrics(pq, factor_filter=target)
t3 = time.time()
print(f"  compute done: {t3-t2:.1f}s", flush=True)
print(f"  records: {len(results)}", flush=True)

for r in results:
    print(f"\n  --- {r['factor_name']} / {r['eval_window']} ---")
    for k in ["ic_mean", "rank_ic_mean", "icir", "rank_icir",
              "ic_positive_ratio",
              "top_annual_return", "top_excess_annual_return",
              "top_sharpe", "top_max_drawdown",
              "top_excess_sharpe", "benchmark_annual_return",
              "group_return_monotonicity", "turnover",
              "coverage", "n_trading_days"]:
        v = r.get(k)
        if v is not None:
            fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
        else:
            fmt = "None"
        print(f"    {k}: {fmt}")

print(f"\nTotal: {t3-t0:.1f}s", flush=True)
