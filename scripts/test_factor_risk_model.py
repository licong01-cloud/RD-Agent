"""Test factor risk model covariance estimation vs sample covariance."""
import pandas as pd
import numpy as np
import time

# Load data
dpv = pd.read_hdf(
    "/mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data/daily_pv.h5",
    key="data",
)
sf = pd.read_parquet(
    "/mnt/f/Dev/RD-Agent-main/git_ignore_folder/factor_implementation_source_data/static_factors.parquet"
)

# Latest cross-section snapshot
latest_date = sf.index.get_level_values("datetime").max()
sf_snap = sf.loc[latest_date].copy()
print(f"Latest date: {latest_date}")
print(f"Stocks in snapshot: {len(sf_snap)}")

# Daily returns (last 250 days)
close = dpv["close"].unstack("instrument")
returns = close.pct_change().tail(250).dropna(axis=1, thresh=200)
print(f"Returns: {returns.shape} (days x stocks)")

# Common stocks
common = sf_snap.index.intersection(returns.columns)
print(f"Common stocks: {len(common)}")

returns_common = returns[common].dropna()
B = sf_snap.loc[common].fillna(0).values  # factor exposure matrix
print(f"B matrix (factor exposures): {B.shape}")

# === Factor Risk Model ===
t0 = time.time()

R = returns_common.values  # T x N
N = R.shape[1]
K = B.shape[1]  # 90 factors

# Cross-section regression to get factor returns
BtB_inv_Bt = np.linalg.pinv(B)  # K x N
factor_returns = R @ BtB_inv_Bt.T  # T x K

# Factor covariance (K x K)
factor_cov = np.cov(factor_returns, rowvar=False)  # 90x90

# Residuals (specific risk)
residuals = R - factor_returns @ B.T  # T x N
specific_var = np.var(residuals, axis=0)  # N-vector
D = np.diag(specific_var)  # N x N diagonal

# Full covariance: Cov = B * Sigma_f * B' + D
stock_cov = B @ factor_cov @ B.T + D  # N x N

t1 = time.time()
print(f"\n=== Factor Risk Model ===")
print(f"Factor cov: {factor_cov.shape}")
print(f"Stock cov:  {stock_cov.shape}")
print(f"Time: {t1-t0:.3f}s")

# === MVO for Top-50 ===
import cvxpy as cp

np.random.seed(42)
alpha = np.random.randn(N) * 0.01
top50_idx = np.argsort(alpha)[-50:]

alpha_50 = alpha[top50_idx]
cov_50 = stock_cov[np.ix_(top50_idx, top50_idx)]
# Ensure symmetry for cvxpy
cov_50 = (cov_50 + cov_50.T) / 2

w = cp.Variable(50)
lam = 0.5
prob = cp.Problem(
    cp.Maximize(alpha_50 @ w - lam * cp.quad_form(w, cov_50)),
    [cp.sum(w) == 1, w >= 0, w <= 0.05],
)
t2 = time.time()
prob.solve(solver=cp.SCS)
t3 = time.time()

print(f"\n=== MVO (Top-50, factor risk model cov) ===")
print(f"Status: {prob.status}")
print(f"Optimization time: {(t3-t2)*1000:.1f}ms")
print(f"Non-zero weights: {np.sum(w.value > 0.001)}")
print(f"Max weight: {w.value.max():.3f}")
print(f"Min non-zero: {w.value[w.value > 0.001].min():.3f}")

# === Summary ===
print(f"\n=== Total ===")
risk_ms = (t1 - t0) * 1000
opt_ms = (t3 - t2) * 1000
total_ms = risk_ms + opt_ms
print(f"Risk model build: {risk_ms:.0f}ms")
print(f"Optimization:     {opt_ms:.0f}ms")
print(f"Total per day:    {total_ms:.0f}ms")
print(f"Backtest 1000 days: {total_ms:.0f}s = {total_ms/60:.1f}min")

# Compare: if we update risk model monthly (not daily)
print(f"\n=== Practical: monthly risk model update ===")
print(f"Risk model: {risk_ms:.0f}ms once per month")
print(f"Daily cost: optimization only = {opt_ms:.0f}ms")
print(f"Backtest 1000 days: {opt_ms:.0f}s = {opt_ms/60:.1f}min")
