"""Single-factor metrics calculation engine (vectorized, pre-unstacked).

Computes 17 metrics per factor from combined_factors_df.parquet + qlib_bin close prices.
Architecture: unstack all data ONCE into (dates × instruments) matrices, then compute
all metrics via pure numpy/pandas matrix ops — zero per-factor unstack overhead.
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from rdagent.app.factor_metrics.qlib_data_reader import read_close_prices

logger = logging.getLogger(__name__)

DEFAULT_RETURN_HORIZON = 2
N_GROUPS = 5
ANNUALIZE_FACTOR = 252

# 多持有期：A股T+1制度下，shift(-N)/shift(-1)-1 表示持有 N-1 天
# 1d=shift(-2)/shift(-1)-1, 5d=shift(-6)/shift(-1)-1, etc.
HOLDING_PERIODS = {
    "1d": 2,   # T+1买 T+2卖，持有1天
    "5d": 6,   # T+1买 T+6卖，持有5天
    "10d": 11, # T+1买 T+11卖，持有10天
    "20d": 21, # T+1买 T+21卖，持有20天
}


# ================================================================
# Matrix-level vectorized computations (operate on pre-unstacked arrays)
# ================================================================

def _pearson_ic_from_matrices(f_arr: np.ndarray, r_arr: np.ndarray, min_obs: int = 6) -> np.ndarray:
    """Vectorized Pearson correlation per row between two (n_dates, n_instruments) arrays.

    Returns 1-D array of length n_dates with IC values (NaN where insufficient data).
    """
    valid = ~(np.isnan(f_arr) | np.isnan(r_arr))
    n = valid.sum(axis=1).astype(float)

    fm = np.where(valid, f_arr, 0.0)
    rm = np.where(valid, r_arr, 0.0)

    sf = fm.sum(axis=1)
    sr = rm.sum(axis=1)
    sff = (fm ** 2).sum(axis=1)
    srr = (rm ** 2).sum(axis=1)
    sfr = (fm * rm).sum(axis=1)

    num = n * sfr - sf * sr
    den = np.sqrt(np.maximum((n * sff - sf**2) * (n * srr - sr**2), 1e-30))

    return np.where(n >= min_obs, num / den, np.nan)


def _rank_matrix(arr: np.ndarray) -> np.ndarray:
    """Row-wise ranking using pandas (C-level, handles NaN).

    Input/output: (n_dates, n_instruments) float arrays.
    """
    return pd.DataFrame(arr).rank(axis=1).values


def _robust_zscore_matrix(arr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Row-wise robust z-score: (x - median) / (1.4826 * MAD), clipped.

    Equivalent to qlib's RobustZScoreNorm. Handles extreme values that
    distort Pearson IC. Returns same shape, NaN preserved.
    """
    med = np.nanmedian(arr, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(arr - med), axis=1, keepdims=True)
    scale = np.where(mad > 1e-12, 1.4826 * mad, 1.0)
    z = (arr - med) / scale
    return np.clip(z, -clip, clip)


def _cs_zscore_matrix(arr: np.ndarray) -> np.ndarray:
    """Row-wise cross-sectional z-score: (x - mean) / std.

    Equivalent to qlib's CSZScoreNorm. Removes market-level mean from
    returns so IC measures pure cross-sectional predictive power.
    """
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (arr - mu) / sd


def _group_returns_from_matrices(
    f_arr: np.ndarray, r_arr: np.ndarray, n_groups: int = N_GROUPS
) -> np.ndarray:
    """Vectorized group returns from pre-unstacked matrices.

    Returns (n_dates, n_groups) array of daily group mean returns. NaN for invalid rows.
    """
    valid = ~(np.isnan(f_arr) | np.isnan(r_arr))
    n_valid = valid.sum(axis=1)
    n_dates = f_arr.shape[0]

    # Rank factor values per row (1-based)
    f_ranked = pd.DataFrame(np.where(valid, f_arr, np.nan)).rank(axis=1, method="first").values
    # Convert to 0-based group: floor((rank-1) * n_groups / n_valid), capped
    nv = n_valid[:, None].astype(float)
    nv = np.where(nv > 0, nv, 1.0)
    groups = np.floor((f_ranked - 1) * n_groups / nv).astype(float)
    groups = np.clip(groups, 0, n_groups - 1)
    groups = np.where(valid, groups, np.nan)

    result = np.full((n_dates, n_groups), np.nan)
    for g in range(n_groups):
        mask = (groups == g) & valid
        counts = mask.sum(axis=1).astype(float)
        sums = np.where(mask, r_arr, 0.0).sum(axis=1)
        result[:, g] = np.where(counts > 0, sums / counts, np.nan)

    # Invalidate rows with < n_groups valid or any missing group
    bad = (n_valid < n_groups) | np.any(np.isnan(result), axis=1)
    result[bad] = np.nan
    return result


def _long_only_from_group_arr(grp: np.ndarray, dates_mask: np.ndarray) -> dict[str, float]:
    """Compute long-only (top group) metrics from (n_dates, n_groups) array.

    A股市场只允许做多，所有收益指标基于top组（因子值最高的20%股票）。
    超额收益 = top组收益 - 全市场平均收益(benchmark)。
    """
    g = grp[dates_mask]
    valid_rows = ~np.any(np.isnan(g), axis=1)
    g = g[valid_rows]
    if len(g) == 0:
        return {}

    top = g[:, -1]                    # top quintile daily returns
    benchmark = g.mean(axis=1)        # market avg daily returns
    excess = top - benchmark          # excess daily returns

    top_mean = top.mean()
    top_std = top.std()
    excess_mean = excess.mean()
    excess_std = excess.std()

    # Top group cumulative & drawdown
    top_cum = np.cumprod(1 + top)
    top_peak = np.maximum.accumulate(top_cum)
    top_dd = (top_cum - top_peak) / np.where(top_peak > 0, top_peak, 1.0)

    return {
        "top_annual_return": float(top_mean * ANNUALIZE_FACTOR),
        "top_excess_annual_return": float(excess_mean * ANNUALIZE_FACTOR),
        "top_sharpe": float(top_mean / top_std * np.sqrt(ANNUALIZE_FACTOR)) if top_std > 0 else 0.0,
        "top_max_drawdown": float(top_dd.min()),
        "top_excess_sharpe": float(excess_mean / excess_std * np.sqrt(ANNUALIZE_FACTOR)) if excess_std > 0 else 0.0,
        "benchmark_annual_return": float(benchmark.mean() * ANNUALIZE_FACTOR),
    }


def _monotonicity_from_group_arr(grp: np.ndarray, dates_mask: np.ndarray) -> float:
    """Group return monotonicity from (n_dates, n_groups) array."""
    g = grp[dates_mask]
    valid_rows = ~np.any(np.isnan(g), axis=1)
    g = g[valid_rows]
    if len(g) == 0:
        return 0.0
    means = g.mean(axis=0)
    n = len(means)
    if n < 2:
        return 0.0
    ideal = np.arange(n, dtype=float)
    actual = np.argsort(np.argsort(means)).astype(float)
    c = np.corrcoef(ideal, actual)[0, 1]
    return float(c) if not np.isnan(c) else 0.0


def _turnover_from_matrix(f_arr: np.ndarray) -> float:
    """Vectorized rank turnover from pre-unstacked (n_dates, n_instruments) array."""
    ranked = pd.DataFrame(f_arr).rank(axis=1, pct=True).values
    prev = ranked[:-1]
    curr = ranked[1:]

    valid = ~(np.isnan(prev) | np.isnan(curr))
    n = valid.sum(axis=1).astype(float)

    pm = np.where(valid, prev, 0.0)
    cm = np.where(valid, curr, 0.0)

    sp = pm.sum(axis=1)
    sc = cm.sum(axis=1)
    spp = (pm ** 2).sum(axis=1)
    scc = (cm ** 2).sum(axis=1)
    spc = (pm * cm).sum(axis=1)

    num = n * spc - sp * sc
    den = np.sqrt(np.maximum((n * spp - sp**2) * (n * scc - sc**2), 1e-30))

    corrs = np.where(n >= 6, num / den, np.nan)
    t = 1.0 - corrs
    t = t[~np.isnan(t)]
    return float(t.mean()) if len(t) > 0 else 0.0


def _ic_decay_half_life(
    f_arr: np.ndarray, close_mat: np.ndarray, horizons: list[int] | None = None
) -> float:
    """IC decay half-life from pre-unstacked matrices. Only 3 horizons for speed."""
    if horizons is None:
        horizons = [1, 5, 20]

    ics = []
    for h in horizons:
        fwd = close_mat.shift(-h) / close_mat.shift(-(h - 1)) - 1 if isinstance(close_mat, pd.DataFrame) else None
        if fwd is None:
            return float("nan")
        r_arr = fwd.values
        f_ranked = _rank_matrix(f_arr)
        r_ranked = _rank_matrix(r_arr)
        ic_vals = _pearson_ic_from_matrices(f_ranked, r_ranked)
        ic_vals = ic_vals[~np.isnan(ic_vals)]
        ics.append(abs(ic_vals.mean()) if len(ic_vals) > 0 else 0.0)

    ics_arr = np.array(ics)
    if ics_arr[0] <= 0 or np.all(ics_arr <= 0):
        return float("nan")

    valid = ics_arr > 0
    if valid.sum() < 2:
        return float("nan")

    log_ic = np.log(ics_arr[valid])
    h_valid = np.array(horizons)[valid].astype(float)
    n = len(h_valid)
    sx, sy = h_valid.sum(), log_ic.sum()
    sxx = (h_valid ** 2).sum()
    sxy = (h_valid * log_ic).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return float("nan")

    lam = -(n * sxy - sx * sy) / denom
    if lam <= 0:
        return float("nan")
    return float(np.log(2) / lam)


# ================================================================
# Helpers: window slicing, instrument alignment
# ================================================================

EVAL_WINDOWS = {
    "full": None,
    "out_sample": ("2024-07-01", None),
    "recent_6m": (-126, None),
    "recent_3m": (-63, None),
    "recent_1m": (-21, None),
}

REQUIRED_DAYS = {
    "full": 0,
    "out_sample": 0,
    "recent_6m": 126,
    "recent_3m": 63,
    "recent_1m": 21,
}


def _resolve_date_mask(dates: pd.DatetimeIndex, window_spec, data_start: str, data_end: str):
    """Return boolean mask over dates array + (w_start, w_end) strings."""
    if window_spec is None:
        return np.ones(len(dates), dtype=bool), data_start, data_end

    offset_or_date, end_override = window_spec
    w_end = data_end if end_override is None else end_override

    if isinstance(offset_or_date, int):
        if len(dates) < abs(offset_or_date):
            w_start = str(dates[0].date())
        else:
            w_start = str(dates[offset_or_date].date())
    else:
        w_start = offset_or_date

    mask = (dates >= pd.Timestamp(w_start)) & (dates <= pd.Timestamp(w_end))
    return mask, w_start, w_end


def _align_instrument_names(close_df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """Align instrument naming between qlib_bin (000001.SZ) and factors (SZ000001)."""
    close_insts = set(close_df.index.get_level_values("instrument").unique())
    factor_insts = set(factors_df.index.get_level_values("instrument").unique())

    if len(close_insts & factor_insts) > len(factor_insts) * 0.5:
        return close_df

    mapping = {}
    for inst in close_insts:
        parts = inst.split(".")
        if len(parts) == 2:
            mapping[inst] = parts[1] + parts[0]

    if len(set(mapping.values()) & factor_insts) > len(factor_insts) * 0.5:
        idx = close_df.index
        new_insts = [mapping.get(i, i) for i in idx.get_level_values("instrument")]
        close_df.index = pd.MultiIndex.from_arrays(
            [idx.get_level_values("datetime"), new_insts],
            names=["datetime", "instrument"],
        )
        return close_df

    logger.warning("Instrument name alignment: no reliable mapping found, using as-is")
    return close_df


# ================================================================
# Standalone per-factor computation (extracted from closure for reuse)
# ================================================================

def _compute_factor_metrics_impl(
    fname: str,
    f_arr_full: np.ndarray,
    dates: pd.DatetimeIndex,
    fwd_arr: np.ndarray,
    fwd_arrs: dict[str, np.ndarray],
    close_unstacked: pd.DataFrame,
    data_start: str,
    data_end: str,
    calc_batch_id: str,
) -> tuple[list, list]:
    """Compute all eval-window metrics for a single factor (thread-safe, read-only).

    Returns (factor_results, factor_reports).
    """
    grp_full = _group_returns_from_matrices(f_arr_full, fwd_arr)

    factor_results = []
    factor_reports = []

    for window_name, window_spec in EVAL_WINDOWS.items():
        try:
            mask, w_start, w_end = _resolve_date_mask(dates, window_spec, data_start, data_end)
            if mask.sum() == 0:
                factor_reports.append({
                    "factor_name": fname, "eval_window": window_name, "status": "skipped",
                    "error_message": f"数据天数不足: 实际0天, 需要{REQUIRED_DAYS.get(window_name, 0)}天",
                    "n_trading_days": 0, "required_days": REQUIRED_DAYS.get(window_name, 0),
                    "data_start": None, "data_end": None,
                    "data_source": "parquet", "calc_engine": "rdagent",
                    "calculated_at": datetime.now(timezone.utc).isoformat(),
                })
                continue

            req_days = REQUIRED_DAYS.get(window_name, 0)
            actual_days = int(mask.sum())
            if req_days > 0 and actual_days < req_days:
                factor_reports.append({
                    "factor_name": fname, "eval_window": window_name, "status": "skipped",
                    "error_message": f"数据天数不足: 实际{actual_days}天, 需要{req_days}天",
                    "n_trading_days": actual_days, "required_days": req_days,
                    "data_start": w_start, "data_end": w_end,
                    "data_source": "parquet", "calc_engine": "rdagent",
                    "calculated_at": datetime.now(timezone.utc).isoformat(),
                })
                continue

            f_w = f_arr_full[mask]
            r_w = fwd_arr[mask]

            f_w_z = _robust_zscore_matrix(f_w)
            r_w_z = _robust_zscore_matrix(r_w)
            ic_p = _pearson_ic_from_matrices(f_w_z, r_w_z)

            r_w_csz = _cs_zscore_matrix(r_w)
            ic_p_csz = _pearson_ic_from_matrices(f_w_z, r_w_csz)

            ic_s = _pearson_ic_from_matrices(_rank_matrix(f_w), _rank_matrix(r_w))

            ic_p_clean = ic_p[~np.isnan(ic_p)]
            ic_s_clean = ic_s[~np.isnan(ic_s)]
            ic_csz_clean = ic_p_csz[~np.isnan(ic_p_csz)]

            ic_mean = float(ic_p_clean.mean()) if len(ic_p_clean) > 0 else None
            ic_std = float(ic_p_clean.std()) if len(ic_p_clean) > 0 else None
            ric_mean = float(ic_s_clean.mean()) if len(ic_s_clean) > 0 else None
            ric_std = float(ic_s_clean.std()) if len(ic_s_clean) > 0 else None
            ic_csz_mean = float(ic_csz_clean.mean()) if len(ic_csz_clean) > 0 else None

            result = {
                "factor_name": fname, "eval_window": window_name,
                "calc_batch_id": calc_batch_id,
                "data_start": w_start, "data_end": w_end,
                "return_horizon": "1d", "universe": "all",
                "calculated_at": datetime.now(timezone.utc).isoformat(),
                "ic_mean": ic_mean, "ic_std": ic_std, "ic_csz_mean": ic_csz_mean,
                "rank_ic_mean": ric_mean, "rank_ic_std": ric_std,
                "icir": float(ic_mean / ic_std) if ic_mean and ic_std and ic_std > 0 else None,
                "rank_icir": float(ric_mean / ric_std) if ric_mean and ric_std and ric_std > 0 else None,
                "ic_positive_ratio": float((ic_p_clean > 0).mean()) if len(ic_p_clean) > 0 else None,
            }

            lo = _long_only_from_group_arr(grp_full, mask)
            result.update(lo)
            result["group_return_monotonicity"] = _monotonicity_from_group_arr(grp_full, mask)

            if window_name == "full":
                result["turnover"] = _turnover_from_matrix(f_arr_full)
                result["ic_decay_half_life"] = _ic_decay_half_life(f_arr_full, close_unstacked)
            else:
                result["turnover"] = None
                result["ic_decay_half_life"] = None

            if window_name == "full":
                f_ranked_full = _rank_matrix(f_w)
                for pname, p_arr in fwd_arrs.items():
                    r_p = p_arr[mask]
                    r_ranked_p = _rank_matrix(r_p)
                    ic_mp = _pearson_ic_from_matrices(f_ranked_full, r_ranked_p)
                    ic_mp_clean = ic_mp[~np.isnan(ic_mp)]
                    result[f"rank_ic_{pname}"] = float(ic_mp_clean.mean()) if len(ic_mp_clean) > 0 else None
            else:
                for pname in HOLDING_PERIODS:
                    result[f"rank_ic_{pname}"] = None

            n_total = f_w.size
            n_valid = np.count_nonzero(~np.isnan(f_w))
            result["coverage"] = float(n_valid / n_total) if n_total > 0 else 0.0
            result["n_trading_days"] = int(mask.sum())

            factor_results.append(result)
            factor_reports.append({
                "factor_name": fname, "eval_window": window_name, "status": "ok",
                "error_message": None, "n_trading_days": int(mask.sum()),
                "required_days": REQUIRED_DAYS.get(window_name, 0),
                "data_start": w_start, "data_end": w_end,
                "data_source": "parquet", "calc_engine": "rdagent",
                "calculated_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.warning(f"Failed {window_name} for {fname}: {e}")
            _n_days = int(mask.sum()) if 'mask' in locals() and mask is not None else None
            _w_start = w_start if 'w_start' in locals() else None
            _w_end = w_end if 'w_end' in locals() else None
            factor_reports.append({
                "factor_name": fname, "eval_window": window_name, "status": "error",
                "error_message": str(e), "n_trading_days": _n_days,
                "required_days": REQUIRED_DAYS.get(window_name, 0),
                "data_start": _w_start, "data_end": _w_end,
                "data_source": "parquet", "calc_engine": "rdagent",
                "calculated_at": datetime.now(timezone.utc).isoformat(),
            })

    return factor_results, factor_reports


# ================================================================
# Streaming API: prepare once, compute per-factor independently
# ================================================================

def prepare_shared_context(
    qlib_bin_path: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    instrument_hint: Optional[set[str]] = None,
) -> dict[str, Any]:
    """One-time shared data preparation: load close prices + compute forward returns.

    Returns context dict for use with compute_single_factor_metrics().
    """
    import time
    t0 = time.time()
    calc_batch_id = str(uuid.uuid4())

    logger.info("prepare_shared_context: loading close prices...")
    close_df = read_close_prices(
        qlib_bin_path, start_date=start_date, end_date=end_date,
        instrument_filter=instrument_hint,
    )
    logger.info(f"Close prices loaded: {len(close_df)} rows, {time.time()-t0:.1f}s")

    close_unstacked = close_df["close"].unstack(level="instrument")

    fwd_ret_mats: dict[str, pd.DataFrame] = {}
    for period_name, shift_n in HOLDING_PERIODS.items():
        fwd_ret_mats[period_name] = close_unstacked.shift(-shift_n) / close_unstacked.shift(-1) - 1

    dates = close_unstacked.index
    data_start = start_date or str(dates.min().date())
    data_end = end_date or str(dates.max().date())

    logger.info(f"Shared context ready: {len(dates)} dates × {len(close_unstacked.columns)} instruments, "
                f"{data_start} ~ {data_end}, {time.time()-t0:.1f}s")

    return {
        "close_unstacked": close_unstacked,
        "close_df": close_df,
        "fwd_ret_mats": fwd_ret_mats,
        "dates": dates,
        "data_start": data_start,
        "data_end": data_end,
        "calc_batch_id": calc_batch_id,
    }


def compute_single_factor_metrics(
    fname: str,
    factor_df: pd.DataFrame,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Compute metrics for a single factor using shared context.

    Parameters
    ----------
    fname : factor name
    factor_df : DataFrame with single column, MultiIndex(datetime, instrument)
    ctx : dict from prepare_shared_context()

    Returns dict with keys: factor_name, metrics (dict of window→metrics), reports.
    """
    import math
    import time
    t0 = time.time()

    close_unstacked = ctx["close_unstacked"]
    fwd_ret_mats = ctx["fwd_ret_mats"]

    # Extract single factor series
    if isinstance(factor_df, pd.DataFrame) and factor_df.shape[1] >= 1:
        factor_series = factor_df.iloc[:, 0]
    else:
        factor_series = factor_df

    factor_unstacked = factor_series.unstack(level="instrument")

    # Align instruments: close (qlib "000001.SZ") vs factor (tushare "SZ000001")
    factor_insts = set(factor_unstacked.columns)
    close_insts = set(close_unstacked.columns)
    overlap = len(factor_insts & close_insts)

    if overlap < len(factor_insts) * 0.3 and overlap < 100:
        # Try "000001.SZ" → "SZ000001" conversion
        mapping = {}
        for c in close_unstacked.columns:
            parts = c.split(".")
            if len(parts) == 2:
                mapping[c] = parts[1] + parts[0]
        if len(set(mapping.values()) & factor_insts) > len(factor_insts) * 0.3:
            logger.info(f"compute_single_factor_metrics({fname}): renaming close instruments (qlib→tushare)")
            close_unstacked = close_unstacked.rename(columns=mapping)
            fwd_ret_mats = {k: v.rename(columns=mapping) for k, v in fwd_ret_mats.items()}
        else:
            raise ValueError(
                f"因子与 close 股票命名不匹配且无法自动转换: "
                f"factor样例={list(factor_insts)[:3]}, close样例={list(close_insts)[:3]}, overlap={overlap}"
            )

    # Align: factor ∩ close (以 close 为统一基准)
    common_insts = sorted(close_unstacked.columns.intersection(factor_unstacked.columns))
    if len(common_insts) < 10:
        raise ValueError(f"股票交集过少: {len(common_insts)}, factor={len(factor_insts)}, close={len(close_insts)}")

    c_aligned = close_unstacked[common_insts]
    f_aligned = factor_unstacked.reindex(index=c_aligned.index, columns=common_insts)

    fwd_ret_mat = fwd_ret_mats["1d"][common_insts]
    fwd_arrs = {pname: df[common_insts].values for pname, df in fwd_ret_mats.items()}

    dates = c_aligned.index
    fwd_arr = fwd_ret_mat.values
    f_arr_full = f_aligned.values

    factor_results, factor_reports = _compute_factor_metrics_impl(
        fname=fname,
        f_arr_full=f_arr_full,
        dates=dates,
        fwd_arr=fwd_arr,
        fwd_arrs=fwd_arrs,
        close_unstacked=c_aligned,
        data_start=ctx["data_start"],
        data_end=ctx["data_end"],
        calc_batch_id=ctx["calc_batch_id"],
    )

    # Sanitize NaN/Inf → None
    for rec in factor_results:
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rec[k] = None

    # Reorganize: group by eval_window
    metrics_by_window = {}
    for rec in factor_results:
        window = rec.get("eval_window", "")
        metrics_by_window[window] = {
            k: v for k, v in rec.items()
            if k not in ("factor_name", "eval_window", "calc_batch_id",
                         "data_source", "calc_engine", "calculated_at")
        }

    duration = time.time() - t0
    logger.info(f"compute_single_factor_metrics({fname}): {len(common_insts)} insts, "
                f"{len(factor_results)} windows ok, {duration:.1f}s")

    return {
        "factor_name": fname,
        "metrics": metrics_by_window,
        "reports": factor_reports,
        "duration": round(duration, 2),
    }


# ================================================================
# Main entry: pre-unstack once, compute all factors on matrices
# ================================================================

def compute_all_factors_metrics(
    parquet_path: Path,
    qlib_bin_path: Optional[Path] = None,
    factor_filter: Optional[list[str]] = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Compute metrics for all factors using pre-unstacked matrix architecture.

    1. Read parquet + close prices, unstack ONCE into (dates × instruments) matrices
    2. Pre-compute forward returns matrix
    3. For each factor: slice columns from pre-built matrices, compute metrics via numpy

    Returns dict with keys: metrics, factor_reports, calc_batch_id, summary.
    """
    import time
    t0 = time.time()
    calc_batch_id = str(uuid.uuid4())

    logger.info(f"Loading factor data from {parquet_path}")
    factors_df = pd.read_parquet(parquet_path)

    if not isinstance(factors_df.index, pd.MultiIndex):
        raise ValueError("parquet must have MultiIndex(datetime, instrument)")
    if factors_df.index.names[0] != "datetime":
        factors_df.index.names = ["datetime", "instrument"]
    if isinstance(factors_df.columns, pd.MultiIndex):
        factors_df.columns = [col[-1] if isinstance(col, tuple) else col for col in factors_df.columns]

    factor_names = list(factors_df.columns)
    if factor_filter:
        factor_names = [f for f in factor_names if f in set(factor_filter)]
    logger.info(f"Computing metrics for {len(factor_names)} factors")

    # Date range
    all_dates_idx = factors_df.index.get_level_values("datetime")
    data_start = str(all_dates_idx.min().date())
    data_end = str(all_dates_idx.max().date())
    factor_instruments = set(factors_df.index.get_level_values("instrument").unique())

    # Load close prices
    logger.info("Loading close prices...")
    close_df = read_close_prices(
        qlib_bin_path, start_date=data_start, end_date=data_end,
        instrument_filter=factor_instruments,
    )
    close_df = _align_instrument_names(close_df, factors_df)
    logger.info(f"Close prices: {len(close_df)} rows, prep: {time.time()-t0:.1f}s")

    # === ONE-TIME UNSTACK: factors + close → (dates × instruments) matrices ===
    t1 = time.time()
    close_unstacked = close_df["close"].unstack(level="instrument")

    # 多持有期前向收益矩阵 (dates × instruments)
    fwd_ret_mats: dict[str, pd.DataFrame] = {}
    for period_name, shift_n in HOLDING_PERIODS.items():
        fwd_ret_mats[period_name] = close_unstacked.shift(-shift_n) / close_unstacked.shift(-1) - 1

    # 默认持有期（1d）用于分组回测和主IC
    fwd_ret_mat = fwd_ret_mats["1d"]

    # Unstack each factor column
    factor_mats = {}
    for fname in factor_names:
        factor_mats[fname] = factors_df[fname].unstack(level="instrument")

    # Align all matrices to common columns
    common_insts = close_unstacked.columns
    for fname in factor_names:
        ci = common_insts.intersection(factor_mats[fname].columns)
        common_insts = ci
    common_insts = sorted(common_insts)

    close_unstacked = close_unstacked[common_insts]
    fwd_ret_mat = fwd_ret_mat[common_insts]
    for pname in fwd_ret_mats:
        fwd_ret_mats[pname] = fwd_ret_mats[pname][common_insts]
    for fname in factor_names:
        factor_mats[fname] = factor_mats[fname].reindex(index=close_unstacked.index, columns=common_insts)

    # Common dates index
    dates = close_unstacked.index
    fwd_arr = fwd_ret_mat.values
    fwd_arrs = {pname: df.values for pname, df in fwd_ret_mats.items()}

    logger.info(f"Unstack done: {len(dates)} dates × {len(common_insts)} instruments, {time.time()-t1:.1f}s")

    # === Compute per-factor metrics ===
    all_results = []
    all_reports = []
    t2 = time.time()

    def _compute_single_factor(fi_fname):
        """Wrapper for parallel dispatch."""
        fi, fname = fi_fname
        f_mat_df = factor_mats[fname]
        f_arr_full = f_mat_df.values
        factor_results, factor_reports = _compute_factor_metrics_impl(
            fname=fname,
            f_arr_full=f_arr_full,
            dates=dates,
            fwd_arr=fwd_arr,
            fwd_arrs=fwd_arrs,
            close_unstacked=close_unstacked,
            data_start=data_start,
            data_end=data_end,
            calc_batch_id=calc_batch_id,
        )
        return fi, factor_results, factor_reports

    # Parallel execution: each factor is independent (read-only shared matrices)
    # numpy/pandas release GIL during matrix ops → ThreadPoolExecutor achieves real parallelism
    if len(factor_names) > 1 and max_workers > 1:
        logger.info(f"Using ThreadPoolExecutor(max_workers={max_workers}) for {len(factor_names)} factors")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_single_factor, (fi, fname)): fname
                for fi, fname in enumerate(factor_names)
            }
            done_count = 0
            for future in as_completed(futures):
                fi, factor_results, factor_reports = future.result()
                all_results.extend(factor_results)
                all_reports.extend(factor_reports)
                done_count += 1
                if done_count % 5 == 0 or done_count == len(factor_names):
                    logger.info(f"  Progress: {done_count}/{len(factor_names)} factors, {time.time()-t2:.1f}s")
    else:
        for fi, fname in enumerate(factor_names):
            _, factor_results, factor_reports = _compute_single_factor((fi, fname))
            all_results.extend(factor_results)
            all_reports.extend(factor_reports)
            if (fi + 1) % 5 == 0 or fi == len(factor_names) - 1:
                logger.info(f"  Progress: {fi+1}/{len(factor_names)} factors, {time.time()-t2:.1f}s")

    # Sanitize NaN/Inf → None for JSON compatibility
    import math
    for rec in all_results:
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rec[k] = None

    logger.info(f"Sanitize done: {len(all_results)} records, {len(factor_names)} factors, total {time.time()-t0:.1f}s")

    ok_count = sum(1 for r in all_reports if r["status"] == "ok")
    skipped_count = sum(1 for r in all_reports if r["status"] == "skipped")
    error_count = sum(1 for r in all_reports if r["status"] == "error")

    logger.info(f"Done: {len(all_results)} metrics, {len(all_reports)} reports "
                f"(ok={ok_count}, skipped={skipped_count}, error={error_count}), "
                f"{len(factor_names)} factors, total {time.time()-t0:.1f}s")

    return {
        "metrics": all_results,
        "factor_reports": all_reports,
        "calc_batch_id": calc_batch_id,
        "summary": {
            "ok_count": ok_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
        }
    }
