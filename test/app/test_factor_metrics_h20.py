# ruff: noqa: SLF001

import numpy as np
import pandas as pd
import pytest
from rdagent.app.factor_metrics import engine


def test_newey_west_icir_uses_long_run_variance_and_rejects_degenerate_samples() -> None:
    positively_autocorrelated = np.repeat([0.01, 0.03, 0.015, 0.035], 25)

    hac_icir = engine._newey_west_icir(positively_autocorrelated)
    naive_icir = positively_autocorrelated.mean() / positively_autocorrelated.std()

    assert hac_icir is not None
    assert 0 < hac_icir < naive_icir
    assert engine._newey_west_icir(np.ones(40)) is None
    assert engine._newey_west_icir(np.arange(engine.H20_HAC_LAG, dtype=float)) is None


def test_h20_metrics_are_additive_and_keep_existing_1d_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "EVAL_WINDOWS", {"full": None})
    monkeypatch.setattr(engine, "REQUIRED_DAYS", {"full": 0})

    rng = np.random.default_rng(20260711)
    n_dates = 80
    n_instruments = 12
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    columns = [f"S{i:02d}" for i in range(n_instruments)]

    cross_section = np.tile(np.linspace(-1.0, 1.0, n_instruments), (n_dates, 1))
    factors = cross_section + rng.normal(scale=0.15, size=cross_section.shape)
    one_day_returns = -0.01 * factors + rng.normal(scale=0.003, size=factors.shape)
    h20_returns = 0.05 * factors + rng.normal(scale=0.04, size=factors.shape)

    close_returns = rng.normal(loc=0.0005, scale=0.01, size=(n_dates, n_instruments))
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + close_returns, axis=0), index=dates, columns=columns)
    forward_returns = {
        "1d": one_day_returns,
        "5d": rng.normal(scale=0.02, size=factors.shape),
        "10d": rng.normal(scale=0.03, size=factors.shape),
        "20d": h20_returns,
    }

    results, reports = engine._compute_factor_metrics_impl(
        fname="test_factor",
        f_arr_full=factors,
        dates=dates,
        fwd_arr=one_day_returns,
        fwd_arrs=forward_returns,
        close_unstacked=close,
        data_start=str(dates.min().date()),
        data_end=str(dates.max().date()),
        calc_batch_id="test-batch",
    )

    assert reports[0]["status"] == "ok"
    assert len(results) == 1
    result = results[0]

    assert result["return_horizon"] == "1d"
    assert result["ic_mean"] < 0
    assert result["h20_return_horizon"] == "T21T1"
    assert result["h20_ic_mean"] > 0
    assert result["h20_rank_ic_mean"] > 0
    assert result["h20_icir_hac"] is not None
    assert result["h20_rank_icir_hac"] is not None
    assert result["h20_n_obs"] == n_dates
    assert result["h20_hac_lag"] == engine.H20_HAC_LAG

    # Preserve the legacy full-window field while providing formal h20 stats
    # for every evaluation window through the new companion fields.
    assert result["rank_ic_20d"] == pytest.approx(result["h20_rank_ic_mean"])
