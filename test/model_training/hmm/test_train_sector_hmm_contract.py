from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM
from model_training.hmm import train_sector_hmm as subject


def _diag_hmm(values):
    return SimpleNamespace(
        n_components=2,
        covariance_type="diag",
        means_=np.zeros((2, 2)),
        covars_=np.asarray(values, dtype=np.float64),
    )


def test_covariance_fix_persists_bounds_and_counts_anomalies() -> None:
    hmm = _diag_hmm([[1e-8, 20.0], [0.5, 2.0]])
    fixed, count = subject.validate_and_fix_covariance(hmm, min_covar=1e-3, max_covar=10.0)
    stats = subject.covariance_bound_stats(hmm, min_covar=1e-3, max_covar=10.0)
    assert fixed is True
    assert count == 2
    assert stats["covariance_min_after"] == pytest.approx(1e-3)
    assert stats["covariance_max_after"] == pytest.approx(10.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_covariance_nonfinite_or_nonpositive_fails_loudly(bad) -> None:
    hmm = _diag_hmm([[bad, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        subject.validate_and_fix_covariance(hmm)


def test_transition_smoothing_normalizes_and_enforces_minimum_self_transition() -> None:
    result = subject.smooth_transition_matrix(
        np.array([[0.05, 0.95], [0.8, 0.2]], dtype=np.float64),
        alpha=0.1,
        min_self_trans=0.3,
    )
    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.all(np.diag(result) >= 0.3)
    assert np.isfinite(result).all()


def _rows(count: int = 12):
    start = date(2026, 1, 1)
    return [
        {
            "trade_date": start + timedelta(days=index),
            "pct_change": float(index + 1),
            "vol": 100.0 + index,
            "amount": 1000.0,
            "mf_net_amt": 10.0,
            "mf_buy_elg_amt": 8.0,
            "mf_sell_elg_amt": 3.0,
        }
        for index in range(count)
    ]


def test_rolling_window_and_optional_limit_down_change_observation_contract() -> None:
    rows = _rows()
    csi = {row["trade_date"]: 0.1 for row in rows}
    market = {row["trade_date"]: 1000.0 for row in rows}
    up = {row["trade_date"]: 0.2 for row in rows}
    down = {row["trade_date"]: 0.3 for row in rows}
    base, base_dates = subject.build_observation_matrix(
        rows, csi, market, up, down, rolling_window=3, use_limit_down=False,
    )
    extended, extended_dates = subject.build_observation_matrix(
        rows, csi, market, up, down, rolling_window=5, use_limit_down=True,
    )
    assert base.shape[1] == 7
    assert extended.shape[1] == 8
    assert base_dates == extended_dates
    assert not np.allclose(base[:, 1], extended[:, 1])
    assert np.allclose(extended[:, 4], 0.3)


def test_observation_missing_required_market_evidence_fails_loudly() -> None:
    rows = _rows(count=3)
    csi = {row["trade_date"]: 0.1 for row in rows}
    market = {row["trade_date"]: 1000.0 for row in rows[:-1]}
    up = {row["trade_date"]: 0.2 for row in rows}

    with pytest.raises(ValueError, match="missing benchmark or market volume"):
        subject.build_observation_matrix(rows, csi, market, up, rolling_window=2)


def test_zscore_on_and_off_are_explicit_and_reproducible() -> None:
    observations = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    standardized, means, stds = subject.apply_zscore(observations)
    assert np.allclose(standardized.mean(axis=0), 0.0)
    assert np.allclose((observations - means) / stds, standardized)
    assert np.array_equal(observations.copy(), observations)  # explicit no-zscore path


def test_same_input_and_seed_produce_identical_hmm_parameters() -> None:
    rng = np.random.default_rng(20260724)
    observations = np.vstack(
        [rng.normal(-1.0, 0.2, size=(80, 2)), rng.normal(1.0, 0.2, size=(80, 2))],
    )
    models = []
    for _ in range(2):
        hmm = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=50,
            min_covar=1e-3,
            random_state=subject.HMM_RANDOM_SEED,
        ).fit(observations)
        models.append((hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_))
    for left, right in zip(models[0], models[1], strict=True):
        assert np.array_equal(left, right)


def test_empty_training_source_fails_instead_of_creating_partial_model(monkeypatch) -> None:
    connection = SimpleNamespace(close=lambda: None)
    cfg = SimpleNamespace(
        db_host="unused",
        db_port=0,
        db_user="unused",
        db_password="",
        db_name="unused",
        train_start=date(2026, 1, 1),
        train_end=date(2026, 2, 1),
        qlib_bin_dir="unused",
        sector_level="L2",
    )
    monkeypatch.setattr(subject, "get_db_conn", lambda *_args: connection)
    monkeypatch.setattr(subject, "load_l2_sector_data", lambda *_args: {})
    monkeypatch.setattr(subject, "load_csi300_daily", lambda *_args: {})
    monkeypatch.setattr(subject, "load_market_total_volume", lambda *_args: {})
    monkeypatch.setattr(subject, "load_sector_stock_mapping", lambda *_args: {})
    monkeypatch.setattr(subject, "read_qlib_calendar", lambda *_args: [])
    monkeypatch.setattr(subject, "get_limit_up_ratio_by_sector", lambda *_args: ({}, {}))

    with pytest.raises(ValueError, match="no sector data"):
        subject.train_all_sectors(cfg)
