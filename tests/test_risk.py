"""Tests for VaR / ES engines and backtesting (synthetic data only, no network)."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, norm, t as student_t

from mcfinance.risk import (
    LEVELS,
    basel_traffic_light,
    cholesky_correlated_normals,
    christoffersen_cc,
    christoffersen_independence,
    estimate_risk,
    ewma_covariance,
    historical,
    kupiec_pof,
    mc_normal,
    mc_student_t,
    parametric_normal,
    portfolio_losses,
    rolling_backtest,
)

SEED = 12345
V = 1_000_000.0
W3 = np.array([0.5, 0.3, 0.2])


def make_cov(vols, corr):
    vols = np.asarray(vols)
    return np.outer(vols, vols) * np.asarray(corr)


CORR = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
# Realistic daily volatilities (1.0% to 2.0%).
COV = make_cov([0.015, 0.010, 0.020], CORR)
# Small volatilities: the closed form linearises exp(r) - 1 ~ r and ignores
# the convexity term r^2 / 2, so its relative error is roughly z * sigma / 2.
# At 0.5% daily volatility that is about 0.4%, more than the 4-standard-error
# tolerance of the test below (about 0.3%, independent of sigma). With
# volatilities around 0.1% the linearisation error drops to about 0.08%.
COV_SMALL = make_cov([0.0010, 0.0008, 0.0012], CORR)


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------
def test_cholesky_draws_have_target_covariance():
    n = 200_000
    X = cholesky_correlated_normals(COV, n, rng=SEED)
    sample = np.cov(X, rowvar=False)
    # Standard error of a sample covariance entry for normal data:
    # sqrt((cov_ij^2 + cov_ii cov_jj) / n).
    se = np.sqrt((COV**2 + np.outer(np.diag(COV), np.diag(COV))) / n)
    assert np.all(np.abs(sample - COV) < 4 * se)
    assert np.allclose(X.mean(axis=0), 0, atol=4 * np.sqrt(np.diag(COV) / n))


def test_non_positive_definite_raises():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    with pytest.raises(ValueError):
        cholesky_correlated_normals(bad, 10, rng=SEED)


# ---------------------------------------------------------------------------
# Normal engines
# ---------------------------------------------------------------------------
def test_mc_normal_matches_closed_form():
    # R independent MC estimates; their mean must be within 4 standard errors
    # (sd / sqrt(R)) of the closed form. COV_SMALL keeps the linearisation
    # error of the closed form well below this tolerance.
    rng = np.random.default_rng(SEED)
    R = 20
    runs = [mc_normal(COV_SMALL, W3, V, n_sims=100_000, rng=rng) for _ in range(R)]
    exact = parametric_normal(COV_SMALL, W3, V)
    for alpha in LEVELS:
        for attr in ("var", "es"):
            values = np.array([getattr(res, attr)[alpha] for res in runs])
            se = values.std(ddof=1) / np.sqrt(R)
            assert abs(values.mean() - getattr(exact, attr)[alpha]) < 4 * se, (attr, alpha)


def test_parametric_horizon_scaling():
    one = parametric_normal(COV, W3, V)
    ten = parametric_normal(COV, W3, V, horizon=10)
    for alpha in LEVELS:
        assert ten.var[alpha] == pytest.approx(np.sqrt(10) * one.var[alpha], rel=1e-12)
        assert ten.es[alpha] == pytest.approx(np.sqrt(10) * one.es[alpha], rel=1e-12)


def test_parametric_reference_values():
    # One asset with sigma = 1%: VaR_99 = 2.3263 % of V, ES_99 = 2.6652 % of V.
    res = parametric_normal(np.array([[0.01**2]]), [1.0], 100.0)
    assert res.var[0.99] == pytest.approx(norm.ppf(0.99), rel=1e-12)
    assert res.es[0.99] == pytest.approx(norm.pdf(norm.ppf(0.99)) / 0.01, rel=1e-12)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        parametric_normal(COV, [0.5, 0.3, 0.3], V)
    with pytest.raises(ValueError):
        mc_normal(COV, [0.5, 0.5, 0.5], V, n_sims=100, rng=SEED)
    with pytest.raises(ValueError):
        portfolio_losses(np.zeros((2, 3)), [1.0, 1.0, 1.0], V)


@pytest.mark.parametrize("engine", [mc_normal, mc_student_t])
def test_same_seed_is_reproducible(engine):
    a = engine(COV, W3, V, n_sims=10_000, rng=SEED)
    b = engine(COV, W3, V, n_sims=10_000, rng=SEED)
    assert a == b


# ---------------------------------------------------------------------------
# Student-t engine
# ---------------------------------------------------------------------------
def _t_draws(cov, nu, n, rng):
    """Same construction as mc_student_t, to inspect the simulated returns."""
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, len(cov)))
    W = rng.chisquare(nu, size=n) / nu
    return np.sqrt((nu - 2) / nu) * (Z @ L.T) / np.sqrt(W)[:, None]


def test_student_t_has_target_covariance_and_fat_tails():
    X = _t_draws(COV, 5, 400_000, np.random.default_rng(SEED))
    # Heavy tails make the sample covariance noisy, so use a relative tolerance.
    assert np.allclose(np.cov(X, rowvar=False), COV, rtol=0.05, atol=1e-6)
    # Theoretical excess kurtosis of a t(5) is 6 / (5 - 4) = 6; any positive
    # value shows fatter tails than the normal (excess kurtosis 0).
    assert kurtosis(X @ W3) > 1.0


def test_student_t_with_large_nu_is_normal():
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((400_000, 3))
    normal = mc_normal(COV, W3, V, Z=Z)
    t_large = mc_student_t(COV, W3, V, nu=1000, Z=Z, rng=rng)
    for alpha in LEVELS:
        assert t_large.var[alpha] == pytest.approx(normal.var[alpha], rel=0.02)
        assert t_large.es[alpha] == pytest.approx(normal.es[alpha], rel=0.02)


def test_student_t_quantile_crossover():
    # Unit-variance t(5) quantiles are sqrt(3/5) * t_quantile:
    #   95%: 0.7746 * 2.015 = 1.561  < normal 1.645
    #   99%: 0.7746 * 3.365 = 2.606  > normal 2.326
    # Same variance, but the t puts more mass both in the centre and in the
    # far tails, and less in the shoulders, so it only dominates far out.
    scale = np.sqrt(3 / 5)
    assert scale * student_t.ppf(0.95, 5) < norm.ppf(0.95)
    assert scale * student_t.ppf(0.99, 5) > norm.ppf(0.99)

    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((500_000, 3))
    normal = mc_normal(COV, W3, V, Z=Z)
    t5 = mc_student_t(COV, W3, V, nu=5, Z=Z, rng=rng)
    assert t5.var[0.99] > normal.var[0.99]
    assert t5.es[0.99] > normal.es[0.99]
    # Not an assertion that the t is "riskier" at 95%: its 95% VaR is lower.
    assert t5.var[0.95] < normal.var[0.95]


def test_student_t_requires_nu_above_two():
    with pytest.raises(ValueError):
        mc_student_t(COV, W3, V, nu=2.0, n_sims=100, rng=SEED)


# ---------------------------------------------------------------------------
# Historical simulation and EWMA
# ---------------------------------------------------------------------------
def test_historical_is_empirical_quantile():
    returns = np.random.default_rng(SEED).standard_t(4, size=(1_000, 3)) * 0.01
    res = historical(returns, W3, V)
    losses = -V * (np.expm1(returns) @ W3)
    for alpha in LEVELS:
        assert res.var[alpha] == pytest.approx(np.quantile(losses, alpha))
        assert res.es[alpha] >= res.var[alpha]
        assert res.var[alpha] > 0  # a loss is a positive number
    # A portfolio that only gains has negative losses.
    assert np.all(portfolio_losses(np.full((5, 3), 0.01), W3, V) < 0)


def test_historical_uses_raw_returns_regardless_of_use_mean():
    # Historical simulation is defined on the raw returns: the use_mean flag
    # of estimate_risk must not demean them.
    returns = 0.002 + np.random.default_rng(SEED).standard_normal((500, 3)) * 0.01
    raw = historical(returns, W3, V)
    for flag in (False, True):
        res = estimate_risk("historical", returns, W3, V, use_mean=flag)
        assert res == raw
    demeaned = historical(returns - returns.mean(axis=0), W3, V)
    assert raw.var[0.99] < demeaned.var[0.99]  # a positive drift lowers the losses


def test_ewma_weights_recent_returns_more():
    calm = np.random.default_rng(SEED).standard_normal((480, 2)) * 0.01
    stressed = np.random.default_rng(SEED + 1).standard_normal((20, 2)) * 0.04
    r = np.vstack([calm, stressed])
    ewma = ewma_covariance(r, 0.94)
    assert np.all(np.diag(ewma) > np.diag(np.cov(r, rowvar=False)))
    # A single return: the EWMA estimate is r r^T.
    one = np.array([[0.01, -0.02]])
    assert np.allclose(ewma_covariance(one), one.T @ one)


# ---------------------------------------------------------------------------
# Backtesting tests
# ---------------------------------------------------------------------------
def test_kupiec_zero_when_rate_is_exact():
    e = np.zeros(1_000, dtype=bool)
    e[::100] = True  # exactly 10 exceptions = 1% of 1000
    lr, pvalue = kupiec_pof(e, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-10)
    assert pvalue == pytest.approx(1.0)


def test_kupiec_hand_computed():
    # 5 exceptions in 250 days at p = 1%: pi = 0.02,
    # LR = -2 [245 ln 0.99 + 5 ln 0.01] + 2 [245 ln 0.98 + 5 ln 0.02]
    #    = 2 [245 ln(0.98 / 0.99) + 5 ln 2] = 1.9568...
    e = np.zeros(250, dtype=bool)
    e[:5] = True
    expected = 2 * (245 * math.log(0.98 / 0.99) + 5 * math.log(2))
    lr, pvalue = kupiec_pof(e, 0.01)
    assert lr == pytest.approx(expected, rel=1e-12)
    assert 0.1 < pvalue < 0.2


def test_kupiec_rejects_wrong_rate_and_handles_edges():
    e = np.zeros(250, dtype=bool)
    e[:15] = True  # 6% instead of 1%
    assert kupiec_pof(e, 0.01)[1] < 0.001
    # 0 exceptions: LR = -2 * 250 ln(0.99) = 5.025
    lr0, p0 = kupiec_pof(np.zeros(250, dtype=bool), 0.01)
    assert lr0 == pytest.approx(-2 * 250 * math.log(0.99))
    assert 0.02 < p0 < 0.03
    # All exceptions: finite statistic, certain rejection.
    lr_all, p_all = kupiec_pof(np.ones(250, dtype=bool), 0.01)
    assert np.isfinite(lr_all) and p_all < 1e-10


def test_christoffersen_independence():
    spread = np.zeros(1_000, dtype=bool)
    spread[5::100] = True  # 10 isolated exceptions
    assert christoffersen_independence(spread)[1] > 0.05

    clustered = np.zeros(1_000, dtype=bool)
    clustered[500:510] = True  # the same 10 exceptions, all consecutive
    assert christoffersen_independence(clustered)[1] < 0.001
    assert christoffersen_cc(clustered, 0.01)[1] < 0.001
    # Same rate, so the POF test alone cannot tell the two apart.
    assert kupiec_pof(spread, 0.01)[0] == pytest.approx(kupiec_pof(clustered, 0.01)[0])


def test_christoffersen_edge_cases():
    lr, pvalue = christoffersen_independence(np.zeros(250, dtype=bool))
    assert lr == pytest.approx(0.0) and pvalue == pytest.approx(1.0)
    lr_cc, p_cc = christoffersen_cc(np.zeros(250, dtype=bool), 0.01)
    assert np.isfinite(lr_cc) and 0 < p_cc < 1
    assert np.isfinite(christoffersen_independence(np.ones(250, dtype=bool))[0])


@pytest.mark.parametrize("n, zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red"), (25, "red")])
def test_traffic_light_boundaries(n, zone):
    assert basel_traffic_light(n) == zone


def _iid_returns(n_days, seed=SEED):
    X = cholesky_correlated_normals(COV, n_days, rng=seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.DataFrame(X, index=dates)


def test_backtest_on_iid_normal_returns_is_calibrated():
    returns = _iid_returns(3_500)
    bt = rolling_backtest(returns, W3, V, methods=("parametric_normal", "mc_normal"),
                          levels=(0.99,), window=500, n_sims=20_000, rng=SEED)
    assert len(bt) == 3_000
    for method in ("parametric_normal", "mc_normal"):
        e = bt[f"exc_{method}_99"].to_numpy()
        # Binomial tolerance: 4 standard deviations around N p.
        expected = len(e) * 0.01
        assert abs(e.sum() - expected) < 4 * np.sqrt(expected * 0.99)
        assert kupiec_pof(e, 0.01)[1] > 0.05


def test_backtest_has_no_look_ahead():
    returns = _iid_returns(700)
    t = 600
    before = rolling_backtest(returns, W3, V, methods=("parametric_normal", "historical"), window=500, rng=SEED)
    shocked = returns.copy()
    shocked.iloc[t] = -0.30  # a crash on day t
    after = rolling_backtest(shocked, W3, V, methods=("parametric_normal", "historical"), window=500, rng=SEED)

    day = returns.index[t]
    cols = [c for c in before.columns if c.startswith(("var_", "es_"))]
    # The forecast FOR day t is unchanged; the realised loss on day t is not.
    assert before.loc[day, cols].equals(after.loc[day, cols])
    assert after.loc[day, "loss"] > before.loc[day, "loss"]
    # The forecast for the next day does see the crash.
    next_day = returns.index[t + 1]
    assert after.loc[next_day, "var_parametric_normal_99"] > before.loc[next_day, "var_parametric_normal_99"]


def test_estimate_risk_rejects_unknown_method():
    with pytest.raises(ValueError):
        estimate_risk("garch", np.zeros((10, 3)) + 0.001, W3, V)
