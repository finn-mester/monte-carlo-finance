"""Tests for the variance reduction estimators."""

import numpy as np
import pytest

from mcfinance.black_scholes import bs_price
from mcfinance.pricing import price_european
from mcfinance.variance_reduction import (
    control_variate_estimate,
    estimator_variance,
    price_european_antithetic,
    price_european_combined,
    price_european_control_variate,
)

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
SEED = 12345

METHODS = [
    price_european_antithetic,
    price_european_control_variate,
    price_european_combined,
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_matches_black_scholes(method, option, q):
    result = method(S0, K, R, SIGMA, T, n=400_000, option=option, q=q, rng=SEED)
    exact = bs_price(S0, K, R, SIGMA, T, option, q=q)
    assert abs(result.price - exact) < 4 * result.std_error


@pytest.mark.parametrize("method", METHODS)
def test_smaller_standard_error_than_plain(method):
    n = 100_000
    plain = price_european(S0, K, R, SIGMA, T, n=n, rng=SEED)
    reduced = method(S0, K, R, SIGMA, T, n=n, rng=SEED)
    assert reduced.n == n
    assert reduced.std_error < plain.std_error


@pytest.mark.parametrize("method", METHODS)
def test_same_seed_is_reproducible(method):
    a = method(S0, K, R, SIGMA, T, n=10_000, rng=SEED)
    b = method(S0, K, R, SIGMA, T, n=10_000, rng=SEED)
    assert a == b


@pytest.mark.parametrize("method", [price_european_antithetic, price_european_combined])
def test_odd_n_raises_for_antithetic(method):
    with pytest.raises(ValueError):
        method(S0, K, R, SIGMA, T, n=10_001)


@pytest.mark.parametrize("method", METHODS)
def test_confidence_interval_coverage(method):
    # If the standard error were computed wrongly (e.g. treating antithetic
    # pairs as independent), coverage would drift far from 95%.
    rng = np.random.default_rng(SEED)
    exact = bs_price(S0, K, R, SIGMA, T, "call")
    n_reps = 500
    hits = sum(
        method(S0, K, R, SIGMA, T, n=5_000, rng=rng).contains(exact) for _ in range(n_reps)
    )
    assert 0.92 <= hits / n_reps <= 0.98


def test_control_variate_factor_matches_theory():
    # Empirical variance ratio plain / control variate should be 1 / (1 - rho^2).
    rng = np.random.default_rng(SEED)
    params = dict(S0=S0, K=K, r=R, sigma=SIGMA, T=T, n=2_000)
    var_plain = estimator_variance(price_european, 2_000, rng=rng, **params)
    var_cv = estimator_variance(price_european_control_variate, 2_000, rng=rng, **params)

    rho = price_european_control_variate(S0, K, R, SIGMA, T, n=1_000_000, rng=rng).rho
    predicted = 1.0 / (1.0 - rho**2)
    assert var_plain / var_cv == pytest.approx(predicted, rel=0.15)


def test_control_variate_identical_payoff_gives_zero_variance():
    # If X equals the control Y, X - 1 * (Y - E[Y]) = E[Y] for every sample.
    Y = np.random.default_rng(SEED).lognormal(size=1_000)
    result = control_variate_estimate(Y, Y, mean_Y=1.5, beta=1.0)
    assert result.price == pytest.approx(1.5)
    assert result.std_error == pytest.approx(0.0, abs=1e-12)
    assert result.rho == pytest.approx(1.0)


def test_control_variate_estimated_beta_is_regression_slope():
    rng = np.random.default_rng(SEED)
    Y = rng.standard_normal(10_000)
    X = 3.0 + 2.0 * Y + 0.1 * rng.standard_normal(10_000)
    result = control_variate_estimate(X, Y, mean_Y=0.0)
    assert result.beta == pytest.approx(2.0, abs=0.01)


def test_estimator_variance_needs_two_reps():
    with pytest.raises(ValueError):
        estimator_variance(price_european, 1, S0=S0, K=K, r=R, sigma=SIGMA, T=T, n=100)
