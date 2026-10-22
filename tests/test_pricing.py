"""Tests for GBM simulation and Monte Carlo pricing."""

import numpy as np
import pytest

from mcfinance.black_scholes import bs_price
from mcfinance.gbm import simulate_paths, simulate_terminal
from mcfinance.pricing import convergence_curve, price_european

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
SEED = 12345


@pytest.mark.parametrize("option", ["call", "put"])
def test_mc_matches_black_scholes(option):
    result = price_european(S0, K, R, SIGMA, T, n=400_000, option=option, rng=SEED)
    exact = bs_price(S0, K, R, SIGMA, T, option)
    assert abs(result.price - exact) < 4 * result.std_error


def test_same_seed_is_reproducible():
    a = price_european(S0, K, R, SIGMA, T, n=10_000, rng=SEED)
    b = price_european(S0, K, R, SIGMA, T, n=10_000, rng=SEED)
    assert a == b


def test_standard_error_scales_as_one_over_sqrt_n():
    # Quadrupling the sample size should halve the standard error.
    small = price_european(S0, K, R, SIGMA, T, n=50_000, rng=1)
    large = price_european(S0, K, R, SIGMA, T, n=200_000, rng=2)
    assert large.std_error / small.std_error == pytest.approx(0.5, rel=0.05)


def test_confidence_interval_coverage():
    # About 95% of the 95% confidence intervals should contain the exact price.
    rng = np.random.default_rng(SEED)
    exact = bs_price(S0, K, R, SIGMA, T, "call")
    n_reps = 500
    hits = sum(
        price_european(S0, K, R, SIGMA, T, n=5_000, rng=rng).contains(exact) for _ in range(n_reps)
    )
    assert 0.92 <= hits / n_reps <= 0.98


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_terminal_mean_is_forward_price(q):
    # Under Q, E[S_T] = S0 * exp((r - q) T).
    n = 400_000
    S_T = simulate_terminal(S0, R, SIGMA, T, n, rng=SEED, q=q)
    std_error = S_T.std(ddof=1) / np.sqrt(n)
    expected = S0 * np.exp((R - q) * T)
    assert abs(S_T.mean() - expected) < 4 * std_error


def test_simulate_paths_shape_and_start():
    paths = simulate_paths(S0, R, SIGMA, T, n_paths=7, n_steps=50, rng=SEED)
    assert paths.shape == (7, 51)
    assert np.all(paths[:, 0] == S0)
    assert np.all(paths > 0)


def test_convergence_curve_last_point_matches_price_european():
    n_max = 20_000
    ns, prices, std_errors = convergence_curve(S0, K, R, SIGMA, T, n_max, rng=SEED)
    direct = price_european(S0, K, R, SIGMA, T, n=n_max, rng=SEED)
    assert ns[-1] == n_max
    assert np.all(np.diff(ns) > 0)
    assert prices[-1] == pytest.approx(direct.price, rel=1e-10)
    assert std_errors[-1] == pytest.approx(direct.std_error, rel=1e-6)


def test_n_equal_one_raises():
    with pytest.raises(ValueError):
        price_european(S0, K, R, SIGMA, T, n=1)


@pytest.mark.parametrize("kwargs", [dict(S0=0.0), dict(sigma=-0.1), dict(T=0.0)])
def test_simulator_rejects_invalid_inputs(kwargs):
    params = dict(S0=S0, r=R, sigma=SIGMA, T=T, n=10)
    params.update(kwargs)
    with pytest.raises(ValueError):
        simulate_terminal(**params)
