"""Tests for path-dependent options: closed forms and Monte Carlo pricers."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from mcfinance.black_scholes import bs_price
from mcfinance.exotics import path_averages, price_asian, price_barrier
from mcfinance.exotics_closed_form import (
    BGK_BETA,
    barrier_closed_form,
    bgk_adjusted_barrier,
    geometric_asian_price,
    geometric_average_expectation,
)
from mcfinance.gbm import simulate_paths
from mcfinance.pricing import payoff, price_european

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
H = 90.0
SEED = 12345


# ---------------------------------------------------------------------------
# Geometric Asian closed form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_geometric_asian_with_one_date_is_european(option, q):
    # With m = 1 the "average" is S_T itself.
    asian = geometric_asian_price(S0, K, R, SIGMA, T, 1, option, q=q)
    assert asian == pytest.approx(bs_price(S0, K, R, SIGMA, T, option, q=q), abs=1e-12)


def test_geometric_asian_call_decreases_with_monitoring_dates():
    # As m grows, both the mean and the variance of ln G decrease when
    # r - q - sigma^2/2 >= 0 (here 0.03), and a call is increasing in both,
    # so the call price must decrease and stay below the European price.
    # For a put the two effects go in opposite directions (lower mean raises
    # the put, lower variance lowers it), so no monotonicity is guaranteed.
    prices = [geometric_asian_price(S0, K, R, SIGMA, T, m, "call") for m in (1, 12, 252)]
    assert prices[0] > prices[1] > prices[2]
    assert all(p <= bs_price(S0, K, R, SIGMA, T, "call") + 1e-12 for p in prices)


def test_geometric_average_expectation_matches_simulation():
    m, n = 12, 200_000
    _, G = path_averages(simulate_paths(S0, R, SIGMA, T, n, m, rng=SEED))
    std_error = G.std(ddof=1) / np.sqrt(n)
    assert abs(G.mean() - geometric_average_expectation(S0, R, SIGMA, T, m)) < 4 * std_error


# ---------------------------------------------------------------------------
# Asian Monte Carlo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("option", ["call", "put"])
def test_mc_geometric_asian_matches_closed_form(option):
    m = 52
    result = price_asian(
        S0, K, R, SIGMA, T, 200_000, m, option=option, average="geometric", rng=SEED
    )
    exact = geometric_asian_price(S0, K, R, SIGMA, T, m, option)
    assert abs(result.price - exact) < 4 * result.std_error


def test_mc_geometric_asian_coverage():
    rng = np.random.default_rng(SEED)
    m = 12
    exact = geometric_asian_price(S0, K, R, SIGMA, T, m)
    n_reps = 400
    hits = sum(
        price_asian(S0, K, R, SIGMA, T, 2_000, m, average="geometric", rng=rng).contains(exact)
        for _ in range(n_reps)
    )
    assert 0.92 <= hits / n_reps <= 0.98


def test_arithmetic_asian_with_one_date_is_european():
    # One monitoring date: the average is S_T, and simulate_paths with one
    # step consumes the same normals as simulate_terminal.
    asian = price_asian(S0, K, R, SIGMA, T, 100_000, 1, rng=SEED)
    european = price_european(S0, K, R, SIGMA, T, 100_000, rng=SEED)
    assert asian.price == pytest.approx(european.price, rel=1e-12)
    assert asian.std_error == pytest.approx(european.std_error, rel=1e-12)


def test_arithmetic_average_dominates_geometric_pathwise():
    # AM-GM: on every path, arithmetic average >= geometric average, so the
    # call payoffs are ordered on every path and so are the MC prices.
    paths = simulate_paths(S0, R, SIGMA, T, 10_000, 52, rng=SEED)
    A, G = path_averages(paths)
    assert np.all(A >= G - 1e-12)
    assert np.all(payoff(A, K, "call") >= payoff(G, K, "call") - 1e-12)

    arith = price_asian(S0, K, R, SIGMA, T, 10_000, 52, rng=SEED)
    geom = price_asian(S0, K, R, SIGMA, T, 10_000, 52, average="geometric", rng=SEED)
    assert arith.price >= geom.price


def test_arithmetic_asian_below_european_call():
    # Averaging lowers the effective volatility, so the Asian call is cheaper.
    asian = price_asian(S0, K, R, SIGMA, T, 50_000, 52, rng=SEED)
    assert asian.price + 4 * asian.std_error < bs_price(S0, K, R, SIGMA, T, "call")


def test_control_variate_agrees_and_reduces_error():
    m, n = 52, 50_000
    plain = price_asian(S0, K, R, SIGMA, T, n, m, rng=1)
    cv = price_asian(S0, K, R, SIGMA, T, n, m, control_variate=True, rng=2)
    combined_se = np.sqrt(plain.std_error**2 + cv.std_error**2)
    assert abs(plain.price - cv.price) < 4 * combined_se
    assert cv.std_error < plain.std_error / 10
    assert cv.rho > 0.99


def test_asian_input_validation():
    with pytest.raises(ValueError):
        price_asian(S0, K, R, SIGMA, T, 1_000, 12, average="harmonic")
    with pytest.raises(ValueError):
        price_asian(S0, K, R, SIGMA, T, 1_000, 12, average="geometric", control_variate=True)


# ---------------------------------------------------------------------------
# Barrier closed form
# ---------------------------------------------------------------------------
def _down_and_out_by_integration(S0, K, H, r, sigma, T, q=0.0):
    """Independent check: integrate the call payoff against the density of
    x = ln(S_T / S0) for paths that never touch b = ln(H / S0), obtained by
    the reflection principle (method of images)."""
    mu = r - q - 0.5 * sigma**2
    s = sigma * np.sqrt(T)
    b = np.log(H / S0)

    def density(x):
        return norm.pdf(x, mu * T, s) - np.exp(2 * mu * b / sigma**2) * norm.pdf(
            x - 2 * b, mu * T, s
        )

    lower = np.log(K / S0)  # since H <= K, the payoff region lies above the barrier
    value, _ = quad(lambda x: (S0 * np.exp(x) - K) * density(x), lower, lower + 12 * s, limit=200)
    return np.exp(-r * T) * value


@pytest.mark.parametrize(
    "barrier, strike, q", [(90.0, 100.0, 0.0), (80.0, 100.0, 0.0), (95.0, 110.0, 0.03)]
)
def test_barrier_formula_matches_reflection_principle(barrier, strike, q):
    closed = barrier_closed_form(S0, strike, barrier, R, SIGMA, T, "down-and-out", q=q)
    assert closed == pytest.approx(
        _down_and_out_by_integration(S0, strike, barrier, R, SIGMA, T, q), abs=1e-8
    )


def test_barrier_in_out_parity_and_limits():
    vanilla = bs_price(S0, K, R, SIGMA, T, "call")
    d_out = barrier_closed_form(S0, K, H, R, SIGMA, T, "down-and-out")
    d_in = barrier_closed_form(S0, K, H, R, SIGMA, T, "down-and-in")
    assert d_out + d_in == pytest.approx(vanilla, abs=1e-12)
    assert 0 < d_out < vanilla
    # A barrier far below the spot is almost never hit.
    assert barrier_closed_form(S0, K, 1.0, R, SIGMA, T) == pytest.approx(vanilla, abs=1e-10)


@pytest.mark.parametrize("barrier, strike", [(100.0, 100.0), (105.0, 120.0), (95.0, 90.0)])
def test_barrier_closed_form_invalid_inputs(barrier, strike):
    # H >= S0 (already knocked out) or H > K (formula not valid).
    with pytest.raises(ValueError):
        barrier_closed_form(S0, strike, barrier, R, SIGMA, T)


def test_bgk_shifts_barrier_away_from_spot():
    m = 52
    shifted = bgk_adjusted_barrier(H, SIGMA, T, m)
    assert shifted == pytest.approx(H * np.exp(-BGK_BETA * SIGMA * np.sqrt(T / m)))
    assert shifted < H
    assert bgk_adjusted_barrier(110.0, SIGMA, T, m, direction="up") > 110.0
    assert bgk_adjusted_barrier(H, SIGMA, T, 10**12) == pytest.approx(H)


# ---------------------------------------------------------------------------
# Barrier Monte Carlo
# ---------------------------------------------------------------------------
def test_barrier_in_plus_out_equals_vanilla_on_same_paths():
    n, m = 50_000, 52
    d_out = price_barrier(S0, K, H, R, SIGMA, T, n, m, kind="down-and-out", rng=SEED)
    d_in = price_barrier(S0, K, H, R, SIGMA, T, n, m, kind="down-and-in", rng=SEED)
    paths = simulate_paths(S0, R, SIGMA, T, n, m, rng=SEED)
    vanilla = np.mean(np.exp(-R * T) * payoff(paths[:, -1], K, "call"))
    assert d_out.price + d_in.price == pytest.approx(vanilla, rel=1e-12)
    assert d_out.price <= vanilla


def test_barrier_far_below_spot_is_vanilla():
    # A barrier at 30 is 6 standard deviations away: no path touches it.
    n, m = 20_000, 52
    d_out = price_barrier(S0, K, 30.0, R, SIGMA, T, n, m, rng=SEED)
    paths = simulate_paths(S0, R, SIGMA, T, n, m, rng=SEED)
    vanilla = np.mean(np.exp(-R * T) * payoff(paths[:, -1], K, "call"))
    assert d_out.price == pytest.approx(vanilla, rel=1e-12)


def test_barrier_invalid_inputs_and_reproducibility():
    with pytest.raises(ValueError):
        price_barrier(S0, K, 100.0, R, SIGMA, T, 1_000, 12)
    with pytest.raises(ValueError):
        price_barrier(S0, K, H, R, SIGMA, T, 1_000, 12, kind="up-and-out")
    a = price_barrier(S0, K, H, R, SIGMA, T, 5_000, 12, rng=SEED)
    b = price_barrier(S0, K, H, R, SIGMA, T, 5_000, 12, rng=SEED)
    assert a == b


def test_fine_monitoring_matches_bgk_closed_form():
    # With m = 1000 dates and the BGK-shifted barrier, the remaining error of
    # the continuity correction is o(1/sqrt(m)); we allow 4 standard errors
    # plus a small tolerance of 0.01 (about 0.1% of the price) for it.
    m = 1_000
    mc = price_barrier(S0, K, H, R, SIGMA, T, 100_000, m, rng=SEED)
    bgk = barrier_closed_form(S0, K, bgk_adjusted_barrier(H, SIGMA, T, m), R, SIGMA, T)
    assert abs(mc.price - bgk) < 4 * mc.std_error + 0.01


def test_continuous_formula_understates_discrete_down_and_out():
    # A continuously monitored barrier is checked at every instant, so it is
    # hit MORE often than one checked at m dates: the continuous down-and-out
    # is worth LESS than the discrete one. Using the unadjusted continuous
    # formula therefore understates the discretely monitored price.
    m = 52
    mc = price_barrier(S0, K, H, R, SIGMA, T, 100_000, m, rng=SEED)
    continuous = barrier_closed_form(S0, K, H, R, SIGMA, T)
    assert continuous < mc.price - 4 * mc.std_error


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def test_result_does_not_depend_on_batch_size():
    # Batching only changes how many paths are generated at once. Different
    # seeds are used so that the check is purely statistical.
    n, m = 40_000, 52
    small = price_asian(S0, K, R, SIGMA, T, n, m, rng=1, batch_size=3_000)
    large = price_asian(S0, K, R, SIGMA, T, n, m, rng=2, batch_size=40_000)
    assert abs(small.price - large.price) < 4 * np.hypot(small.std_error, large.std_error)

    small = price_barrier(S0, K, H, R, SIGMA, T, n, m, rng=1, batch_size=3_000)
    large = price_barrier(S0, K, H, R, SIGMA, T, n, m, rng=2, batch_size=40_000)
    assert abs(small.price - large.price) < 4 * np.hypot(small.std_error, large.std_error)


def test_same_seed_is_identical_across_batch_sizes():
    # Stronger property that holds with NumPy's Generator: normals are drawn
    # sequentially in row-major order, so batches of rows consume exactly the
    # same stream as one big array and the per-path payoffs are identical.
    n, m = 40_000, 52
    a = price_asian(S0, K, R, SIGMA, T, n, m, rng=SEED, batch_size=3_000)
    b = price_asian(S0, K, R, SIGMA, T, n, m, rng=SEED, batch_size=40_000)
    assert a.price == pytest.approx(b.price, rel=1e-12)
