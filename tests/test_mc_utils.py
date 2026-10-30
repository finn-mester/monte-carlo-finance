"""Essential checks of mc_utils.py. Synthetic data only: no network, no data file."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

# mc_utils.py sits at the repository root, one level above this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mc_utils as mc

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
H = 90.0
SEED = 42

# Portfolio used in the risk tests: three assets with daily volatilities of 1-2%.
WEIGHTS = np.array([0.5, 0.3, 0.2])
V = 1_000_000.0
CORR = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
VOLS = np.array([0.015, 0.010, 0.020])
COV = np.outer(VOLS, VOLS) * CORR


def within_4_se(result, exact):
    return abs(result["price"] - exact) < 4 * result["std_error"]


# --- Black-Scholes -----------------------------------------------------------
def test_black_scholes_reference_values():
    # Textbook values for the at-the-money contract.
    assert mc.bs_price(S0, K, R, SIGMA, T, "call") == pytest.approx(10.4506, abs=1e-4)
    assert mc.bs_price(S0, K, R, SIGMA, T, "put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    # C - P = S0 exp(-qT) - K exp(-rT), for any strike and dividend yield.
    for strike in (80.0, 100.0, 120.0):
        for q in (0.0, 0.03):
            call = mc.bs_price(S0, strike, R, SIGMA, T, "call", q)
            put = mc.bs_price(S0, strike, R, SIGMA, T, "put", q)
            forward_value = S0 * np.exp(-q * T) - strike * np.exp(-R * T)
            assert call - put == pytest.approx(forward_value, abs=1e-10)


# --- Plain Monte Carlo -------------------------------------------------------
def test_mc_within_4_standard_errors_of_black_scholes():
    for option in ("call", "put"):
        result = mc.price_european(S0, K, R, SIGMA, T, 400_000, option, rng=SEED)
        assert within_4_se(result, mc.bs_price(S0, K, R, SIGMA, T, option))


def test_same_seed_gives_same_result():
    a = mc.price_european(S0, K, R, SIGMA, T, 10_000, rng=SEED)
    b = mc.price_european(S0, K, R, SIGMA, T, 10_000, rng=SEED)
    assert a == b


def test_confidence_interval_covers_about_95_percent():
    rng = np.random.default_rng(SEED)
    exact = mc.bs_price(S0, K, R, SIGMA, T)
    results = [mc.price_european(S0, K, R, SIGMA, T, 5_000, rng=rng) for _ in range(500)]
    coverage = np.mean([res["ci_low"] <= exact <= res["ci_high"] for res in results])
    assert 0.92 <= coverage <= 0.98


def test_mean_of_terminal_price_is_the_forward():
    n = 400_000
    S_T = mc.simulate_terminal(S0, R, SIGMA, T, n, rng=SEED)
    assert abs(S_T.mean() - S0 * np.exp(R * T)) < 4 * S_T.std(ddof=1) / np.sqrt(n)


def test_simulate_paths_shape_and_start():
    paths = mc.simulate_paths(S0, R, SIGMA, T, n_paths=7, n_steps=50, rng=SEED)
    assert paths.shape == (7, 51)
    assert np.all(paths[:, 0] == S0)


# --- Variance reduction ------------------------------------------------------
REDUCED = [mc.price_antithetic, mc.price_control_variate, mc.price_combined]


@pytest.mark.parametrize("method", REDUCED)
def test_variance_reduction_is_unbiased_and_reduces_the_error(method):
    n = 100_000
    plain = mc.price_european(S0, K, R, SIGMA, T, n, rng=SEED)
    for option in ("call", "put"):
        assert within_4_se(
            method(S0, K, R, SIGMA, T, n, option, rng=SEED), mc.bs_price(S0, K, R, SIGMA, T, option)
        )
    assert method(S0, K, R, SIGMA, T, n, rng=SEED)["std_error"] < plain["std_error"]


def test_variance_reduction_confidence_intervals_cover_95_percent():
    # Wrong standard errors, e.g. antithetic pairs treated as independent,
    # would push the coverage far from 95%.
    exact = mc.bs_price(S0, K, R, SIGMA, T)
    for method in REDUCED:
        rng = np.random.default_rng(SEED)
        results = [method(S0, K, R, SIGMA, T, 5_000, rng=rng) for _ in range(500)]
        coverage = np.mean([res["ci_low"] <= exact <= res["ci_high"] for res in results])
        assert 0.92 <= coverage <= 0.98


def test_antithetic_needs_an_even_n():
    with pytest.raises(ValueError):
        mc.price_antithetic(S0, K, R, SIGMA, T, 10_001)


# --- Asian options -----------------------------------------------------------
def test_geometric_asian_with_one_date_is_black_scholes():
    for option in ("call", "put"):
        asian = mc.geometric_asian_price(S0, K, R, SIGMA, T, 1, option)
        assert asian == pytest.approx(mc.bs_price(S0, K, R, SIGMA, T, option), abs=1e-12)


def test_mc_geometric_asian_matches_closed_form():
    result = mc.price_asian(S0, K, R, SIGMA, T, 200_000, 52, average="geometric", rng=SEED)
    assert within_4_se(result, mc.geometric_asian_price(S0, K, R, SIGMA, T, 52))


def test_arithmetic_average_is_above_geometric_on_every_path():
    arithmetic, geometric = mc.asian_averages(S0, R, SIGMA, T, 10_000, 52, rng=SEED)
    assert np.all(arithmetic >= geometric - 1e-12)  # AM-GM inequality


def test_asian_control_variate_agrees_and_reduces_the_error():
    plain = mc.price_asian(S0, K, R, SIGMA, T, 50_000, 52, rng=1)
    cv = mc.price_asian(S0, K, R, SIGMA, T, 50_000, 52, control_variate=True, rng=2)
    assert abs(plain["price"] - cv["price"]) < 4 * np.hypot(plain["std_error"], cv["std_error"])
    assert cv["std_error"] < plain["std_error"] / 10
    assert cv["rho"] > 0.99


def test_arithmetic_asian_with_one_date_is_the_european_on_the_same_draws():
    asian = mc.price_asian(S0, K, R, SIGMA, T, 100_000, 1, rng=SEED)
    european = mc.price_european(S0, K, R, SIGMA, T, 100_000, rng=SEED)
    assert asian["price"] == pytest.approx(european["price"], rel=1e-12)


# --- Barrier options ---------------------------------------------------------
def test_barrier_formula_matches_reflection_principle():
    # Independent check: integrate the call payoff against the density of
    # x = ln(S_T / S0) for paths that never touched b = ln(H / S0).
    mu, s, b = R - 0.5 * SIGMA**2, SIGMA * np.sqrt(T), np.log(H / S0)

    def density(x):
        image = np.exp(2 * mu * b / SIGMA**2) * norm.pdf(x - 2 * b, mu * T, s)
        return norm.pdf(x, mu * T, s) - image

    low = np.log(K / S0)
    value, _ = quad(lambda x: (S0 * np.exp(x) - K) * density(x), low, low + 12 * s, limit=200)
    assert mc.barrier_closed_form(S0, K, H, R, SIGMA, T) == pytest.approx(
        np.exp(-R * T) * value, abs=1e-8
    )


def test_knock_in_plus_knock_out_is_the_vanilla_on_the_same_paths():
    out = mc.price_barrier(S0, K, H, R, SIGMA, T, 50_000, 52, "down-and-out", rng=SEED)
    knock_in = mc.price_barrier(S0, K, H, R, SIGMA, T, 50_000, 52, "down-and-in", rng=SEED)
    assert out["price"] + knock_in["price"] == pytest.approx(out["vanilla"], rel=1e-12)


def test_discrete_barrier_bias_and_bgk_correction():
    # A barrier checked at m dates is hit LESS often than a continuous one, so
    # the discrete down-and-out is worth MORE: the continuous formula understates it.
    continuous = mc.barrier_closed_form(S0, K, H, R, SIGMA, T)
    weekly = mc.price_barrier(S0, K, H, R, SIGMA, T, 100_000, 52, rng=SEED)
    assert continuous < weekly["price"] - 4 * weekly["std_error"]
    # With the BGK barrier the closed form is close; 0.01 is left for the
    # small error of the correction itself.
    fine = mc.price_barrier(S0, K, H, R, SIGMA, T, 100_000, 1_000, rng=SEED)
    bgk = mc.barrier_closed_form(S0, K, mc.bgk_barrier(H, SIGMA, T, 1_000), R, SIGMA, T)
    assert abs(fine["price"] - bgk) < 4 * fine["std_error"] + 0.01


# --- VaR and ES --------------------------------------------------------------
def test_cholesky_reproduces_the_covariance():
    n = 200_000
    sample = np.cov(mc.cholesky_normals(COV, n, rng=SEED), rowvar=False)
    # Standard error of a sample covariance for normal data.
    se = np.sqrt((COV**2 + np.outer(np.diag(COV), np.diag(COV))) / n)
    assert np.all(np.abs(sample - COV) < 4 * se)


def test_mc_normal_is_close_to_the_parametric_formula():
    # The formula linearises exp(r) - 1 ~ r, with a relative error of about
    # z * sigma / 2. Volatilities near 0.1% keep this below the 4-SE tolerance.
    small_cov = np.outer(VOLS / 10, VOLS / 10) * CORR
    rng = np.random.default_rng(SEED)
    runs = [mc.mc_normal_var_es(small_cov, WEIGHTS, V, n_sims=100_000, rng=rng) for _ in range(20)]
    exact = mc.parametric_var_es(small_cov, WEIGHTS, V)
    for level in mc.LEVELS:
        for measure in ("var", "es"):
            values = np.array([run[measure][level] for run in runs])
            se = values.std(ddof=1) / np.sqrt(len(values))
            assert abs(values.mean() - exact[measure][level]) < 4 * se


def test_student_t_is_above_the_normal_at_99_but_below_at_95():
    # Same variance, but the t has more mass in the centre and the far tails.
    # Unit-variance t(5) quantiles: 1.561 at 95% (normal 1.645), 2.606 at 99% (normal 2.326).
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((500_000, 3))
    normal = mc.mc_normal_var_es(COV, WEIGHTS, V, Z=Z)
    student = mc.mc_student_t_var_es(COV, WEIGHTS, V, nu=5, Z=Z, rng=rng)
    assert student["var"][0.99] > normal["var"][0.99]
    assert student["es"][0.99] > normal["es"][0.99]
    assert student["var"][0.95] < normal["var"][0.95]


def test_historical_var_is_the_quantile_of_the_raw_losses():
    returns = 0.002 + np.random.default_rng(SEED).standard_t(4, size=(1_000, 3)) * 0.01
    losses = -V * (np.expm1(returns) @ WEIGHTS)
    result = mc.historical_var_es(returns, WEIGHTS, V)
    for level in mc.LEVELS:
        assert result["var"][level] == pytest.approx(np.quantile(losses, level))
        assert result["es"][level] >= result["var"][level] > 0


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        mc.portfolio_losses(np.zeros((2, 3)), [0.5, 0.5, 0.5], V)


def test_parametric_var_scales_with_the_square_root_of_time():
    one_day = mc.parametric_var_es(COV, WEIGHTS, V)
    ten_days = mc.parametric_var_es(COV, WEIGHTS, V, horizon=10)
    assert ten_days["var"][0.99] == pytest.approx(np.sqrt(10) * one_day["var"][0.99])


# --- Backtesting -------------------------------------------------------------
def test_kupiec_hand_computed_case():
    # 5 exceptions in 250 days at p = 1%: pi = 0.02 and
    # LR = 2 [245 ln(0.98 / 0.99) + 5 ln 2].
    exceptions = np.zeros(250, dtype=bool)
    exceptions[:5] = True
    expected = 2 * (245 * math.log(0.98 / 0.99) + 5 * math.log(2))
    assert mc.kupiec_test(exceptions, 0.01)[0] == pytest.approx(expected, rel=1e-12)
    # 10 exceptions in 1000 days is exactly the expected rate: LR = 0, p-value = 1.
    exact_rate = np.zeros(1_000, dtype=bool)
    exact_rate[::100] = True
    assert mc.kupiec_test(exact_rate, 0.01)[1] == pytest.approx(1.0)


def test_christoffersen_rejects_clustered_exceptions_only():
    spread = np.zeros(1_000, dtype=bool)
    spread[5::100] = True
    clustered = np.zeros(1_000, dtype=bool)
    clustered[500:510] = True  # the same 10 exceptions, all consecutive
    assert mc.christoffersen_independence(spread)[1] > 0.05
    assert mc.christoffersen_independence(clustered)[1] < 0.001
    assert mc.christoffersen_cc(clustered, 0.01)[1] < 0.001


def test_traffic_light_boundaries():
    zones = [mc.traffic_light(n) for n in (4, 5, 9, 10)]
    assert zones == ["green", "yellow", "yellow", "red"]


def test_backtest_has_no_look_ahead():
    dates = pd.bdate_range("2010-01-01", periods=700)
    returns = pd.DataFrame(mc.cholesky_normals(COV, 700, rng=SEED), index=dates)
    t = 600
    before = mc.rolling_backtest(returns, WEIGHTS, V, n_sims=2_000, rng=SEED)
    crashed = returns.copy()
    crashed.iloc[t] = -0.30
    after = mc.rolling_backtest(crashed, WEIGHTS, V, n_sims=2_000, rng=SEED)

    day, next_day = dates[t], dates[t + 1]
    forecasts = [c for c in before.columns if c.startswith(("var_", "es_"))]
    # The forecast FOR day t ignores the crash of day t. A tiny tolerance is
    # used because the copy can round differently in the last digit.
    np.testing.assert_allclose(
        after.loc[day, forecasts].to_numpy(dtype=float),
        before.loc[day, forecasts].to_numpy(dtype=float),
        rtol=1e-10,
    )
    assert after.loc[day, "loss"] > before.loc[day, "loss"]
    # The forecast for the NEXT day does see it.
    assert (
        after.loc[next_day, "var_parametric_normal_99"]
        > before.loc[next_day, "var_parametric_normal_99"]
    )
