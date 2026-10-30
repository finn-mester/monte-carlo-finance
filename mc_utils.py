"""Shared functions for the four notebooks of this project.

The file is organised in sections, in the same order as the notebooks:

    1. Plot style
    2. Simulating stock prices (geometric Brownian motion)
    3. Black-Scholes closed form
    4. Plain Monte Carlo pricing
    5. Variance reduction
    6. Asian options
    7. Barrier options
    8. Value at Risk and Expected Shortfall
    9. Backtesting
    10. Market data

Conventions used everywhere:
    S0, K, H   spot, strike and barrier, in the same currency
    r, q       risk-free rate and dividend yield, per year, continuously compounded
    sigma      annual volatility (0.2 means 20%)
    T          maturity in years
    n          number of simulated paths
    rng        a numpy Generator, an integer seed, or None
               (np.random.default_rng accepts all three)

Pricing functions return a dict: {"price", "std_error", "ci_low", "ci_high"}.
Losses are POSITIVE numbers: a loss of 100 is written +100, a gain is negative.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

# 97.5% quantile of the normal distribution: a 95% confidence interval leaves
# 2.5% in each tail, so it is price +/- 1.96 standard errors.
Z_95 = float(norm.ppf(0.975))

# ---------------------------------------------------------------------------
# 1. Plot style
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
GREY = "#666666"


def set_plot_style():
    """Clean, minimal matplotlib style shared by all figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


# ---------------------------------------------------------------------------
# 2. Simulating stock prices (geometric Brownian motion)
# ---------------------------------------------------------------------------
def terminal_price(S0, r, sigma, T, Z, q=0.0):
    """Stock price at T for standard normal draws Z.

    To price an option we simulate in the risk-neutral world, where the stock
    grows at r - q. Geometric Brownian motion has an exact solution:

        S_T = S0 * exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z)

    so S_T can be sampled directly, with no time-stepping error. The
    -sigma^2 / 2 term keeps the mean of S_T equal to S0 exp((r - q) T).
    """
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_terminal(S0, r, sigma, T, n, q=0.0, rng=None):
    """Draw n independent terminal prices S_T."""
    rng = np.random.default_rng(rng)
    return terminal_price(S0, r, sigma, T, rng.standard_normal(n), q)


def simulate_paths(S0, r, sigma, T, n_paths, n_steps, q=0.0, rng=None):
    """Simulate full price paths; returns an array (n_paths, n_steps + 1).

    Column 0 is S0. Each step uses the exact GBM formula over dt = T / n_steps.
    The whole matrix is kept in memory, so this is meant for a few paths (plots,
    tests). The pricers in sections 6 and 7 keep only what the payoff needs.
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_steps = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    # Adding up the log-returns gives log(S_t / S0) at each date.
    return np.column_stack([np.full(n_paths, S0), S0 * np.exp(np.cumsum(log_steps, axis=1))])


# ---------------------------------------------------------------------------
# 3. Black-Scholes closed form
# ---------------------------------------------------------------------------
def bs_price(S0, K, r, sigma, T, option="call", q=0.0):
    """Black-Scholes price of a European call or put: the exact benchmark.

    call = S0 e^(-qT) N(d1) - K e^(-rT) N(d2)
    put  = K e^(-rT) N(-d2) - S0 e^(-qT) N(-d1)

    N(d2) is the risk-neutral probability that the call ends in the money.
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discounted_S = S0 * np.exp(-q * T)  # stock today, net of the dividends paid until T
    discounted_K = K * np.exp(-r * T)  # value today of paying K at T
    if option == "call":
        return float(discounted_S * norm.cdf(d1) - discounted_K * norm.cdf(d2))
    return float(discounted_K * norm.cdf(-d2) - discounted_S * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# 4. Plain Monte Carlo pricing
# ---------------------------------------------------------------------------
def payoff(S, K, option="call"):
    """Option payoff at expiry: max(S - K, 0) for a call, max(K - S, 0) for a put."""
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def summarise(samples):
    """Monte Carlo estimate from i.i.d. samples: mean, standard error, 95% CI.

    By the central limit theorem the sample mean is close to normal, with
    standard error s / sqrt(n). The samples must be independent.
    """
    price = float(np.mean(samples))
    std_error = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    return {
        "price": price,
        "std_error": std_error,
        "ci_low": price - Z_95 * std_error,
        "ci_high": price + Z_95 * std_error,
    }


def price_european(S0, K, r, sigma, T, n, option="call", q=0.0, rng=None):
    """Plain Monte Carlo: the option price is the discounted average payoff.

    Risk-neutral pricing says V0 = exp(-rT) E[payoff(S_T)], and the sample mean
    of n simulated payoffs estimates that expectation.
    """
    S_T = simulate_terminal(S0, r, sigma, T, n, q, rng)
    return summarise(np.exp(-r * T) * payoff(S_T, K, option))


def convergence_curve(S0, K, r, sigma, T, n_max, option="call", q=0.0, n_points=60, rng=None):
    """Running estimate as the sample grows: one long sample read at growing n.

    Returns (ns, prices, std_errors), with ns log-spaced between 100 and n_max.
    """
    S_T = simulate_terminal(S0, r, sigma, T, n_max, q, rng)
    x = np.exp(-r * T) * payoff(S_T, K, option)
    ns = np.unique(np.geomspace(100, n_max, n_points).round().astype(int))
    # Cumulative sums give the mean and variance of the first n samples.
    sums, sums_of_squares = np.cumsum(x)[ns - 1], np.cumsum(x**2)[ns - 1]
    prices = sums / ns
    variances = (sums_of_squares - ns * prices**2) / (ns - 1)
    return ns, prices, np.sqrt(variances / ns)


# ---------------------------------------------------------------------------
# 5. Variance reduction
# ---------------------------------------------------------------------------
# In all three methods n is the TOTAL number of payoff evaluations, so that
# they are compared with plain Monte Carlo at the same budget.
def price_antithetic(S0, K, r, sigma, T, n, option="call", q=0.0, rng=None):
    """Antithetic variates: use each draw twice, as Z and as -Z.

    When Z gives a high stock price, -Z gives a low one, so the two payoffs
    partly offset each other and their average is less noisy.
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even for antithetic sampling, got {n}")
    rng = np.random.default_rng(rng)
    Z = rng.standard_normal(n // 2)
    up = payoff(terminal_price(S0, r, sigma, T, Z, q), K, option)
    down = payoff(terminal_price(S0, r, sigma, T, -Z, q), K, option)
    # The standard error must use the n/2 PAIR averages, which are independent.
    # The n payoffs are not: the two members of a pair are negatively correlated.
    return summarise(np.exp(-r * T) * 0.5 * (up + down))


def control_variate_adjust(X, Y, mean_Y):
    """Control variate estimate of E[X], using Y whose exact mean is known.

    On a given path, the error of Y is known (Y - mean_Y). If X and Y are
    correlated, X probably made a similar error, so we subtract it:

        X_adj = X - beta (Y - mean_Y),  with beta = Cov(X, Y) / Var(Y)

    The variance is divided by 1 / (1 - rho^2). beta is estimated on the same
    sample, which adds a bias of order 1/n, negligible next to the 1/sqrt(n)
    statistical error.
    """
    beta = np.cov(X, Y, ddof=1)[0, 1] / np.var(Y, ddof=1)
    result = summarise(X - beta * (Y - mean_Y))
    result["beta"] = float(beta)
    result["rho"] = float(np.corrcoef(X, Y)[0, 1])
    return result


def price_control_variate(S0, K, r, sigma, T, n, option="call", q=0.0, rng=None):
    """Control variate: the discounted stock price, Y = exp(-rT) S_T.

    Its exact mean is S0 exp(-qT): the discounted stock is a martingale under
    the risk-neutral measure (up to the dividends). So we know how far each
    simulated sample is from the truth.
    """
    S_T = simulate_terminal(S0, r, sigma, T, n, q, rng)
    discount = np.exp(-r * T)
    return control_variate_adjust(
        discount * payoff(S_T, K, option), discount * S_T, S0 * np.exp(-q * T)
    )


def price_combined(S0, K, r, sigma, T, n, option="call", q=0.0, rng=None):
    """Both methods together: the control variate applied to antithetic pair averages."""
    if n % 2 != 0:
        raise ValueError(f"n must be even for antithetic sampling, got {n}")
    rng = np.random.default_rng(rng)
    Z = rng.standard_normal(n // 2)
    S_up = terminal_price(S0, r, sigma, T, Z, q)
    S_down = terminal_price(S0, r, sigma, T, -Z, q)
    discount = np.exp(-r * T)
    X_pairs = discount * 0.5 * (payoff(S_up, K, option) + payoff(S_down, K, option))
    Y_pairs = discount * 0.5 * (S_up + S_down)  # still has mean S0 exp(-qT)
    return control_variate_adjust(X_pairs, Y_pairs, S0 * np.exp(-q * T))


def price_variance(pricer, n_reps, rng, **params):
    """Variance of a pricer's estimate, measured over n_reps independent runs.

    This measures the true spread of the price, so it does not rely on the
    standard error reported by the pricer. Variance reduction factor of a method:
    price_variance(plain) / price_variance(method), at the same n.
    """
    prices = [pricer(**params, rng=rng)["price"] for _ in range(n_reps)]
    return float(np.var(prices, ddof=1))


# ---------------------------------------------------------------------------
# 6. Asian options
# ---------------------------------------------------------------------------
# Monitoring dates are t_i = i T / m for i = 1..m. S0 itself is not a date.
def geometric_asian_price(S0, K, r, sigma, T, m, option="call", q=0.0):
    """Exact price of an Asian option on the GEOMETRIC average of m dates.

    The log of the geometric average G is an average of normal variables, so it
    is normal, with

        mean     = ln S0 + (r - q - sigma^2 / 2) T (m + 1) / (2m)
        variance = sigma^2 T (m + 1)(2m + 1) / (6 m^2)

    and the price has a Black-Scholes form. With m = 1, G is S_T and this IS
    Black-Scholes. As m grows the variance falls: averaging lowers the volatility.
    """
    mean = np.log(S0) + (r - q - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    variance = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m**2)
    expected_G = np.exp(mean + 0.5 * variance)
    d1 = (mean + variance - np.log(K)) / np.sqrt(variance)
    d2 = d1 - np.sqrt(variance)
    if option == "call":
        return float(np.exp(-r * T) * (expected_G * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(np.exp(-r * T) * (K * norm.cdf(-d2) - expected_G * norm.cdf(-d1)))


def asian_averages(S0, r, sigma, T, n, m, q=0.0, rng=None):
    """Arithmetic and geometric averages of n paths over m dates.

    All paths move forward one date at a time. Only running sums are kept, so
    memory does not grow with m. The average of the logs gives the geometric
    average without multiplying hundreds of prices together.
    """
    rng = np.random.default_rng(rng)
    dt = T / m
    drift, vol = (r - q - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)
    log_S = np.full(n, np.log(S0))
    sum_S, sum_log_S = np.zeros(n), np.zeros(n)
    for _ in range(m):
        log_S += drift + vol * rng.standard_normal(n)
        sum_S += np.exp(log_S)
        sum_log_S += log_S
    return sum_S / m, np.exp(sum_log_S / m)


def price_asian(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    m,
    option="call",
    average="arithmetic",
    control_variate=False,
    q=0.0,
    rng=None,
):
    """Monte Carlo price of an Asian option, payoff(average, K).

    The arithmetic Asian has no closed form: a sum of lognormals is not
    lognormal. With control_variate=True the geometric Asian on the same paths
    is the control. Its exact price is known, and on each path both averages are
    almost equal (rho close to 1), so nearly all the noise is removed.
    """
    arithmetic, geometric = asian_averages(S0, r, sigma, T, n, m, q, rng)
    discount = np.exp(-r * T)
    Y = discount * payoff(geometric, K, option)
    if average == "geometric":
        return summarise(Y)
    X = discount * payoff(arithmetic, K, option)
    if control_variate:
        return control_variate_adjust(X, Y, geometric_asian_price(S0, K, r, sigma, T, m, option, q))
    return summarise(X)


# ---------------------------------------------------------------------------
# 7. Barrier options
# ---------------------------------------------------------------------------
# -zeta(1/2) / sqrt(2 pi), where zeta is the Riemann zeta function.
BGK_BETA = 0.5826


def barrier_closed_form(S0, K, H, r, sigma, T, kind="down-and-out", q=0.0):
    """Down barrier call with CONTINUOUS monitoring and H <= K (Reiner-Rubinstein).

    A down-and-in call only exists if the stock touches H. Together, a
    down-and-in and a down-and-out always pay exactly one vanilla call, so
    down-and-out = vanilla - down-and-in (in-out parity).
    """
    if H >= S0 or H > K:
        raise ValueError(f"the formula needs H < S0 and H <= K, got H={H}, S0={S0}, K={K}")
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    vol = sigma * np.sqrt(T)
    y = np.log(H**2 / (S0 * K)) / vol + lam * vol
    down_in = S0 * np.exp(-q * T) * (H / S0) ** (2 * lam) * norm.cdf(y) - K * np.exp(-r * T) * (
        H / S0
    ) ** (2 * lam - 2) * norm.cdf(y - vol)
    if kind == "down-and-in":
        return float(down_in)
    return float(bs_price(S0, K, r, sigma, T, "call", q) - down_in)


def bgk_barrier(H, sigma, T, m):
    """Broadie-Glasserman-Kou correction: the barrier to use for m monitoring dates.

    A barrier checked at m dates is hit less often than a continuous one, as
    if it were further from the spot. So we can reuse the continuous formula
    with the barrier moved away from S0: H exp(-0.5826 sigma sqrt(T / m)).
    """
    return float(H * np.exp(-BGK_BETA * sigma * np.sqrt(T / m)))


def price_barrier(S0, K, H, r, sigma, T, n, m, kind="down-and-out", q=0.0, rng=None):
    """Monte Carlo price of a down barrier call, checked at m dates.

    A path is knocked out (or in) if any date has S <= H. The exact GBM step is
    used between dates, so the only bias compared with the continuous formula
    is the discrete monitoring itself, which is what the contract specifies.

    "vanilla" in the result is the plain call on the same paths (in-out parity).
    """
    if H >= S0:
        raise ValueError(f"H = {H} >= S0 = {S0}: the barrier is already hit")
    rng = np.random.default_rng(rng)
    dt = T / m
    drift, vol = (r - q - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)
    log_S = np.full(n, np.log(S0))
    hit = np.zeros(n, dtype=bool)
    for _ in range(m):
        log_S += drift + vol * rng.standard_normal(n)
        hit |= log_S <= np.log(H)  # once hit, a path stays hit
    vanilla = np.exp(-r * T) * payoff(np.exp(log_S), K, "call")
    alive = ~hit if kind == "down-and-out" else hit
    result = summarise(vanilla * alive)
    result["vanilla"] = float(vanilla.mean())
    return result


# ---------------------------------------------------------------------------
# 8. Value at Risk and Expected Shortfall
# ---------------------------------------------------------------------------
# VaR(a) is the a-quantile of the loss: on a fraction a of days we lose less.
# ES(a) is the average loss on the remaining days, beyond the VaR.
# The mean return is set to zero: over one day it is tiny next to the
# volatility, and estimating it from data adds noise rather than information.
LEVELS = (0.95, 0.975, 0.99)


def portfolio_losses(log_returns, weights, value):
    """Exact portfolio loss for each row of daily log returns.

    Stocks compound, so a log return r is a simple return exp(r) - 1:
    loss = -V * sum_i w_i (exp(r_i) - 1).
    """
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {weights.sum()}")
    return -value * (np.expm1(log_returns) @ weights)


def var_es(losses, levels=LEVELS):
    """Empirical VaR and ES of a set of losses, as {"var": {level: x}, "es": {level: x}}."""
    result = {"var": {}, "es": {}}
    for level in levels:
        quantile = float(np.quantile(losses, level))
        result["var"][level] = quantile
        result["es"][level] = float(losses[losses >= quantile].mean())
    return result


def cholesky_normals(cov, n, rng=None):
    """Draw n zero-mean normal vectors with covariance cov (shape (n, d)).

    Write cov = L L^T. If z has independent standard normal entries, then
    x = L z has covariance L I L^T = cov: multiplying by L turns independent
    shocks into shocks with the right volatilities and correlations.
    Stacked in rows, x = L z becomes X = Z L^T.
    """
    rng = np.random.default_rng(rng)
    L = np.linalg.cholesky(cov)
    return rng.standard_normal((n, len(cov))) @ L.T


def ewma_covariance(log_returns, lam=0.94):
    """Exponentially weighted covariance (RiskMetrics): recent days count more.

    The last return has weight (1 - lam), the one before (1 - lam) lam, and so
    on, rescaled to sum to 1. The memory is about 1 / (1 - lam) = 17 days, so
    the estimate reacts quickly when volatility changes.
    """
    r = np.asarray(log_returns, dtype=float)
    ages = np.arange(len(r))[::-1]  # 0 for the most recent row
    weights = (1 - lam) * lam**ages
    weights /= weights.sum()
    return (r * weights[:, None]).T @ r


def parametric_var_es(cov, weights, value, levels=LEVELS, horizon=1):
    """Closed-form VaR and ES for normal returns (delta-normal method).

    Approximation: the portfolio return is linearised to sum_i w_i r_i, which
    is normal with sigma_p = sqrt(w' cov w), and then
        VaR = V z sigma_p           ES = V sigma_p phi(z) / (1 - level)
    Over h days the covariance is multiplied by h (square-root-of-time rule),
    which assumes i.i.d. returns.
    """
    weights = np.asarray(weights, dtype=float)
    sigma_p = float(np.sqrt(weights @ np.asarray(cov) @ weights * horizon))
    result = {"var": {}, "es": {}}
    for level in levels:
        z = norm.ppf(level)
        result["var"][level] = value * z * sigma_p
        result["es"][level] = value * sigma_p * norm.pdf(z) / (1 - level)
    return result


def mc_normal_var_es(
    cov, weights, value, levels=LEVELS, n_sims=100_000, horizon=1, Z=None, rng=None
):
    """Monte Carlo VaR and ES with multivariate normal returns.

    Same model as parametric_var_es, but each scenario is revalued exactly.
    Passing Z (standard normals, shape (n_sims, d)) reuses the same draws
    across calls; this is what the backtest does.
    """
    L = np.linalg.cholesky(cov)
    if Z is None:
        Z = np.random.default_rng(rng).standard_normal((n_sims, L.shape[0]))
    X = np.sqrt(horizon) * (Z @ L.T)
    return var_es(portfolio_losses(X, weights, value), levels)


def mc_student_t_var_es(
    cov, weights, value, levels=LEVELS, nu=5, n_sims=100_000, horizon=1, Z=None, W=None, rng=None
):
    """Monte Carlo VaR and ES with multivariate Student-t returns (fat tails).

    X = sqrt((nu - 2) / nu) * L z / sqrt(W), with W ~ chi2(nu) / nu.

    L z / sqrt(W) has covariance cov * nu / (nu - 2). The first factor brings
    it back to exactly cov, so the normal and t models have the same volatilities
    and correlations and differ only in the shape of the tails.

    One W scales all assets at once: a small W is a turbulent day for every
    stock. This is tail dependence, which the multivariate normal does not have.
    """
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(rng)
    if Z is None:
        Z = rng.standard_normal((n_sims, L.shape[0]))
    if W is None:
        W = rng.chisquare(nu, size=len(Z)) / nu
    scale = np.sqrt((nu - 2) / nu)
    X = np.sqrt(horizon) * scale * (Z @ L.T) / np.sqrt(W)[:, None]
    return var_es(portfolio_losses(X, weights, value), levels)


def historical_var_es(log_returns, weights, value, levels=LEVELS):
    """Historical simulation: apply each past day's RAW returns to today's portfolio.

    No distribution is assumed and nothing is simulated or demeaned: VaR and
    ES come directly from the losses the window would have produced.
    """
    return var_es(portfolio_losses(log_returns, weights, value), levels)


# ---------------------------------------------------------------------------
# 9. Backtesting
# ---------------------------------------------------------------------------
def rolling_backtest(
    returns,
    weights,
    value,
    levels=(0.95, 0.99),
    window=500,
    n_sims=20_000,
    nu=5,
    lam=0.94,
    rng=None,
):
    """Forecast VaR and ES every day and compare with the realised loss.

    For day t, the models only see the previous `window` days, never day t
    itself (no look-ahead). An exception is a day when the loss exceeds the VaR.

    The normals Z and chi-squares W are drawn ONCE and reused every day (common
    random numbers); only the covariance changes. This is fast and removes
    simulation noise from the VaR series, but it makes the Monte Carlo error of
    consecutive days correlated.

    Returns one row per day, with columns loss, var_<method>_<level>,
    es_<method>_<level> and exc_<method>_<level>.
    """
    rng = np.random.default_rng(rng)
    r = returns.to_numpy()
    Z = rng.standard_normal((n_sims, r.shape[1]))
    W = rng.chisquare(nu, size=n_sims) / nu
    realised = portfolio_losses(r, weights, value)

    rows = []
    for t in range(window, len(r)):
        past = r[t - window : t]  # stops at t - 1
        cov = np.cov(past, rowvar=False)
        forecasts = {
            "parametric_normal": parametric_var_es(cov, weights, value, levels),
            "mc_normal": mc_normal_var_es(cov, weights, value, levels, Z=Z),
            "mc_student_t": mc_student_t_var_es(cov, weights, value, levels, nu, Z=Z, W=W),
            "mc_normal_ewma": mc_normal_var_es(
                ewma_covariance(past, lam), weights, value, levels, Z=Z
            ),
            "historical": historical_var_es(past, weights, value, levels),
        }
        row = {"loss": realised[t]}
        for method, forecast in forecasts.items():
            for level in levels:
                key = f"{method}_{100 * level:g}"
                row[f"var_{key}"] = forecast["var"][level]
                row[f"es_{key}"] = forecast["es"][level]
                row[f"exc_{key}"] = realised[t] > forecast["var"][level]
        rows.append(row)
    return pd.DataFrame(rows, index=returns.index[window:])


def kupiec_test(exceptions, p):
    """Kupiec test: is the exception rate equal to p? Returns (LR, p-value).

    Compares the likelihood of the data under p and under the observed rate
    pi = x / N. xlogy applies the convention 0 * log(0) = 0, so that 0
    exceptions works. The statistic is chi-square with 1 degree of freedom.
    """
    exceptions = np.asarray(exceptions, dtype=bool)
    N, x = len(exceptions), int(exceptions.sum())
    pi = x / N
    log_l_p = xlogy(N - x, 1 - p) + xlogy(x, p)
    log_l_pi = xlogy(N - x, 1 - pi) + xlogy(x, pi)
    lr = max(-2 * (log_l_p - log_l_pi), 0.0)
    return float(lr), float(chi2.sf(lr, df=1))


def transition_counts(exceptions):
    """(n00, n01, n10, n11): n01 counts days without an exception followed by one."""
    e = np.asarray(exceptions, dtype=bool)
    before, after = e[:-1], e[1:]
    return (
        int(np.sum(~before & ~after)),
        int(np.sum(~before & after)),
        int(np.sum(before & ~after)),
        int(np.sum(before & after)),
    )


def christoffersen_independence(exceptions):
    """Christoffersen test: does an exception today make one tomorrow likelier?

    Compares pi01 = P(exception | none yesterday) and pi11 = P(exception |
    exception yesterday) with a single rate pi. Clustered exceptions give
    pi11 >> pi01. The statistic is chi-square with 1 degree of freedom.
    Returns (LR, p-value).
    """
    n00, n01, n10, n11 = transition_counts(exceptions)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_l_one_rate = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l_two_rates = (
        xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    )
    lr = max(-2 * (log_l_one_rate - log_l_two_rates), 0.0)
    return float(lr), float(chi2.sf(lr, df=1))


def christoffersen_cc(exceptions, p):
    """Conditional coverage: Kupiec + independence, chi-square with 2 degrees of freedom."""
    lr = kupiec_test(exceptions, p)[0] + christoffersen_independence(exceptions)[0]
    return float(lr), float(chi2.sf(lr, df=2))


def traffic_light(n_exceptions):
    """Basel (1996) zones for 99% VaR over 250 days, used here as an illustration."""
    if n_exceptions <= 4:
        return "green"
    return "yellow" if n_exceptions <= 9 else "red"


def backtest_report(bt, methods, levels=(0.99, 0.95)):
    """Table of backtest statistics, one row per method and level.

    If the ES is right, the average loss on exception days should be close to
    the average ES predicted for those days.
    """
    rows = []
    for level in levels:
        p = 1 - level
        for method in methods:
            key = f"{method}_{100 * level:g}"
            e = bt[f"exc_{key}"].to_numpy(dtype=bool)
            last_250 = int(e[-250:].sum())
            rows.append(
                {
                    "method": method,
                    "level": level,
                    "exceptions": int(e.sum()),
                    "expected": len(e) * p,
                    "rate": e.mean(),
                    "kupiec_p": kupiec_test(e, p)[1],
                    "independence_p": christoffersen_independence(e)[1],
                    "cc_p": christoffersen_cc(e, p)[1],
                    "last_250": last_250,
                    "zone": traffic_light(last_250) if level == 0.99 else "",
                    "avg_loss_on_exceptions": bt.loc[e, "loss"].mean(),
                    "avg_predicted_es": bt.loc[e, f"es_{key}"].mean(),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. Market data
# ---------------------------------------------------------------------------
# Five CAC 40 stocks from different sectors, all quoted in EUR.
TICKERS = ["MC.PA", "TTE.PA", "SAN.PA", "BNP.PA", "AIR.PA"]


def load_prices(path="data/prices.csv", start="2012-01-01"):
    """Read daily adjusted close prices, downloading them the first time.

    Yahoo Finance data should not be redistributed, so data/ is not committed.
    If the download fails this function raises an error: it never falls back
    on made-up prices.
    """
    path = Path(path)
    if not path.exists():
        import yfinance as yf

        raw = yf.download(TICKERS, start=start, auto_adjust=True, progress=False, threads=False)
        prices = raw["Close"][TICKERS].dropna(how="any")  # keep days with all five prices
        if prices.empty:
            raise RuntimeError("yfinance returned no data, check the network and the tickers")
        prices.index.name = "Date"
        path.parent.mkdir(exist_ok=True)
        prices.to_csv(path)
    return pd.read_csv(path, index_col="Date", parse_dates=True)
