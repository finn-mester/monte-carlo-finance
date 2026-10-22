"""Variance reduction for Monte Carlo pricing of European options.

Plain Monte Carlo has standard error s / sqrt(n): to halve the error we need
four times as many paths. Variance reduction techniques instead shrink s, the
standard deviation of each observation, while keeping the estimator unbiased
(or almost unbiased).

Every estimator here returns an MCResult, and n is always the TOTAL number of
payoff evaluations. This way all methods are compared at an equal budget.

Methods
-------
- Antithetic variates: pair each normal draw Z with -Z.
- Control variate: correct the payoff with the discounted terminal price,
  whose expectation is known exactly.
- Combined: control variate applied to antithetic pair averages.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .gbm import _get_rng, _validate, terminal_from_normals
from .pricing import MCResult, payoff


@dataclass(frozen=True)
class ControlVariateResult(MCResult):
    """MCResult with control-variate diagnostics.

    Attributes
    ----------
    beta : coefficient used in X_adj = X - beta * (Y - E[Y]).
    rho : sample correlation between the payoff X and the control Y. The
        theoretical variance reduction factor versus plain MC is 1 / (1 - rho^2).
    """

    beta: float
    rho: float

    @property
    def predicted_factor(self) -> float:
        """Theoretical variance reduction factor 1 / (1 - rho^2)."""
        return 1.0 / (1.0 - self.rho**2)


def control_variate_estimate(
    X: np.ndarray,
    Y: np.ndarray,
    mean_Y: float,
    beta: float | None = None,
) -> ControlVariateResult:
    """Control-variate estimate of E[X] from paired samples (X_i, Y_i).

    X_adj_i = X_i - beta * (Y_i - E[Y])

    E[X_adj] = E[X] for any fixed beta. The variance is minimised by
    beta* = Cov(X, Y) / Var(Y), which leaves Var(X) * (1 - rho^2).

    If beta is None it is estimated from the same sample. That makes beta
    slightly dependent on the data and introduces a bias of order 1/n, while the
    standard error is of order 1/sqrt(n). For the sample sizes used here
    (thousands of paths or more) the bias is negligible next to the statistical
    error, which is why this standard practice is acceptable. A pilot run to
    estimate beta would remove the bias at the cost of extra simulations.

    The returned n is the number of observations len(X). Callers whose
    observations are themselves averages (antithetic pairs) override it.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape or X.ndim != 1:
        raise ValueError("X and Y must be 1-D arrays of the same length")
    m = len(X)
    if m < 2:
        raise ValueError(f"need at least 2 observations, got {m}")

    var_Y = Y.var(ddof=1)
    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
    var_X = X.var(ddof=1)
    if beta is None:
        if var_Y == 0:
            raise ValueError("the control variate has zero variance, beta is undefined")
        beta = cov_XY / var_Y
    rho = cov_XY / np.sqrt(var_X * var_Y) if var_X > 0 and var_Y > 0 else 0.0

    adjusted = X - beta * (Y - mean_Y)
    price = adjusted.mean()
    std_error = adjusted.std(ddof=1) / np.sqrt(m)
    return ControlVariateResult(
        price=float(price), std_error=float(std_error), n=m, beta=float(beta), rho=float(rho)
    )


def _check_even(n: int) -> None:
    """Raise ValueError unless n is an even number >= 4."""
    if n < 4 or n % 2 != 0:
        raise ValueError(f"n must be an even number >= 4 for antithetic sampling, got {n}")


def _antithetic_pairs(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    q: float,
    rng: np.random.Generator | int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Terminal prices for n/2 normals Z and their mirrors -Z."""
    _validate(S0, sigma, T)
    _check_even(n)
    rng = _get_rng(rng)
    Z = rng.standard_normal(n // 2)
    S_plus = terminal_from_normals(S0, r, sigma, T, Z, q=q)
    S_minus = terminal_from_normals(S0, r, sigma, T, -Z, q=q)
    return S_plus, S_minus


def price_european_antithetic(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    option: str = "call",
    q: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> MCResult:
    """Price a European option with antithetic variates.

    n (even) payoffs are evaluated from n/2 normal draws: each Z is used once
    as Z and once as -Z, and the two payoffs of a pair are averaged.

    Returns
    -------
    MCResult whose n is the total number of payoff evaluations and whose
    standard error is computed from the n/2 pair averages.
    """
    S_plus, S_minus = _antithetic_pairs(S0, r, sigma, T, n, q, rng)
    disc = np.exp(-r * T)
    pair_avg = 0.5 * disc * (payoff(S_plus, K, option) + payoff(S_minus, K, option))

    # The n/2 pair averages are i.i.d., but the two payoffs inside a pair are
    # not: they are negatively correlated by construction (that is the whole
    # point). Treating all n payoffs as independent would give a wrong standard
    # error, so the standard error is computed from the pair averages only.
    m = n // 2
    price = pair_avg.mean()
    std_error = pair_avg.std(ddof=1) / np.sqrt(m)
    return MCResult(price=float(price), std_error=float(std_error), n=n)


def price_european_control_variate(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    option: str = "call",
    q: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> ControlVariateResult:
    """Price a European option using the discounted terminal price as control.

    Control variate Y = e^{-rT} S_T. Under the risk-neutral measure the
    discounted asset (with dividends reinvested) is a martingale, so
    E[Y] = S0 e^{-qT} exactly. Beta is estimated from the same sample.

    Returns
    -------
    ControlVariateResult (price, standard error, n, beta, rho).
    """
    if n < 2:
        raise ValueError(f"n must be at least 2 to estimate a standard error, got {n}")
    _validate(S0, sigma, T)
    rng = _get_rng(rng)

    Z = rng.standard_normal(n)
    S_T = terminal_from_normals(S0, r, sigma, T, Z, q=q)
    disc = np.exp(-r * T)
    X = disc * payoff(S_T, K, option)
    Y = disc * S_T
    return control_variate_estimate(X, Y, mean_Y=S0 * np.exp(-q * T))


def price_european_combined(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    option: str = "call",
    q: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> ControlVariateResult:
    """Antithetic variates and the control variate together.

    The control variate is applied to the antithetic pair averages: the
    observations are the n/2 i.i.d. pairs (X_pair, Y_pair) with
    X_pair = (X(Z) + X(-Z)) / 2 and Y_pair = (Y(Z) + Y(-Z)) / 2. Y_pair still
    has the known expectation S0 e^{-qT}.

    Returns
    -------
    ControlVariateResult with n = total number of payoff evaluations.
    """
    S_plus, S_minus = _antithetic_pairs(S0, r, sigma, T, n, q, rng)
    disc = np.exp(-r * T)
    X_pair = 0.5 * disc * (payoff(S_plus, K, option) + payoff(S_minus, K, option))
    Y_pair = 0.5 * disc * (S_plus + S_minus)

    result = control_variate_estimate(X_pair, Y_pair, mean_Y=S0 * np.exp(-q * T))
    # Report the total number of payoff evaluations, not the number of pairs.
    return dataclasses.replace(result, n=n)


def estimator_variance(
    estimator: Callable[..., MCResult],
    n_reps: int,
    rng: np.random.Generator | int | None = None,
    **kwargs,
) -> float:
    """Empirical variance of an estimator's price over independent runs.

    Runs estimator(**kwargs, rng=rng) n_reps times and returns the sample
    variance (ddof=1) of the prices. The variance reduction factor of a method
    versus plain MC at the same n is then
        estimator_variance(price_european, ...) / estimator_variance(method, ...).
    """
    if n_reps < 2:
        raise ValueError(f"n_reps must be at least 2, got {n_reps}")
    rng = _get_rng(rng)
    prices = np.array([estimator(**kwargs, rng=rng).price for _ in range(n_reps)])
    return float(prices.var(ddof=1))


def time_per_call(
    estimator: Callable[..., MCResult],
    n_calls: int = 200,
    n_repeats: int = 5,
    rng: np.random.Generator | int | None = None,
    **kwargs,
) -> float:
    """Wall-clock seconds per call of estimator(**kwargs, rng=rng).

    The estimator is called n_calls times in a row, this block is repeated
    n_repeats times, and the fastest block is kept (as Python's timeit does):
    the minimum is the measurement least affected by other processes.

    Returns
    -------
    Seconds per call (float).
    """
    rng = _get_rng(rng)
    timings = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        for _ in range(n_calls):
            estimator(**kwargs, rng=rng)
        timings.append(time.perf_counter() - start)
    return min(timings) / n_calls
