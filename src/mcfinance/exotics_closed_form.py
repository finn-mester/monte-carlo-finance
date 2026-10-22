"""Closed-form benchmarks for path-dependent options.

- Discretely monitored GEOMETRIC Asian option: exact, because the geometric
  average of lognormal prices is itself lognormal. It validates the Monte
  Carlo engine and serves as a control variate for the arithmetic Asian.
- Continuously monitored down barrier call (Reiner-Rubinstein, as in Hull).
- Broadie-Glasserman-Kou (BGK) continuity correction, which adapts the
  continuous barrier formula to discrete monitoring.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .black_scholes import bs_price

# BGK constant beta = -zeta(1/2) / sqrt(2 pi), where zeta is the Riemann zeta
# function (zeta(1/2) = -1.4603545...). Rounded to 4 decimals as in the paper.
BGK_BETA = 0.5826


def _validate_common(S0: float, K: float, sigma: float, T: float) -> None:
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")


def geometric_average_params(
    S0: float, r: float, sigma: float, T: float, m: int, q: float = 0.0
) -> tuple[float, float]:
    """Mean and variance of ln G, where G = (prod_{i=1..m} S_{t_i})^(1/m).

    With t_i = i T / m and ln S_t = ln S0 + (r - q - sigma^2/2) t + sigma W_t:

    mean: (1/m) sum t_i = T (m + 1) / (2m)
        -> mu_G = ln S0 + (r - q - sigma^2/2) T (m + 1) / (2m)

    variance: Var((1/m) sum W_{t_i}) = (1/m^2) sum_i sum_j min(t_i, t_j)
              = (T / m^3) * m (m + 1) (2m + 1) / 6
        -> var_G = sigma^2 T (m + 1)(2m + 1) / (6 m^2)

    For m = 1 this is ln S_T itself: mu_G = ln S0 + (r - q - sigma^2/2) T and
    var_G = sigma^2 T, so the geometric Asian reduces to the European option.
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    mu_G = np.log(S0) + (r - q - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    var_G = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m**2)
    return float(mu_G), float(var_G)


def geometric_average_expectation(
    S0: float, r: float, sigma: float, T: float, m: int, q: float = 0.0
) -> float:
    """Exact E[G] = exp(mu_G + var_G / 2) under the risk-neutral measure."""
    mu_G, var_G = geometric_average_params(S0, r, sigma, T, m, q)
    return float(np.exp(mu_G + 0.5 * var_G))


def geometric_asian_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    m: int,
    option: str = "call",
    q: float = 0.0,
) -> float:
    """Price of a discretely monitored geometric-average Asian option.

    Since ln G ~ N(mu_G, var_G), this is a Black-Scholes-type formula:

    call = e^{-rT} [ E[G] N(d1) - K N(d2) ]
    put  = e^{-rT} [ K N(-d2) - E[G] N(-d1) ]

    with d1 = (mu_G + var_G - ln K) / sqrt(var_G), d2 = d1 - sqrt(var_G).

    This price is also the exact expectation of the discounted geometric
    payoff, which is what the control variate in exotics.price_asian needs.
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    _validate_common(S0, K, sigma, T)

    mu_G, var_G = geometric_average_params(S0, r, sigma, T, m, q)
    sd_G = np.sqrt(var_G)
    mean_G = np.exp(mu_G + 0.5 * var_G)
    d1 = (mu_G + var_G - np.log(K)) / sd_G
    d2 = d1 - sd_G

    if option == "call":
        undiscounted = mean_G * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        undiscounted = K * norm.cdf(-d2) - mean_G * norm.cdf(-d1)
    return float(np.exp(-r * T) * undiscounted)


def barrier_closed_form(
    S0: float,
    K: float,
    H: float,
    r: float,
    sigma: float,
    T: float,
    kind: str = "down-and-out",
    q: float = 0.0,
) -> float:
    """Continuously monitored down barrier CALL with H <= K (Reiner-Rubinstein).

    With lambda = (r - q + sigma^2/2) / sigma^2 and
    y = ln(H^2 / (S0 K)) / (sigma sqrt(T)) + lambda sigma sqrt(T):

    down-and-in  = S0 e^{-qT} (H/S0)^{2 lambda} N(y)
                   - K e^{-rT} (H/S0)^{2 lambda - 2} N(y - sigma sqrt(T))
    down-and-out = vanilla call - down-and-in   (in-out parity)

    Reference: J. Hull, Options, Futures, and Other Derivatives, chapter on
    exotic options. The tests check it independently by integrating the
    payoff against the absorbed density given by the reflection principle.
    """
    if kind not in ("down-and-out", "down-and-in"):
        raise ValueError(f"kind must be 'down-and-out' or 'down-and-in', got {kind!r}")
    _validate_common(S0, K, sigma, T)
    if H <= 0:
        raise ValueError(f"H must be positive, got {H}")
    if H >= S0:
        raise ValueError(f"H = {H} >= S0 = {S0}: a down barrier at or above the spot is already hit")
    if H > K:
        raise ValueError(
            f"H = {H} > K = {K}: this closed form is only valid for a down barrier "
            "at or below the strike (H <= K)"
        )

    sqrt_T = np.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (S0 * K)) / (sigma * sqrt_T) + lam * sigma * sqrt_T

    down_in = S0 * np.exp(-q * T) * (H / S0) ** (2 * lam) * norm.cdf(y) - K * np.exp(
        -r * T
    ) * (H / S0) ** (2 * lam - 2) * norm.cdf(y - sigma * sqrt_T)

    if kind == "down-and-in":
        return float(down_in)
    return float(bs_price(S0, K, r, sigma, T, "call", q=q) - down_in)


def bgk_adjusted_barrier(
    H: float, sigma: float, T: float, m: int, direction: str = "down"
) -> float:
    """Broadie-Glasserman-Kou continuity correction for discrete monitoring.

    A barrier checked only at m dates is crossed less often than a continuous
    one, so the discrete option behaves like a continuous one whose barrier
    is shifted AWAY from the spot by a factor exp(beta sigma sqrt(T/m)):

        down barrier: H_eff = H exp(-beta sigma sqrt(T/m))
        up barrier:   H_eff = H exp(+beta sigma sqrt(T/m))

    Pricing with the continuous formula at H_eff approximates the discrete
    price with an error of order o(1/sqrt(m)).
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    shift = BGK_BETA * sigma * np.sqrt(T / m)
    if direction == "down":
        return float(H * np.exp(-shift))
    if direction == "up":
        return float(H * np.exp(shift))
    raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
