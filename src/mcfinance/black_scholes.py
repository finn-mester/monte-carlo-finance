"""Closed-form Black-Scholes(-Merton) prices for European options.

These prices are the benchmark used to validate every Monte Carlo estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes price of a European call or put with dividend yield q.

    call = S0 e^{-qT} N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S0 e^{-qT} N(-d1)

    with d1 = [ln(S0/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)) and
    d2 = d1 - sigma sqrt(T).

    Returns
    -------
    The option price (same currency unit as S0 and K). Raises ValueError for an
    unknown option type or non-positive S0, K, sigma or T.
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc_S = S0 * np.exp(-q * T)  # present value of the stock net of dividends
    disc_K = K * np.exp(-r * T)  # present value of the strike

    if option == "call":
        price = disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    else:
        price = disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    return float(price)
