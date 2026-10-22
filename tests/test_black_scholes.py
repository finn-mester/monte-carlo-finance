"""Tests for the closed-form Black-Scholes benchmark."""

import numpy as np
import pytest

from mcfinance.black_scholes import bs_price

# Reference contract used throughout the project.
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0


def test_reference_values():
    # Textbook values for the at-the-money contract (Hull).
    assert bs_price(S0, K, R, SIGMA, T, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(S0, K, R, SIGMA, T, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("q", [0.0, 0.02, 0.05])
def test_put_call_parity(strike, q):
    # C - P = S0 e^{-qT} - K e^{-rT}
    call = bs_price(S0, strike, R, SIGMA, T, "call", q=q)
    put = bs_price(S0, strike, R, SIGMA, T, "put", q=q)
    expected = S0 * np.exp(-q * T) - strike * np.exp(-R * T)
    assert call - put == pytest.approx(expected, abs=1e-10)


def test_call_increasing_in_volatility():
    vols = [0.05, 0.1, 0.2, 0.3, 0.5]
    prices = [bs_price(S0, K, R, v, T, "call") for v in vols]
    assert all(a < b for a, b in zip(prices, prices[1:]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(option="straddle"),
        dict(sigma=0.0),
        dict(sigma=-0.1),
        dict(T=0.0),
        dict(S0=0.0),
        dict(K=-1.0),
    ],
)
def test_invalid_inputs_raise(kwargs):
    params = dict(S0=S0, K=K, r=R, sigma=SIGMA, T=T, option="call")
    params.update(kwargs)
    with pytest.raises(ValueError):
        bs_price(**params)
