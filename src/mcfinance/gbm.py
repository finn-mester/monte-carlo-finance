"""Simulation of geometric Brownian motion (GBM) under the risk-neutral measure.

Under the risk-neutral measure Q, the stock price follows

    dS_t = (r - q) S_t dt + sigma S_t dW_t,

whose exact solution is

    S_t = S_0 * exp((r - q - sigma^2 / 2) t + sigma W_t).

Because W_t ~ N(0, t), we can sample S_t exactly at any time, with no
time-discretisation error (unlike an Euler scheme).
"""

from __future__ import annotations

import numpy as np


def _get_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Return a numpy Generator from a Generator, an integer seed, or None."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _validate(S0: float, sigma: float, T: float) -> None:
    """Check the model parameters shared by both simulators."""
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")


def simulate_terminal(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    rng: np.random.Generator | int | None = None,
    q: float = 0.0,
) -> np.ndarray:
    """Sample n independent terminal prices S_T under the risk-neutral measure.

    Parameters
    ----------
    S0 : initial stock price (> 0).
    r : continuously compounded risk-free rate.
    sigma : volatility (>= 0).
    T : maturity in years (> 0).
    n : number of samples.
    rng : numpy Generator, integer seed, or None.
    q : continuous dividend yield.

    Returns
    -------
    Array of shape (n,) with samples of S_T.
    """
    _validate(S0, sigma, T)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    rng = _get_rng(rng)

    Z = rng.standard_normal(n)
    return terminal_from_normals(S0, r, sigma, T, Z, q=q)


def terminal_from_normals(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    Z: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    """Map standard normal draws Z to terminal prices S_T (exact GBM solution).

    Exposed separately so that estimators which need control over the normals
    (e.g. antithetic variates, which reuse Z and -Z) share the same formula.
    """
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S0 * np.exp(drift + diffusion)


def simulate_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator | int | None = None,
    q: float = 0.0,
) -> np.ndarray:
    """Simulate full GBM paths on a uniform time grid.

    Each step uses the exact GBM transition, so the values at the grid points
    have exactly the right distribution regardless of the step size.

    Returns
    -------
    Array of shape (n_paths, n_steps + 1). Column 0 equals S0 and column j
    is the price at time j * T / n_steps.
    """
    _validate(S0, sigma, T)
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be at least 1")
    rng = _get_rng(rng)

    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum of log-increments gives log(S_t / S0) at each grid point.
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths
