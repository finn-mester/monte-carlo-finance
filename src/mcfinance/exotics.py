"""Monte Carlo pricing of path-dependent options: Asian and barrier calls.

Conventions
-----------
Monitoring dates are t_i = i T / m for i = 1..m, with m = n_steps. The
initial price S0 is NOT a monitoring date: it is neither part of the average
nor checked against the barrier.

Paths are generated with simulate_paths, which uses the exact GBM transition
between dates. The path values at the monitoring dates therefore have exactly
the right joint distribution: there is no time-discretisation error. For the
barrier, the only bias is that the barrier is checked at m dates only
(discrete monitoring), which is the contract being priced.

Memory
------
n paths x m steps at once can be large (1e5 x 252 floats is about 200 MB).
Paths are simulated in batches of batch_size; only the per-path discounted
payoffs are kept, and the mean and standard error are computed at the end.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .exotics_closed_form import geometric_asian_price
from .gbm import _get_rng, simulate_paths
from .pricing import MCResult, payoff
from .variance_reduction import control_variate_estimate


def path_averages(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic and geometric averages over the monitoring dates of each path.

    paths has shape (n_paths, m + 1) as returned by simulate_paths; column 0
    (S0) is excluded. The geometric average is computed as exp(mean(log S))
    for numerical stability (no product of 252 numbers).
    """
    monitored = paths[:, 1:]
    arithmetic = monitored.mean(axis=1)
    geometric = np.exp(np.log(monitored).mean(axis=1))
    return arithmetic, geometric


def _simulate_in_batches(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    n_steps: int,
    q: float,
    rng: np.random.Generator | int | None,
    batch_size: int,
    payoffs_fn: Callable[[np.ndarray], tuple[np.ndarray, ...]],
) -> tuple[np.ndarray, ...]:
    """Simulate n paths batch by batch and keep only per-path payoffs.

    payoffs_fn maps a batch of paths (shape (b, n_steps + 1)) to a tuple of
    per-path arrays (e.g. the discounted payoff and a control variate). The
    arrays from all batches are concatenated in order.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2 to estimate a standard error, got {n}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    rng = _get_rng(rng)

    chunks: list[tuple[np.ndarray, ...]] = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        paths = simulate_paths(S0, r, sigma, T, b, n_steps, rng=rng, q=q)
        chunks.append(payoffs_fn(paths))
        remaining -= b
    return tuple(np.concatenate(parts) for parts in zip(*chunks, strict=True))


def price_asian(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    n_steps: int,
    option: str = "call",
    average: str = "arithmetic",
    control_variate: bool = False,
    q: float = 0.0,
    rng: np.random.Generator | int | None = None,
    batch_size: int = 20_000,
) -> MCResult:
    """Price a discretely monitored Asian option on the average of S_{t_1..t_m}.

    Payoff: max(A - K, 0) for a call, max(K - A, 0) for a put, where A is the
    arithmetic or geometric average over the n_steps monitoring dates.

    With control_variate=True (arithmetic average only), the discounted
    GEOMETRIC Asian payoff on the same paths is used as control variate. Its
    exact expectation is geometric_asian_price, and beta is estimated from the
    sample exactly as in Part 2. The result is then a ControlVariateResult,
    which also exposes the sample correlation rho.

    Returns
    -------
    MCResult (price in the currency unit of S0 and K, standard error, n), or a
    ControlVariateResult (also beta and rho) when control_variate=True.
    """
    if average not in ("arithmetic", "geometric"):
        raise ValueError(f"average must be 'arithmetic' or 'geometric', got {average!r}")
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    if control_variate and average != "arithmetic":
        raise ValueError("the geometric control variate only applies to the arithmetic average")
    disc = np.exp(-r * T)

    def payoffs_fn(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-path discounted payoffs for one batch of paths."""
        arithmetic, geometric = path_averages(paths)
        A = arithmetic if average == "arithmetic" else geometric
        X = disc * payoff(A, K, option)
        Y = disc * payoff(geometric, K, option)
        return X, Y

    X, Y = _simulate_in_batches(S0, r, sigma, T, n, n_steps, q, rng, batch_size, payoffs_fn)

    if not control_variate:
        return MCResult.from_samples(X)

    # Why the geometric Asian is an excellent control: on every path the
    # arithmetic and geometric averages of the same prices are very close
    # (by AM-GM, A >= G, and the gap is small when the prices along a path do
    # not move too much). Their payoffs therefore move together almost
    # one-for-one (rho close to 1), while the geometric Asian has an exact
    # price. Subtracting beta * (Y - E[Y]) removes nearly all the noise that
    # the two payoffs share, leaving only the small arithmetic-geometric gap.
    mean_Y = geometric_asian_price(S0, K, r, sigma, T, n_steps, option, q=q)
    return control_variate_estimate(X, Y, mean_Y=mean_Y)


def price_barrier(
    S0: float,
    K: float,
    H: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    n_steps: int,
    kind: str = "down-and-out",
    option: str = "call",
    q: float = 0.0,
    rng: np.random.Generator | int | None = None,
    batch_size: int = 20_000,
) -> MCResult:
    """Price a discretely monitored down barrier call.

    A path is knocked out (down-and-out) or knocked in (down-and-in) if ANY
    monitoring date has S_{t_i} <= H. The payoff is the vanilla call payoff
    max(S_T - K, 0) times the knock-out or knock-in indicator. Since the two
    indicators sum to 1 on every path, down-and-in + down-and-out = vanilla.

    Returns
    -------
    MCResult with the price of the discretely monitored contract. Raises
    ValueError if H >= S0 (already knocked out) or for an unsupported kind.
    """
    if kind not in ("down-and-out", "down-and-in"):
        raise ValueError(f"kind must be 'down-and-out' or 'down-and-in', got {kind!r}")
    if option != "call":
        raise ValueError("only barrier calls are implemented")
    if H <= 0:
        raise ValueError(f"H must be positive, got {H}")
    if H >= S0:
        raise ValueError(
            f"H = {H} >= S0 = {S0}: a down barrier at or above the spot is already hit"
        )
    disc = np.exp(-r * T)

    def payoffs_fn(paths: np.ndarray) -> tuple[np.ndarray]:
        """Per-path discounted payoffs for one batch of paths."""
        monitored = paths[:, 1:]
        hit = (monitored <= H).any(axis=1)
        vanilla = disc * payoff(monitored[:, -1], K, "call")
        alive = ~hit if kind == "down-and-out" else hit
        return (vanilla * alive,)

    (X,) = _simulate_in_batches(S0, r, sigma, T, n, n_steps, q, rng, batch_size, payoffs_fn)
    return MCResult.from_samples(X)
