"""Monte Carlo pricing of European options with confidence intervals.

The price of a European option is the discounted risk-neutral expectation of
its payoff:

    V_0 = e^{-rT} E_Q[ payoff(S_T) ].

We estimate it by the sample mean over n independent draws of S_T. By the
central limit theorem the estimator is approximately normal with standard
error s / sqrt(n), where s is the sample standard deviation of the
discounted payoffs. This gives a 95% confidence interval price +/- 1.96 SE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gbm import _get_rng, simulate_terminal

# 97.5% quantile of the standard normal distribution (two-sided 95% interval).
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo estimate together with its standard error.

    Attributes
    ----------
    price : the Monte Carlo point estimate.
    std_error : estimated standard error of the point estimate.
    n : number of samples used.
    """

    price: float
    std_error: float
    n: int

    @property
    def ci_low(self) -> float:
        """Lower bound of the 95% confidence interval."""
        return self.price - Z_95 * self.std_error

    @property
    def ci_high(self) -> float:
        """Upper bound of the 95% confidence interval."""
        return self.price + Z_95 * self.std_error

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> MCResult:
        """Build a result from i.i.d. discounted payoffs: mean and std / sqrt(n)."""
        samples = np.asarray(samples, dtype=float)
        n = len(samples)
        if n < 2:
            raise ValueError(f"n must be at least 2 to estimate a standard error, got {n}")
        std_error = samples.std(ddof=1) / np.sqrt(n)
        return cls(price=float(samples.mean()), std_error=float(std_error), n=n)

    def contains(self, value: float) -> bool:
        """Return True if value lies inside the 95% confidence interval."""
        return self.ci_low <= value <= self.ci_high

    def __str__(self) -> str:
        return (
            f"{self.price:.4f} +/- {self.std_error:.4f} "
            f"(95% CI [{self.ci_low:.4f}, {self.ci_high:.4f}], n={self.n:,})"
        )


def payoff(S_T: np.ndarray, K: float, option: str = "call") -> np.ndarray:
    """Payoff of a European call max(S_T - K, 0) or put max(K - S_T, 0).

    Returns
    -------
    Array of undiscounted payoffs, same shape as S_T (never negative).
    """
    if option == "call":
        return np.maximum(S_T - K, 0.0)
    if option == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def price_european(
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
    """Price a European option by plain Monte Carlo.

    Returns an MCResult with the discounted sample mean of the payoffs and
    its standard error (sample std with ddof=1, divided by sqrt(n)).
    """
    if n < 2:
        raise ValueError(f"n must be at least 2 to estimate a standard error, got {n}")

    S_T = simulate_terminal(S0, r, sigma, T, n, rng=rng, q=q)
    discounted = np.exp(-r * T) * payoff(S_T, K, option)
    return MCResult.from_samples(discounted)


def convergence_curve(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_max: int,
    option: str = "call",
    q: float = 0.0,
    rng: np.random.Generator | int | None = None,
    n_points: int = 60,
    n_min: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running Monte Carlo estimate and standard error as the sample grows.

    We draw ONE sample of size n_max and read the estimate based on its first
    n draws, for log-spaced values of n. Cumulative sums make this O(n_max):

        mean_n     = (sum of first n x) / n
        variance_n = (sum of x^2 - n * mean_n^2) / (n - 1)

    Returns
    -------
    ns : sample sizes (integers, increasing, last one equals n_max).
    prices : running estimates at each n.
    std_errors : running standard errors at each n.
    """
    if n_min < 2 or n_max < n_min:
        raise ValueError("need 2 <= n_min <= n_max")
    rng = _get_rng(rng)

    S_T = simulate_terminal(S0, r, sigma, T, n_max, rng=rng, q=q)
    discounted = np.exp(-r * T) * payoff(S_T, K, option)

    # Log-spaced sample sizes; rounding can create duplicates, so drop them.
    ns = np.unique(np.geomspace(n_min, n_max, n_points).round().astype(int))

    cum_sum = np.cumsum(discounted)
    cum_sum_sq = np.cumsum(discounted**2)
    sums = cum_sum[ns - 1]
    sums_sq = cum_sum_sq[ns - 1]

    prices = sums / ns
    variances = (sums_sq - ns * prices**2) / (ns - 1)
    # Guard against tiny negative values from floating-point cancellation.
    std_errors = np.sqrt(np.maximum(variances, 0.0) / ns)
    return ns, prices, std_errors
