"""Value at Risk (VaR) and Expected Shortfall (ES) of a linear equity portfolio.

Conventions
-----------
- Losses are POSITIVE numbers (a gain is a negative loss).
- VaR_alpha is the alpha-quantile of the loss distribution, and ES_alpha is
  the mean loss given that the loss is at least VaR_alpha.
- Returns are daily LOG returns r_i, but the portfolio P&L is computed
  exactly from simple returns:
      loss = -V * sum_i w_i (exp(r_i) - 1),
  where V is the portfolio value and w_i are the initial weights (sum to 1).
- By default the mean return is set to ZERO (use_mean=False). At a 1-day
  horizon the mean is tiny compared with the volatility, and estimating it
  from data adds noise rather than information. use_mean=True uses the sample
  mean of the estimation window instead. The flag only affects the parametric
  and Monte Carlo engines: historical simulation always uses the raw returns.
- Horizon h days: mean and covariance are scaled by h. This assumes i.i.d.
  daily returns (square-root-of-time rule for the volatility), which is only
  an approximation: real returns show volatility clustering.

Engines
-------
parametric_normal  closed form for normal returns, linearised portfolio
mc_normal          Monte Carlo, multivariate normal returns via Cholesky
mc_student_t       Monte Carlo, multivariate Student-t returns (fat tails)
historical         empirical quantile of past portfolio losses
mc_normal + ewma_covariance gives the RiskMetrics-style "mc_normal_ewma".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

from .gbm import _get_rng

LEVELS = (0.95, 0.975, 0.99)


@dataclass(frozen=True)
class RiskResult:
    """VaR and ES (in currency units, positive = loss) per confidence level."""

    var: dict[float, float]
    es: dict[float, float]

    def to_frame(self) -> pd.DataFrame:
        """VaR and ES as a DataFrame indexed by confidence level."""
        return pd.DataFrame({"VaR": self.var, "ES": self.es}).rename_axis("level")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def check_weights(weights: Sequence[float]) -> np.ndarray:
    """Return the weights as an array; raise ValueError unless they sum to 1."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1-D sequence")
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError(f"weights must sum to 1 (fractions of the portfolio value), got {w.sum()}")
    return w


def portfolio_losses(log_returns: np.ndarray, weights: Sequence[float], value: float) -> np.ndarray:
    """Exact portfolio loss for each row of daily log returns.

    loss = -V * sum_i w_i (exp(r_i) - 1). expm1 is exp(x) - 1 computed
    accurately for small x.
    """
    w = check_weights(weights)
    log_returns = np.asarray(log_returns, dtype=float)
    return -value * (np.expm1(log_returns) @ w)


def var_es_from_losses(losses: np.ndarray, levels: Sequence[float] = LEVELS) -> RiskResult:
    """Empirical VaR (alpha-quantile) and ES (mean of losses >= VaR).

    Returns
    -------
    RiskResult with VaR and ES per level, in the unit of the losses (positive =
    loss).
    """
    losses = np.asarray(losses, dtype=float)
    var, es = {}, {}
    for alpha in levels:
        q = float(np.quantile(losses, alpha))
        var[alpha] = q
        es[alpha] = float(losses[losses >= q].mean())
    return RiskResult(var=var, es=es)


def cholesky_factor(cov: np.ndarray) -> np.ndarray:
    """Lower-triangular L with L L^T = cov; ValueError if cov is not positive definite.

    Returns
    -------
    The lower-triangular Cholesky factor L, shape (d, d).
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix")
    if not np.allclose(cov, cov.T):
        raise ValueError("cov must be symmetric")
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "covariance matrix is not positive definite, Cholesky factorisation failed"
        ) from exc


def cholesky_correlated_normals(
    cov: np.ndarray, n: int, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """n draws of a zero-mean multivariate normal with covariance cov (shape (n, d)).

    Each row of Z is a vector z of d independent standard normals,
    Cov(z) = I. The row x = L z (i.e. X = Z L^T for all rows at once) has
        Cov(x) = L Cov(z) L^T = L I L^T = L L^T = cov,
    so multiplying by the Cholesky factor turns independent shocks into
    shocks with the target variances and correlations.

    Returns
    -------
    Array of shape (n, d) of correlated draws.
    """
    L = cholesky_factor(cov)
    rng = _get_rng(rng)
    Z = rng.standard_normal((n, L.shape[0]))
    return Z @ L.T


def sample_covariance(log_returns: np.ndarray) -> np.ndarray:
    """Equal-weighted sample covariance (ddof=1) of the rows of log_returns."""
    return np.cov(np.asarray(log_returns, dtype=float), rowvar=False, ddof=1)


def ewma_covariance(log_returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance (RiskMetrics), forecast for the next day.

    The most recent return gets weight (1 - lam), the one before (1 - lam) lam,
    and so on: recent days count more, so the estimate reacts quickly to a
    change in volatility. As in RiskMetrics the mean is taken to be zero.
    The weights over the finite window are rescaled to sum to 1.
    """
    if not 0 < lam < 1:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    r = np.asarray(log_returns, dtype=float)
    n = len(r)
    ages = np.arange(n)[::-1]  # 0 for the last row, n - 1 for the first
    weights = (1 - lam) * lam**ages
    weights /= weights.sum()
    return (r * weights[:, None]).T @ r


def _mean_vector(mu: np.ndarray | None, d: int) -> np.ndarray:
    """Zero vector when mu is None (default zero-mean convention), else mu."""
    return np.zeros(d) if mu is None else np.asarray(mu, dtype=float)


# ---------------------------------------------------------------------------
# VaR / ES engines
# ---------------------------------------------------------------------------
def parametric_normal(
    cov: np.ndarray,
    weights: Sequence[float],
    value: float,
    levels: Sequence[float] = LEVELS,
    mu: np.ndarray | None = None,
    horizon: int = 1,
) -> RiskResult:
    """Closed-form VaR and ES for normal returns (delta-normal method).

    Approximation: the portfolio return is LINEARISED, sum_i w_i r_i, instead
    of the exact sum_i w_i (exp(r_i) - 1). It is then normal with
        mu_p = w . mu * h,   sigma_p = sqrt(w' cov w * h),
    and, with z_alpha the standard normal alpha-quantile and phi its density,
        VaR = V (z_alpha sigma_p - mu_p)
        ES  = V (sigma_p phi(z_alpha) / (1 - alpha) - mu_p).
    """
    w = check_weights(weights)
    mu_p = float(_mean_vector(mu, len(w)) @ w) * horizon
    sigma_p = float(np.sqrt(w @ np.asarray(cov) @ w * horizon))
    var, es = {}, {}
    for alpha in levels:
        z = norm.ppf(alpha)
        var[alpha] = value * (z * sigma_p - mu_p)
        es[alpha] = value * (sigma_p * norm.pdf(z) / (1 - alpha) - mu_p)
    return RiskResult(var=var, es=es)


def mc_normal(
    cov: np.ndarray,
    weights: Sequence[float],
    value: float,
    levels: Sequence[float] = LEVELS,
    n_sims: int = 100_000,
    rng: np.random.Generator | int | None = None,
    mu: np.ndarray | None = None,
    horizon: int = 1,
    Z: np.ndarray | None = None,
) -> RiskResult:
    """Monte Carlo VaR and ES with multivariate normal log returns.

    Returns are mu h + sqrt(h) L z (Cholesky), and each scenario is revalued
    exactly from simple returns. Z (shape (n_sims, d) of standard normals)
    can be passed in to reuse the same draws across calls (common random
    numbers, used in the backtest); otherwise it is drawn from rng.
    """
    L = cholesky_factor(cov)
    if Z is None:
        Z = _get_rng(rng).standard_normal((n_sims, L.shape[0]))
    X = _mean_vector(mu, L.shape[0]) * horizon + np.sqrt(horizon) * (Z @ L.T)
    return var_es_from_losses(portfolio_losses(X, weights, value), levels)


def mc_student_t(
    cov: np.ndarray,
    weights: Sequence[float],
    value: float,
    levels: Sequence[float] = LEVELS,
    nu: float = 5.0,
    n_sims: int = 100_000,
    rng: np.random.Generator | int | None = None,
    mu: np.ndarray | None = None,
    horizon: int = 1,
    Z: np.ndarray | None = None,
    W: np.ndarray | None = None,
) -> RiskResult:
    """Monte Carlo VaR and ES with multivariate Student-t log returns.

    X = sqrt((nu - 2) / nu) * L z / sqrt(W), with z ~ N(0, I) and
    W ~ chi2(nu) / nu, one W per scenario shared by all assets.

    Scaling: L z / sqrt(W) is a multivariate t whose covariance is
    cov * nu / (nu - 2) (finite only for nu > 2). Multiplying by
    sqrt((nu - 2) / nu) brings the covariance back to exactly cov, so the
    normal and t models have the SAME variances and correlations and differ
    only in the shape of the distribution.

    Tail dependence: because a single W scales all assets, a small W (a
    "turbulent day") makes every asset move a lot at the same time. Extreme
    losses therefore tend to happen together even for moderately correlated
    assets. The multivariate normal has no tail dependence: joint extremes
    become independent in the limit.

    Z and W can be passed in for common random numbers (see mc_normal).
    """
    if nu <= 2:
        raise ValueError(f"nu must be > 2 for the covariance to exist, got {nu}")
    L = cholesky_factor(cov)
    rng = _get_rng(rng)
    if Z is None:
        Z = rng.standard_normal((n_sims, L.shape[0]))
    if W is None:
        W = rng.chisquare(nu, size=len(Z)) / nu
    scale = np.sqrt((nu - 2) / nu)
    X = (
        _mean_vector(mu, L.shape[0]) * horizon
        + np.sqrt(horizon) * scale * (Z @ L.T) / np.sqrt(W)[:, None]
    )
    return var_es_from_losses(portfolio_losses(X, weights, value), levels)


def historical(
    log_returns: np.ndarray,
    weights: Sequence[float],
    value: float,
    levels: Sequence[float] = LEVELS,
) -> RiskResult:
    """Historical simulation: VaR and ES of the losses the window actually produced.

    No distribution is assumed and nothing is simulated: each past day's RAW
    returns (not demeaned, the standard definition) are applied to today's
    portfolio, and VaR and ES are the empirical quantile and tail mean of the
    resulting losses.
    """
    r = np.asarray(log_returns, dtype=float)
    return var_es_from_losses(portfolio_losses(r, weights, value), levels)


def estimate_risk(
    method: str,
    log_returns: np.ndarray,
    weights: Sequence[float],
    value: float,
    levels: Sequence[float] = LEVELS,
    use_mean: bool = False,
    horizon: int = 1,
    nu: float = 5.0,
    lam: float = 0.94,
    n_sims: int = 100_000,
    rng: np.random.Generator | int | None = None,
    Z: np.ndarray | None = None,
    W: np.ndarray | None = None,
) -> RiskResult:
    """Estimate VaR/ES from a window of log returns with the named method.

    method: "parametric_normal", "mc_normal", "mc_student_t",
    "mc_normal_ewma" or "historical". The covariance is the equal-weighted
    sample covariance of the window, except for mc_normal_ewma (EWMA, lam).
    use_mean only affects the parametric and Monte Carlo engines; historical
    simulation always uses the raw returns.
    """
    r = np.asarray(log_returns, dtype=float)
    mu = r.mean(axis=0) if use_mean else None
    if method == "historical":
        if horizon != 1:
            raise ValueError("historical simulation is implemented for a 1-day horizon only")
        return historical(r, weights, value, levels)
    if method == "mc_normal_ewma":
        cov = ewma_covariance(r, lam)
    else:
        cov = sample_covariance(r)
    common = dict(levels=levels, mu=mu, horizon=horizon)
    if method == "parametric_normal":
        return parametric_normal(cov, weights, value, **common)
    if method in ("mc_normal", "mc_normal_ewma"):
        return mc_normal(cov, weights, value, n_sims=n_sims, rng=rng, Z=Z, **common)
    if method == "mc_student_t":
        return mc_student_t(cov, weights, value, nu=nu, n_sims=n_sims, rng=rng, Z=Z, W=W, **common)
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
METHODS = ("parametric_normal", "mc_normal", "mc_student_t", "mc_normal_ewma", "historical")


def rolling_backtest(
    log_returns: pd.DataFrame,
    weights: Sequence[float],
    value: float,
    methods: Sequence[str] = METHODS,
    levels: Sequence[float] = (0.95, 0.99),
    window: int = 500,
    n_sims: int = 20_000,
    nu: float = 5.0,
    lam: float = 0.94,
    use_mean: bool = False,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Rolling 1-day VaR/ES forecasts compared with realised losses.

    For each day t from index `window` to the end, VaR and ES are estimated
    from the previous `window` days ONLY (rows t - window .. t - 1), so the
    forecast never uses day t itself (no look-ahead). An exception is a day
    on which the realised loss exceeds the VaR forecast.

    Common random numbers: the standard normal matrix Z and the chi-square
    vector W are drawn ONCE and reused every day; only the covariance changes.
    This makes the backtest much faster and removes day-to-day simulation
    noise from the VaR series, but the Monte Carlo error of consecutive days
    is then correlated (the same draws are reused).

    Returns a DataFrame indexed by date with the realised loss and, for each
    method and level, columns var_<method>_<level>, es_<method>_<level> and
    exc_<method>_<level> (bool), with level written as e.g. 99 or 97.5.
    """
    w = check_weights(weights)
    r = np.asarray(log_returns, dtype=float)
    if len(r) <= window:
        raise ValueError(f"need more than window={window} observations, got {len(r)}")
    rng = _get_rng(rng)
    Z = rng.standard_normal((n_sims, r.shape[1]))
    W = rng.chisquare(nu, size=n_sims) / nu

    realised = portfolio_losses(r, w, value)
    rows = []
    for t in range(window, len(r)):
        past = r[t - window : t]  # excludes day t
        row = {"loss": realised[t]}
        for method in methods:
            res = estimate_risk(
                method, past, w, value, levels, use_mean=use_mean, nu=nu, lam=lam, Z=Z, W=W
            )
            for alpha in levels:
                key = f"{method}_{level_label(alpha)}"
                row[f"var_{key}"] = res.var[alpha]
                row[f"es_{key}"] = res.es[alpha]
                row[f"exc_{key}"] = realised[t] > res.var[alpha]
        rows.append(row)
    index = log_returns.index[window:] if isinstance(log_returns, pd.DataFrame) else None
    return pd.DataFrame(rows, index=index)


def level_label(alpha: float) -> str:
    """0.99 -> '99', 0.975 -> '97.5' (used in column names and tables).

    Returns
    -------
    The level as a percentage string without trailing zeros.
    """
    return f"{100 * alpha:g}"


def kupiec_pof(exceptions: Sequence[bool], p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value).

    H0: the probability of an exception is p. With x exceptions in N days
    and pi = x / N,
        LR = -2 ln[(1 - p)^(N - x) p^x] + 2 ln[(1 - pi)^(N - x) pi^x]
    which is asymptotically chi-square with 1 degree of freedom. xlogy uses
    the convention 0 * log(0) = 0, which handles x = 0 and x = N.
    """
    e = np.asarray(exceptions, dtype=bool)
    N, x = len(e), int(e.sum())
    if N == 0:
        raise ValueError("no observations")
    pi = x / N
    log_l0 = xlogy(N - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(N - x, 1 - pi) + xlogy(x, pi)
    lr = max(-2 * (log_l0 - log_l1), 0.0)  # guard against -1e-15 from rounding
    return float(lr), float(chi2.sf(lr, df=1))


def _transition_counts(e: np.ndarray) -> tuple[int, int, int, int]:
    """n_ij = number of days with state i followed by state j (1 = exception)."""
    prev, curr = e[:-1], e[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))
    return n00, n01, n10, n11


def christoffersen_independence(exceptions: Sequence[bool]) -> tuple[float, float]:
    """Christoffersen (1998) independence test. Returns (LR statistic, p-value).

    H0: an exception today does not change the probability of an exception
    tomorrow. With pi01 = P(exception | no exception yesterday),
    pi11 = P(exception | exception yesterday) and pi the unconditional rate,
        LR = -2 ln L(pi) + 2 ln L(pi01, pi11)
    is asymptotically chi-square with 1 degree of freedom. Clustered
    exceptions (pi11 >> pi01) give a large statistic.
    """
    e = np.asarray(exceptions, dtype=bool)
    if len(e) < 2:
        raise ValueError("need at least 2 observations")
    n00, n01, n10, n11 = _transition_counts(e)

    def ratio(a: int, b: int) -> float:
        """a / b, or 0 when b = 0 (no transitions of that kind)."""
        return a / b if b > 0 else 0.0

    pi01 = ratio(n01, n00 + n01)
    pi11 = ratio(n11, n10 + n11)
    pi = ratio(n01 + n11, n00 + n01 + n10 + n11)
    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2 * (log_l0 - log_l1), 0.0)
    return float(lr), float(chi2.sf(lr, df=1))


def christoffersen_cc(exceptions: Sequence[bool], p: float) -> tuple[float, float]:
    """Conditional coverage test: LR_cc = LR_pof + LR_ind, chi-square with 2 dof.

    Rejects if the exception rate is wrong OR exceptions are clustered.

    Returns
    -------
    (LR statistic, p-value).
    """
    lr = kupiec_pof(exceptions, p)[0] + christoffersen_independence(exceptions)[0]
    return float(lr), float(chi2.sf(lr, df=2))


def basel_traffic_light(n_exceptions: int) -> str:
    """Basel (1996) market-risk backtesting zones for 99% VaR over 250 days.

    green: 0-4 exceptions, yellow: 5-9, red: 10 or more. Used here for
    illustration only.

    Returns
    -------
    'green', 'yellow' or 'red'.
    """
    if n_exceptions < 0:
        raise ValueError("n_exceptions must be non-negative")
    if n_exceptions <= 4:
        return "green"
    if n_exceptions <= 9:
        return "yellow"
    return "red"
