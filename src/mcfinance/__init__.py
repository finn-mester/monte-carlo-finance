"""Monte Carlo methods in finance: option pricing and risk measurement."""

from .black_scholes import bs_price
from .exotics import path_averages, price_asian, price_barrier
from .exotics_closed_form import (
    barrier_closed_form,
    bgk_adjusted_barrier,
    geometric_asian_price,
    geometric_average_expectation,
)
from .gbm import simulate_paths, simulate_terminal
from .risk import (
    RiskResult,
    basel_traffic_light,
    cholesky_correlated_normals,
    christoffersen_cc,
    christoffersen_independence,
    estimate_risk,
    ewma_covariance,
    historical,
    kupiec_pof,
    mc_normal,
    mc_student_t,
    parametric_normal,
    portfolio_losses,
    rolling_backtest,
)
from .pricing import MCResult, convergence_curve, payoff, price_european
from .variance_reduction import (
    ControlVariateResult,
    control_variate_estimate,
    estimator_variance,
    price_european_antithetic,
    price_european_combined,
    price_european_control_variate,
    time_per_call,
)

__all__ = [
    "bs_price",
    "simulate_terminal",
    "simulate_paths",
    "MCResult",
    "payoff",
    "price_european",
    "convergence_curve",
    "ControlVariateResult",
    "control_variate_estimate",
    "price_european_antithetic",
    "price_european_control_variate",
    "price_european_combined",
    "estimator_variance",
    "time_per_call",
    "geometric_asian_price",
    "geometric_average_expectation",
    "barrier_closed_form",
    "bgk_adjusted_barrier",
    "path_averages",
    "price_asian",
    "price_barrier",
    "RiskResult",
    "cholesky_correlated_normals",
    "ewma_covariance",
    "portfolio_losses",
    "parametric_normal",
    "mc_normal",
    "mc_student_t",
    "historical",
    "estimate_risk",
    "rolling_backtest",
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_cc",
    "basel_traffic_light",
]

__version__ = "0.1.0"
