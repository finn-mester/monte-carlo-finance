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
]

__version__ = "0.1.0"
