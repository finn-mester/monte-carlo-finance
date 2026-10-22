"""Monte Carlo methods in finance: option pricing and risk measurement."""

from .black_scholes import bs_price
from .gbm import simulate_paths, simulate_terminal
from .pricing import MCResult, convergence_curve, payoff, price_european

__all__ = [
    "bs_price",
    "simulate_terminal",
    "simulate_paths",
    "MCResult",
    "payoff",
    "price_european",
    "convergence_curve",
]

__version__ = "0.1.0"
