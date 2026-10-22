"""Part 1: price a European option by Monte Carlo and validate it against Black-Scholes.

Run from the repository root:

    python scripts/run_part1.py

Outputs
-------
results/part1_pricing_table.csv
results/figures/01_convergence.png
results/figures/02_error_scaling.png
results/figures/03_paths_and_distribution.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm

# Make the package importable without installing it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcfinance import bs_price, convergence_curve, price_european  # noqa: E402
from mcfinance.gbm import simulate_paths, simulate_terminal  # noqa: E402

# Contract and model parameters.
S0, K, R, SIGMA, T, Q = 100.0, 100.0, 0.05, 0.2, 1.0, 0.0
SEED = 42

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#222222"
MUTED = "#666666"


def set_style() -> None:
    """Clean, minimal matplotlib style shared by all figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependency)."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---:" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


# ---------------------------------------------------------------------------
# 1. Pricing table
# ---------------------------------------------------------------------------
def pricing_table(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for option in ("call", "put"):
        exact = bs_price(S0, K, R, SIGMA, T, option, q=Q)
        for n in (1_000, 10_000, 100_000, 1_000_000):
            res = price_european(S0, K, R, SIGMA, T, n, option=option, q=Q, rng=rng)
            rows.append(
                {
                    "option": option,
                    "n": n,
                    "mc_price": res.price,
                    "std_error": res.std_error,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "bs_price": exact,
                    "abs_error": abs(res.price - exact),
                    "ci_contains_bs": res.contains(exact),
                }
            )
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable version of the pricing table."""
    out = pd.DataFrame(
        {
            "Option": df["option"].str.capitalize(),
            "n": df["n"].map("{:,}".format),
            "MC price": df["mc_price"].map("{:.4f}".format),
            "Std. error": df["std_error"].map("{:.4f}".format),
            "95% CI": [f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(df["ci_low"], df["ci_high"])],
            "Black-Scholes": df["bs_price"].map("{:.4f}".format),
            "Abs. error": df["abs_error"].map("{:.4f}".format),
            "CI contains BS": df["ci_contains_bs"].map({True: "yes", False: "no"}),
        }
    )
    return out


# ---------------------------------------------------------------------------
# 2. Figure 1: convergence of the running estimate
# ---------------------------------------------------------------------------
def figure_convergence(rng: np.random.Generator) -> None:
    exact = bs_price(S0, K, R, SIGMA, T, "call", q=Q)
    ns, prices, ses = convergence_curve(
        S0, K, R, SIGMA, T, n_max=1_000_000, option="call", q=Q, rng=rng, n_points=200
    )
    lo, hi = prices - 1.959963984540054 * ses, prices + 1.959963984540054 * ses

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(ns, lo, hi, color=BLUE, alpha=0.18, linewidth=0, label="95% confidence interval")
    ax.plot(ns, prices, color=BLUE, label="Monte Carlo estimate")
    ax.axhline(exact, color=ORANGE, linestyle="--", label=f"Black-Scholes price ({exact:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths n (log scale)")
    ax.set_ylabel("European call price (USD)")
    ax.set_title("The Monte Carlo estimate converges to the Black-Scholes price")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_convergence.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Figure 2: error scaling (RMSE vs n)
# ---------------------------------------------------------------------------
def figure_error_scaling(rng: np.random.Generator) -> float:
    exact = bs_price(S0, K, R, SIGMA, T, "call", q=Q)
    ns = np.unique(np.geomspace(1e2, 1e5, 13).round().astype(int))
    n_runs = 300

    rmse, mean_se = [], []
    for n in ns:
        runs = [price_european(S0, K, R, SIGMA, T, int(n), q=Q, rng=rng) for _ in range(n_runs)]
        errors = np.array([res.price - exact for res in runs])
        rmse.append(np.sqrt(np.mean(errors**2)))
        mean_se.append(np.mean([res.std_error for res in runs]))
    rmse, mean_se = np.array(rmse), np.array(mean_se)

    # Fit log(RMSE) = a + b log(n); theory says b = -1/2.
    slope, intercept = np.polyfit(np.log(ns), np.log(rmse), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ns, rmse, "o", color=BLUE, markersize=7, label=f"Empirical RMSE over {n_runs} runs")
    ax.plot(ns, mean_se, color=ORANGE, label="Mean reported standard error")
    ax.plot(
        ns,
        np.exp(intercept) * ns**slope,
        color=MUTED,
        linestyle=":",
        linewidth=1.5,
        label=f"Fitted slope {slope:.3f}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of simulated paths n (log scale)")
    ax.set_ylabel("Pricing error (USD, log scale)")
    ax.set_title(r"Error shrinks like $1/\sqrt{n}$: 100x more paths, 10x less error")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_error_scaling.png")
    plt.close(fig)
    return slope


# ---------------------------------------------------------------------------
# 4. Coverage of the 95% confidence interval
# ---------------------------------------------------------------------------
def coverage_check(rng: np.random.Generator, n_runs: int = 2_000, n: int = 10_000) -> float:
    exact = bs_price(S0, K, R, SIGMA, T, "call", q=Q)
    hits = sum(
        price_european(S0, K, R, SIGMA, T, n, q=Q, rng=rng).contains(exact) for _ in range(n_runs)
    )
    return hits / n_runs


# ---------------------------------------------------------------------------
# 5. Figure 3: sample paths and terminal distribution
# ---------------------------------------------------------------------------
def figure_paths_and_distribution(rng: np.random.Generator) -> None:
    n_steps = 252  # daily steps over one year
    paths = simulate_paths(S0, R, SIGMA, T, n_paths=40, n_steps=n_steps, rng=rng, q=Q)
    t = np.linspace(0.0, T, n_steps + 1)
    S_T = simulate_terminal(S0, R, SIGMA, T, 200_000, rng=rng, q=Q)

    # Exact law: ln S_T ~ N(ln S0 + (r - q - sigma^2/2) T, sigma^2 T).
    mu = np.log(S0) + (R - Q - 0.5 * SIGMA**2) * T
    density = lognorm(s=SIGMA * np.sqrt(T), scale=np.exp(mu))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

    ax1.plot(t, paths.T, color=BLUE, alpha=0.35, linewidth=0.9)
    ax1.plot([], [], color=BLUE, alpha=0.6, linewidth=0.9, label="Simulated GBM paths (40)")
    ax1.axhline(K, color=ORANGE, linestyle="--", label=f"Strike K = {K:.0f}")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Stock price (USD)")
    ax1.set_title("Paths fan out: the call pays only above the strike")
    # Leave headroom above the paths so the legend does not cover them.
    ax1.set_ylim(top=paths.max() + 0.35 * (paths.max() - paths.min()))
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

    x = np.linspace(S_T.min(), np.quantile(S_T, 0.999), 400)
    ax2.hist(S_T, bins=120, range=(x[0], x[-1]), density=True, color=BLUE, alpha=0.45,
             label="Simulated $S_T$ (200,000)")
    ax2.plot(x, density.pdf(x), color=ORANGE, label="Exact lognormal density")
    ax2.axvline(K, color=MUTED, linestyle=":", linewidth=1.2, label="Strike")
    ax2.set_xlabel("Terminal price $S_T$ (USD)")
    ax2.set_ylabel("Probability density (1/USD)")
    ax2.set_title("Simulated $S_T$ matches the lognormal law")
    ax2.set_ylim(0, 1.4 * density.pdf(x).max())  # headroom for the legend
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIGURES / "03_paths_and_distribution.png")
    plt.close(fig)


def main() -> None:
    set_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)  # one generator for the whole run

    print(f"Contract: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}, q={Q}, seed={SEED}")
    print(f"Black-Scholes call = {bs_price(S0, K, R, SIGMA, T, 'call'):.4f}, "
          f"put = {bs_price(S0, K, R, SIGMA, T, 'put'):.4f}\n")

    table = pricing_table(rng)
    table.to_csv(RESULTS / "part1_pricing_table.csv", index=False)
    pretty = format_table(table)
    print("Pricing table")
    print(pretty.to_string(index=False))
    print("\nMarkdown version (for the README):\n")
    print(to_markdown(pretty))

    figure_convergence(rng)
    print("\nSaved results/figures/01_convergence.png")

    slope = figure_error_scaling(rng)
    print(f"Saved results/figures/02_error_scaling.png  (fitted log-log slope = {slope:.4f}, theory -0.5)")

    coverage = coverage_check(rng)
    print(f"95% CI coverage over 2,000 runs with n=10,000: {coverage:.2%}")

    figure_paths_and_distribution(rng)
    print("Saved results/figures/03_paths_and_distribution.png")


if __name__ == "__main__":
    main()
