"""Part 2: variance reduction (antithetic variates and control variate).

Compares four estimators of a European option price at the SAME number of
payoff evaluations n: plain Monte Carlo, antithetic variates, control variate
(discounted terminal price) and both combined.

Run from the repository root:

    python scripts/run_part2.py

Outputs
-------
results/part2_pricing_table.csv
results/part2_variance_reduction.csv
results/figures/04_error_vs_n.png
results/figures/05_variance_reduction_by_strike.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # file output only, no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the package importable without installing it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcfinance import (  # noqa: E402
    bs_price,
    estimator_variance,
    price_european,
    price_european_antithetic,
    price_european_combined,
    price_european_control_variate,
    time_per_call,
)

# Reuse the Part 1 plotting style and markdown helper so all figures match.
sys.path.insert(0, str(ROOT / "scripts"))
from run_part1 import BLUE, MUTED, ORANGE, set_style, to_markdown  # noqa: E402

# Contract and model parameters (same as Part 1).
S0, R, SIGMA, T, Q = 100.0, 0.05, 0.2, 1.0, 0.0
K_ATM = 100.0
STRIKES = (80.0, 100.0, 120.0)
SEED = 42

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

AQUA = "#1baf7a"
MAGENTA = "#e87ba4"

# Fixed order and colour per method, used in every table and figure.
METHODS = {
    "Plain MC": price_european,
    "Antithetic": price_european_antithetic,
    "Control variate": price_european_control_variate,
    "Combined": price_european_combined,
}
COLORS = {"Plain MC": BLUE, "Antithetic": ORANGE, "Control variate": AQUA, "Combined": MAGENTA}
# Secondary encoding so the methods stay distinguishable without colour.
MARKERS = {"Plain MC": "o", "Antithetic": "s", "Control variate": "^", "Combined": "D"}


def moneyness(option: str, K: float) -> str:
    """In / at / out of the money label for spot S0."""
    if K == S0:
        return "ATM"
    in_the_money = K < S0 if option == "call" else K > S0
    return "ITM" if in_the_money else "OTM"


# ---------------------------------------------------------------------------
# 1. Pricing table (at-the-money call)
# ---------------------------------------------------------------------------
def pricing_table(rng: np.random.Generator) -> pd.DataFrame:
    exact = bs_price(S0, K_ATM, R, SIGMA, T, "call", q=Q)
    rows = []
    for n in (10_000, 100_000):
        for name, method in METHODS.items():
            res = method(S0, K_ATM, R, SIGMA, T, n, option="call", q=Q, rng=rng)
            rows.append(
                {
                    "method": name,
                    "n": n,
                    "mc_price": res.price,
                    "std_error": res.std_error,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "bs_price": exact,
                    "ci_contains_bs": res.contains(exact),
                }
            )
    return pd.DataFrame(rows)


def format_pricing_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Method": df["method"],
            "n": df["n"].map("{:,}".format),
            "MC price": df["mc_price"].map("{:.4f}".format),
            "Std. error": df["std_error"].map("{:.4f}".format),
            "95% CI": [f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(df["ci_low"], df["ci_high"])],
            "Black-Scholes": df["bs_price"].map("{:.4f}".format),
            "CI contains BS": df["ci_contains_bs"].map({True: "yes", False: "no"}),
        }
    )


# ---------------------------------------------------------------------------
# 2-3. Variance reduction factors and efficiency, by strike
# ---------------------------------------------------------------------------
def variance_reduction_study(
    rng: np.random.Generator, n: int = 10_000, n_reps: int = 500
) -> pd.DataFrame:
    """Empirical variance of each method's price over n_reps runs, per contract.

    variance_factor = Var(plain MC price) / Var(method price) at the same n.
    """
    rows = []
    for option in ("call", "put"):
        for K in STRIKES:
            params = dict(S0=S0, K=K, r=R, sigma=SIGMA, T=T, n=n, option=option, q=Q)
            variances = {
                name: estimator_variance(method, n_reps, rng=rng, **params)
                for name, method in METHODS.items()
            }
            for name in METHODS:
                rows.append(
                    {
                        "option": option,
                        "K": K,
                        "moneyness": moneyness(option, K),
                        "method": name,
                        "n": n,
                        "n_reps": n_reps,
                        "price_variance": variances[name],
                        "variance_factor": variances["Plain MC"] / variances[name],
                    }
                )
    return pd.DataFrame(rows)


def measure_cost(
    rng: np.random.Generator, n: int = 10_000, n_runs: int = 200, n_repeats: int = 5
) -> dict[str, float]:
    """Wall-clock seconds per estimator call at the same n.

    The cost does not depend on the strike, so it is measured once per method
    on the at-the-money call (best of n_repeats blocks of n_runs calls).
    """
    params = dict(S0=S0, K=K_ATM, r=R, sigma=SIGMA, T=T, n=n, option="call", q=Q)
    return {
        name: time_per_call(method, n_runs, n_repeats, rng=rng, **params)
        for name, method in METHODS.items()
    }


def add_efficiency(study: pd.DataFrame, cost: dict[str, float]) -> pd.DataFrame:
    """Efficiency gain = variance factor / (time_method / time_plain)."""
    study = study.copy()
    study["seconds_per_run"] = study["method"].map(cost)
    study["time_ratio"] = study["seconds_per_run"] / cost["Plain MC"]
    study["efficiency_gain"] = study["variance_factor"] / study["time_ratio"]
    return study


def format_study_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per contract: variance factor and efficiency gain per method."""
    out = []
    for (option, K), g in df.groupby(["option", "K"], sort=False):
        g = g.set_index("method")
        row = {"Option": option.capitalize(), "K": f"{K:.0f}", "Moneyness": g["moneyness"].iloc[0]}
        for name in ("Antithetic", "Control variate", "Combined"):
            row[f"{name} VR"] = f"{g.loc[name, 'variance_factor']:.2f}"
        for name in ("Antithetic", "Control variate", "Combined"):
            row[f"{name} eff."] = f"{g.loc[name, 'efficiency_gain']:.2f}"
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 6. Correlation between payoff and control: predicted vs measured factor
# ---------------------------------------------------------------------------
def correlation_check(rng: np.random.Generator, study: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for option in ("call", "put"):
        for K in STRIKES:
            # A single large run gives a precise estimate of rho.
            res = price_european_control_variate(S0, K, R, SIGMA, T, 1_000_000, option=option, q=Q, rng=rng)
            measured = study.query(
                "option == @option and K == @K and method == 'Control variate'"
            )["variance_factor"].iloc[0]
            rows.append(
                {
                    "Option": option.capitalize(),
                    "K": f"{K:.0f}",
                    "Moneyness": moneyness(option, K),
                    "rho": f"{res.rho:.4f}",
                    "Predicted 1/(1-rho^2)": f"{res.predicted_factor:.2f}",
                    "Measured factor": f"{measured:.2f}",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Figure 4: RMSE vs n for the four methods
# ---------------------------------------------------------------------------
def figure_error_vs_n(rng: np.random.Generator, n_runs: int = 300) -> dict[str, float]:
    exact = bs_price(S0, K_ATM, R, SIGMA, T, "call", q=Q)
    # Even sample sizes so that the antithetic estimators accept them.
    ns = np.unique(2 * np.round(np.geomspace(1e3, 1e5, 9) / 2).astype(int))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    slopes = {}
    for name, method in METHODS.items():
        rmse = []
        for n in ns:
            errors = [
                method(S0, K_ATM, R, SIGMA, T, int(n), q=Q, rng=rng).price - exact
                for _ in range(n_runs)
            ]
            rmse.append(np.sqrt(np.mean(np.square(errors))))
        rmse = np.array(rmse)
        slope = np.polyfit(np.log(ns), np.log(rmse), 1)[0]
        slopes[name] = slope
        ax.plot(
            ns,
            rmse,
            marker=MARKERS[name],
            markersize=7,
            color=COLORS[name],
            label=f"{name} (slope {slope:.2f})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of payoff evaluations n (log scale)")
    ax.set_ylabel("RMSE vs Black-Scholes (USD, log scale)")
    ax.set_title("Same $1/\\sqrt{n}$ rate, much lower error: variance reduction shifts the line down")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_error_vs_n.png")
    plt.close(fig)
    return slopes


# ---------------------------------------------------------------------------
# 5. Figure 5: variance reduction factor by strike
# ---------------------------------------------------------------------------
def figure_vr_by_strike(study: pd.DataFrame) -> None:
    methods = ["Antithetic", "Control variate", "Combined"]
    width = 0.26
    x = np.arange(len(STRIKES))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, option in zip(axes, ("call", "put")):
        sub = study[study["option"] == option]
        for i, name in enumerate(methods):
            factors = [
                sub[(sub["K"] == K) & (sub["method"] == name)]["variance_factor"].iloc[0]
                for K in STRIKES
            ]
            pos = x + (i - 1) * width
            # White edges create a visible gap between adjacent bars.
            ax.bar(pos, factors, width, color=COLORS[name], edgecolor="white", linewidth=2,
                   label=name, zorder=3)
            for xi, f in zip(pos, factors):
                ax.text(xi, f * 1.08, f"{f:.1f}x", ha="center", va="bottom", fontsize=8, color="#222222")
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2, label="Plain MC (factor 1)", zorder=4)
        ax.set_xticks(x, [f"K = {K:.0f}\n({moneyness(option, K)})" for K in STRIKES])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.grid(axis="x", visible=False)
        ax.set_title(f"European {option}")
    axes[0].set_ylabel("Variance reduction factor vs plain MC (log scale)")
    axes[0].set_ylim(0.5, study["variance_factor"].max() * 2.5)
    # Upper left of the put panel is empty (the out-of-the-money bars are short).
    axes[1].legend(loc="upper left")
    fig.suptitle(
        "Combining both methods wins for every strike; each method alone helps most in the money",
        x=0.01, ha="left", fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "05_variance_reduction_by_strike.png")
    plt.close(fig)


def main() -> None:
    set_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)  # one generator for the whole run

    print(f"Contract: S0={S0}, r={R}, sigma={SIGMA}, T={T}, q={Q}, seed={SEED}\n")

    table = pricing_table(rng)
    table.to_csv(RESULTS / "part2_pricing_table.csv", index=False)
    pretty = format_pricing_table(table)
    print("Pricing table: at-the-money call (K = 100)")
    print(pretty.to_string(index=False))
    print("\nMarkdown version:\n")
    print(to_markdown(pretty))

    study = variance_reduction_study(rng)
    cost = measure_cost(rng)
    study = add_efficiency(study, cost)
    study.to_csv(RESULTS / "part2_variance_reduction.csv", index=False)
    print("\nVariance reduction (VR) factor and efficiency gain vs plain MC "
          "(n = 10,000, 500 repetitions per method)")
    print("efficiency = VR factor / (time_method / time_plain)\n")
    print(to_markdown(format_study_table(study)))

    print("\nCost per estimator call at n = 10,000 (best of 5 repeats of 200 calls)\n")
    cost_table = pd.DataFrame(
        {
            "Method": list(cost),
            "Time per call (ms)": [f"{1e3 * c:.3f}" for c in cost.values()],
            "Time ratio vs plain": [f"{c / cost['Plain MC']:.2f}" for c in cost.values()],
        }
    )
    print(to_markdown(cost_table))

    print("\nControl variate: predicted factor 1/(1-rho^2) vs measured factor\n")
    print(to_markdown(correlation_check(rng, study)))

    slopes = figure_error_vs_n(rng)
    print("\nSaved results/figures/04_error_vs_n.png")
    print("Fitted log-log slopes of RMSE vs n (theory -0.5): "
          + ", ".join(f"{k} {v:.4f}" for k, v in slopes.items()))

    figure_vr_by_strike(study)
    print("Saved results/figures/05_variance_reduction_by_strike.png")


if __name__ == "__main__":
    main()
