"""Part 4: Value at Risk and Expected Shortfall of an equity portfolio, with a backtest.

Run from the repository root, after downloading the data:

    python scripts/download_data.py
    python scripts/run_part4.py

Portfolio: equal weights in five CAC 40 stocks, V = 1,000,000 EUR. In the
backtest the portfolio is reset to equal weights and V every day, so each
day's loss is comparable with the VaR forecast for that day.

Outputs
-------
results/part4_data_summary.csv
results/part4_correlation.csv
results/part4_var_es_1day.csv
results/part4_var_es_10day.csv
results/part4_student_t_nu.csv
results/part4_mc_uncertainty.csv
results/part4_backtest.csv
results/figures/09_return_distribution.png
results/figures/10_var_es_comparison.png
results/figures/11_convergence.png
results/figures/12_backtest_99.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

# Make the package importable without installing it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcfinance.risk import (  # noqa: E402
    LEVELS,
    METHODS,
    _transition_counts,  # noqa: E402  (diagnostic print only)
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    estimate_risk,
    kupiec_pof,
    level_label,
    mc_normal,
    parametric_normal,
    rolling_backtest,
    sample_covariance,
)

# Reuse the plotting style, palette and markdown helper of the earlier parts.
sys.path.insert(0, str(ROOT / "scripts"))
from run_part1 import BLUE, MUTED, ORANGE, set_style, to_markdown  # noqa: E402
from run_part2 import AQUA, MAGENTA  # noqa: E402

DATA_FILE = ROOT / "data" / "prices.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

VALUE = 1_000_000.0  # EUR
WINDOW = 500  # estimation window in trading days (about 2 years)
SEED = 42

NAMES = {
    "parametric_normal": "Parametric normal",
    "mc_normal": "MC normal",
    "mc_student_t": "MC Student-t (nu = 5)",
    "mc_normal_ewma": "MC normal, EWMA cov.",
    "historical": "Historical simulation",
}
# Fixed colour per method in every figure.
COLORS = {
    "parametric_normal": MUTED,
    "mc_normal": BLUE,
    "mc_student_t": ORANGE,
    "mc_normal_ewma": MAGENTA,
    "historical": AQUA,
}


def eur(x: float) -> str:
    return f"{x:,.0f}"


def load_log_returns() -> pd.DataFrame:
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        print("Run `python scripts/download_data.py` first to download the prices.")
        sys.exit(1)
    prices = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    return np.log(prices).diff().dropna()


# ---------------------------------------------------------------------------
# A. Data summary
# ---------------------------------------------------------------------------
def data_summary(returns: pd.DataFrame, weights: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    port = pd.Series(np.log1p(np.expm1(returns.to_numpy()) @ weights), index=returns.index)
    table = returns.assign(Portfolio=port)
    summary = pd.DataFrame(
        {
            "Annualised volatility": table.std(ddof=1) * np.sqrt(252),
            "Skewness": table.apply(lambda s: skew(s)),
            "Excess kurtosis": table.apply(lambda s: kurtosis(s)),
            "Worst day": table.min(),
            "Best day": table.max(),
        }
    )
    return summary, returns.corr()


# ---------------------------------------------------------------------------
# B. Single-date comparison
# ---------------------------------------------------------------------------
def single_date(
    rng: np.random.Generator, window: np.ndarray, weights: np.ndarray, n_sims: int = 500_000
):
    one_day = {
        m: estimate_risk(m, window, weights, VALUE, LEVELS, n_sims=n_sims, rng=rng) for m in METHODS
    }

    ten_day = {
        m: estimate_risk(m, window, weights, VALUE, (0.99,), horizon=10, n_sims=n_sims, rng=rng)
        for m in METHODS
        if m != "historical"
    }
    # Historical simulation has no covariance to scale: apply sqrt(10) directly.
    hist = one_day["historical"]
    ten_day_hist = {"VaR": np.sqrt(10) * hist.var[0.99], "ES": np.sqrt(10) * hist.es[0.99]}

    nu_table = []
    for nu in (4, 5, 8, 30):
        res = estimate_risk(
            "mc_student_t", window, weights, VALUE, (0.99,), nu=nu, n_sims=n_sims, rng=rng
        )
        nu_table.append({"nu": nu, "VaR 99%": res.var[0.99], "ES 99%": res.es[0.99]})
    normal = one_day["mc_normal"]
    nu_table.append(
        {"nu": "normal (infinite)", "VaR 99%": normal.var[0.99], "ES 99%": normal.es[0.99]}
    )
    return one_day, ten_day, ten_day_hist, pd.DataFrame(nu_table)


def mc_uncertainty(
    rng: np.random.Generator,
    cov: np.ndarray,
    weights: np.ndarray,
    n_sims: int = 500_000,
    n_reps: int = 100,
) -> pd.DataFrame:
    runs = [mc_normal(cov, weights, VALUE, LEVELS, n_sims=n_sims, rng=rng) for _ in range(n_reps)]
    rows = []
    for alpha in LEVELS:
        v = np.array([r.var[alpha] for r in runs])
        e = np.array([r.es[alpha] for r in runs])
        rows.append(
            {
                "Level": f"{level_label(alpha)}%",
                "Mean VaR": eur(v.mean()),
                "Std VaR": f"{v.std(ddof=1):,.1f}",
                "Mean ES": eur(e.mean()),
                "Std ES": f"{e.std(ddof=1):,.1f}",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C. Figures
# ---------------------------------------------------------------------------
def figure_return_distribution(returns: pd.DataFrame, weights: np.ndarray) -> None:
    # Portfolio simple returns in percent, full sample.
    port = 100 * (np.expm1(returns.to_numpy()) @ weights)
    mu, sd = port.mean(), port.std(ddof=1)
    var_normal = -(mu + norm.ppf(0.01) * sd)
    var_hist = -np.quantile(port, 0.01)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.linspace(port.min(), port.max(), 600)
    ax1.hist(port, bins=150, density=True, color=BLUE, alpha=0.45, label="Portfolio daily returns")
    ax1.plot(x, norm.pdf(x, mu, sd), color=ORANGE, label="Fitted normal density")
    ax1.axvline(
        -var_normal,
        color=ORANGE,
        linestyle="--",
        linewidth=1.3,
        label=f"99% VaR, normal fit ({var_normal:.2f}%)",
    )
    ax1.axvline(
        -var_hist,
        color=AQUA,
        linestyle="--",
        linewidth=1.3,
        label=f"99% VaR, empirical ({var_hist:.2f}%)",
    )
    ax1.set_xlim(-8, 8)
    ax1.set_xlabel("Daily portfolio return (%), axis cut at +/-8%")
    ax1.set_ylabel("Density (1/%)")
    ax1.set_title("Peakier centre and fatter tails than the normal")
    ax1.legend(loc="upper right", fontsize=8.5)

    z = (np.sort(port) - mu) / sd
    theo = norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    ax2.plot(
        theo, z, "o", color=BLUE, markersize=3, alpha=0.6, label="Standardised portfolio returns"
    )
    xlim = abs(theo).max() + 0.5
    ax2.plot(
        [-xlim, xlim],
        [-xlim, xlim],
        color=ORANGE,
        linestyle="--",
        label="Normal distribution (45-degree line)",
    )
    ax2.set_xlim(-xlim, xlim)
    ax2.set_xlabel("Normal quantile (standard deviations)")
    ax2.set_ylabel("Empirical quantile (standard deviations)")
    ax2.set_title("QQ plot: extreme days lie far beyond the normal")
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIGURES / "09_return_distribution.png")
    plt.close(fig)


def figure_var_es(one_day: dict) -> None:
    methods = list(METHODS)
    var = [one_day[m].var[0.99] / 1e3 for m in methods]
    es = [one_day[m].es[0.99] / 1e3 for m in methods]
    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, values, color, label in (
        (-width / 2, var, BLUE, "VaR 99%"),
        (width / 2, es, ORANGE, "ES 99%"),
    ):
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=color,
            edgecolor="white",
            linewidth=2,
            label=label,
            zorder=3,
        )
        for bar, v in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    ax.set_xticks(x, [NAMES[m].replace(", ", ",\n").replace(" (", "\n(") for m in methods])
    ax.set_ylabel("1-day loss (thousand EUR), V = 1,000,000 EUR")
    ax.set_ylim(0, max(es) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title(
        "Fat tails matter at 99%: historical and Student-t ES are far above the normal models"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "10_var_es_comparison.png")
    plt.close(fig)


def figure_convergence(
    rng: np.random.Generator, cov: np.ndarray, weights: np.ndarray, n_reps: int = 100
) -> pd.DataFrame:
    exact = parametric_normal(cov, weights, VALUE, (0.99,))
    sigma_p = float(np.sqrt(weights @ cov @ weights)) * VALUE
    ns = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    rows = []
    for n in ns:
        runs = [mc_normal(cov, weights, VALUE, (0.99,), n_sims=n, rng=rng) for _ in range(n_reps)]
        v = np.array([r.var[0.99] for r in runs])
        e = np.array([r.es[0.99] for r in runs])
        rows.append(
            {
                "n": n,
                "var_mean": v.mean(),
                "var_std": v.std(ddof=1),
                "es_mean": e.mean(),
                "es_std": e.std(ddof=1),
                # Std of the estimate in units of sigma_p V / sqrt(n), i.e. relative
                # to the standard error a simple MEAN would have with the same n.
                "var_std_vs_mean_se": v.std(ddof=1) / (sigma_p / np.sqrt(n)),
                "es_std_vs_mean_se": e.std(ddof=1) / (sigma_p / np.sqrt(n)),
            }
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, label in ((axes[0], "var", "VaR 99%"), (axes[1], "es", "ES 99%")):
        mean, sd = df[f"{key}_mean"] / 1e3, df[f"{key}_std"] / 1e3
        ax.fill_between(
            df["n"],
            mean - 2 * sd,
            mean + 2 * sd,
            color=BLUE,
            alpha=0.18,
            linewidth=0,
            label=f"+/- 2 std over {n_reps} repetitions",
        )
        ax.plot(df["n"], mean, color=BLUE, marker="o", markersize=6, label="Mean MC estimate")
        ax.axhline(
            getattr(exact, key)[0.99] / 1e3,
            color=ORANGE,
            linestyle="--",
            label="Closed form (parametric normal, linearised P&L)",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Number of simulated scenarios n (log scale)")
        ax.set_ylabel(f"{label}, 1 day (thousand EUR)")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8.5)
    fig.suptitle(
        "Tail estimates converge slowly: thousands of scenarios still leave visible noise",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "11_convergence.png")
    plt.close(fig)
    return df


def figure_backtest(bt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        bt.index,
        bt["loss"] / 1e3,
        color="#bbbbbb",
        linewidth=0.6,
        label="Realised daily loss",
        zorder=1,
    )
    markers = {"mc_normal": "o", "mc_student_t": "s", "historical": "^"}
    for method, marker in markers.items():
        col = f"var_{method}_99"
        exc = bt[f"exc_{method}_99"]
        ax.plot(
            bt.index,
            bt[col] / 1e3,
            color=COLORS[method],
            linewidth=1.3,
            label=f"{NAMES[method]} 99% VaR: {int(exc.sum())} exceptions",
            zorder=3,
        )
        ax.plot(
            bt.index[exc],
            bt.loc[exc, "loss"] / 1e3,
            marker,
            color=COLORS[method],
            markersize=5,
            linestyle="none",
            zorder=4,
            markerfacecolor="none",
            markeredgewidth=1.2,
        )
    ax.set_ylabel("1-day loss (thousand EUR), positive = loss\n(gains beyond 60 not shown)")
    ax.set_xlabel("Date")
    ax.set_ylim(-60, max(bt["loss"].max() / 1e3 + 10, 100))
    ax.set_title("Exceptions cluster in stress periods: a static 2-year window reacts too slowly")
    ax.legend(
        loc="upper right",
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "12_backtest_99.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# D. Backtest table
# ---------------------------------------------------------------------------
def backtest_table(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha in (0.99, 0.95):
        p = 1 - alpha
        for method in METHODS:
            key = f"{method}_{level_label(alpha)}"
            e = bt[f"exc_{key}"].to_numpy()
            on_exc = bt[f"exc_{key}"]
            rows.append(
                {
                    "method": method,
                    "level": alpha,
                    "days": len(e),
                    "exceptions": int(e.sum()),
                    "expected": len(e) * p,
                    "rate": e.mean(),
                    "kupiec_p": kupiec_pof(e, p)[1],
                    "independence_p": christoffersen_independence(e)[1],
                    "cc_p": christoffersen_cc(e, p)[1],
                    "exceptions_last_250": int(e[-250:].sum()),
                    "zone_last_250": basel_traffic_light(int(e[-250:].sum()))
                    if alpha == 0.99
                    else "",
                    "avg_loss_on_exceptions": bt.loc[on_exc, "loss"].mean(),
                    "avg_predicted_es_on_exceptions": bt.loc[on_exc, f"es_{key}"].mean(),
                }
            )
    return pd.DataFrame(rows)


def format_backtest(df: pd.DataFrame) -> pd.DataFrame:
    def pval(p):
        return "<0.0001" if p < 1e-4 else f"{p:.4f}"

    return pd.DataFrame(
        {
            "Method": df["method"].map(NAMES),
            "Level": df["level"].map(lambda a: f"{level_label(a)}%"),
            "Exceptions": df["exceptions"],
            "Expected": df["expected"].map("{:.1f}".format),
            "Rate": df["rate"].map("{:.2%}".format),
            "Kupiec p": df["kupiec_p"].map(pval),
            "Indep. p": df["independence_p"].map(pval),
            "CC p": df["cc_p"].map(pval),
            "Last 250 d": [
                f"{n} ({z})" if z else str(n)
                for n, z in zip(df["exceptions_last_250"], df["zone_last_250"], strict=True)
            ],
            "Avg loss on exc.": df["avg_loss_on_exceptions"].map(eur),
            "Avg predicted ES": df["avg_predicted_es_on_exceptions"].map(eur),
        }
    )


def risk_table(results: dict, levels) -> pd.DataFrame:
    rows = []
    for method, res in results.items():
        row = {"Method": NAMES.get(method, method)}
        for alpha in levels:
            row[f"VaR {level_label(alpha)}%"] = eur(res.var[alpha])
            row[f"ES {level_label(alpha)}%"] = eur(res.es[alpha])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    set_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)  # one generator for the whole run

    returns = load_log_returns()
    tickers = list(returns.columns)
    weights = np.full(len(tickers), 1 / len(tickers))
    print(
        f"Data: {', '.join(tickers)}; {returns.index[0].date()} to {returns.index[-1].date()}, "
        f"{len(returns)} daily log returns. Equal weights, V = {eur(VALUE)} EUR.\n"
    )

    # ---- A ----
    summary, corr = data_summary(returns, weights)
    summary.to_csv(RESULTS / "part4_data_summary.csv")
    corr.to_csv(RESULTS / "part4_correlation.csv")
    print("A. Data summary (full sample, daily log returns)\n")
    print(
        to_markdown(
            summary.reset_index(names="Asset").assign(
                **{
                    "Annualised volatility": summary["Annualised volatility"]
                    .map("{:.1%}".format)
                    .values,
                    "Skewness": summary["Skewness"].map("{:.2f}".format).values,
                    "Excess kurtosis": summary["Excess kurtosis"].map("{:.2f}".format).values,
                    "Worst day": summary["Worst day"].map("{:.2%}".format).values,
                    "Best day": summary["Best day"].map("{:.2%}".format).values,
                }
            )
        )
    )
    print("\nCorrelation matrix\n")
    print(to_markdown(corr.round(2).reset_index(names="")))

    # ---- B ----
    window = returns.iloc[-WINDOW:]
    print(
        f"\nB. Single-date comparison: estimation window {window.index[0].date()} to "
        f"{window.index[-1].date()} ({WINDOW} days), 1-day horizon, EUR, "
        "MC with 500,000 scenarios\n"
    )
    one_day, ten_day, ten_day_hist, nu_table = single_date(rng, window.to_numpy(), weights)
    one = risk_table(one_day, LEVELS)
    one.to_csv(RESULTS / "part4_var_es_1day.csv", index=False)
    print(to_markdown(one))

    ten = risk_table(ten_day, (0.99,))
    ten.loc[len(ten)] = {
        "Method": "Historical simulation x sqrt(10)",
        "VaR 99%": eur(ten_day_hist["VaR"]),
        "ES 99%": eur(ten_day_hist["ES"]),
    }
    ten.to_csv(RESULTS / "part4_var_es_10day.csv", index=False)
    print("\n10-day 99% VaR and ES (square-root-of-time: covariance x 10)\n")
    print(to_markdown(ten))

    nu_fmt = nu_table.assign(
        **{"VaR 99%": nu_table["VaR 99%"].map(eur), "ES 99%": nu_table["ES 99%"].map(eur)}
    )
    nu_fmt.to_csv(RESULTS / "part4_student_t_nu.csv", index=False)
    print("\nStudent-t sensitivity to nu (1-day, 99%, same covariance)\n")
    print(to_markdown(nu_fmt))

    cov = sample_covariance(window.to_numpy())
    unc = mc_uncertainty(rng, cov, weights)
    unc.to_csv(RESULTS / "part4_mc_uncertainty.csv", index=False)
    print("\nMC normal uncertainty: 100 independent repetitions of 500,000 scenarios (EUR)\n")
    print(to_markdown(unc))

    # ---- C ----
    figure_return_distribution(returns, weights)
    print("\nSaved results/figures/09_return_distribution.png")
    figure_var_es(one_day)
    print("Saved results/figures/10_var_es_comparison.png")
    conv = figure_convergence(rng, cov, weights)
    print("Saved results/figures/11_convergence.png")
    print("\nConvergence of MC normal 99% VaR / ES (100 repetitions per n)\n")
    print(
        to_markdown(
            pd.DataFrame(
                {
                    "n": conv["n"].map("{:,}".format),
                    "Mean VaR": conv["var_mean"].map(eur),
                    "Std VaR": conv["var_std"].map("{:,.0f}".format),
                    "Mean ES": conv["es_mean"].map(eur),
                    "Std ES": conv["es_std"].map("{:,.0f}".format),
                    "Std VaR / (sigma_p V / sqrt n)": conv["var_std_vs_mean_se"].map(
                        "{:.2f}".format
                    ),
                    "Std ES / (sigma_p V / sqrt n)": conv["es_std_vs_mean_se"].map("{:.2f}".format),
                }
            )
        )
    )
    exact = parametric_normal(cov, weights, VALUE, (0.99,))
    print(
        f"Closed form (linearised): VaR 99% = {eur(exact.var[0.99])}, "
        f"ES 99% = {eur(exact.es[0.99])}"
    )

    # ---- D ----
    print("\nD. Rolling backtest (window 500 days, 20,000 scenarios, common random numbers) ...")
    bt = rolling_backtest(
        returns,
        weights,
        VALUE,
        methods=METHODS,
        levels=(0.95, 0.99),
        window=WINDOW,
        n_sims=20_000,
        rng=rng,
    )
    print(f"Backtest period: {bt.index[0].date()} to {bt.index[-1].date()}, {len(bt)} days\n")
    table = backtest_table(bt)
    table.to_csv(RESULTS / "part4_backtest.csv", index=False)
    print(to_markdown(format_backtest(table)))
    figure_backtest(bt)
    print("\nSaved results/figures/12_backtest_99.png")

    # Where do the 99% exceptions of the normal model fall?
    exc = bt["exc_mc_normal_99"]
    by_year = exc.groupby(bt.index.year).sum()
    print(
        "\nMC normal 99% exceptions per calendar year: "
        + ", ".join(f"{y}: {int(n)}" for y, n in by_year.items())
    )
    for method in METHODS:
        e = bt[f"exc_{method}_99"].to_numpy()
        print(f"{NAMES[method]}: 99% transitions (n00, n01, n10, n11) = {_transition_counts(e)}")


if __name__ == "__main__":
    main()
