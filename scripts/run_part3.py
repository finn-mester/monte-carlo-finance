"""Part 3: path-dependent exotic options (Asian and barrier calls).

Run from the repository root:

    python scripts/run_part3.py

Outputs
-------
results/part3_geometric_asian_validation.csv
results/part3_arithmetic_asian.csv
results/part3_asian_vs_monitoring.csv
results/part3_barrier_monitoring.csv
results/part3_barrier_vs_level.csv
results/figures/06_asian_vs_monitoring.png
results/figures/07_barrier_monitoring_bias.png
results/figures/08_barrier_price_vs_level.png
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the package importable without installing it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcfinance import (  # noqa: E402
    barrier_closed_form,
    bgk_adjusted_barrier,
    bs_price,
    estimator_variance,
    geometric_asian_price,
    payoff,
    price_asian,
    price_barrier,
    simulate_paths,
    time_per_call,
)
from mcfinance.pricing import Z_95  # noqa: E402

# Reuse the plotting style, palette and markdown helper of the earlier parts.
sys.path.insert(0, str(ROOT / "scripts"))
from run_part1 import BLUE, MUTED, ORANGE, set_style, to_markdown  # noqa: E402
from run_part2 import AQUA  # noqa: E402

# Contract and model parameters (same as Parts 1 and 2).
S0, K, R, SIGMA, T, Q = 100.0, 100.0, 0.05, 0.2, 1.0, 0.0
H = 90.0
SEED = 42

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

ASIAN_DATES = (12, 52, 252)


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def ci_text(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


# ---------------------------------------------------------------------------
# A1. Geometric Asian: Monte Carlo vs closed form
# ---------------------------------------------------------------------------
def geometric_validation(rng: np.random.Generator, n: int = 200_000) -> pd.DataFrame:
    rows = []
    for m in ASIAN_DATES:
        res = price_asian(S0, K, R, SIGMA, T, n, m, average="geometric", q=Q, rng=rng)
        exact = geometric_asian_price(S0, K, R, SIGMA, T, m, "call", q=Q)
        rows.append(
            {
                "m": m,
                "n": n,
                "mc_price": res.price,
                "std_error": res.std_error,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "closed_form": exact,
                "ci_contains_exact": res.contains(exact),
            }
        )
    return pd.DataFrame(rows)


def format_geometric(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "m": df["m"],
            "n": df["n"].map("{:,}".format),
            "MC price": df["mc_price"].map(fmt),
            "Std. error": df["std_error"].map(fmt),
            "95% CI": [ci_text(lo, hi) for lo, hi in zip(df["ci_low"], df["ci_high"], strict=True)],
            "Closed form": df["closed_form"].map(fmt),
            "CI contains exact": df["ci_contains_exact"].map({True: "yes", False: "no"}),
        }
    )


# ---------------------------------------------------------------------------
# A2. Arithmetic Asian: plain MC vs geometric control variate
# ---------------------------------------------------------------------------
def arithmetic_asian_study(
    rng: np.random.Generator,
    n_price: int = 50_000,
    n_study: int = 10_000,
    n_reps: int = 200,
) -> pd.DataFrame:
    """Prices at n_price, then variance factor and cost at n_study.

    As in Part 2: the variance reduction factor is measured from the spread of
    n_reps independent prices, and the cost is the best of several blocks of
    calls (time_per_call). Efficiency = factor / (time_cv / time_plain).
    """
    rows = []
    for m in ASIAN_DATES:
        base = dict(S0=S0, K=K, r=R, sigma=SIGMA, T=T, n_steps=m, option="call", q=Q)
        plain = price_asian(**base, n=n_price, rng=rng)
        cv = price_asian(**base, n=n_price, control_variate=True, rng=rng)

        var_plain = estimator_variance(price_asian, n_reps, rng=rng, **base, n=n_study)
        var_cv = estimator_variance(
            price_asian, n_reps, rng=rng, **base, n=n_study, control_variate=True
        )
        t_plain = time_per_call(price_asian, 20, 3, rng=rng, **base, n=n_study)
        t_cv = time_per_call(price_asian, 20, 3, rng=rng, **base, n=n_study, control_variate=True)

        factor = var_plain / var_cv
        for name, res in (("Plain MC", plain), ("Control variate", cv)):
            is_cv = name == "Control variate"
            rows.append(
                {
                    "m": m,
                    "method": name,
                    "n": n_price,
                    "mc_price": res.price,
                    "std_error": res.std_error,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "rho": res.rho if is_cv else np.nan,
                    "predicted_factor": res.predicted_factor if is_cv else np.nan,
                    "variance_factor": factor if is_cv else 1.0,
                    "seconds_per_run": t_cv if is_cv else t_plain,
                    "time_ratio": t_cv / t_plain if is_cv else 1.0,
                    "efficiency_gain": factor / (t_cv / t_plain) if is_cv else 1.0,
                }
            )
    return pd.DataFrame(rows)


def format_arithmetic(df: pd.DataFrame) -> pd.DataFrame:
    def opt(values, f):
        return [f(v) if np.isfinite(v) else "" for v in values]

    return pd.DataFrame(
        {
            "m": df["m"],
            "Method": df["method"],
            "MC price": df["mc_price"].map(fmt),
            "Std. error": df["std_error"].map(fmt),
            "95% CI": [ci_text(lo, hi) for lo, hi in zip(df["ci_low"], df["ci_high"], strict=True)],
            "rho": opt(df["rho"], lambda v: f"{v:.5f}"),
            "Predicted 1/(1-rho^2)": opt(df["predicted_factor"], lambda v: f"{v:.0f}"),
            "Measured VR factor": df["variance_factor"].map(
                lambda v: f"{v:.0f}" if v != 1 else "1"
            ),
            "Time ratio": df["time_ratio"].map("{:.2f}".format),
            "Efficiency gain": df["efficiency_gain"].map(lambda v: f"{v:.0f}" if v != 1 else "1"),
        }
    )


# ---------------------------------------------------------------------------
# A3. Figure 6: Asian price vs number of monitoring dates
# ---------------------------------------------------------------------------
def figure_asian_vs_monitoring(rng: np.random.Generator, n: int = 50_000) -> pd.DataFrame:
    ms = [1, 2, 4, 12, 52, 252]
    european = bs_price(S0, K, R, SIGMA, T, "call", q=Q)
    rows = []
    for m in ms:
        res = price_asian(S0, K, R, SIGMA, T, n, m, control_variate=True, q=Q, rng=rng)
        rows.append(
            {
                "m": m,
                "arithmetic_cv_price": res.price,
                "arithmetic_cv_std_error": res.std_error,
                "geometric_closed_form": geometric_asian_price(S0, K, R, SIGMA, T, m, "call", q=Q),
                "european_bs": european,
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(
        european,
        color=MUTED,
        linestyle="--",
        linewidth=1.5,
        label=f"European call, Black-Scholes ({european:.4f})",
    )
    ax.plot(
        df["m"],
        df["geometric_closed_form"],
        color=ORANGE,
        marker="s",
        markersize=7,
        label="Geometric Asian call (closed form)",
    )
    ax.errorbar(
        df["m"],
        df["arithmetic_cv_price"],
        yerr=Z_95 * df["arithmetic_cv_std_error"],
        color=BLUE,
        marker="o",
        markersize=7,
        capsize=4,
        label="Arithmetic Asian call (MC + control variate;\n95% CI smaller than the marker)",
    )
    ax.set_xscale("log")
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.minorticks_off()
    ax.set_xlabel("Number of monitoring dates m (log scale)")
    ax.set_ylabel("Option price (USD)")
    ax.set_ylim(df["geometric_closed_form"].min() - 0.8, european + 1.2)
    ax.set_title("Averaging lowers the effective volatility, so Asian calls are cheaper")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_asian_vs_monitoring.png")
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# B1-B2. Barrier: discrete monitoring bias and BGK correction
# ---------------------------------------------------------------------------
def barrier_monitoring_study(
    rng: np.random.Generator, n: int = 200_000
) -> tuple[pd.DataFrame, float]:
    ms = [4, 12, 26, 52, 126, 252, 504, 1000]
    continuous = barrier_closed_form(S0, K, H, R, SIGMA, T, "down-and-out", q=Q)
    rows = []
    for m in ms:
        res = price_barrier(S0, K, H, R, SIGMA, T, n, m, kind="down-and-out", q=Q, rng=rng)
        bgk = barrier_closed_form(S0, K, bgk_adjusted_barrier(H, SIGMA, T, m), R, SIGMA, T, q=Q)
        rows.append(
            {
                "m": m,
                "n": n,
                "mc_price": res.price,
                "std_error": res.std_error,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "continuous_closed_form": continuous,
                "continuous_bias": continuous - res.price,
                "bgk_barrier": bgk_adjusted_barrier(H, SIGMA, T, m),
                "bgk_closed_form": bgk,
                "bgk_bias": bgk - res.price,
                "bgk_in_ci": res.contains(bgk),
            }
        )
    df = pd.DataFrame(rows)

    # Bias of the continuous formula should scale like 1/sqrt(m): fit the
    # log-log slope of |MC - continuous| against m.
    slope = np.polyfit(np.log(df["m"]), np.log(np.abs(df["continuous_bias"])), 1)[0]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    m_grid = np.geomspace(ms[0], ms[-1], 200)
    bgk_curve = [
        barrier_closed_form(S0, K, bgk_adjusted_barrier(H, SIGMA, T, m), R, SIGMA, T, q=Q)
        for m in m_grid
    ]
    ax.axhline(
        continuous,
        color=ORANGE,
        linestyle="--",
        label=f"Continuous closed form, no correction ({continuous:.4f})",
    )
    ax.plot(m_grid, bgk_curve, color=AQUA, label="BGK-adjusted closed form")
    ax.errorbar(
        df["m"],
        df["mc_price"],
        yerr=Z_95 * df["std_error"],
        color=BLUE,
        marker="o",
        markersize=7,
        capsize=4,
        linestyle="none",
        label="Monte Carlo, discrete monitoring (95% CI)",
    )
    ax.set_xscale("log")
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.minorticks_off()
    ax.set_xlabel("Number of monitoring dates m (log scale)")
    ax.set_ylabel("Down-and-out call price (USD)")
    ax.set_title("Monitoring bias decays like $1/\\sqrt{m}$; the BGK correction removes most of it")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "07_barrier_monitoring_bias.png")
    plt.close(fig)
    return df, slope


def format_barrier_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["m"].isin([12, 52, 252, 1000])]
    return pd.DataFrame(
        {
            "m": df["m"],
            "MC price": df["mc_price"].map(fmt),
            "Std. error": df["std_error"].map(fmt),
            "95% CI": [ci_text(lo, hi) for lo, hi in zip(df["ci_low"], df["ci_high"], strict=True)],
            "Continuous CF": df["continuous_closed_form"].map(fmt),
            "Bias (continuous - MC)": df["continuous_bias"].map(fmt),
            "BGK CF": df["bgk_closed_form"].map(fmt),
            "Bias (BGK - MC)": df["bgk_bias"].map(fmt),
            "BGK in CI": df["bgk_in_ci"].map({True: "yes", False: "no"}),
        }
    )


# ---------------------------------------------------------------------------
# B3. Figure 8: price vs barrier level
# ---------------------------------------------------------------------------
def figure_barrier_vs_level(
    rng: np.random.Generator, n: int = 100_000, m: int = 252
) -> pd.DataFrame:
    levels = [60, 65, 70, 75, 80, 85, 90, 93, 95, 97, 99]
    vanilla = bs_price(S0, K, R, SIGMA, T, "call", q=Q)
    H_eff = lambda h: bgk_adjusted_barrier(h, SIGMA, T, m)  # noqa: E731
    rows = []
    for level in levels:
        res = price_barrier(S0, K, float(level), R, SIGMA, T, n, m, q=Q, rng=rng)
        bgk = barrier_closed_form(S0, K, H_eff(level), R, SIGMA, T, q=Q)
        rows.append(
            {
                "H": level,
                "mc_price": res.price,
                "std_error": res.std_error,
                "bgk_closed_form": bgk,
                "bgk_minus_mc": bgk - res.price,
                "bgk_in_ci": res.contains(bgk),
            }
        )
    df = pd.DataFrame(rows)

    H_grid = np.linspace(60, 99.5, 300)
    curve = [barrier_closed_form(S0, K, H_eff(h), R, SIGMA, T, q=Q) for h in H_grid]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(
        vanilla, color=MUTED, linestyle="--", linewidth=1.5, label=f"Vanilla call ({vanilla:.4f})"
    )
    ax.plot(H_grid, curve, color=AQUA, label=f"BGK-adjusted closed form (m = {m})")
    ax.errorbar(
        df["H"],
        df["mc_price"],
        yerr=Z_95 * df["std_error"],
        color=BLUE,
        marker="o",
        markersize=7,
        capsize=4,
        linestyle="none",
        label=f"Monte Carlo, m = {m} (95% CI smaller than the marker)",
    )
    ax.set_xlabel("Barrier level H (USD), spot S0 = 100")
    ax.set_ylabel("Down-and-out call price (USD)")
    ax.set_xlim(58, 101)
    ax.set_ylim(0, vanilla * 1.25)
    ax.set_title("A low barrier leaves a vanilla call; a barrier near spot makes it worthless")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_barrier_price_vs_level.png")
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# B4. In-out parity on one set of paths
# ---------------------------------------------------------------------------
def in_out_parity(
    rng: np.random.Generator, n: int = 100_000, m: int = 52
) -> tuple[float, float, float]:
    """Price the knock-out, the knock-in and the vanilla on the SAME paths.

    The generator state is copied before each call so that all three consume
    exactly the same normal draws; the main generator then moves on.
    """
    d_out = price_barrier(
        S0, K, H, R, SIGMA, T, n, m, kind="down-and-out", q=Q, rng=copy.deepcopy(rng)
    )
    d_in = price_barrier(
        S0, K, H, R, SIGMA, T, n, m, kind="down-and-in", q=Q, rng=copy.deepcopy(rng)
    )
    paths = simulate_paths(S0, R, SIGMA, T, n, m, rng=rng, q=Q)
    vanilla = float(np.mean(np.exp(-R * T) * payoff(paths[:, -1], K, "call")))
    return d_out.price, d_in.price, vanilla


def main() -> None:
    set_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)  # one generator for the whole run

    print(f"Contract: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}, q={Q}, seed={SEED}\n")

    # ---- A. Asian options ----
    geo = geometric_validation(rng)
    geo.to_csv(RESULTS / "part3_geometric_asian_validation.csv", index=False)
    print("A1. Geometric Asian call: Monte Carlo vs closed form\n")
    print(to_markdown(format_geometric(geo)))

    arith = arithmetic_asian_study(rng)
    arith.to_csv(RESULTS / "part3_arithmetic_asian.csv", index=False)
    print(
        "\nA2. Arithmetic Asian call: plain MC vs geometric control variate "
        "(prices at n = 50,000; VR factor from 200 repetitions at n = 10,000; "
        "time = best of 3 blocks of 20 calls at n = 10,000)\n"
    )
    print(to_markdown(format_arithmetic(arith)))
    print(
        "\nTime per call at n = 10,000 (ms): "
        + ", ".join(
            f"m={row.m} {row.method} {1e3 * row.seconds_per_run:.2f}" for row in arith.itertuples()
        )
    )

    asian_m = figure_asian_vs_monitoring(rng)
    asian_m.to_csv(RESULTS / "part3_asian_vs_monitoring.csv", index=False)
    print(
        "\nA3. Asian call price vs number of monitoring dates (arithmetic: MC + CV, n = 50,000)\n"
    )
    print(
        to_markdown(
            pd.DataFrame(
                {
                    "m": asian_m["m"],
                    "Arithmetic (MC + CV)": asian_m["arithmetic_cv_price"].map(fmt),
                    "Std. error": asian_m["arithmetic_cv_std_error"].map(fmt),
                    "Geometric (closed form)": asian_m["geometric_closed_form"].map(fmt),
                    "European (BS)": asian_m["european_bs"].map(fmt),
                }
            )
        )
    )
    print("Saved results/figures/06_asian_vs_monitoring.png")

    # ---- B. Barrier options ----
    barrier, slope = barrier_monitoring_study(rng)
    barrier.to_csv(RESULTS / "part3_barrier_monitoring.csv", index=False)
    print(f"\nB1. Down-and-out call, K = {K:.0f}, H = {H:.0f}, n = 200,000\n")
    print(to_markdown(format_barrier_table(barrier)))
    print("\nAll monitoring frequencies (figure 7):")
    print(
        barrier[["m", "mc_price", "std_error", "continuous_bias", "bgk_bias", "bgk_in_ci"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"Fitted log-log slope of |continuous - MC| vs m: {slope:.4f} (theory -0.5)")
    print("Saved results/figures/07_barrier_monitoring_bias.png")

    level = figure_barrier_vs_level(rng)
    level.to_csv(RESULTS / "part3_barrier_vs_level.csv", index=False)
    print("\nB3. Down-and-out call vs barrier level (m = 252, n = 100,000)\n")
    print(
        to_markdown(
            pd.DataFrame(
                {
                    "H": level["H"],
                    "MC price": level["mc_price"].map(fmt),
                    "Std. error": level["std_error"].map(fmt),
                    "BGK CF": level["bgk_closed_form"].map(fmt),
                    "BGK - MC": level["bgk_minus_mc"].map(fmt),
                    "BGK in CI": level["bgk_in_ci"].map({True: "yes", False: "no"}),
                }
            )
        )
    )
    print("Saved results/figures/08_barrier_price_vs_level.png")

    d_out, d_in, vanilla = in_out_parity(rng)
    print(f"\nB4. In-out parity on the same 100,000 paths (m = 52, H = {H:.0f}):")
    print(
        f"down-and-out {d_out:.6f} + down-and-in {d_in:.6f} = {d_out + d_in:.6f}; "
        f"vanilla on the same paths {vanilla:.6f}; difference {d_out + d_in - vanilla:.2e}"
    )


if __name__ == "__main__":
    main()
