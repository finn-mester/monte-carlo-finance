# Monte Carlo Methods in Finance: option pricing and risk measurement

[![Tests](https://github.com/<GITHUB_USERNAME>/monte-carlo-finance/actions/workflows/tests.yml/badge.svg)](https://github.com/<GITHUB_USERNAME>/monte-carlo-finance/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A from-scratch Python library and study of Monte Carlo methods for derivatives pricing and market risk. It covers European options, variance reduction, path-dependent exotics, and VaR and Expected Shortfall on real market data. The guiding principle: **every Monte Carlo result is validated against a closed-form benchmark or a statistical test, reported with a confidence interval or its sampling uncertainty, and covered by automated tests** (115 tests, all offline). A Monte Carlo number without an error bar is not a result.

## Key results

- **Pricing is validated against Black-Scholes.** With 1,000,000 paths the MC call price is 10.4578 (standard error 0.0147, 95% CI [10.4289, 10.4866]), against an exact price of 10.4506. The empirical error decays with slope -0.4983 (theory: -0.5), and 95.10% of 2,000 independent 95% confidence intervals contain the exact price.
- **Variance reduction at equal budget** (at-the-money call, $n = 10{,}000$, 500 repetitions): the variance falls by 2.41x with antithetic variates, 7.36x with a control variate and 31.92x with both combined. Combined, that is 25.90x less variance per unit of CPU time.
- **Arithmetic Asian option** (no closed form): using the geometric Asian as a control variate divides the variance by 1,010 at $m = 52$ monitoring dates, for 2% extra runtime (correlation 0.99960).
- **Barrier option with discrete monitoring** ($H = 90$, $m = 52$): the textbook continuous formula is off by -0.4810, about 15 standard errors. The Broadie-Glasserman-Kou correction brings the error down to +0.0271, inside the Monte Carlo noise.
- **VaR and ES on five CAC 40 stocks** (1-day, 99%, 1,000,000 EUR): the normal model gives VaR 23,717 and ES 27,071 EUR. The fat-tailed models give VaR 26,497 / ES 34,880 (Student-t) and VaR 28,154 / ES 43,331 (historical simulation): the normal model understates the tail.
- **The backtest is reported honestly** (3,285 days): **no model passes at 99%**. The models have 48 to 75 exceptions against 32.9 expected, so Kupiec rejects all five, and every model fails the independence test because of volatility clustering. At 95%, four of five models pass Kupiec.

<table>
  <tr>
    <td width="50%"><img src="results/figures/02_error_scaling.png" alt="Error scaling"><br><sub>MC error decays like 1/√n and matches the reported standard error.</sub></td>
    <td width="50%"><img src="results/figures/05_variance_reduction_by_strike.png" alt="Variance reduction by strike"><br><sub>Variance reduction factors by strike: combining both methods wins everywhere.</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="results/figures/07_barrier_monitoring_bias.png" alt="Barrier monitoring bias"><br><sub>Discrete-monitoring bias of a barrier option and the BGK correction.</sub></td>
    <td width="50%"><img src="results/figures/12_backtest_99.png" alt="VaR backtest"><br><sub>99% VaR backtest: exceptions cluster in stress periods (2015, 2020).</sub></td>
  </tr>
</table>

## Roadmap

| Part | Topic | Status | Write-up |
|---|---|---|---|
| 1 | European options by Monte Carlo, validated against Black-Scholes | Done | [docs/part1_european_options.md](docs/part1_european_options.md) |
| 2 | Variance reduction (antithetic variates, control variate), compared at equal budget | Done | [docs/part2_variance_reduction.md](docs/part2_variance_reduction.md) |
| 3 | Path-dependent exotics: arithmetic Asian (geometric control variate), discretely monitored barrier (BGK correction) | Done | [docs/part3_exotic_options.md](docs/part3_exotic_options.md) |
| 4 | VaR and Expected Shortfall of a real equity portfolio, with a rolling backtest | Done | [docs/part4_var_es.md](docs/part4_var_es.md) |

A short runnable tour of the library is in [notebooks/demo.ipynb](notebooks/demo.ipynb).

## Quick start

```bash
git clone https://github.com/<GITHUB_USERNAME>/monte-carlo-finance.git
cd monte-carlo-finance
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev,data]"

pytest                                 # 115 tests, offline, runs in seconds
python scripts/download_data.py        # Part 4 prices from Yahoo Finance (not committed)
python scripts/run_part1.py            # each script writes its tables to results/
python scripts/run_part2.py            # and its figures to results/figures/
python scripts/run_part3.py
python scripts/run_part4.py
python scripts/run_all.py              # or: everything above in order (skips the download if data exists)
```

```python
from mcfinance import bs_price, price_european

res = price_european(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, n=1_000_000, rng=42)
print(res)                              # price +/- standard error, 95% CI, n
print(bs_price(100, 100, 0.05, 0.2, 1.0))
```

## Project structure

```
.
├── src/mcfinance/
│   ├── gbm.py                  # exact GBM simulation (terminal values and full paths)
│   ├── black_scholes.py        # closed-form benchmark
│   ├── pricing.py              # MC estimator, MCResult with confidence interval
│   ├── variance_reduction.py   # antithetic, control variate, combined, VR and timing helpers
│   ├── exotics_closed_form.py  # geometric Asian, continuous barrier, BGK correction
│   ├── exotics.py              # batched MC for Asian and barrier options
│   └── risk.py                 # VaR/ES engines, EWMA, backtest and coverage tests
├── tests/                      # pytest suite (synthetic data only)
├── scripts/                    # download_data.py, run_part1..4.py, run_all.py
├── docs/                       # detailed write-up of each part
├── notebooks/demo.ipynb        # short demo with stored outputs
├── results/                    # generated CSV tables and figures
├── data/                       # downloaded prices (not committed)
└── pyproject.toml
```

## Methodology and honesty notes

- **How results are validated.**
  - European options are checked against Black-Scholes, and confidence-interval coverage is tested.
  - Variance-reduction factors are measured from repeated independent runs, not just from reported standard errors.
  - The geometric Asian and barrier formulas are checked against independent derivations.
  - VaR and ES are backtested against realised losses with Kupiec and Christoffersen tests.
  - Every figure and table is produced by a script with a fixed seed (42).
- **Data.** Daily adjusted close prices of LVMH, TotalEnergies, Sanofi, BNP Paribas and Airbus from Yahoo Finance (via `yfinance`), 2012-01-02 to 2026-10-20, downloaded on 2026-10-22. The data is not redistributed: `scripts/download_data.py` regenerates it, and a later download will extend the period and slightly change the Part 4 numbers.
- **Main limitations.**
  - All models assume Black-Scholes dynamics: constant volatility, no jumps, no smile.
  - The VaR models assume i.i.d. returns and use square-root-of-time scaling, which the backtest itself contradicts.
  - The risk portfolio is five linear equity positions, with no options.
  - The BGK correction is asymptotic and is inaccurate at very coarse monitoring ($m = 4$).
  - The closed-form VaR linearises the P&L, so it differs from the exact-revaluation Monte Carlo by about 1.8%.
  - This is a learning project, not a validated or regulatory model.

## License

MIT, see [LICENSE](LICENSE).
