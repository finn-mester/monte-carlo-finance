# Monte Carlo Methods in Finance: option pricing and risk measurement

A from-scratch Python implementation of Monte Carlo methods for derivatives pricing and risk management. The guiding principle: **every Monte Carlo result is validated against a closed-form benchmark, reported with a confidence interval, and covered by automated tests.** A Monte Carlo number without an error bar is not a result. This project shows how big the error bar is, checks that it shrinks at the rate theory predicts, and checks that the 95% interval contains the true value about 95% of the time.

## Roadmap

| Part | Topic | Status |
|---|---|---|
| 1 | European option pricing by Monte Carlo, validated against Black-Scholes | Done |
| 2 | Variance reduction (antithetic variates, control variates) | Planned |
| 3 | Exotic option pricing (path-dependent payoff) | Planned |
| 4 | Value at Risk and Expected Shortfall | Planned |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                             # run the test suite
python scripts/run_part1.py        # reproduce the Part 1 table and figures
```

```python
from mcfinance import bs_price, price_european

res = price_european(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, n=1_000_000, rng=42)
print(res)                          # price +/- standard error, 95% CI, n
print(bs_price(100, 100, 0.05, 0.2, 1.0))
```

## Part 1: European option pricing

### Method

**Model.** Under the risk-neutral measure $\mathbb{Q}$ the stock follows a geometric Brownian motion with risk-free rate $r$, dividend yield $q$ and volatility $\sigma$:

$$dS_t = (r - q)\,S_t\,dt + \sigma\,S_t\,dW_t .$$

This SDE has an exact solution, so $S_T$ can be sampled directly with no time-discretisation error:

$$S_T = S_0 \exp\!\Big(\big(r - q - \tfrac{1}{2}\sigma^2\big)T + \sigma\sqrt{T}\,Z\Big), \qquad Z \sim \mathcal{N}(0,1).$$

**Estimator.** The option price is the discounted expected payoff, $V_0 = e^{-rT}\,\mathbb{E}_{\mathbb{Q}}[h(S_T)]$ with $h(S) = (S-K)^+$ for a call and $(K-S)^+$ for a put. With $n$ independent draws $S_T^{(i)}$ and $Y_i = e^{-rT}h(S_T^{(i)})$:

$$\hat V_n = \frac{1}{n}\sum_{i=1}^{n} Y_i .$$

**Standard error and confidence interval.** By the central limit theorem $\hat V_n$ is approximately normal with standard error

$$\mathrm{SE} = \frac{s_n}{\sqrt{n}}, \qquad s_n^2 = \frac{1}{n-1}\sum_{i=1}^{n}\big(Y_i - \hat V_n\big)^2,$$

and the 95% confidence interval is $\hat V_n \pm 1.96\,\mathrm{SE}$. The error therefore decreases like $1/\sqrt{n}$: 100 times more paths give 10 times less error.

**Benchmark.** Each estimate is compared with the Black-Scholes closed form, $C = S_0 e^{-qT}N(d_1) - K e^{-rT}N(d_2)$, $P = K e^{-rT}N(-d_2) - S_0 e^{-qT}N(-d_1)$.

### Results

Contract: $S_0 = 100$, $K = 100$, $r = 5\%$, $\sigma = 20\%$, $T = 1$ year, no dividends, seed 42. Black-Scholes prices: call **10.4506**, put **5.5735**. Full table in [`results/part1_pricing_table.csv`](results/part1_pricing_table.csv).

| Option | n | MC price | Std. error | 95% CI | Black-Scholes | Abs. error | CI contains BS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Call | 1,000 | 9.9120 | 0.4483 | [9.0332, 10.7907] | 10.4506 | 0.5386 | yes |
| Call | 10,000 | 10.3887 | 0.1481 | [10.0983, 10.6790] | 10.4506 | 0.0619 | yes |
| Call | 100,000 | 10.4260 | 0.0468 | [10.3342, 10.5178] | 10.4506 | 0.0246 | yes |
| Call | 1,000,000 | 10.4578 | 0.0147 | [10.4289, 10.4866] | 10.4506 | 0.0072 | yes |
| Put | 1,000 | 5.3208 | 0.2668 | [4.7978, 5.8438] | 5.5735 | 0.2527 | yes |
| Put | 10,000 | 5.5536 | 0.0866 | [5.3839, 5.7232] | 5.5735 | 0.0200 | yes |
| Put | 100,000 | 5.5543 | 0.0273 | [5.5008, 5.6078] | 5.5735 | 0.0192 | yes |
| Put | 1,000,000 | 5.5687 | 0.0087 | [5.5517, 5.5856] | 5.5735 | 0.0049 | yes |

Additional checks produced by the same script:

- **Convergence rate.** Over 300 independent runs per sample size ($n$ from $10^2$ to $10^5$), the fitted log-log slope of the RMSE against $n$ is **-0.4983** (theory: -0.5).
- **Interval calibration.** Over 2,000 independent runs with $n = 10{,}000$, **95.10%** of the 95% confidence intervals contain the Black-Scholes price.

**Convergence of the running estimate** (single sample of 1,000,000 paths):

![Convergence](results/figures/01_convergence.png)

**Error scaling**: empirical RMSE against Black-Scholes and the mean standard error reported by the estimator:

![Error scaling](results/figures/02_error_scaling.png)

**Simulated paths and terminal distribution**: the histogram of $S_T$ matches the exact lognormal density:

![Paths and distribution](results/figures/03_paths_and_distribution.png)

## Project structure

```
.
├── src/mcfinance/
│   ├── gbm.py              # exact GBM simulation (terminal values and full paths)
│   ├── black_scholes.py    # closed-form benchmark
│   └── pricing.py          # MC estimator, MCResult with CI, convergence curve
├── tests/                  # pytest suite
├── scripts/run_part1.py    # reproduces the Part 1 table and figures
├── results/                # generated table and figures
├── pyproject.toml
└── requirements.txt
```

## Testing

```bash
pytest
```

The suite (30 tests) checks:

- Black-Scholes against textbook values (call 10.4506, put 5.5735), put-call parity across strikes and dividend yields, monotonicity in volatility, and input validation.
- Monte Carlo call and put prices within 4 standard errors of Black-Scholes ($n = 400{,}000$).
- Reproducibility with a fixed seed.
- Standard error halves when the sample size is multiplied by 4.
- 95% CI coverage between 92% and 98% over 500 repetitions.
- $\mathbb{E}[S_T] = S_0 e^{(r-q)T}$ (martingale property of the discounted price).
- Path shapes, consistency between `convergence_curve` and `price_european`, and invalid inputs.

## Assumptions and limitations

- Black-Scholes world: constant volatility and interest rate, continuous dividend yield, lognormal prices, no transaction costs. Real markets show volatility smiles and jumps.
- European exercise only in Part 1. American options need methods such as Longstaff-Schwartz.
- The confidence interval relies on the central limit theorem. It is approximate for small $n$ or for heavy-tailed payoffs.
- Plain Monte Carlo is deliberately used here as the baseline. It converges slowly ($1/\sqrt{n}$), which is what Part 2 (variance reduction) addresses.
- Pseudo-random numbers come from NumPy's PCG64 generator. Results are reproducible for a given seed and NumPy version.
