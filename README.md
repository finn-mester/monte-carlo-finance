# Monte Carlo Methods in Finance: option pricing and risk measurement

A from-scratch Python implementation of Monte Carlo methods for derivatives pricing and risk management. The guiding principle: **every Monte Carlo result is validated against a closed-form benchmark (or, for risk measures, backtested against realised losses), reported with its uncertainty, and covered by automated tests.** A Monte Carlo number without an error bar is not a result. This project shows how big the error bar is, checks that it shrinks at the rate theory predicts, and checks that the 95% interval contains the true value about 95% of the time.

## Roadmap

| Part | Topic | Status |
|---|---|---|
| 1 | European option pricing by Monte Carlo, validated against Black-Scholes | Done |
| 2 | Variance reduction (antithetic variates, control variate), compared at equal budget | Done |
| 3 | Path-dependent exotics: arithmetic Asian (geometric control variate) and discretely monitored barrier (BGK correction) | Done |
| 4 | Value at Risk and Expected Shortfall of a real equity portfolio, with a 13-year backtest | Done |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py    # Part 4 data from Yahoo Finance (not committed)

pytest                             # run the test suite (no data or network needed)
python scripts/run_part1.py        # reproduce the Part 1 table and figures
python scripts/run_part2.py        # reproduce the Part 2 tables and figures
python scripts/run_part3.py        # reproduce the Part 3 tables and figures (about 1 minute)
python scripts/run_part4.py        # Part 4 VaR/ES tables, figures and backtest (about 1 minute)
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

## Part 2: Variance reduction

Plain Monte Carlo error falls like $\sigma_X/\sqrt{n}$. Halving it costs four times as many paths. Variance reduction lowers $\sigma_X$ instead. All methods below are compared at the **same budget**: $n$ is always the total number of payoff evaluations.

### Method

**Antithetic variates.** Draw $n/2$ normals $Z_i$ and use each one twice, as $Z_i$ and $-Z_i$. With $X(Z) = e^{-rT}h\big(S_T(Z)\big)$:

$$\hat V^{\text{AV}} = \frac{1}{n/2}\sum_{i=1}^{n/2} \frac{X(Z_i) + X(-Z_i)}{2}.$$

The $n/2$ pair averages are i.i.d., but the two members of a pair are negatively correlated. The standard error must therefore be computed from the pair averages, with $n/2$ observations, and not from the $n$ individual payoffs. The method helps when the payoff is close to monotone in $Z$: writing $X = X_{\text{even}} + X_{\text{odd}}$, the pair average removes the odd part $X_{\text{odd}}$ exactly.

**Control variate.** Use $Y = e^{-rT}S_T$, whose expectation is known exactly because the discounted asset is a martingale under $\mathbb{Q}$: $\mathbb{E}[Y] = S_0e^{-qT}$. The adjusted estimator

$$\hat V^{\text{CV}} = \frac{1}{n}\sum_{i=1}^{n}\Big(X_i - \beta\,\big(Y_i - \mathbb{E}[Y]\big)\Big)$$

is unbiased for any fixed $\beta$. Its variance is minimised by

$$\beta^* = \frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(Y)}, \qquad \operatorname{Var}\big(X - \beta^*(Y - \mathbb{E}[Y])\big) = (1-\rho^2)\operatorname{Var}(X),$$

so the variance reduction factor is $1/(1-\rho^2)$, where $\rho = \operatorname{Corr}(X, Y)$. $\beta^*$ is estimated from the same sample, which introduces a bias of order $1/n$, negligible next to the $1/\sqrt{n}$ statistical error at these sample sizes.

**Combined.** The control variate is applied to the antithetic pair averages $\big(\tfrac{X(Z)+X(-Z)}{2}, \tfrac{Y(Z)+Y(-Z)}{2}\big)$. The averaged control still has expectation $S_0e^{-qT}$.

### Results

**At-the-money call** ($K = 100$, same contract and seed as Part 1). Full table in [`results/part2_pricing_table.csv`](results/part2_pricing_table.csv).

| Method | n | MC price | Std. error | 95% CI | Black-Scholes | CI contains BS |
|---:|---:|---:|---:|---:|---:|---:|
| Plain MC | 10,000 | 10.3452 | 0.1477 | [10.0558, 10.6346] | 10.4506 | yes |
| Antithetic | 10,000 | 10.4289 | 0.1042 | [10.2246, 10.6333] | 10.4506 | yes |
| Control variate | 10,000 | 10.4111 | 0.0557 | [10.3019, 10.5203] | 10.4506 | yes |
| Combined | 10,000 | 10.4539 | 0.0276 | [10.3999, 10.5079] | 10.4506 | yes |
| Plain MC | 100,000 | 10.3799 | 0.0468 | [10.2883, 10.4716] | 10.4506 | yes |
| Antithetic | 100,000 | 10.4738 | 0.0329 | [10.4092, 10.5384] | 10.4506 | yes |
| Control variate | 100,000 | 10.4352 | 0.0177 | [10.4004, 10.4699] | 10.4506 | yes |
| Combined | 100,000 | 10.4606 | 0.0086 | [10.4438, 10.4775] | 10.4506 | yes |

The combined estimator with 10,000 paths (standard error 0.0276) is more precise than plain Monte Carlo with 100,000 paths (0.0468).

**Variance reduction by strike.** Each method was run 500 times with $n = 10{,}000$. The VR factor is $\operatorname{Var}(\text{plain price}) / \operatorname{Var}(\text{method price})$, measured from the spread of the 500 prices, not from the reported standard errors. The efficiency gain is $\text{VR factor} / (t_{\text{method}} / t_{\text{plain}})$, i.e. the variance reduction per unit of CPU time. Full data in [`results/part2_variance_reduction.csv`](results/part2_variance_reduction.csv).

| Option | K | Moneyness | Antithetic VR | Control variate VR | Combined VR | Antithetic eff. | Control variate eff. | Combined eff. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Call | 80 | ITM | 7.84 | 61.34 | 537.63 | 9.41 | 40.31 | 436.27 |
| Call | 100 | ATM | 2.41 | 7.36 | 31.92 | 2.89 | 4.83 | 25.90 |
| Call | 120 | OTM | 1.18 | 2.50 | 55.10 | 1.41 | 1.64 | 44.71 |
| Put | 80 | OTM | 1.07 | 1.31 | 9.61 | 1.28 | 0.86 | 7.80 |
| Put | 100 | ATM | 1.58 | 2.19 | 9.34 | 1.90 | 1.44 | 7.58 |
| Put | 120 | ITM | 13.17 | 6.21 | 188.54 | 15.81 | 4.08 | 152.99 |

**Cost per estimator call** at $n = 10{,}000$ (best of 5 repeats of 200 calls; the cost does not depend on the strike):

| Method | Time per call (ms) | Time ratio vs plain |
|---:|---:|---:|
| Plain MC | 0.240 | 1.00 |
| Antithetic | 0.200 | 0.83 |
| Control variate | 0.366 | 1.52 |
| Combined | 0.296 | 1.23 |

**Control variate: theory vs measurement.** The predicted factor $1/(1-\rho^2)$, with $\rho$ estimated from one run of 1,000,000 paths, matches the measured factor:

| Option | K | Moneyness | rho | Predicted 1/(1-rho^2) | Measured factor |
|---:|---:|---:|---:|---:|---:|
| Call | 80 | ITM | 0.9922 | 63.97 | 61.34 |
| Call | 100 | ATM | 0.9242 | 6.86 | 7.36 |
| Call | 120 | OTM | 0.7517 | 2.30 | 2.50 |
| Put | 80 | OTM | -0.4448 | 1.25 | 1.31 |
| Put | 100 | ATM | -0.7613 | 2.38 | 2.19 |
| Put | 120 | ITM | -0.9228 | 6.74 | 6.21 |

**Error vs sample size** (at-the-money call, 300 runs per $n$). All four methods converge at the same $1/\sqrt{n}$ rate. The fitted slopes are -0.4916 (plain), -0.4788 (antithetic), -0.4993 (control variate) and -0.5017 (combined). Variance reduction lowers the constant, not the rate:

![Error vs n](results/figures/04_error_vs_n.png)

![Variance reduction by strike](results/figures/05_variance_reduction_by_strike.png)

### Discussion

- **Where the control variate helps most: in-the-money calls.** A deep in-the-money call behaves almost like the stock itself, so its payoff is almost perfectly correlated with $S_T$ ($\rho = 0.9922$ at $K = 80$) and the variance falls by a factor of 61.
- **Where it helps little: out-of-the-money options.** For the $K = 120$ call, most payoffs are zero whatever the exact value of $S_T$, so the payoff is only partly linear in $S_T$ ($\rho = 0.7517$, factor 2.50). For the $K = 80$ put, the payoff is non-zero only in the lower tail and $\rho = -0.4448$. The factor is then 1.31, and after the extra cost of estimating $\beta$ the efficiency gain is **0.86, below 1**: on this contract the control variate alone is slightly worse than plain Monte Carlo per unit of CPU time.
- **Antithetic variates** follow the same pattern. They remove the part of the payoff that is odd in $Z$, so they help most when the payoff is close to linear in $Z$ (deep in the money: 7.84 for the $K = 80$ call, 13.17 for the $K = 120$ put). For out-of-the-money options the payoff is mostly convex and one-sided, so the factor is close to 1 (1.18 and 1.07). Because antithetics need only $n/2$ normal draws, they are also slightly cheaper than plain MC (time ratio 0.83), so their efficiency gain is above 1 in every case measured.
- **Why combining works much better than either method alone.** On the out-of-the-money call, the two methods alone give 1.18 and 2.50 but together give 55.10. This is not a coincidence. The antithetic pair average removes the odd part of the payoff and leaves its even, convex part. The averaged control $\tfrac{1}{2}\big(Y(Z)+Y(-Z)\big)$ is proportional to $\cosh(\sigma\sqrt{T}Z)$, which is also even and convex. It is therefore strongly correlated with what remains of the payoff. The two methods attack different components of the variance, so their effects compound.
- **Runtime trade-off.** At $n = 10{,}000$ the control variate costs 1.52 times plain MC per call, because of the extra arrays and the covariance estimate. Combined costs 1.23 times. These overheads are small next to the variance reductions, except for the out-of-the-money put case above. Timings depend on the machine; the ratios are the meaningful part.

## Part 3: Path-dependent exotic options

Asian and barrier options depend on the whole price path, not only on $S_T$. This is where Monte Carlo becomes genuinely necessary. Monitoring dates are $t_i = iT/m$, $i = 1, \dots, m$; $S_0$ is not a monitoring date. Paths use the exact GBM transition between dates, so the path itself has no discretisation error. Paths are simulated in batches of 20,000 and only per-path payoffs are kept, so memory does not grow with $n \times m$.

### Asian options

**Why Monte Carlo is needed.** The arithmetic Asian call pays $\big(A - K\big)^+$ with $A = \frac{1}{m}\sum_{i=1}^{m} S_{t_i}$. A sum of lognormal variables is not lognormal and has no tractable distribution, so there is no closed-form price.

**Geometric Asian: an exact benchmark.** The geometric average $G = \big(\prod_{i=1}^{m} S_{t_i}\big)^{1/m}$ *is* lognormal, because $\ln G$ is an average of jointly normal variables. Using $\frac{1}{m}\sum_i t_i = T\frac{m+1}{2m}$ and $\frac{1}{m^2}\sum_{i,j}\min(t_i,t_j) = T\frac{(m+1)(2m+1)}{6m^2}$:

$$\ln G \sim \mathcal{N}\big(\mu_G, \sigma_G^2\big), \quad \mu_G = \ln S_0 + \big(r - q - \tfrac{1}{2}\sigma^2\big)T\,\frac{m+1}{2m}, \quad \sigma_G^2 = \sigma^2 T\,\frac{(m+1)(2m+1)}{6m^2},$$

$$C_{\text{geo}} = e^{-rT}\Big(e^{\mu_G + \sigma_G^2/2}\,N(d_1) - K\,N(d_2)\Big), \qquad d_1 = \frac{\mu_G + \sigma_G^2 - \ln K}{\sigma_G}, \quad d_2 = d_1 - \sigma_G .$$

For $m = 1$ this is exactly the Black-Scholes formula, which the tests check. As $m$ grows, $\sigma_G^2$ falls from $\sigma^2T$ towards $\sigma^2T/3$: averaging lowers the effective volatility.

**Geometric Asian: a control variate.** On the same paths, take $X = e^{-rT}(A-K)^+$ and $Y = e^{-rT}(G-K)^+$, with $\mathbb{E}[Y] = C_{\text{geo}}$ known exactly, and use the Part 2 estimator

$$\hat V^{\text{CV}} = \frac{1}{n}\sum_{i=1}^{n}\Big(X_i - \hat\beta\,\big(Y_i - C_{\text{geo}}\big)\Big), \qquad \hat\beta = \frac{\widehat{\operatorname{Cov}}(X,Y)}{\widehat{\operatorname{Var}}(Y)} .$$

On every path the two averages of the same prices are very close ($A \ge G$ by AM-GM, with a small gap), so $\rho = \operatorname{Corr}(X,Y)$ is almost 1 and the variance is divided by $1/(1-\rho^2)$.

**Validation: Monte Carlo geometric Asian call vs closed form** ($K = 100$):

| m | n | MC price | Std. error | 95% CI | Closed form | CI contains exact |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 200,000 | 5.9542 | 0.0185 | [5.9179, 5.9904] | 5.9402 | yes |
| 52 | 200,000 | 5.6411 | 0.0175 | [5.6068, 5.6753] | 5.6374 | yes |
| 252 | 200,000 | 5.5748 | 0.0173 | [5.5408, 5.6087] | 5.5655 | yes |

**Arithmetic Asian call: plain MC vs geometric control variate.** Prices are at $n = 50{,}000$. The VR factor is measured from 200 independent repetitions at $n = 10{,}000$, and the cost is the best of 3 blocks of 20 calls at $n = 10{,}000$, as in Part 2.

| m | Method | MC price | Std. error | 95% CI | rho | Predicted 1/(1-rho^2) | Measured VR factor | Time ratio | Efficiency gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | Plain MC | 6.1906 | 0.0382 | [6.1157, 6.2655] |  |  | 1 | 1.00 | 1 |
| 12 | Control variate | 6.1568 | 0.0011 | [6.1547, 6.1589] | 0.99961 | 1286 | 1152 | 1.06 | 1084 |
| 52 | Plain MC | 5.8768 | 0.0366 | [5.8051, 5.9486] |  |  | 1 | 1.00 | 1 |
| 52 | Control variate | 5.8544 | 0.0010 | [5.8524, 5.8563] | 0.99960 | 1261 | 1010 | 1.02 | 994 |
| 252 | Plain MC | 5.7833 | 0.0358 | [5.7132, 5.8534] |  |  | 1 | 1.00 | 1 |
| 252 | Control variate | 5.7819 | 0.0010 | [5.7800, 5.7838] | 0.99962 | 1304 | 1145 | 1.01 | 1139 |

The control variate divides the variance by about 1,000, or equivalently it gives the precision of roughly 1,000 times as many plain paths. It costs almost nothing, because path simulation dominates the runtime: computing the geometric payoff on paths that already exist costs 1 to 6% extra. All three measured factors are below the predicted $1/(1-\rho^2)$, by 10 to 20%. With 200 repetitions, each variance estimate has a relative uncertainty of about $\sqrt{2/199} \approx 10\%$, so their ratio is uncertain by about 14%. The gaps are therefore consistent with sampling noise, but more repetitions would be needed to rule out a small systematic effect.

**Price vs number of monitoring dates** (arithmetic: MC with control variate, $n = 50{,}000$):

| m | Arithmetic (MC + CV) | Std. error | Geometric (closed form) | European (BS) |
|---:|---:|---:|---:|---:|
| 1 | 10.4506 | 0.0000 | 10.4506 | 10.4506 |
| 2 | 8.1121 | 0.0015 | 7.9434 | 10.4506 |
| 4 | 6.9394 | 0.0012 | 6.7335 | 10.4506 |
| 12 | 6.1552 | 0.0011 | 5.9402 | 10.4506 |
| 52 | 5.8545 | 0.0010 | 5.6374 | 10.4506 |
| 252 | 5.7815 | 0.0010 | 5.5655 | 10.4506 |

With $m = 1$ the arithmetic and geometric averages are both $S_T$, so the control variate is exact (standard error 0). Both Asian prices then coincide with the European price, as they must.

![Asian price vs monitoring dates](results/figures/06_asian_vs_monitoring.png)

### Barrier options

**Contract.** A down-and-out call pays $(S_T - K)^+\,\mathbf{1}\{S_{t_i} > H \text{ for all } i\}$. The down-and-in call pays when the barrier *is* hit, so on every path knock-in + knock-out = vanilla.

**Continuous closed form** (Reiner-Rubinstein, $H \le K$). With $\lambda = \frac{r - q + \sigma^2/2}{\sigma^2}$ and $y = \frac{\ln\big(H^2/(S_0K)\big)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}$:

$$C_{\text{DI}} = S_0e^{-qT}\Big(\frac{H}{S_0}\Big)^{2\lambda}N(y) - Ke^{-rT}\Big(\frac{H}{S_0}\Big)^{2\lambda-2}N\big(y - \sigma\sqrt{T}\big), \qquad C_{\text{DO}} = C_{\text{BS}} - C_{\text{DI}} .$$

The tests check this formula against an independent numerical integration of the payoff over the absorbed density given by the reflection principle (agreement to $10^{-8}$).

**Discrete-monitoring bias and the BGK correction.** The formula above assumes the barrier is checked at every instant. Real contracts check it at $m$ dates, so a path can dip below $H$ between two dates without being knocked out. The discrete option is therefore knocked out *less* often and is worth *more*: the continuous formula understates the discrete down-and-out price. Broadie, Glasserman and Kou (1997) showed that the discrete price is well approximated by the continuous formula with the barrier shifted away from the spot:

$$H_{\text{eff}} = H\,e^{-\beta\sigma\sqrt{T/m}}, \qquad \beta = -\frac{\zeta(1/2)}{\sqrt{2\pi}} \approx 0.5826,$$

where $\zeta$ is the Riemann zeta function. The shift is of order $\sqrt{T/m}$, so the uncorrected bias decays like $1/\sqrt{m}$.

**Down-and-out call, $K = 100$, $H = 90$, $n = 200{,}000$.** The bias is measured against the Monte Carlo price of the discretely monitored contract:

| m | MC price | Std. error | 95% CI | Continuous CF | Bias (continuous - MC) | BGK CF | Bias (BGK - MC) | BGK in CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 9.5571 | 0.0330 | [9.4925, 9.6218] | 8.6655 | -0.8916 | 9.5802 | 0.0231 | yes |
| 52 | 9.1465 | 0.0328 | [9.0822, 9.2108] | 8.6655 | -0.4810 | 9.1736 | 0.0271 | yes |
| 252 | 8.8977 | 0.0327 | [8.8336, 8.9617] | 8.6655 | -0.2322 | 8.9139 | 0.0162 | yes |
| 1000 | 8.7871 | 0.0326 | [8.7231, 8.8510] | 8.6655 | -0.1216 | 8.7941 | 0.0070 | yes |

Even with daily monitoring ($m = 252$), the continuous formula is off by 0.2322, about 7 standard errors. After the BGK correction, the remaining difference is within the Monte Carlo noise at every $m$ in this table.

![Barrier monitoring bias](results/figures/07_barrier_monitoring_bias.png)

Two results in the full sweep behind Figure 7 are worth reporting as they are:

- **The fitted slope is -0.4385, not -0.5.** This is the log-log slope of $|\text{continuous} - \text{MC}|$ against $m$, over $m = 4$ to $1000$. The $1/\sqrt{m}$ law is the leading-order term: higher-order terms matter at coarse monitoring. At fine monitoring the bias (0.1434 at $m = 504$, 0.1216 at $m = 1000$) is only about 4 standard errors of the Monte Carlo estimate (0.033), so those points are noisy.
- **At $m = 4$ (quarterly monitoring), the BGK price is outside the 95% CI** (BGK - MC = 0.0660, about 2 standard errors). BGK is an asymptotic correction for frequent monitoring, and with only 4 dates the error it neglects is no longer small.

**Price vs barrier level** ($m = 252$, $n = 100{,}000$): the BGK-adjusted formula is within the Monte Carlo 95% CI for every barrier from $H = 60$ to $H = 99$.

| H | MC price | Std. error | BGK CF | BGK - MC | BGK in CI |
|---:|---:|---:|---:|---:|---:|
| 60 | 10.4878 | 0.0465 | 10.4506 | -0.0373 | yes |
| 65 | 10.4186 | 0.0464 | 10.4506 | 0.0320 | yes |
| 70 | 10.4106 | 0.0465 | 10.4499 | 0.0393 | yes |
| 75 | 10.4643 | 0.0468 | 10.4407 | -0.0236 | yes |
| 80 | 10.4319 | 0.0470 | 10.3706 | -0.0613 | yes |
| 85 | 10.0742 | 0.0466 | 10.0322 | -0.0420 | yes |
| 90 | 8.9326 | 0.0464 | 8.9139 | -0.0187 | yes |
| 93 | 7.5655 | 0.0451 | 7.5340 | -0.0316 | yes |
| 95 | 6.2274 | 0.0429 | 6.1960 | -0.0315 | yes |
| 97 | 4.4254 | 0.0379 | 4.4598 | 0.0344 | yes |
| 99 | 2.3133 | 0.0292 | 2.2799 | -0.0333 | yes |

![Barrier price vs level](results/figures/08_barrier_price_vs_level.png)

**In-out parity** on the same 100,000 paths ($m = 52$, $H = 90$): down-and-out 9.127808 + down-and-in 1.292498 = 10.420305, equal to the vanilla price on the same paths (10.420305) up to $1.78 \times 10^{-15}$.

### Limitations of Part 3

- Constant volatility and interest rate (Black-Scholes dynamics). With a volatility smile, barrier prices in particular can change substantially.
- Only down barriers on calls, and the closed form requires $H \le K$. Up barriers, puts and rebates are not implemented.
- The Monte Carlo prices the *discretely* monitored contract. To price a *continuously* monitored barrier by simulation, one would add a Brownian-bridge correction: the probability that the path crossed $H$ between two dates. This is not implemented, so the continuous formula is used only as a closed-form benchmark.
- BGK is an asymptotic correction. It is accurate here from $m = 12$ upwards but not at $m = 4$.
- The Asian control variate uses a beta estimated from the same sample (bias of order $1/n$, as in Part 2).

## Part 4: Value at Risk and Expected Shortfall

The last part moves from pricing to risk measurement, on **real market data**. It computes the 1-day VaR and ES of an equally weighted portfolio of five CAC 40 stocks with four methods, then backtests them over 13 years.

### Definitions and conventions

Losses are **positive** numbers. For a confidence level $\alpha$ (95%, 97.5% and 99%; 97.5% is the level used for ES in the Basel FRTB framework):

$$\mathrm{VaR}_\alpha = \inf\{\ell : P(L \le \ell) \ge \alpha\}, \qquad \mathrm{ES}_\alpha = \mathbb{E}\big[L \mid L \ge \mathrm{VaR}_\alpha\big].$$

VaR says "on 99% of days we lose less than this". ES answers the question VaR ignores: *how much do we lose on the other 1%?*

Daily **log** returns $r_i$ are modelled, but the portfolio P&L is computed exactly from simple returns, with value $V = 1{,}000{,}000$ EUR and weights $w_i = 1/5$:

$$L = -V\sum_{i} w_i\,\big(e^{r_i} - 1\big).$$

The mean return is set to zero by default (`use_mean=False`) in the parametric and Monte Carlo engines. At a 1-day horizon it is negligible next to the volatility, and estimating it adds noise. For an $h$-day horizon the covariance is scaled by $h$ (square-root-of-time rule). **This is an approximation**: it assumes i.i.d. returns, which volatility clustering contradicts.

### Methods

**Correlated returns by Cholesky.** Write the covariance as $\Sigma = LL^\top$ with $L$ lower triangular. If $z$ is a vector of independent standard normals ($\operatorname{Cov}(z) = I$), then $x = Lz$ satisfies $\operatorname{Cov}(x) = L\,I\,L^\top = \Sigma$. Multiplying by $L$ turns independent shocks into shocks with the right variances *and* correlations.

1. **Parametric normal (closed form).** The portfolio return is *linearised*, $\sum_i w_i r_i \approx \sum_i w_i(e^{r_i}-1)$. It is then normal with $\sigma_p = \sqrt{w^\top\Sigma w}$, so that
   $$\mathrm{VaR}_\alpha = V\,(z_\alpha\sigma_p - \mu_p), \qquad \mathrm{ES}_\alpha = V\Big(\sigma_p\,\frac{\varphi(z_\alpha)}{1-\alpha} - \mu_p\Big).$$
2. **Monte Carlo normal.** $x = Lz$ through Cholesky, with each scenario revalued exactly.
3. **Monte Carlo Student-t** ($\nu = 5$ by default). $x = \sqrt{\tfrac{\nu-2}{\nu}}\,\dfrac{Lz}{\sqrt{W}}$, with $W \sim \chi^2_\nu/\nu$. The vector $Lz/\sqrt{W}$ has covariance $\Sigma\,\nu/(\nu-2)$, and the factor $\sqrt{(\nu-2)/\nu}$ brings it back to exactly $\Sigma$. The normal and t models therefore share variances and correlations and differ only in shape. Because a single $W$ scales all assets, a "turbulent" scenario (small $W$) hits every stock at once. This is **tail dependence**, which the multivariate normal does not have.
4. **Historical simulation.** Apply each of the last 500 days of actual (raw, not demeaned) returns to today's portfolio and take the empirical quantile and tail mean. Nothing is simulated or assumed, and the `use_mean` flag does not apply.
5. **Monte Carlo normal with EWMA covariance** (RiskMetrics, $\lambda = 0.94$). Same as 2, but the covariance is $\Sigma = \sum_k (1-\lambda)\lambda^k\, r_{t-k}r_{t-k}^\top$ (weights rescaled to sum to 1 over the window). Recent days count more, with an effective memory of about $1/(1-\lambda) \approx 17$ days.

**The 95% vs 99% crossover.** A unit-variance $t_5$ has its 95% quantile at $\sqrt{3/5}\times 2.015 = 1.561$, *below* the normal's 1.645. Its 99% quantile is $\sqrt{3/5}\times 3.365 = 2.606$, *above* the normal's 2.326. With the same variance, the t puts more mass in the centre and in the far tails and less in the "shoulders", so it is only riskier far out. A fat-tailed model can therefore give a *lower* 95% VaR and a *higher* 99% VaR than the normal.

### Data

| Ticker | Company | Sector |
|---|---|---|
| MC.PA | LVMH | Luxury |
| TTE.PA | TotalEnergies | Energy |
| SAN.PA | Sanofi | Healthcare |
| BNP.PA | BNP Paribas | Banking |
| AIR.PA | Airbus | Aerospace |

- **Source:** Yahoo Finance via `yfinance` (version 1.7.0), daily adjusted close prices (`auto_adjust=True`) in EUR.
- **Download date:** 2026-10-22.
- **Period:** 2012-01-02 to 2026-10-20, giving 3,786 aligned prices (no rows dropped) and 3,785 daily returns.
- **Sanity checks:** no non-positive prices. One return is flagged by the $|r| > 25\%$ check and kept after review: Airbus on 2020-03-18 (log return -25.06%), during the COVID-19 crash.
- **Regeneration:** the data is not committed (`data/` is in `.gitignore`), because Yahoo Finance data should not be redistributed. Run `python scripts/download_data.py` to regenerate it. Re-downloading later extends the period and can change adjusted prices slightly, so the numbers below are those of the 2026-10-22 download. The tests do not use this data.

**Summary (full sample, daily log returns):**

| Asset | Annualised volatility | Skewness | Excess kurtosis | Worst day | Best day |
|---:|---:|---:|---:|---:|---:|
| MC.PA | 27.2% | 0.16 | 3.93 | -9.08% | 12.06% |
| TTE.PA | 25.7% | -0.60 | 13.26 | -18.16% | 14.04% |
| SAN.PA | 22.4% | -1.29 | 17.01 | -20.99% | 8.10% |
| BNP.PA | 31.3% | -0.45 | 7.48 | -19.12% | 16.54% |
| AIR.PA | 33.1% | -0.39 | 14.91 | -25.06% | 18.62% |
| Portfolio | 20.7% | -0.59 | 9.89 | -13.02% | 11.55% |

Correlations range from 0.28 (Sanofi-Airbus) to 0.54 (BNP-TotalEnergies and BNP-Airbus); the full matrix is in [`results/part4_correlation.csv`](results/part4_correlation.csv). Every asset and the portfolio have strongly positive excess kurtosis (a normal has 0): **returns are fat-tailed**.

![Return distribution](results/figures/09_return_distribution.png)

### Results on the last date

Estimation window: the last 500 days (2024-11-04 to 2026-10-20). 1-day horizon, EUR, Monte Carlo with 500,000 scenarios.

| Method | VaR 95% | ES 95% | VaR 97.5% | ES 97.5% | VaR 99% | ES 99% |
|---:|---:|---:|---:|---:|---:|---:|
| Parametric normal | 17,077 | 21,415 | 20,348 | 24,271 | 24,152 | 27,670 |
| MC normal | 16,811 | 21,013 | 19,998 | 23,796 | 23,717 | 27,071 |
| MC Student-t (nu = 5) | 15,903 | 22,738 | 20,278 | 27,661 | 26,497 | 34,880 |
| MC normal, EWMA cov. | 13,199 | 16,563 | 15,761 | 18,764 | 18,650 | 21,379 |
| Historical simulation | 14,708 | 23,729 | 20,070 | 30,408 | 28,154 | 43,331 |

![VaR and ES comparison](results/figures/10_var_es_comparison.png)

What the numbers show:

- **MC normal reproduces the closed form, up to linearisation.** The MC normal 99% VaR (23,717) is 1.8% below the parametric one (24,152). Across 100 repetitions, the standard deviation of the MC estimate is only 54.5 EUR, so this gap is not noise. It comes from the closed form's linear P&L. Since $e^{r} - 1 > r$, a stock that falls loses slightly less than the linear approximation says, and the exact revaluation in the Monte Carlo captures this.
- **The Student-t crossover appears on real data.** Compared with the normal (same covariance), the t model gives a *lower* 95% VaR (15,903 vs 16,811) and a *higher* 99% VaR (26,497 vs 23,717). Its ES is higher at every level, because ES looks into the tail itself.
- **Historical simulation gives the largest 99% VaR and ES** (28,154 and 43,331). The window contains a cluster of large losses in spring 2025 (Figure 12), which no normal model with the same variance reproduces. Its 99% ES is 1.54 times its VaR, against 1.14 for the normal model: the empirical tail beyond the VaR is much heavier.
- **EWMA gives the lowest risk** (18,650 at 99%): the most recent weeks were calmer than the two-year average.

**10-day 99% VaR and ES** (covariance $\times 10$; historical simulation scaled by $\sqrt{10}$):

| Method | VaR 99% | ES 99% |
|---:|---:|---:|
| Parametric normal | 76,376 | 87,501 |
| MC normal | 72,425 | 82,523 |
| MC Student-t (nu = 5) | 80,450 | 104,370 |
| MC normal, EWMA cov. | 57,221 | 65,362 |
| Historical simulation x sqrt(10) | 89,031 | 137,025 |

The parametric 10-day VaR is exactly $\sqrt{10}$ times the 1-day value. The MC normal 10-day VaR (72,425) is below $\sqrt{10} \times 23{,}717 \approx 75{,}000$, because the convexity of $e^r - 1$ grows with the size of the moves.

**Student-t sensitivity to $\nu$** (1-day, 99%, same covariance). Lower $\nu$ means fatter tails and higher VaR and ES; the effect on ES is much larger:

| nu | VaR 99% | ES 99% |
|---:|---:|---:|
| 4 | 27,138 | 37,336 |
| 5 | 26,327 | 34,835 |
| 8 | 25,441 | 31,420 |
| 30 | 24,275 | 28,214 |
| normal (infinite) | 23,717 | 27,071 |

**Monte Carlo uncertainty** (MC normal, 100 independent repetitions of 500,000 scenarios, EUR):

| Level | Mean VaR | Std VaR | Mean ES | Std ES |
|---:|---:|---:|---:|---:|
| 95% | 16,837 | 36.8 | 21,079 | 38.1 |
| 97.5% | 20,042 | 42.8 | 23,868 | 46.1 |
| 99% | 23,754 | 54.5 | 27,178 | 60.2 |

**Convergence: tail estimates need many scenarios.** The last two columns give the standard deviation of the estimate divided by $\sigma_pV/\sqrt{n}$, which is the standard error of a simple *mean* with the same $n$.

| n | Mean VaR | Std VaR | Mean ES | Std ES | Std VaR / (sigma_p V / sqrt n) | Std ES / (sigma_p V / sqrt n) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 23,576 | 1,234 | 27,129 | 1,478 | 3.76 | 4.50 |
| 3,000 | 23,631 | 642 | 27,078 | 847 | 3.39 | 4.47 |
| 10,000 | 23,698 | 395 | 27,143 | 475 | 3.80 | 4.58 |
| 30,000 | 23,761 | 209 | 27,204 | 289 | 3.49 | 4.82 |
| 100,000 | 23,761 | 123 | 27,174 | 145 | 3.73 | 4.41 |
| 300,000 | 23,754 | 69 | 27,172 | 83 | 3.62 | 4.38 |
| 1,000,000 | 23,752 | 38 | 27,162 | 39 | 3.70 | 3.77 |

The 99% VaR estimate is about 3.4 to 3.8 times noisier than a mean estimated from the same number of scenarios, and ES about 3.8 to 4.8 times. (The asymptotic value for a normal quantile is $\sqrt{0.99 \times 0.01}/\varphi(z_{0.99}) \approx 3.73$.) Matching the precision of a mean therefore takes roughly $3.7^2 \approx 14$ times more scenarios for VaR, and more for ES. The estimates still converge at the $1/\sqrt{n}$ rate. They converge to the exact-P&L value, not to the dashed linearised closed form, which explains the visible gap in Figure 11.

![Convergence](results/figures/11_convergence.png)

### Backtest

Rolling 1-day forecasts from 2013-12-18 to 2026-10-20 (3,285 days). Each forecast uses only the previous 500 days (no look-ahead; a test checks this), with 20,000 scenarios per Monte Carlo forecast. **Common random numbers:** the same normal matrix and chi-square vector are reused every day and only the covariance changes. This removes day-to-day simulation noise from the VaR series, but it makes the Monte Carlo error of consecutive days correlated.

**Tests used:**

- **Kupiec (POF):** is the exception rate equal to $1-\alpha$?
- **Christoffersen independence:** is an exception today independent of yesterday's?
- **Conditional coverage (CC):** both at once.
- **Basel (1996) traffic light:** green, yellow or red zone, from the number of 99% exceptions over the last 250 days. Used here for illustration only.
- **ES diagnostic** (no formal test): if the ES is right, the average loss on exception days should be close to the average ES predicted for those days.

| Method | Level | Exceptions | Expected | Rate | Kupiec p | Indep. p | CC p | Last 250 d | Avg loss on exc. | Avg predicted ES |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parametric normal | 99% | 63 | 32.9 | 1.92% | <0.0001 | 0.0002 | <0.0001 | 1 (green) | 41,223 | 31,701 |
| MC normal | 99% | 72 | 32.9 | 2.19% | <0.0001 | 0.0002 | <0.0001 | 1 (green) | 39,731 | 31,080 |
| MC Student-t (nu = 5) | 99% | 48 | 32.9 | 1.46% | 0.0129 | <0.0001 | <0.0001 | 1 (green) | 45,811 | 40,074 |
| MC normal, EWMA cov. | 99% | 75 | 32.9 | 2.28% | <0.0001 | 0.0351 | <0.0001 | 5 (yellow) | 34,258 | 28,405 |
| Historical simulation | 99% | 51 | 32.9 | 1.55% | 0.0032 | 0.0001 | <0.0001 | 2 (green) | 42,739 | 42,226 |
| Parametric normal | 95% | 150 | 164.3 | 4.57% | 0.2473 | <0.0001 | <0.0001 | 7 | 30,533 | 24,054 |
| MC normal | 95% | 155 | 164.3 | 4.72% | 0.4549 | <0.0001 | <0.0001 | 7 | 30,117 | 23,522 |
| MC Student-t (nu = 5) | 95% | 174 | 164.3 | 5.30% | 0.4393 | <0.0001 | <0.0001 | 8 | 28,939 | 25,455 |
| MC normal, EWMA cov. | 95% | 194 | 164.3 | 5.91% | 0.0204 | <0.0001 | <0.0001 | 13 | 25,306 | 21,384 |
| Historical simulation | 95% | 176 | 164.3 | 5.36% | 0.3522 | <0.0001 | <0.0001 | 10 | 28,975 | 27,513 |

![Backtest](results/figures/12_backtest_99.png)

**Discussion.**

- **No model passes at 99%.** Every 99% VaR has too many exceptions, and Kupiec rejects all five at the 5% level. The normal models are worst (63 and 72 exceptions against 32.9 expected). The fat-tailed models come closest: Student-t 48 (p = 0.0129) and historical 51 (p = 0.0032).
- **At 95%, the exception rate is fine for four models out of five.** Kupiec does not reject the parametric normal, MC normal, Student-t or historical models; it does reject EWMA (p = 0.0204). This matches the crossover: the normal is only too thin in the far tail. At 95% the Student-t has *more* exceptions than the normal (174 vs 155), exactly as its lower 95% VaR predicts.
- **Every model fails the independence test, at both levels.** Exceptions cluster in stress periods: the MC normal model has 17 of its 72 exceptions in 2020 and 13 in 2015, against 0 in 2017. This is **volatility clustering**: calm and turbulent periods alternate, and a 500-day equally weighted window reacts slowly to both. It underestimates risk when a crisis starts and overestimates it afterwards. Figure 12 shows the reverse "ghost" effect: the equally weighted VaR drops sharply in early 2022, when the March 2020 crash leaves the 500-day window, without anything happening in the market that day.
- **EWMA reduces clustering but does not fix the tails.** With a 17-day memory, EWMA has fewer back-to-back exceptions than the MC normal model (5 pairs of consecutive exceptions vs 8). Its independence p-value rises from 0.0002 to 0.0351, still a rejection at 5%. But it has the *most* exceptions (75 at 99%, 194 at 95%). EWMA fixes the *level* of volatility but keeps a normal *shape*: after calm stretches its forecast volatility is low, and the conditional tails are still thinner than reality. Combining EWMA with fat tails (Student-t innovations or filtered historical simulation) would address both problems; it is not implemented here.
- **The normal ES is too optimistic.** On 99% exception days, the average realised loss exceeds the average predicted ES for every model: 39,731 vs 31,080 for the MC normal, 45,811 vs 40,074 for the Student-t, and only 42,739 vs 42,226 for historical simulation. Historical simulation is the only model whose ES is close to what actually happened in the tail.
- **The parametric and MC normal models differ in the backtest** (63 vs 72 exceptions) even though their VaRs differ by only about 2% (exact P&L, plus the noise of 20,000 scenarios). Near the 99% quantile, small changes in VaR move the exception count noticeably.
- **The traffic light over the last 250 days is green for all models except EWMA** (5 exceptions, yellow). Taken alone, that window would hide the problems that the full 13-year backtest reveals.

### Limitations of Part 4

- **i.i.d. returns and square-root-of-time scaling.** The backtest shows that returns are *not* i.i.d. (clustered exceptions), so 10-day figures obtained by $\sqrt{10}$ scaling are approximations.
- **Portfolio scope.** It is a linear portfolio of five cash equities with no options, rebalanced to equal weights every day in the backtest. Options would require full revaluation of non-linear payoffs in each scenario.
- **Covariance estimation error.** A 500-day sample covariance is estimated with error, and that error is ignored. Only five assets are used, so diversification and estimation issues are much smaller than in a real book.
- **Common random numbers** in the backtest make the Monte Carlo errors of consecutive days correlated.
- **No formal ES backtest** (for example Acerbi-Szekely); only the average-loss diagnostic above.
- **For illustration only.** This is not a regulatory model: the traffic light is used only as an illustration, and FRTB uses ES at 97.5% with liquidity horizons, stressed periods and other features not covered here.

## Project structure

```
.
├── src/mcfinance/
│   ├── gbm.py              # exact GBM simulation (terminal values and full paths)
│   ├── black_scholes.py    # closed-form benchmark
│   ├── pricing.py          # MC estimator, MCResult with CI, convergence curve
│   ├── variance_reduction.py  # antithetic, control variate, combined, VR and timing helpers
│   ├── exotics_closed_form.py # geometric Asian, continuous barrier, BGK correction
│   ├── exotics.py          # batched MC for Asian and barrier options
│   └── risk.py             # VaR/ES engines, EWMA, backtest and coverage tests
├── tests/                  # pytest suite
├── scripts/
│   ├── run_part1.py        # reproduces the Part 1 table and figures
│   ├── run_part2.py        # reproduces the Part 2 tables and figures
│   ├── run_part3.py        # reproduces the Part 3 tables and figures
│   ├── download_data.py    # downloads the Part 4 prices (Yahoo Finance)
│   └── run_part4.py        # Part 4 VaR/ES, figures and backtest
├── data/                   # downloaded prices (not committed, see .gitignore)
├── results/                # generated tables and figures
├── pyproject.toml
└── requirements.txt
```

## Testing

```bash
pytest
```

The suite (114 tests) checks:

- Black-Scholes against textbook values (call 10.4506, put 5.5735), put-call parity across strikes and dividend yields, monotonicity in volatility, and input validation.
- Monte Carlo call and put prices within 4 standard errors of Black-Scholes ($n = 400{,}000$).
- Reproducibility with a fixed seed.
- Standard error halves when the sample size is multiplied by 4.
- 95% CI coverage between 92% and 98% over 500 repetitions.
- $\mathbb{E}[S_T] = S_0 e^{(r-q)T}$ (martingale property of the discounted price).
- Path shapes, consistency between `convergence_curve` and `price_european`, and invalid inputs.
- Variance reduction: each estimator (antithetic, control variate, combined) within 4 standard errors of Black-Scholes for calls and puts, with and without dividends; a smaller standard error than plain MC at the same $n$; reproducibility; odd $n$ rejected for antithetic sampling.
- 95% CI coverage between 92% and 98% for each variance reduction method. This checks that their standard errors are computed correctly, in particular that antithetic pairs are not treated as independent.
- The measured control-variate factor matches $1/(1-\rho^2)$ within 15%; a payoff identical to the control gives zero variance; the estimated $\beta$ equals the regression slope.
- Exotics, closed forms: the geometric Asian with $m = 1$ equals Black-Scholes (call and put, with and without dividends), and the geometric Asian call decreases with $m$ and stays below the European call. The barrier formula matches an independent reflection-principle integration, satisfies in-out parity and tends to the vanilla as $H \to 0$. Invalid barriers ($H \ge S_0$, $H > K$) are rejected, and the BGK shift goes in the right direction.
- Exotics, Monte Carlo: the geometric Asian is within 4 standard errors of its closed form, with 95% CI coverage in [0.92, 0.98]. The arithmetic Asian with one date equals the European MC price on the same draws; AM-GM holds pathwise ($A \ge G$); the arithmetic Asian is below the European call. The control variate agrees with plain MC and reduces the standard error more than tenfold.
- Barrier Monte Carlo: knock-in + knock-out = vanilla on the same paths; a far barrier gives the vanilla; with $m = 1000$ the price matches the BGK formula; the unadjusted continuous formula understates the discrete down-and-out price. Results do not depend on the batch size, statistically and, with the same seed, exactly.
- Risk (synthetic data only, no network): Cholesky draws have the target covariance and a non positive definite matrix is rejected. MC normal VaR/ES match the closed form at 95/97.5/99% within 4 standard errors (with small volatilities, so that the linearisation error of the closed form stays below the tolerance). The parametric VaR scales exactly with $\sqrt{h}$.
- Student-t engine: it has the target covariance and positive excess kurtosis, it tends to the normal for $\nu = 1000$, and it shows the 95%/99% crossover (higher VaR than the normal at 99%, lower at 95%). Historical VaR equals `np.quantile` of the losses, ES is always at least VaR, and weights must sum to 1.
- Backtest tests: Kupiec is 0 when the exception count equals its expectation, matches a hand-computed value (5 exceptions in 250 days at 1%) and rejects a wrong rate. Christoffersen rejects clustered exceptions but not spread-out ones. Edge cases (0 or only exceptions) are handled, and traffic-light boundaries are checked. On i.i.d. normal data the rolling backtest is calibrated (Kupiec not rejected), and a crash on day $t$ does not change the forecast for day $t$ (no look-ahead).

## Assumptions and limitations

- Black-Scholes world: constant volatility and interest rate, continuous dividend yield, lognormal prices, no transaction costs. Real markets show volatility smiles and jumps.
- European exercise only in Part 1. American options need methods such as Longstaff-Schwartz.
- The confidence interval relies on the central limit theorem. It is approximate for small $n$ or for heavy-tailed payoffs.
- Plain Monte Carlo converges slowly ($1/\sqrt{n}$). Part 2 reduces the constant, not the rate. Quasi-Monte Carlo (low-discrepancy sequences) could improve the rate itself and is not covered.
- In Part 2, the control-variate $\beta$ is estimated from the same sample (bias of order $1/n$). Runtime measurements depend on the machine and on NumPy's implementation; only the ratios between methods are meaningful.
- Pseudo-random numbers come from NumPy's PCG64 generator. Results are reproducible for a given seed and NumPy version.
