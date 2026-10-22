# Part 2: Variance reduction

[Back to the README](../README.md) · [Part 1](part1_european_options.md) · [Part 2](part2_variance_reduction.md) · [Part 3](part3_exotic_options.md) · [Part 4](part4_var_es.md)

Reproduce with `python scripts/run_part2.py` from the repository root.

Plain Monte Carlo error falls like $\sigma_X/\sqrt{n}$. Halving it costs four times as many paths. Variance reduction lowers $\sigma_X$ instead. All methods below are compared at the **same budget**: $n$ is always the total number of payoff evaluations.

## Method

**Antithetic variates.** Draw $n/2$ normals $Z_i$ and use each one twice, as $Z_i$ and $-Z_i$. With $X(Z) = e^{-rT}h\big(S_T(Z)\big)$:

$$\hat V^{\text{AV}} = \frac{1}{n/2}\sum_{i=1}^{n/2} \frac{X(Z_i) + X(-Z_i)}{2}.$$

The $n/2$ pair averages are i.i.d., but the two members of a pair are negatively correlated. The standard error must therefore be computed from the pair averages, with $n/2$ observations, and not from the $n$ individual payoffs. The method helps when the payoff is close to monotone in $Z$: writing $X = X_{\text{even}} + X_{\text{odd}}$, the pair average removes the odd part $X_{\text{odd}}$ exactly.

**Control variate.** Use $Y = e^{-rT}S_T$, whose expectation is known exactly because the discounted asset is a martingale under $\mathbb{Q}$: $\mathbb{E}[Y] = S_0e^{-qT}$. The adjusted estimator

$$\hat V^{\text{CV}} = \frac{1}{n}\sum_{i=1}^{n}\Big(X_i - \beta\,\big(Y_i - \mathbb{E}[Y]\big)\Big)$$

is unbiased for any fixed $\beta$. Its variance is minimised by

$$\beta^* = \frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(Y)}, \qquad \operatorname{Var}\big(X - \beta^*(Y - \mathbb{E}[Y])\big) = (1-\rho^2)\operatorname{Var}(X),$$

so the variance reduction factor is $1/(1-\rho^2)$, where $\rho = \operatorname{Corr}(X, Y)$. $\beta^*$ is estimated from the same sample, which introduces a bias of order $1/n$, negligible next to the $1/\sqrt{n}$ statistical error at these sample sizes.

**Combined.** The control variate is applied to the antithetic pair averages $\big(\tfrac{X(Z)+X(-Z)}{2}, \tfrac{Y(Z)+Y(-Z)}{2}\big)$. The averaged control still has expectation $S_0e^{-qT}$.

## Results

**At-the-money call** ($K = 100$, same contract and seed as Part 1). Full table in [`results/part2_pricing_table.csv`](../results/part2_pricing_table.csv).

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

**Variance reduction by strike.** Each method was run 500 times with $n = 10{,}000$. The VR factor is $\operatorname{Var}(\text{plain price}) / \operatorname{Var}(\text{method price})$, measured from the spread of the 500 prices, not from the reported standard errors. The efficiency gain is $\text{VR factor} / (t_{\text{method}} / t_{\text{plain}})$, i.e. the variance reduction per unit of CPU time. Full data in [`results/part2_variance_reduction.csv`](../results/part2_variance_reduction.csv).

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

![Error vs n](../results/figures/04_error_vs_n.png)

![Variance reduction by strike](../results/figures/05_variance_reduction_by_strike.png)

## Discussion

- **Where the control variate helps most: in-the-money calls.** A deep in-the-money call behaves almost like the stock itself, so its payoff is almost perfectly correlated with $S_T$ ($\rho = 0.9922$ at $K = 80$) and the variance falls by a factor of 61.
- **Where it helps little: out-of-the-money options.** For the $K = 120$ call, most payoffs are zero whatever the exact value of $S_T$, so the payoff is only partly linear in $S_T$ ($\rho = 0.7517$, factor 2.50). For the $K = 80$ put, the payoff is non-zero only in the lower tail and $\rho = -0.4448$. The factor is then 1.31, and after the extra cost of estimating $\beta$ the efficiency gain is **0.86, below 1**: on this contract the control variate alone is slightly worse than plain Monte Carlo per unit of CPU time.
- **Antithetic variates** follow the same pattern. They remove the part of the payoff that is odd in $Z$, so they help most when the payoff is close to linear in $Z$ (deep in the money: 7.84 for the $K = 80$ call, 13.17 for the $K = 120$ put). For out-of-the-money options the payoff is mostly convex and one-sided, so the factor is close to 1 (1.18 and 1.07). Because antithetics need only $n/2$ normal draws, they are also slightly cheaper than plain MC (time ratio 0.83), so their efficiency gain is above 1 in every case measured.
- **Why combining works much better than either method alone.** On the out-of-the-money call, the two methods alone give 1.18 and 2.50 but together give 55.10. This is not a coincidence. The antithetic pair average removes the odd part of the payoff and leaves its even, convex part. The averaged control $\tfrac{1}{2}\big(Y(Z)+Y(-Z)\big)$ is proportional to $\cosh(\sigma\sqrt{T}Z)$, which is also even and convex. It is therefore strongly correlated with what remains of the payoff. The two methods attack different components of the variance, so their effects compound.
- **Runtime trade-off.** At $n = 10{,}000$ the control variate costs 1.52 times plain MC per call, because of the extra arrays and the covariance estimate. Combined costs 1.23 times. These overheads are small next to the variance reductions, except for the out-of-the-money put case above. Timings depend on the machine; the ratios are the meaningful part.

## Tests

`tests/test_variance_reduction.py` checks:

- Each estimator (antithetic, control variate, combined) is within 4 standard errors of Black-Scholes for calls and puts, with and without dividends, and has a smaller standard error than plain MC at the same $n$. Results are reproducible, and an odd $n$ is rejected for antithetic sampling.
- 95% CI coverage is between 92% and 98% for each method. This checks that the standard errors are computed correctly, in particular that antithetic pairs are not treated as independent.
- The measured control-variate factor matches $1/(1-\rho^2)$ within 15%; a payoff identical to the control gives zero variance; the estimated $\beta$ equals the regression slope.

## Assumptions and limitations

- Variance reduction lowers the constant, not the $1/\sqrt{n}$ rate. Quasi-Monte Carlo (low-discrepancy sequences) could improve the rate itself and is not covered.
- The control-variate $\beta$ is estimated from the same sample (bias of order $1/n$).
- Runtime measurements depend on the machine and on NumPy's implementation; only the ratios between methods are meaningful.
