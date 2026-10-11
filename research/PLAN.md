# Pre-registration: optimal vs naive diversification for a Canadian investor

**Status: DRAFT** (committed 2026-10-09, before any strategy was compared on dev data).
It is frozen in its last commit before the tag `prereg-v1` (blueprint Step 7a). After the
freeze it changes only through a dated, signed-off amendment at the end, and the holdout is
run once (Step 7b).

## 1. Question

Do long-only **minimum variance** and **equal risk contribution (ERC)** portfolios beat the
naive **1/N** portfolio after transaction costs, risk-adjusted (Sharpe ratio), for a USD
investor and for a Canadian investor who holds the same US assets unhedged, out of sample?

This replicates and extends DeMiguel, Garlappi and Uppal (2009), "Optimal versus naive
diversification: how inefficient is the 1/N portfolio strategy?", *Review of Financial
Studies* 22(5). The extension adds shrinkage covariance estimators, ERC, a CAD view,
transaction costs, a time-series-robust Sharpe test, a luck benchmark, and risk-model
validation.

## 2. Data and fixed design

These were set in the blueprint (2026-10-01) before any development.

- **Assets:** the Fama-French 12 industry portfolios, daily and value-weighted (Kenneth R.
  French Data Library). The 49 industries are a robustness check.
- **Factors:** FF5 plus momentum, daily. The US risk-free rate is French's `RF`.
- **FX:** USD/CAD, CAD per USD, the daily noon New York rate (FRED `DEXCAUS`).
- **Canadian rates:** the 3-month T-bill (Statistics Canada table 10-10-0122-01) and the
  10-year benchmark yield (Bank of Canada Valet).
  - The 3-month T-bill is converted to a daily rate as the return from rolling bills,
    (1 + y·91/365)^(d/91) − 1.
- **Views:**
  - USD.
  - CAD unhedged: (1 + r_usd)(1 + fx) − 1.
  - CAD hedged by covered interest parity: r_usd + rf_cad − rf_usd.
  - Excess returns use each view's own risk-free rate.
  - The hedged view's excess return equals the USD one exactly, so it is never tested. It
    adds only total-return metrics.
- **Estimation currency:** USD-view strategies estimate on USD returns, and CAD-view
  strategies on CAD unhedged returns.
- **Periods:**
  - Dev: USD from 1963-07, CAD from 1971-01, to **DEV_END = 2010-12-31**.
  - **Holdout: 2011-01-01 → HOLDOUT_END.** HOLDOUT_END is the last day of the last complete
    calendar month present in all of the 12 industries, the factors, USD/CAD and the CA
    T-bill, at the vintage frozen in §12. It is set in §13.
- **Execution:**
  - Rebalancing is monthly. Each strategy estimates through the close of the last trading
    day of the month and trades at the next close.
  - Costs are 5 bp per side, charged on Σ|w_target − w_drifted|.
  - The backtest starts invested, with no opening cost.
- **Guard:** the data loaders stop at DEV_END unless `allow_holdout` is set. Only the
  `reproduce` command (`src/portfolio_lab/cli.py`) sets it, and a test scans for it. The
  `--holdout` mode of `reproduce` refuses to run unless the tag `prereg-v1` exists and this
  file is identical to the tagged copy.

## 3. Hypotheses (confirmatory)

For strategy S and view V, Δ(S, V) = SR(S) − SR(1/N), the difference in Sharpe ratios of
monthly net excess returns over the holdout months.

| | S | V | H0 | H1 |
|---|---|---|---|---|
| **H1** | minimum variance (primary configuration) | USD | Δ = 0 | Δ ≠ 0 |
| **H2** | ERC (primary configuration) | USD | Δ = 0 | Δ ≠ 0 |
| **H3** | minimum variance (primary configuration) | CAD unhedged | Δ = 0 | Δ ≠ 0 |
| **H4** | ERC (primary configuration) | CAD unhedged | Δ = 0 | Δ ≠ 0 |

- **Family:** the four tests together, Holm at a **family-wise error rate of 5%**,
  two-sided.
- **How results read:**
  - A rejection with Δ̂ > 0 reads "beats 1/N", and with Δ̂ < 0 "loses to 1/N".
  - No rejection reads "no detectable difference at this power" (§8), never "equal".
- **Evaluation sample:**
  - Each strategy runs once, continuously, from the common first rebalance of the strategy
    set (the first month end where every strategy has its lookback) to HOLDOUT_END.
  - Its months 2011-01 → HOLDOUT_END are then cut out.
  - The January 2011 portfolio is December 2010's drifted holdings, and January's trade
    pays its cost.

## 4. Primary configuration

- **Window:** 252 trading days. **Cap:** none (long-only, sum to 1). **Cost:** 5 bp.
- **Estimator:** one estimator serves both strategies and both views, chosen on dev data by
  this rule, written before the dev comparison:
  - The candidates are `sample`, `lw_identity` (Ledoit-Wolf 2004, scaled identity target)
    and `lw_constant_correlation` (Ledoit-Wolf 2004, constant-correlation target).
  - Choose the candidate whose `min variance (<estimator>, 252d)` has the **lowest annualized
    volatility of monthly net excess returns over the dev months, USD view**. The start is
    the common first rebalance of the full strategy set.
  - The rule never looks at a mean or a Sharpe ratio. Lowest out-of-sample variance is
    minimum variance's own objective and the criterion of the Ledoit-Wolf papers.
  - EWMA is not a candidate. Its decay of 0.94 is a 1-day RiskMetrics factor, and its
    minimum variance turns over about 29% a month. It stays exploratory (§6).
- The chosen estimator is recorded in §13.

## 5. Test procedure

- **Returns:** monthly simple excess returns. Daily net returns are compounded within each
  calendar month, and the view's risk-free rate is compounded the same way and subtracted.
  - Ledoit and Wolf used log returns in their examples. Simple returns keep the test
    consistent with every other metric here.
  - The reported Sharpe ratio is the monthly one × √12, so it differs slightly from the
    daily-based Sharpe ratio in the performance tables.
- **Statistic** (Ledoit and Wolf 2008, "Robust performance hypothesis testing with the
  Sharpe ratio", *Journal of Empirical Finance* 15(5)):
  - Δ̂ = f(v̂), with v = (μ1, μ2, γ1, γ2), γ = E[r²] and f = μ1/√(γ1 − μ1²) − μ2/√(γ2 − μ2²).
  - s(Δ̂) = √(∇f(v̂)' Ψ̂ ∇f(v̂) / T).
- **Ψ̂ on the original data:** a HAC estimate of the long-run covariance of
  y_t = (r1 − μ1, r2 − μ2, r1² − γ1, r2² − γ2), as in the authors' own code (CRAN package
  `bootTimeInference`, `compute.se.Parzen.pw`).
  - **Prewhitening** (Andrews and Monahan 1992): fit a VAR(1) without intercept by OLS. Cap
    the singular values of its matrix A at 0.97. Take the residuals e_t (n = T − 1 rows).
  - **Kernel:** Parzen, with bandwidth S = 2.6614 (α̂ n)^(1/5). α̂ is Andrews' (1991) AR(1)
    plug-in on the columns of e, with equal weights.
  - **Formula:** Ψ̂_e = n/(n − 4) Σ_{|j| < S} k(j/S) Γ̂(j), with Γ̂(j) = (1/n) Σ e_t e'_{t−j}.
    Then Ψ̂ = (I − A)⁻¹ Ψ̂_e (I − A)⁻¹'.
  - The paper's text proposes the quadratic-spectral kernel and says the Parzen kernel
    "yields very similar performance" (footnote 8). Its authors' code uses Parzen.
- **Bootstrap:**
  - A circular block bootstrap (Politis and Romano 1992) of the monthly return pairs, with
    block size b.
  - A bootstrap sample has l = ⌊T/b⌋ blocks (l·b rows), each starting at a uniform position,
    with wrap-around.
  - Each bootstrap sample is studentized by its own block standard error.
    - y* is centred at the bootstrap sample's own moments.
    - ζ_j = b^(−1/2) Σ_{t=1..b} y*_{(j−1)b+t} and Ψ̂* = (1/l) Σ ζ_j ζ_j'.
    - s(Δ̂*) uses the gradient at the bootstrap moments and the original T.
  - With d = |Δ̂|/s(Δ̂) and d̃*_m = |Δ̂*_m − Δ̂|/s(Δ̂*_m), the p-value is
    (#{d̃*_m ≥ d} + 1)/(M + 1), with **M = 4999**.
- **Block size: Algorithm 3.1 of Ledoit and Wolf (2008), as a rule.**
  - Fit a VAR(1) with intercept to the return pair by OLS.
  - Generate **K = 1000** pseudo-series of length T. Each starts from the first observation,
    is iterated with residuals resampled by a stationary bootstrap (mean block 5), and drops
    a burn-in of 50.
  - For each b in **{1, 2, 4, 6, 8, 10, 12}**, test H0: Δ = Δ̂ (the original estimate, the
    pseudo-parameter) on every pseudo-series (M = 499), and count the rejections at 5%.
  - Take the b whose rejection rate is closest to 5%, which is the same as coverage closest
    to 95%. Ties go to the smaller b.
  - This follows the authors' `block.size.calibrate`, with a finer grid and M = 499 for this
    longer sample. Their defaults are {1, 3, 6, 10} and M = 199.
  - The p-values for every b in the grid are reported as a sensitivity.
  - The calibration runs for each primary pair. Exploratory pairs (§6) use the b calibrated
    for the same view's minimum variance pair.
- **Seeds:** every random draw uses NumPy's PCG64, seeded from **SEED = 20261009** and a
  stable label for the pair. The manifest records the versions.

## 6. Variants and trial count

Twelve industries, 252-day windows and 5 bp unless stated. These are the strategy names used
in the code.

- **Benchmark:** `1/N`.
- **Minimum variance:**
  - `min variance (sample, 252d)`
  - `min variance (lw_identity, 252d)`
  - `min variance (lw_constant_correlation, 252d)`
  - `min variance (ewma, 252d)`
  - `min variance (sample, 252d, cap 0.25)`
  - `min variance (lw_identity, 252d, cap 0.25)`
  - `min variance (lw_constant_correlation, 252d, cap 0.25)`
  - `min variance (ewma, 252d, cap 0.25)`
- **ERC:**
  - `ERC (sample, 252d)`
  - `ERC (lw_identity, 252d)`
  - `ERC (lw_constant_correlation, 252d)`
  - `ERC (ewma, 252d)`
- **Tilts of 1/N:**
  - `momentum tilt (strength 0.5) on 1/N`
  - `momentum tilt (strength 1) on 1/N`
  - `low-vol tilt (strength 0.5, 252d) on 1/N`
  - `low-vol tilt (strength 1, 252d) on 1/N`

That is **K = 16 strategies besides 1/N**, the primary pair among them, so **32 comparisons
with 1/N** over the USD and CAD views.

- **The exploratory comparisons** (the 32 minus the primary four):
  - They are reported with raw p-values and with Holm-adjusted p-values over all 32.
  - They are reported with the luck benchmark (§7).
  - **None is claimed as a finding.**
- **Robustness, 7b, primary pair only, no new hypotheses:**
  - windows of 126 and 504 days
  - costs of 0, 10 and 20 bp (the same trades repriced)
  - the CAD-hedged view (total-return metrics)
  - the 49 industries (cut-able)

## 7. Luck benchmark

For each strategy, R = 1000 random long-only shadow strategies with the strategy's exact
turnover, on the same days and view, at the same cost.

- **How a shadow trades:**
  - It starts at a Dirichlet(1) portfolio at the strategy's first trade.
  - At each later trade it draws u ~ Dirichlet(1) and trades to w + λ(u − w), where w is
    its drifted weights and λ = min(1, 2τ_t / Σ|u − w|). τ_t is the strategy's turnover at
    that trade.
  - When λ caps at 1 the turnover falls short. The shortfalls are counted and reported.
- **Outputs:**
  - Each strategy's and 1/N's percentile among their own shadows' Sharpe ratios.
  - **Best of K:** the share of the R replicates in which the best of K shadows (one per
    strategy) reaches the best actual Sharpe ratio among the K strategies.
- It is run on the dev period (where the selection of §4 happened) and on the holdout.
- **What it measures** (stated after the dev run, by the owner's decision, with no change to
  the method):
  - A fresh random target at every trade gives a shadow no persistent exposure, so the
    shadows' Sharpe ratios cluster near 1/N's. On dev, in USD, the median is 0.39 and the
    95th percentile 0.41.
  - A strategy's percentile therefore says whether it beat random rebalancing of the same
    assets at the same turnover and cost. **It is not a data-snooping correction** for the K
    strategies tried; Holm over the 32 comparisons (§6) is.
  - Best of K is near 0 for any persistent tilt. It is conservative, because shadows of
    different strategies are independent while the strategies are correlated.
  - Where the cap binds, shadows trade less than their strategy; the luck table reports the
    turnover ratio. On dev it binds often only for the EWMA strategies, at 24-30% of trades.

## 8. Power: minimum detectable difference

- **Formula:** MDE = (z_{1−α/2} + z_{0.80}) × SE, for 80% power.
  - The SE is the HAC standard error of Δ̂ measured on dev, scaled to the holdout's number of
    months, and annualized (× √12).
  - It is stated at α = 0.05 and at 0.0125, Holm's first step.
- **Values measured while planning** (2026-10-09, 188 holdout months, Bartlett HAC, all three
  candidate estimators):

  | Pair | Correlation with 1/N | MDE, α = 0.05 | MDE, α = 0.0125 |
  |---|---|---|---|
  | Minimum variance vs 1/N, USD | 0.78-0.81 | 0.43-0.46 | 0.51-0.54 |
  | Minimum variance vs 1/N, CAD | 0.76-0.78 | 0.46-0.48 | 0.55-0.57 |
  | ERC vs 1/N, USD and CAD | 0.997 | 0.05-0.06 | 0.06-0.07 |

  The values for the chosen estimator, on the frozen vintage, go in §13.
- **Stated in advance:**
  - The minimum variance tests can only detect annual Sharpe differences of about half a
    unit. A null result there is the expected outcome and is not evidence that the
    strategies are equal.
  - The ERC tests can detect differences of about 0.06.

## 9. Secondary measures (descriptive, no family-wise claims)

- **Jobson-Korkie with Memmel's (2003) correction.** It assumes iid normal returns, which
  monthly returns violate (volatility clustering, fat tails). It is reported for comparison
  with DeMiguel et al.
- **Certainty-equivalent return** with γ = 1, as in DeMiguel et al.
- **CVaR:** the historical expected shortfall at 95% of daily net returns, as a positive loss.
- **Max drawdown**, annual turnover and annual cost.
- The full metric table: CAGR, volatility, Sharpe, Sortino, Calmar, skew, kurtosis, and
  tracking error and information ratio against the market.

## 10. Risk-model validation (secondary)

- **VaR backtests:**
  - The series are the daily net returns of 1/N and the primary pair, in the USD and CAD
    views, over the holdout days.
  - The forecasts are one day ahead, and their windows may use dev history.
  - The methods are historical, normal, Cornish-Fisher, and filtered historical simulation
    (EWMA λ = 0.94), all with a **500-day window**, at 95% and 99%.
  - The tests are Kupiec (the LR and an exact binomial p-value), Christoffersen independence,
    conditional coverage, and the Basel traffic-light zones (99%, 250-day windows).
  - FHS with a 1000-day window is a declared sensitivity.
- **Stress:**
  - The portfolios are 1/N and the primary pair, at their HOLDOUT_END holdings.
  - Historical scenarios are replayed buy-and-hold: GFC 2008-09-01 → 2008-11-30, COVID
    2020-02-19 → 2020-03-23, rates 2022-01-01 → 2022-12-31. The strategies' realized paths
    are shown next to them.
  - Hypothetical one-variable shocks use weekly partial macro betas, estimated over 2011-01 →
    HOLDOUT_END. The 2001 → DEV_END betas are the comparison.
    - market −20%
    - momentum −35%
    - Canadian 10-year +200 bp
    - CAD ±10%

## 11. Prior knowledge and dev looks already taken

- **Prior knowledge:** the owner and the assistant both know the literature:
  - 1/N is hard to beat out of sample (DeMiguel et al. 2009).
  - The low-volatility anomaly.
  - Shrinkage helps minimum variance.
- **The holdout period is known history.** Both also know the broad history of 2011-2026: a
  long US equity bull market led by technology, the 2020 crash and recovery, the 2022
  inflation and rate rise, and a stronger US dollar against the Canadian dollar. The holdout
  guards against tuning on these specific series, not against that knowledge.
- **Dev results seen before this draft** (steps 3-6, all logged in NOTES.md):
  - Step 3: weights only. Minimum variance is concentrated in Utilities (100% in 59 of 559
    months with the sample estimator).
  - Step 4: 1/N gross and net CAGR and Sharpe (a sanity check against a daily-rebalanced
    1/N), and every strategy's turnover. No strategy-vs-1/N performance table.
  - Step 5a: VaR exception counts and backtests on 1/N and minimum variance (sample) return
    series.
  - Step 6: the 2008 replay and the realized 2008 paths of 1/N and minimum variance (sample),
    where minimum variance lost less (−22.1% vs −30.4% in USD).
- **Planning this step** (2026-10-09) measured only dates, the autocorrelation of 1/N's
  monthly returns, correlations with 1/N and standard errors (§8). No Sharpe ratio, mean,
  volatility or difference of any strategy was printed.

## 12. Vintage and reproduction

- **Refresh:** every source is downloaded once, just before the dev analysis that this file
  records. That vintage is frozen.
- **The manifest** (`research/results/manifest.json`, committed with the freeze) records, for
  each source file:
  - the URL
  - the file's header line
  - its SHA-256, size and retrieval time
- **It also records:**
  - the Python and package versions
  - the git commit
  - the seed, DEV_END and HOLDOUT_END
  - the command
- **The command:** `uv run portfolio-lab reproduce` regenerates every number.
  - On the same vintage the numbers must match to 1e-6 (checked in 7b).
  - On a newer vintage it warns, names the changed files and reports the drift.
- **Post-registration data:**
  - Months the French library publishes after the tag `prereg-v1` were seen by nobody at the
    freeze.
  - The frozen strategies may be run on them later, unchanged.
  - That run is descriptive only, too short to test, reported apart from the holdout and
    never merged into H1-H4.
  - It uses only the sources above. No other data is spliced in.

## 13. Dev decisions (to be filled in before the freeze)

- Vintage refreshed on: _to be filled_
- HOLDOUT_END: _to be filled_ (expected 2026-08-31, 188 months)
- Estimator chosen by the rule of §4: _to be filled_, with every candidate's dev volatility
- Calibrated block sizes, by view: _to be filled_
- MDE for the chosen configuration: _to be filled_
- Dev results (descriptive), all with the luck benchmark:
  - performance
  - the Sharpe tests on dev
  - the VaR backtests
  - the GFC stress
  - _to be filled_

## Change log

- 2026-10-09: draft committed before any dev comparison (blueprint Step 7a, plan
  `plans/step-7a-plan.md`, all twelve decisions approved by the owner).
- 2026-10-09, still before any dev comparison: §5 now follows the authors' published code
  (CRAN `bootTimeInference`) where the paper is silent or differs.
  - The prewhitened Parzen kernel replaces the quadratic-spectral one.
  - The exact bootstrap sample length, centring and calibration steps are spelled out.
