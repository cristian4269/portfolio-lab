# Step 7a plan: pre-registration, dev analysis, luck benchmark, reproduce CLI

Approved by the owner on 2026-10-09 (all twelve decisions as recommended, plus the
post-registration addendum in decision 6). Read with the blueprint's Step 7a section (and 7b,
which this step pre-registers). Branch `step-7a-prereg`.

Before drafting, a scratch script measured the inputs below on dev data. It printed no Sharpe
ratio, mean, volatility or difference of any strategy: only dates, autocorrelations,
correlations with 1/N and standard errors. Ranking the strategies on dev is session 3's job, and
it must come **after** the draft PLAN.md is committed. The numbers are under "Grounding
2026-10-09".

## The order that makes this a pre-registration

1. The owner approves this plan.
2. **Session 1 opens by committing the draft `research/PLAN.md`** (the decisions below as the
   study's rules), before any strategy is compared on dev.
3. Sessions 1-2 build the tools, on synthetic data.
4. Session 3 refreshes the data, then runs the dev analysis. It applies the dev rules written
   in the draft (decisions 2 and 6), records the results in PLAN.md, and finishes the reviews.
   The last commit freezes PLAN.md and gets the local tag `prereg-v1`.
5. The owner pushes the tag (its own gate). No holdout number exists until 7b.

## What gets built

- `research/PLAN.md`:
  - the question, hypotheses and primary tests
  - the variants and trial count
  - metrics, the minimum detectable difference, the prior-knowledge statement
  - HOLDOUT_END
  - the dev decisions (filled in session 3)
- `report/inference.py`: the Sharpe-ratio tests (decisions 3-5).
  - `sharpe_diff(r1, r2)`: Δ̂ = SR1 − SR2, with SR = μ/√(γ − μ²) and γ = E[r²], as in Ledoit
    and Wolf (2008), plus its gradient.
  - `hac_psi(y)`: the HAC covariance with the prewhitened QS kernel and the Andrews-Monahan
    (1992) automatic bandwidth.
  - `lw_test(r1, r2, *, block, resamples, rng)`: the studentized circular block bootstrap,
    giving Δ̂, its SE, the p-value and the CI.
  - `calibrate_block(r1, r2, *, blocks, pseudo, rng)`: LW's Algorithm 3.1.
  - `jobson_korkie_memmel(r1, r2)`: the secondary test.
  - `holm(pvalues)`, `min_detectable_difference(se, *, alpha, power)`, and `ceq(r, gamma=1)`
    (DeMiguel et al.'s certainty equivalent).
- `risk/backtest.kupiec_exact(exc, level)`: the exact two-sided binomial p-value next to the
  chi-square LR (5a's note: the chi-square size at 99% over 250 days is 9.5%).
- `backtest/luck.py`: random long-only "shadow" strategies with each strategy's exact turnover
  (decision 8). The simulator is vectorized over shadows, and a test checks it against
  `engine.simulate`.
- `study.py`: the design as code.
  - `PRIMARY`, `EXPLORATORY`, `ROBUSTNESS`, `SEED`, the test settings and the dev selection
    rule.
  - The functions that turn data into result tables: performance, tests, luck, VaR backtests
    and stress.
  - It takes `allow_holdout` as a parameter and never contains the literal `=True`.
  - A test checks that every strategy name in `study.py` appears in PLAN.md.
- `cli.py reproduce` (decision 9): the only `allow_holdout=True` in `src/`.
  - It writes `research/results/<period>/*.csv` and `research/results/manifest.json`.
  - Holdout mode is locked behind the tag.
- `config.HOLDOUT_END`, set in session 3 (decision 6).

## Decisions for the owner (recommendation first)

1. **Primary hypotheses: four confirmatory tests, two-sided, Holm at a 5% family-wise error
   rate.**
   - H1 and H2: minimum variance vs 1/N and ERC vs 1/N, in the USD view. H3 and H4: the same
     pair in the CAD (unhedged) view.
   - The statistic is the difference in Sharpe ratios of **monthly net excess returns**, at
     5 bp, over the holdout months 2011-01 → HOLDOUT_END.
     - Each strategy runs once, continuously from its first dev rebalance (the common start of
       `compare`) to HOLDOUT_END. Its holdout months are then cut out.
     - The January 2011 portfolio is therefore December's drifted holdings, and January's
       trade pays its cost.
   - A rejection with Δ̂ > 0 reads "beats 1/N" and with Δ̂ < 0 "loses to 1/N". No rejection
     reads "no detectable difference", never "equal".
   - Alternatives:
     - One-sided tests: more power, but the blueprint's question allows losing.
     - A separate family per view: less conservative, but USD and CAD are the same strategies
       seen through FX.
   - The CAD-hedged view gets no test. Its excess return equals the USD one exactly (CIP,
     Step 2's property test), so it only adds total-return metrics.
2. **Primary configuration: a 252-day window, no cap, 5 bp. One estimator for both strategies,
   chosen on dev by a rule that never looks at a Sharpe ratio.**
   - **The rule:** among `sample`, `lw_identity` and `lw_constant_correlation`, take the one
     whose 252-day minimum variance has the lowest dev volatility of monthly net excess
     returns in the USD view. The common start is the one `compare` sets.
   - The same estimator then serves ERC and both views.
   - Lowest out-of-sample variance is minimum variance's own objective and the criterion of
     the Ledoit-Wolf shrinkage papers.
   - EWMA is left out of the primary candidates. Its decay 0.94 is a 1-day RiskMetrics
     factor, and it turns over 29% a month (Step 4). It stays exploratory.
   - Alternatives:
     - Fix `lw_constant_correlation` a priori: simplest, and no dev step.
     - Pick the best dev Sharpe: more data snooping, which the luck benchmark would then have
       to absorb.
     - `sample`: DeMiguel et al.'s own shortsale-constrained portfolio, which is 100%
       Utilities in 59 of 559 months.
3. **The test: Ledoit and Wolf (2008), as published.**
   - **Data:** monthly simple excess returns. LW used log returns in their examples; simple
     returns keep the test consistent with every other metric here (stated in PLAN.md).
   - **The SE on the original data:** a HAC estimate with VAR(1) prewhitening and the
     Andrews AR(1) plug-in bandwidth. It is implemented here, since statsmodels has neither.
     - **Amended in session 1, before any dev comparison:** the kernel is **Parzen**
       (bandwidth 2.6614 (αn)^(1/5), singular values of A capped at 0.97), as in the authors'
       own R code (CRAN `bootTimeInference`).
     - The paper's text names QS, and its footnote 8 says Parzen performs the same.
     - The code also settles the bootstrap sample length (l·b), the centring (bootstrap
       moments) and the calibration loop (rejection rate at H0: Δ = Δ̂).
     - The details are in PLAN.md §5.
   - **The bootstrap:**
     - It is a circular block bootstrap of the return pairs.
     - Each bootstrap sample gets its own block SE: l = ⌊T/b⌋, ζ_j = b^(−1/2) Σ y*, and
       Ψ* = (1/l) Σ ζ_j ζ_j'.
     - The p-value is (#{d̃* ≥ d} + 1)/(M + 1), with M = 4999.
   - **The block size: LW's Algorithm 3.1, pre-registered as a rule.**
     - The rule fits a VAR(1) to the pair and resamples its residuals by a stationary
       bootstrap with mean block 5.
     - It simulates K = 1000 pseudo-series and, for each b in {1, 2, 4, 6, 8, 10, 12} (with
       M = 499 inside), the coverage of a 95% interval.
     - It takes the b whose coverage is closest to 95%.
     - The p-value for every b in the grid is reported as a sensitivity.
   - **This amends the blueprint's "block length pre-registered":** the rule is fixed, not
     the number, as LW's own method requires.
   - **Exploratory pairs** (decision 7) use the b calibrated for the same view's minimum
     variance pair, so the calibration is not run 32 times.
   - Seeds are fixed and derived from `SEED` and the pair's name.
   - Alternatives:
     - A fixed b (6, or ⌊T^(1/3)⌋ = 5): simpler and one session lighter, but not LW's method.
     - Daily data: LW recommend a bivariate GARCH model for calibrating daily data, which is
       much harder, and the power comes from the span, not the frequency.
4. **Secondary measures, reported without family-wise claims:**
   - Jobson-Korkie with Memmel's correction, with its iid-normal caveat.
   - CEQ (γ = 1, as DeMiguel et al.).
   - CVaR: the historical ES at 95% of daily net returns, as a positive loss.
   - Max drawdown, annual turnover, and the full `Result.summary` table.
5. **Minimum detectable difference, stated before the holdout.**
   - MDE = (z_{1−α/2} + z_{0.80}) × SE.
     - The SE is the dev HAC SE of Δ̂, scaled to the holdout's T, and annualized.
     - α is 0.05, and also 0.0125, Holm's first step.
   - Grounding values for the annual Sharpe difference:
     - **Minimum variance vs 1/N: 0.43-0.48** (0.51-0.57 at Holm's first step).
     - **ERC vs 1/N: 0.05-0.06.** ERC's correlation with 1/N is 0.997.
   - PLAN.md therefore says this in advance:
     - The minimum variance tests can only detect a very large difference. A null result
       there is the expected outcome and is not evidence of equality.
     - The ERC tests can detect small differences.
6. **HOLDOUT_END and the vintage: refresh every source once, at the start of session 3,
   before the dev run.**
   - HOLDOUT_END is the last day of the last complete calendar month present in all of: the
     12 industries, the factors, USD/CAD and the CA T-bill.
   - The current cache gives **2026-08-31** (the French files end there), so 188 holdout
     months.
   - The value goes into `config.HOLDOUT_END`, PLAN.md and the manifest, all in the freeze
     commit.
   - Refreshing first matters: the French library revises history. The dev decisions must be
     made on the vintage that gets frozen.
   - The 49-industry file is refreshed too (its cached copy is from a later day than the
     others).
   - The alternative is to freeze from the cache as it stands, whose files were retrieved on
     different days.
   - **Post-registration addendum (owner, 2026-10-09):**
     - Months the French library publishes after the `prereg-v1` tag are data nobody had
       seen at the freeze.
     - The frozen strategies may be run on them later, with no changes to the design.
     - That run is descriptive only: a few months are far too short to test anything.
     - It is reported apart from the holdout and never merged into the four main tests.
     - The data must come from the same sources. Splicing in other data (ETFs, personal-use
       feeds) is ruled out by invariant 3 and would mix different return series.
7. **Exploratory variants and the trial count.** Twelve industries and 252 days unless
   stated:
   - Minimum variance × {sample, lw_identity, lw_constant_correlation, ewma} × cap {none,
     0.25}: 8. The 0.25 cap took effective N from 1.8 to 4.6 in Step 3.
   - ERC × the 4 estimators: 4.
   - Momentum 12-1 and low-volatility tilts of 1/N at strength {0.5, 1}: 4.
   - **That is K = 16 strategies besides 1/N** (the primary pair among them), so 32
     comparisons with 1/N over the two views.
     - They are reported with raw and Holm-adjusted p-values over the 32 and with the luck
       benchmark.
     - Nothing exploratory is claimed as a finding.
   - **Robustness (7b, the primary pair only, no new hypotheses):**
     - windows of 126 and 504 days
     - costs of 0, 10 and 20 bp (`reprice`)
     - the CAD-hedged view (total-return metrics)
     - 49 industries, if not cut
   - Step 4's momentum tilt on a 126-day minimum variance (a timing probe) is dropped.
   - **The trial disclosure in PLAN.md also lists the dev looks already taken:**
     - Step 4: the 1/N CAGR and Sharpe sanity table, and every strategy's turnover.
     - Step 5a: the VaR exceptions of 1/N and minimum variance.
     - Step 6: the GFC replay and realized 2008 paths of 1/N and minimum variance (sample),
       where minimum variance lost less.
8. **Luck benchmark: turnover-matched random shadows, R = 1000 per strategy.**
   - A shadow starts at a Dirichlet(1) portfolio at the strategy's first trade.
   - At each later trade, it draws u ~ Dirichlet(1) and trades to w + λ(u − w), with w the
     shadow's drifted weights and λ = min(1, 2τ_t / Σ|u − w|).
     - Its turnover then equals the strategy's τ_t at that trade, so the costs match too.
     - The days and the view are the same as the strategy's.
     - When λ caps at 1 (rare: P < 0.0005 for minimum variance, about 3% at 29% turnover),
       the shortfall is counted and reported.
   - **Outputs:**
     - Each strategy's percentile among its own shadows' Sharpe ratios.
     - 1/N's percentile.
     - **Best of K:** the share of the R replicates where the best of the K shadows (one per
       strategy) reaches the best actual Sharpe ratio.
   - It runs on dev and on the holdout. `SEED` = 20261009, and NumPy's PCG64 is fixed in the
     manifest.
   - Alternatives:
     - The deflated Sharpe ratio (Bailey and López de Prado 2014): analytic, but it assumes
       the trials' Sharpe ratios are exchangeable, and the blueprint asks for this benchmark.
     - Random portfolios without turnover matching: their cost drag would differ.
9. **`reproduce`: dev by default, the holdout locked behind the tag.**
   - The command is `portfolio-lab reproduce [--refresh] [--holdout]`.
   - `--holdout` refuses unless:
     - the git tag `prereg-v1` exists
     - `research/PLAN.md` is byte-identical to `git show prereg-v1:research/PLAN.md`
     - `config.HOLDOUT_END` is set
   - In 7a only dev mode ever runs. The lock is tested in a temporary git repo.
   - **The manifest:**
     - per source: URL, the file's header line, SHA-256, size and retrieval time (from the
       cache sidecar)
     - Python and package versions
     - the git commit and a dirty flag
     - the seed, DEV_END, HOLDOUT_END, the command and the run time
   - **On a newer vintage** (a SHA differs from the committed manifest), it warns, names the
     files and carries on. 7b reports the drift.
   - The manifest from session 3's dev run is committed with the freeze, which timestamps the
     vintage.
10. **VaR backtests (secondary, pre-registered for 7b):**
    - The series are the daily net returns of 1/N and the primary pair, in USD and CAD.
    - All four methods, **window 500 for every method**, at 95% and 99%.
    - The forecasts for holdout days may use dev history.
    - The tests are Kupiec (LR plus the exact binomial p), Christoffersen, conditional
      coverage, and the Basel zones (99%, 250 days).
    - FHS with a 1000-day window is a declared sensitivity: it did slightly better on dev in
      5a, and choosing it now would be a dev selection.
11. **Stress, for 7b:**
    - The portfolios are the primary pair and 1/N at their HOLDOUT_END final weights.
    - Scenarios gfc, covid and rates, through `historical.table(..., allow_holdout=True)` from
      `reproduce`, plus the realized paths.
    - Shocks use weekly partial macro betas over 2011-01 → HOLDOUT_END, the period the
      current weights live in. Step 6's 2001 → DEV_END span is the comparison.
12. **Speed:** no SLSQP warm start unless the dev `reproduce` run takes over 10 minutes.
    - The grounding ran 7 strategies on both views in 17 s.
    - The full grid plus robustness is about 22 strategies over 1964 → 2026, an estimated 1-2
      minutes, plus the luck and calibration runs, measured in session 2.

## Conventions

- **Monthly returns:** daily net returns compounded within each calendar month, minus the
  view's risk-free rate compounded the same way. Annualized Sharpe = monthly × √12. These
  differ slightly from the daily Sharpe ratios of `Result.summary`, which is stated.
- **LW notation:** y_t = (r1 − μ1, r2 − μ2, r1² − γ1, r2² − γ2), and s(Δ̂) = √(∇f' Ψ ∇f / T).
  Ψ is a population (1/T) covariance. At b = 1 the block formula is the sample covariance of
  the bootstrap data (LW footnote 9), which is tested.
- **Holm:** sort the p-values ascending, compare p_(k) with α/(m − k + 1), and stop at the
  first that fails. Adjusted p-values are monotone and capped at 1.
- **Seeds:** every random draw comes from `np.random.default_rng` seeded from `SEED` and a
  stable label (`zlib.crc32` of the name), never from global state.

## Sessions

**Session 1: draft PLAN.md, then the inference tools.** Tests first, then code.
- Commit the draft `research/PLAN.md` first (decisions 1-12, with dev results "to be filled
  in session 3").
- `sharpe_diff` and its gradient: hand values on 6 monthly returns, and the gradient against
  finite differences.
- `hac_psi`:
  - The QS kernel's weights at hand points.
  - The Andrews (1991) AR(1) plug-in bandwidth, by hand, on a 2-column example.
  - Prewhitening on a hand VAR(1).
  - With a bandwidth near 0 and no prewhitening it reduces to the 1/T sample covariance.
- `lw_test`:
  - Ψ* at b = 1 equals the sample covariance.
  - The block Ψ* by hand for T = 8, b = 2.
  - The p-value formula on a fixed draw.
  - The circular blocks wrap around.
  - Seeded runs are repeatable.
- **The size check:** iid normal pairs with equal Sharpe ratios, T = 120. Over 300
  simulations with M = 199, the rejection rate at 5% falls inside the 99% binomial band.
  This is a slow test, run in CI only if it takes under 30 s.
- `calibrate_block`:
  - On iid data it picks a small b.
  - On a strongly autocorrelated VAR(1) it picks a larger one.
  - It is seeded.
- JK-Memmel by hand; Holm on a textbook example; MDE in closed form; CEQ by hand;
  `kupiec_exact` against `scipy.stats.binomtest`.
- RED and GREEN commits. Then a mutation run:
  - γ centred instead of raw
  - the gradient's sign
  - b^(−1/2) dropped
  - ≥ replaced by >
  - Holm's divisor off by one
  - no prewhitening recolouring
  - the QS weight formula
  - the Memmel term dropped
- As built (2026-10-09, see NOTES.md):
  - Draft PLAN.md `0f3c664`. §5 amended to follow the authors' code (`d8d8efd`).
  - RED `bfb79ae`, GREEN `05a5f0a`, survivor tests `bedd6ce`.
  - 58 inference tests, 100% coverage of the module. A loop-based reference HAC matches to
    1e-10. The iid size check is inside its 99% band.
  - The mutation run caught 21 of 21: the plan's eight, plus the bandwidth constant, the cap,
    n/(n − 4), the bootstrap centre, ties, the binomial rate, the geometric mean, the
    burn-in, wrap-around, the calibration null, rf compounding, the lag direction and the
    Andrews exponent.
  - Timing: `lw_test` (M = 4999) 0.12 s; `calibrate_block` about 2 minutes per pair at
    T = 188.

**Session 2: luck benchmark, study and reproduce.**
- `luck.py`:
  - The vectorized shadows equal `engine.simulate` on 5 shadows to 1e-12, with costs.
  - Turnover equals τ_t exactly when λ < 1, and the shortfall is reported when it caps.
  - Weights stay ≥ 0 and sum to 1.
  - The seed is repeatable.
  - Best-of-K by hand on 3 strategies × 4 replicates.
- `study.py`:
  - The registry: names unique and all present in PLAN.md.
  - The pipeline on a small synthetic universe produces every table.
  - The dev selection rule (decision 2) by hand.
- `cli.py reproduce`:
  - The manifest's fields come from a fake cache.
  - The drift warning appears on a changed SHA.
  - `--holdout` refuses without the tag, with an edited PLAN.md, and without HOLDOUT_END, in a
    temporary git repo.
  - The text scan still passes.
- **Timing of a dev run** on the current cache (performance tables stay unread until session
  3; only the clock and the shape of the output are checked).
- RED, GREEN and a mutation run:
  - λ without the cap
  - turnover with ½ missing
  - shadows sharing one RNG stream
  - best-of-K taking the mean instead of the max
  - the lock skipped
  - the manifest reading the wrong SHA

- As built (2026-10-09, see NOTES.md):
  - luck RED `c7b169d` / GREEN `c344831`; bootstrap speed-up `7b60031`; `report(start=)`
    `2b939c7`.
  - study RED `a61d955` / GREEN `8970045`, written in one pass.
  - reproduce RED `ab72d21` / GREEN `d66aa6c`; survivor tests `b397b62`.
  - Mutation run: 19 of 19 (16 at first). Session 1's run is still 21 of 21 on the new
    bootstrap.
  - Dev `reproduce` timing: 571 s, so no warm start. Only the clock and the table shapes
    were read.

**Session 3: the dev analysis, the reviews, the freeze.**
- Refresh every source (network; keyless public sources, polite delays) and set HOLDOUT_END
  (decision 6).
- Run `reproduce` (dev).
  - Apply decision 2's rule and record the estimator.
  - Record the dev results: performance, tests, luck, VaR backtests, the GFC stress.
  - Record the MDE from the final vintage and the trial disclosure.
- python-reviewer, `/code-review` (high), then verification-loop. Fix the findings. Re-run the
  dev analysis and check the numbers are unchanged.
- Check invariants 1-6. Update NOTES.md, the blueprint's status line and the change log.
- **Last commit:** PLAN.md status "FROZEN", `config.HOLDOUT_END` and `manifest.json`. Then the
  local tag `prereg-v1` on that commit.
  - A `--merge` merge keeps the commit, so the tag stays valid on main.
- Ask the owner before pushing the branch and opening the PR, and separately before pushing
  the tag.
- As built (2026-10-10/11, see NOTES.md):
  - Vintage refreshed (French files unchanged); HOLDOUT_END 2026-08-31 (`5b9144b`).
  - The dev rule chose lw_identity. No primary difference: all Holm-adjusted p-values are
    1.0.
  - The luck benchmark's null is degenerate for persistent tilts. The owner chose to keep it
    and state what it measures (`746a348`, `9e87fdb`).
  - python-reviewer fixes `d488950` / `e11e2fc`; `/code-review` fixes `3a5ed3b` / `aedd063`
    / `0cffd64`.
  - Verification loop passes. The final dev run is on `9e87fdb`. Then the freeze commit and
    the local tag `prereg-v1`.
- Extra-high review (2026-10-11, owner asked): 11 findings, all fixed (RED `fd2c58f`, GREEN
  `9e2df18`).
  - The dev re-run on `9e2df18` reproduced every table.
  - Mutation runs: 24 of 24 (session 2, with five new) and 21 of 21 (session 1).
  - Re-frozen; the tag moved from `9492cb5` (never pushed).

## Checks every session

`uv run pytest --cov`; `uv run ruff check .` and `uv run ruff format --check .`; `uv run mypy`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dev strategy results seen before the draft is committed | Low, irreversible | Grounding printed no performance; the draft is the first commit of session 1; session 2 only times the run |
| The QS / Andrews-Monahan HAC is wrong, with no reference implementation to hand (no R here) | Medium | Hand-computed tests for each piece; the iid size simulation; the Bartlett HAC as a sanity bound (grounding: within 3% of iid) |
| Algorithm 3.1 too slow | Medium | Vectorized over resamples; primary pairs only; K = 1000 is LW's lower limit; timed in session 1 |
| Holdout opened early | Low, irreversible | The tag-and-PLAN lock on `--holdout`; the text scan; only dev mode runs in 7a |
| The refresh changes the dev numbers in NOTES | Likely (French revisions) | The refresh comes before the dev run; NOTES gets the old→new difference |
| The holdout period is known history | Certain | Prior-knowledge statement: the holdout guards against tuning on these series, not against knowing 2011-2026 |
| A null result read as failure | Certain for minimum variance | The MDE is stated in advance (decision 5) |

## Considered and rejected

- **Fixed block length:** simpler, but not LW's method (decision 3).
- **Daily-frequency tests:** strong volatility clustering, and LW's calibration needs a
  bivariate GARCH model there (decision 3).
- **Bonferroni:** Holm is uniformly more powerful with the same guarantee.
- **The Romano-Wolf StepM for the exploratory family:** better power under dependence, but it
  is another bootstrap to build and test for results that are exploratory anyway.
- **Benjamini-Hochberg:** it controls false discoveries, not the error rate behind a claim.
- **Deflated Sharpe ratio instead of the luck benchmark** (decision 8).
- **Testing the CAD-hedged view:** its Sharpe ratio is identical to USD's.

## Grounding 2026-10-09 (dev data; no strategy performance printed)

Script `step7a_ground.py` (session scratchpad).

- **The vintage** (cache, last dates only):
  - 12 industries and factors: 2026-08-31.
  - USD/CAD: 2026-09-25.
  - CA T-bill and 10-year: 2026-10-01.
  - So HOLDOUT_END would be 2026-08-31, with 188 holdout months.
- **Dependence in monthly 1/N excess returns (dev):**

  | View | T | AC(1) of r | AC of r², lags 1, 2, 3, 6, 12 |
  |---|---|---|---|
  | USD | 558 | +0.09 | +0.11, +0.07, +0.13, +0.09, +0.02 |
  | CAD | 468 | +0.06 | +0.11, +0.06, +0.12, +0.03, +0.02 |

  The serial dependence is small but present, which is why the block bootstrap and the HAC
  are used.
- **Power** (dev SE of the annualized Sharpe difference vs 1/N, scaled to 188 months; Bartlett
  HAC):

  | Pair | ρ with 1/N | SE (iid) | SE (HAC) | MDE α = 0.05 | MDE α = 0.0125 |
  |---|---|---|---|---|---|
  | Minimum variance, 3 estimators, USD | 0.78-0.81 | 0.16-0.17 | 0.15-0.16 | 0.43-0.46 | 0.51-0.54 |
  | Minimum variance, 3 estimators, CAD | 0.76-0.78 | 0.17-0.18 | 0.16-0.17 | 0.46-0.48 | 0.55-0.57 |
  | ERC, 3 estimators, both views | 0.997 | 0.019-0.020 | 0.019-0.020 | 0.05-0.06 | 0.06-0.07 |

- **Turnover matching:** the chance that one Dirichlet(1) step cannot reach τ.
  - It is 0 for 1/N and ERC, and 0.0005 for minimum variance (sample; mean τ 5.4%, max 32.5%).
  - It is 0.0001 at τ = 15%, 3% at τ = 29% (EWMA minimum variance) and 59% at τ = 50%.
- **Timing:** `compare` of 7 strategies on both dev views took 17.0 s.
