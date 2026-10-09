# Step 5b plan: factor and macro regressions

Approved by the owner on 2026-10-09, after a review the same day that the owner asked for
(session 1 of 2). Read with the blueprint's Step 5b section.
Branch `step-5b-factors`.

The review tested the plan's claims with scratch scripts on synthetic and dev data. It looked at
model behaviour only and ranked no strategies. What changed is listed at the end ("Review
2026-10-09").

## What gets built

- `risk/factors.py`:
  - `nw_lags(T)`: the blueprint's Newey-West lag rule, floor(4(T/100)^(2/9)).
  - `regress(y, X, *, lags=None)`: OLS with an intercept and Newey-West (HAC) standard errors,
    via statsmodels. Returns a `Fit`: coefficients, HAC standard errors, t-stats, p-values, R²,
    adjusted R², the number of observations, the lag used, and the first and last dates.
  - `factor_betas(excess, factors, *, fx=None, frequency="daily")`:
    - USD view: excess returns on FF5 + momentum.
    - CAD view: the same factors plus the USD/CAD return (`fx`).
    - Takes a Series or a DataFrame (one regression per column) and returns a tidy table:
      coefficient, SE, t and p per term, plus R², observations, lag and dates per column.
  - `macro_betas(excess, d_cad10y, fx, factors, *, frequency="daily")`: the CAD factor model
    plus the change in the Canadian 10-year yield. Its betas on Δy and fx are partial
    sensitivities: the factors are held fixed (decision 6). These are the inputs for Step 6.
    - Deviation from the blueprint: the blueprint's `macro_betas(returns, d_cad10y, fx)` has
      no `factors`. The change gets logged.
  - `to_weekly(df, *, how="compound")`: compounds daily returns into weekly ones; `how="sum"`
    adds yield changes (decision 4).
  - `rolling_betas(excess, factors, *, fx=None, window=252)`: rolling OLS betas.
- `data/views.py`: `yield_changes(levels, days)` gives the daily change in a yield on US trading
  days. It carries levels exactly as `fx_returns` does (`align_levels`, stale levels raise).
- The docstring carries the caveat that the daily FX fix (noon New York) is not synchronous with
  US closes, which is why the weekly regressions are reported alongside the daily ones.
- Every formula gets a hand-computed or synthetic test (invariant 6). Dev data is used only for
  sanity checks.

## Decisions for the owner (recommendation first)

1. **What gets regressed: any daily excess-return Series or DataFrame, in the view's own
   currency.**
   - USD: the return minus French RF. CAD (unhedged): the return minus the Canadian T-bill
     rate. That is `View.excess`, Step 2's definition. Strategies use `Result.net`, so the
     alphas are net of costs.
   - The CAD-hedged view needs no regression of its own: its excess return equals the USD
     excess return exactly (Step 2, CIP).
   - The dev check uses the 12 industries plus 1/N, minimum variance (sample) and the market
     benchmark, in the USD and CAD views.
   - The market benchmark is a plumbing check, measured in the review:
     - USD view: a Mkt-RF beta of 1 to 2e-15 and R² = 1.
     - CAD view: fx 1.000, Mkt-RF 1.001, R² 0.9996.
2. **The FX regressor is Step 2's `fx_returns`:** fx = S_t/S_{t−1} − 1, with S in CAD per USD.
   - A positive beta means the CAD investor gains when the USD strengthens. Unhedged USD
     assets show a beta near 1.
   - For Step 6, "CAD +10%" (the CAD appreciates 10%) means fx = 1/1.10 − 1 = −9.09%, not
     −10%. This goes in a docstring and the Step 6 notes.
3. **HAC: Bartlett kernel, lag floor(4(T/100)^(2/9)), no small-sample correction, normal
   p-values.**
   - This is Newey and West's (1994) rule as the blueprint sets it. The rest is statsmodels'
     default (`use_correction=False`, `use_t=False`).
   - The review checked the sandwich formula under Conventions against statsmodels to 4e-16.
   - Lags: 11 for both daily views (T = 11,959 USD and 10,094 CAD); 7 for weekly (T ≈ 2,090);
     4 at T = 250.
   - `lags=` overrides the rule. The lag used is reported.
   - Alternative: correct by T/(T − k). It changes the SEs by 0.04% to 0.3% across the planned
     regressions (the most on weekly macro, T ≈ 520).
4. **Weekly regressions: Wednesday-to-Wednesday weeks. Returns are compounded and yield
   changes are summed.**
   - **Why Wednesday:** a Friday is a market holiday far more often than a Wednesday. From
     1963 to 2010 there are 91 Friday holidays (Good Friday every year) and 46 Wednesday ones,
     24 of them the 1968 Wednesday closures. Friday weeks are the alternative.
   - **Returns:** each column compounds as Π(1 + r) − 1 over the trading days in the week,
     excess returns included. This is the same convention for every column.
     - Exact weekly excess returns would compound the total return and the risk-free rate
       separately. The difference is at most 1.3e-4 on dev data, against a weekly SD of 1.9%,
       and computing it would need the total returns as an extra input.
   - **Yield changes are summed, never compounded:** the weekly Δy = y(Wed) − y(previous Wed).
   - Order: align the daily inputs, drop the NaN edges (decision 7), then make the weeks. That
     way y and X always cover the same days, the partial weeks at either end included.
   - A week is labelled by its Wednesday, and a week with no trading days does not appear.
5. **Rolling betas: a 252-day window, point estimates only, each indexed by the last day of its
   window.**
   - They are descriptive (for the dashboard), not forecasts, so there are no HAC errors and no
     shift to a target day. This differs from `var.forecast`, and the docstring says so.
   - **Implementation:** numpy `sliding_window_view` with per-window normal equations solved
     in one batch, the same technique as 5a.
     - Review timing on 11,959 days with 7 regressors: 0.15 s, matching a lstsq loop to 1e-14.
     - statsmodels `RollingOLS` takes 0.75 s (1e-13). It updates XᵀX by adding and removing
       rows, so its error can grow with the length of the series.
   - Tested against a per-window loop to 1e-10.
6. **`macro_betas`: partial betas. The CAD excess return is regressed on Δy, fx and the six
   factors, and Step 6 shocks one variable with the others held at 0.** (Changed in review:
   the draft regressed on Δy and fx alone.)
   - **Why:** in the review, from 2001 to 2010 the yield beta without controls was positive
     for every portfolio checked. A total beta would make a "+200 bp" stress a *gain*.

     | CAD excess return | Δy alone | Δy + fx | + Mkt-RF | + all six factors (daily / weekly) |
     |---|---|---|---|---|
     | 1/N | +8.45 | +9.15 | −0.12 | −0.03 / −0.10 |
     | Market | +9.01 | +9.73 | +0.00 | +0.00 / −0.02 |
     | Utilities | +4.86 | +5.63 | −1.40 | −1.21 / −3.14 |
     | Money (financials) | +11.40 | +12.03 | −0.59 | −0.19 / −0.22 |

     - The total beta is the stock-bond correlation, not duration: corr(Δy, Mkt-RF) = +0.36.
       Yields rose on good equity days in both halves of the decade (+7.7 in 2001-05, +9.2 in
       2006-10).
     - That correlation flipped in 2022, one of Step 6's registered stresses. A total beta from
       the dev period would point the wrong way there.
     - With the factors held fixed, Δy measures the direct rate sensitivity: about 0 for the
       market, and negative for utilities, which act as bond proxies.
     - fx behaves the same way. Alone it is 0.53 for 1/N, because the USD rises when US stocks
       fall: a natural hedge for a Canadian investor. Partial, it is the mechanical 0.97-1.0.
   - **Consistency:** Step 6's factor shocks (MKT −20%, UMD −35%) use the partial betas of the
     same FF5 + momentum model. So every hypothetical shock means "this variable moves, the
     others stay at 0", and a combined scenario (2022-like: rates up and equities down) is the
     sum of its parts.
   - **Units:** Δy is in decimal yield units (100 bp = 0.01), so a +200 bp shock moves P&L by
     0.02 × beta. Δy comes from `yield_changes`, which carries levels onto US trading days, so
     a Canadian holiday gives a change of 0.
   - **Sample: daily from 2001 only** (the daily Valet series starts then), about 2,500 dev
     days, or about 520 weeks.
     - The monthly `CA_GOVT_LONG` back to 1936 is a different instrument (the average yield of
       bonds over 10 years). Its daily values change once a month, so it cannot give daily
       changes.
     - A monthly regression on it from 1971 is left to 7b, as a robustness check if needed.
   - **Daily and weekly both reported:** the utilities' weekly yield beta (−3.1) is 2.6 times
     the daily one (−1.2). Step 6 chooses between them, and I'd recommend weekly.
   - Alternatives: the market alone as the control (simpler, close to the full model for
     diversified portfolios), or the draft's total betas (rejected above).
7. **Alignment: strict, with NaN allowed only at the edges.** (Changed in review: the draft
   inner-joined the inputs and dropped every NaN row.)
   - Each input is trimmed to the dates the inputs share. Within that span the dates must
     match exactly (`check_aligned`).
     - This catches a monthly y passed with daily X, which an inner join would silently
       accept.
   - A leading or trailing block of NaN is dropped, such as Δy before 2001 or an industry
     listed late. A NaN between two observed values raises, naming the day: HAC lags and
     rolling windows count rows, so a gap would make non-adjacent days look adjacent.
   - Values that are not finite (other than NaN) raise, through `_validate`.
   - The `Fit` reports the number of observations and the first and last dates, so a short
     sample is visible.
   - The dev inputs have no interior NaN: the 12 industries, the views and the carried 10-year
     levels are all complete.
8. **The functions are pure and unguarded,** like the risk functions in 5a (5a's decision 6).
   They take no `allow_holdout`.

## Conventions

- **Signs and units:** returns are decimals, and betas are per unit of regressor.
- **The intercept (alpha) is per period** (daily or weekly) and not annualized.
  - The CAD alpha also absorbs the average US-Canada rate gap. CAD excess = USD excess + fx +
    r_usd·fx + (rf_usd − rf_cad).
  - So compare alphas within a view, not across views.
- **HAC formula** (the review checked it against statsmodels to 4e-16): with residuals e and
  rows x_t,
  Ω = Σ_t e_t² x_t x_tᵀ + Σ_{j=1}^{L} (1 − j/(L+1)) Σ_t e_t e_{t−j} (x_t x_{t−j}ᵀ + x_{t−j} x_tᵀ),
  and V = (XᵀX)⁻¹ Ω (XᵀX)⁻¹. There is no 1/T factor and no correction.
- **Synthetic checks** (fixed seeds):
  - y = 0.5·MKT + 0.2·SMB + ε, T = 5,000: each beta within 4 HAC SEs of the truth.
  - CAD: r_cad = (1 + r_usd)(1 + fx) − 1, with r_usd = 0.5·MKT + ε. The fx beta comes out
    near 1, and the MKT beta near 0.5.
  - Macro: y = −7·Δy + 1·fx + 0.8·MKT + ε, where Δy is correlated with MKT. The partial
    betas come out near −7, 1 and 0.8, while a regression without MKT is biased. The test
    shows that bias.

## Sessions

**Session 1: the regression core and the factor regressions.** Tests first, then code.
- `nw_lags`: T = 100 → 4, T = 250 → 4, T = 2,000 → 7, T = 11,959 → 11, and T ≤ 0 raises.
- `regress`:
  - The HAC SEs equal the hand-computed lag-1 sandwich on a 6-row example to 1e-12. They also
    match a general-lag numpy version of the formula on random data.
  - The coefficients equal `numpy.linalg.lstsq`.
  - Invariances (hypothesis): scaling y by c scales the betas and SEs by |c|. Adding c·X_k to
    y adds c to beta_k and leaves the SEs unchanged.
  - A singular X raises, as do too few rows (T ≤ the number of parameters).
- Alignment (decision 7):
  - the span is trimmed
  - a monthly y with daily X raises
  - leading and trailing NaN are dropped and reported
  - interior NaN raises, naming the day
- `factor_betas`: the USD and CAD synthetic checks, one row per column for a DataFrame, and the
  `frequency="weekly"` path.
- `to_weekly`:
  - a 2-week hand example with a holiday
  - compounding vs summing
  - the Wednesday labels
  - a partial edge week
- RED and GREEN commits. Then a mutation run:
  - the wrong Bartlett weight
  - the lag off by one
  - fx inverted
  - compounding that sums instead of multiplying
  - Δy compounded instead of summed
  - the week boundary off by a day
  - the alignment check skipped

**Session 2: rolling and macro betas, dev check, review.**
- `rolling_betas` equals a per-window loop to 1e-10 and has no look-ahead (changing day k
  leaves every beta before day k unchanged). The first beta is on day `window`.
- `yield_changes`:
  - a hand example across a Canadian holiday
  - NaN before the first level
  - stale levels raise, as in `fx_returns`
- `macro_betas`:
  - the macro synthetic check, including the bias without controls
  - daily and weekly
  - the sample starts in 2001 and is reported
- Dev check (model behaviour only, no ranking):
  - the market plumbing check (decision 1)
  - industry betas
  - daily vs weekly fx and Δy betas (the asynchrony caveat in numbers)
  - macro betas from 2001 to 2010, total vs partial (the decision-6 table)
  - timing
- python-reviewer, then `/code-review`, then verification-loop. Fix the findings.
- Check invariants 1-6. Update NOTES.md, the blueprint's status line and the change log,
  including the `macro_betas` signature change.
- Ask the owner before pushing and opening the PR.

## Checks every session

`uv run pytest --cov`; `uv run ruff check .` and `uv run ruff format --check .`; `uv run mypy`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| The FX sign inverted between Step 2 and Step 6 | Medium, silent | Use Step 2's `fx_returns`; unhedged synthetic beta ≈ +1; Step 6 conversion written down |
| Stress betas that carry the dev-period stock-bond correlation | High in the draft | Partial betas with the factors held fixed (decision 6); total vs partial shown in the dev check |
| Yield changes treated like returns | Medium, silent | `how="sum"` for Δy; a mutation run for compounding Δy |
| Misaligned or gappy inputs regressed silently | Medium | Strict alignment, edge-only NaN, reported nobs and dates (decision 7) |
| HAC options drift in statsmodels | Low | The hand-computed sandwich test pins the numbers |
| Daily 10-year history short (2001+) | Certain | nobs and dates reported; weekly and the monthly long-bond alternative noted |
| Weekly compounding of excess returns is approximate | Low | Measured at ≤ 1.3e-4 vs a 1.9% weekly SD; stated |

## Considered and rejected

- **Total (univariate) macro betas:** they give a rates shock the wrong sign (decision 6).
- **Dimson lead-lag regressions** (adding fx_{t+1}) for the FX asynchrony: the blueprint
  already chooses weekly regressions. With factors in the model, daily and weekly fx betas
  differ by about 0.01 (1/N: 0.981 daily, 0.991 weekly).
- **statsmodels `RollingOLS`:** 5 times slower, and its incremental updates can drift
  (decision 5).
- **Inner join and dropping every NaN row:** it hides misalignment (decision 7).

## Review 2026-10-09 (what changed from the first draft)

1. Decision 6: `macro_betas` holds the six factors fixed and gives partial betas. The draft's
   total betas made a rates shock a gain on dev data. `macro_betas` gains a `factors`
   argument and a `frequency`.
2. Decision 4: yield changes are summed, not compounded (the draft would have compounded Δy
   in the weekly path). The Wednesday rationale now has numbers, an unverified French
   attribution is gone, and the excess-return approximation is measured.
3. Decision 7: strict alignment and edge-only NaN replace the inner join and NaN drop.
4. Decision 5: numpy sliding windows replace `RollingOLS`.
5. Decision 3: the correction size is fixed (0.04% to 0.3%, not "under 0.1%"). Normal p-values
   are stated, and the formula is verified against statsmodels.
6. New: `yield_changes` in `data/views.py`; the CAD alpha caveat; the market plumbing check
   with measured tolerances.
