# Step 5a plan: VaR/ES + VaR backtesting

Approved by the owner as drafted on 2026-10-08 (session 1 of 3). Read with the blueprint's
Step 5a section.
Branch `step-5a-var`.

## What gets built

- `risk/var.py`: one-day value at risk (VaR) and expected shortfall (ES) by four methods:
  - historical
  - parametric normal
  - Cornish-Fisher
  - filtered historical simulation (FHS: EWMA volatility, λ = 0.94)
  - It gives point estimates on one sample and rolling one-day-ahead forecasts on a return
    series. A helper scales to 10 days by √10, labeled an approximation.
- `risk/backtest.py`: the exception series, Kupiec POF, Christoffersen independence and
  conditional coverage, and Basel traffic-light zones (99% over 250 observations only).
  `report` puts every test for every method in one table, for 7a and the dashboard.
- Every formula has a hand-computed test (invariant 6). Tests use synthetic data, and dev data
  is used only for timing and sanity checks.

## Decisions for the owner (recommendation first)

1. **Input: any daily simple-return Series. The dev checks use total net returns in the view's
   currency** (`Result.net`), not excess returns.
   - VaR is a loss of portfolio value, so the investor's actual P&L is the natural input;
     excess returns would subtract a risk-free rate the investor still earns.
   - Dev checks use 1/N, minimum variance (sample matrix) and the market benchmark, each in the
     USD and CAD views.
   - Alternative: excess returns (`view.excess`-style), which 5b's regressions will use anyway.
2. **The rolling window is one parameter, `window`, defaulting to 500 trading days (~2 years).**
   - 250 days leaves the 99% historical VaR resting on the 2nd-3rd worst day, which is very
     noisy. 1000 days reacts slowly to new regimes.
   - The session-3 dev check also shows 250 and 1000 as sensitivity. 7a pre-registers the window
     it reports.
   - Every method uses the same window, so the methods are compared on equal information.
3. **Quantile and ES conventions.**
   - Historical VaR = −(numpy `quantile(..., method="linear")` at 1 − level), Hyndman-Fan type 7,
     the numpy and pandas default.
   - Historical ES = the mean of the returns at or below −VaR (the tail mean, Hull's
     definition). "At or below" keeps the tail non-empty, even for a constant sample, and
     ES ≥ VaR always holds.
   - Alternative: the Acerbi-Tasche estimator (the average of the empirical quantile over the
     tail). It is coherent in theory, but harder to explain, and mixed with an interpolated VaR
     it can fall below VaR in edge cases.
   - Parametric ES uses closed forms. Normal: ES = −μ + σ φ(z)/(1 − level). Cornish-Fisher: the
     exact tail mean of the CF quantile, from truncated-normal moments (below). This is not the
     Boudt-Peterson-Croux "modified ES", which comes from a different (Edgeworth) expansion.
4. **FHS (filtered historical simulation)**
   - **Volatility filter:** a zero-mean EWMA, λ = 0.94: σ²(s+1) = λ σ²(s) + (1 − λ) r(s)².
     Zero mean is RiskMetrics' convention, the same as Step 3's `ewma_cov`.
   - **One causal pass over the whole series.** The standardized residual on day s is
     z(s) = r(s)/σ(s), and σ(s) uses returns through s − 1 only.
   - **Seed:** σ²(1) = the mean of r² over the first window, so the seed uses only data seen by
     the first forecast. Its weight decays as 0.94^k: 2e-7 after 250 days and 4e-14 after 500.
   - **Forecast at close t:**
     - Take the residuals of the last `window` days and scale them by σ(t+1).
     - VaR and ES come from the same historical estimator.
     - Because σ(t+1) > 0, this equals σ(t+1) × the historical VaR/ES of the residuals. That
       makes FHS as fast as the historical method.
   - **Alternative:** re-run the filter inside each window. It is slower, and every window would
     need its own seed.
5. **Basel zones are given only for 99% and exactly 250 observations:** green 0-4 exceptions,
   yellow 5-9, red 10 or more. Any other level or length raises an error.
   - `basel_zones` gives the zone of each trailing 250-day window, for the dashboard's history.
   - The plus factors (0.40-1.00) are not modeled.
6. **The risk functions are pure and unguarded, like `simulate`** (the Step 4 guard design).
   - They price whatever Series they are given.
   - That Series comes from the guarded `run`/`compare`/`reprice`.
   - No `allow_holdout` parameter, so the holdout text scan has nothing new to cover.
7. **Speed: vectorized windows, checked by a loop.**
   - Rolling estimates use `numpy.lib.stride_tricks.sliding_window_view`, which builds no copies
     (12,000 × 1,000 floats is ~96 MB at most if a copy is ever needed).
   - Pandas' online rolling skew and kurtosis are avoided: they drift numerically.
   - A plain per-window loop in the tests must agree to 1e-12.
   - Target: all four methods at both levels on 12,000 days in under 2 s.

## Conventions

- **Signs.**
  - VaR and ES are positive losses: VaR = −q, where q is the return quantile at 1 − level.
  - `level` is the confidence (0.95, 0.99) and must lie in (0.5, 1). The exception probability
    is p = 1 − level.
- **Timing.**
  - The forecast for day t+1 uses returns t − window + 1, ..., t and is indexed by day t+1, so
    it lines up with the return it is tested against.
  - The first forecast is for day `window` + 1.
  - The forecast made at the last close (for a day not yet in the index) comes from the point
    functions on the last window, for Step 8's "today's VaR".
- **Exceptions.** Day t is an exception when the loss is strictly larger than the forecast:
  r(t) < −VaR(t).
- **Moments (normal and CF):**
  - μ is the window mean and σ the sample standard deviation (ddof = 1).
  - S and K are Step 2's bias-corrected skew and excess kurtosis (`metrics.skew`,
    `metrics.excess_kurtosis`), so the library has one definition.
  - A window with σ = 0 raises an error (CF is undefined there, and the normal method would
    claim zero risk).
- **Cornish-Fisher.**
  - Let z = Φ⁻¹(1 − level), which is negative.
  - z_cf = z + (z² − 1) S/6 + (z³ − 3z) K/24 − (2z³ − 5z) S²/36.
  - VaR = −(μ + z_cf σ).
  - ES = −μ + σ (φ(z)/(1 − level)) [1 + z S/6 + (z² − 1) K/24 − (2z² − 1) S²/36]. This is the exact
    mean of the CF quantile over the tail, from E[Zᵏ | Z < z]. It was checked against numerical
    integration to 1e-11 (S = −0.8, K = 6, 99%).
  - With S = K = 0, both reduce to the normal formulas.
  - When skew and kurtosis are extreme, the CF quantile is not monotone and ES can fall below
    VaR. That case is documented and counted in the dev check, not "corrected".
- **Inputs.** Returns must be finite, ≥ −100%, complete, and on a sorted, unique
  `DatetimeIndex`, using the existing `_validate` helpers. The series needs more than `window`
  rows.
- **Backtests** (T observations, x exceptions, p = 1 − level). `0·ln 0 = 0` throughout
  (`scipy.special.xlogy`).
  - Kupiec POF: LR_uc = −2 ln[(1 − p)^(T − x) p^x] + 2 ln[(1 − x/T)^(T − x) (x/T)^x], χ²(1).
    It works for x = 0 and x = T.
  - Christoffersen independence uses the transition counts n00, n01, n10 and n11 over
    consecutive days:
    - π01 = n01/(n00 + n01), π11 = n11/(n10 + n11), π = (n01 + n11)/(T − 1).
    - LR_ind = −2 ln[(1 − π)^(n00 + n10) π^(n01 + n11)]
      + 2 ln[(1 − π01)^n00 π01^n01 (1 − π11)^n10 π11^n11], χ²(1).
    - When n11 = 0, or when a denominator is 0, the terms with a zero count vanish.
  - Conditional coverage: LR_cc = LR_uc + LR_ind, χ²(2). Its LR_uc is computed on the T − 1
    transition days (Christoffersen 1998), so the sum is exact. The stand-alone Kupiec test uses
    all T days.
  - Each test returns `stat`, `pvalue` and `df`, plus `x` and `T`.
- **Hand examples.**
  - **20-value list:** the sorted returns start −10%, −6%, followed by 18 values from −5% up.
    - 95%: q = −0.10 + 0.95 × 0.04, so VaR = 6.2% and ES = 10%.
    - 99%: q = −0.10 + 0.19 × 0.04, so VaR = 9.24% and ES = 10%.
  - **Normal,** μ = 0.1%, σ = 1%, 99%: VaR = 2.2263% and ES = 2.5652%. The sign is checked: a
    larger μ gives a smaller VaR.
  - **Kupiec,** T = 250, x = 8, p = 0.01: LR = 7.7336, p-value 0.0054. The test computes this
    itself with `math.log`, independently.
  - **Christoffersen:** a hand-built sequence with clustered exceptions, and one with n11 = 0
    (for example 0,1,0,1,0,0,0,0,0,0).

## Sessions

**Session 1 (max effort): plan, then the point estimators.** Tests first, then code:
- `var_es(sample, level)` (historical; also FHS's core):
  - the 20-value list
  - the tail is never empty
  - ES ≥ VaR (hypothesis)
  - a positive scale factor commutes with it (the FHS identity)
- `normal_var_es(mu, sigma, level)`:
  - the hand values
  - checked against `scipy.stats.norm`
  - the sign check
- `cornish_fisher_var_es(mu, sigma, skew, exkurt, level)`:
  - equals the normal method when S = K = 0, exactly
  - ES matches numerical integration of the CF quantile
  - negative skew raises the 99% VaR
- `ewma_vol(returns, decay, seed)`:
  - a 3-day recursion by hand
  - causal: σ(s) is unchanged when r(s) changes
- `scale_horizon(x, days=10)`: √10.
- RED and GREEN commits.
- As built (2026-10-08, see NOTES.md):
  - Every function returns a `VarEs(var, es)` NamedTuple, and the filter's seed argument is
    `seed_var` (a variance).
  - Each formula lives in a private array core, so session 2 can reuse it.
  - ES is clamped to ≥ VaR, because a tail mean can round 1 ulp past q.
  - 88 tests; the mutation run caught all 13 planted bugs.

**Session 2 (max effort): rolling forecasts and backtests.**
- `forecast(returns, *, method, level, window=500)` returns a DataFrame with `var` and `es`,
  indexed by target day. Methods: `METHODS = ("historical", "normal", "cornish_fisher", "fhs")`.
- Tests:
  - the vectorized version equals a per-window loop to 1e-12 (hypothesis)
  - no look-ahead: perturbing returns after day k never changes a forecast for a day ≤ k + 1
  - the first forecast is for day `window` + 1
  - FHS equals σ(t+1) × the historical estimate of the residuals
- `exceptions`, `kupiec`, `christoffersen`, `conditional_coverage`, `basel_zone`,
  `basel_zones`, `report`:
  - the hand values above
  - the LR_cc identity
  - x = 0 and x = T
  - n11 = 0, and the case where the only exception falls on the last day
  - Basel boundaries at 4/5 and 9/10, and errors for the wrong level or length
- Size check (Monte Carlo, fixed seed): with true normal VaR on iid normal returns, Kupiec at
  5% rejects close to 5% of the time.
- Mutation run on the rolling loop and the tests: a window ending at t+1, `≤` versus `<` for an
  exception, a missing (1 − λ), and so on.
- RED and GREEN commits.
- As built (2026-10-09, see NOTES.md):
  - `next_day` gives today's VaR by the same rule.
  - The tests return a `LikelihoodRatio`; `transitions` is public.
  - `report` has one row per (level, method).
  - The minimum window is 4 for every method.
  - `tail_prob` and `return_series` are shared in `_validate`.
  - The Kupiec test's exact size at 99% over 250 days is 9.5%, not 5% (for 7a).
  - 73 tests; the mutation run caught all 22 planted bugs.

**Session 3 (extra effort): dev data, then review.**
- Dev checks: 1/N, minimum variance and the market, in the USD and CAD views:
  - exception rates and the three tests per method at 95% and 99%
  - Basel zones over time
  - sensitivity to `window` (250/500/1000)
  - timing (under 2 s per series)
  - how often the CF quantile is non-monotone
  - Model behavior only; no strategy ranking (Step 4's decision 6).
- python-reviewer, then `/code-review`, then verification-loop; fix the findings.
- Check invariants 1-6. Add NOTES.md entries, the blueprint status line and the change log.
- Ask the owner before pushing and opening the PR.

## Checks every session

`uv run pytest --cov`; `uv run ruff check .` and `uv run ruff format --check .`; `uv run mypy`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Off-by-one: a forecast that uses the day it is tested on | Medium, fatal | Index by target day, perturbation test, loop reference, mutation run |
| Sign errors (VaR as a return, z taken as positive) | Medium | Sign-checked hand values; scipy cross-check |
| Christoffersen undefined at n11 = 0 or with no exceptions | High if not handled | `xlogy`, dedicated hand cases |
| CF misbehaves at the extreme moments around 1987 | Medium | Documented; non-monotone windows counted in the dev check |
| Numerical drift in rolling moments | Low | No pandas online moments; loop reference at 1e-12 |
| Session 3 overruns (dev checks plus review) | Medium | Dev checks are scripted; review may spill into a 4th session |
