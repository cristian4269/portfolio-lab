# Notes

Running log of decisions, data quirks and findings. Newest first.

## 2026-10-04: step 3, session 2 of 4 (weight rules)

- `construct/weights.py`: `equal_weight(assets)`, `min_variance(cov, *, cap=1.0)`,
  `equal_risk_contribution(cov)`. Each returns a Series labeled by the assets, >= 0, summing
  to 1. Both covariance rules check the matrix (square, same assets in the same order on both
  axes, finite, symmetric to 1e-12, positive variances) and require its correlation matrix to be
  numerically positive definite (smallest eigenvalue > 1e-10 x largest).
- **Two additions to the approved plan**, both behind the same interface:
  - *Polish after SLSQP.* SLSQP alone stopped up to 6.6e-8 from the exact weights (300 random
    small cases). It does find the active set (assets
    at 0 and at the cap). With those fixed, the free assets solve the equality-constrained
    problem exactly, a linear system. The result is kept only if it is feasible and passes the
    KKT check (every asset that could be sold has a marginal variance no higher than every asset
    that could be bought), otherwise SLSQP's weights are returned. Result: within 2e-15 of an
    exhaustive 3^N search, with no fallback in 400 random cases. Dropped assets get exactly 0
    and capped ones exactly the cap, so the engine trades no dust.
  - *Minimum variance also rejects singular matrices* (the plan said this for ERC only). This
    keeps the optimum unique and the polish's linear system solvable. A T < N sample matrix raises
    with "use a shrinkage estimator", and the identity-target Ledoit-Wolf estimate of the same
    window works (tested).
- **Scaling matters, as the plan's risk table said.** On the raw daily matrix (variances ~1e-4)
  SLSQP's absolute tolerance stops it early: at N = 49 the weights were off by 3.4e-4. The
  matrix is divided by its mean variance first; without that, the KKT property test fails. Time per call at `ftol` 1e-12 (synthetic): N = 12 about 3-5 ms, N = 49 about 30-40 ms.
- **ERC** follows Spinu (2013) as described by Griveau-Billion, Richard and Roncalli (2013):
  Newton on f(y) = y'Cy / 2 - sum ln y_i with C the correlation matrix, then w proportional to
  y_i / sigma_i. Budgets b_i = 1 instead of Spinu's sum-to-1 give the same weights and make f
  standard self-concordant. Damped steps run while the decrement is >= 0.95 (3 - sqrt 5) / 2,
  starting from the best multiple of equal weights; it takes 10-20 iterations on realistic
  matrices. Rounding sets a floor on the Newton decrement (about 1e-12 at condition number
  1e8), so a fixed 1e-12 stop could loop forever. Instead: once the decrement is below 1e-8,
  one more full step (which squares it) and stop. Risk contributions are equal to 1e-10 on
  random matrices and to 1e-8 at condition 1e8.
- Hand examples (module docstring, exact fractions in the tests): 2 assets 8/11; the corner
  (1, 0) where the closed form says 7/6; a 3-asset case where an asset with unconstrained weight
  -1/5 gets exactly 0; inverse variance (36, 9, 4)/49 and, capped at 1/2, (1/2, 9/26, 4/26); ERC
  (1/2, 1/3, 1/6) for sd 1, 2, 4% with correlations 1/2, 1/2, -1/16 (every risk contribution
  7/12); ERC proportional to 1/sigma for uncorrelated and for equally correlated assets
  (Maillard et al. 2010); 2-asset ERC = sigma2 / (sigma1 + sigma2) for any correlation. Weights
  are the same for the sample matrix divided by T or by T - 1 (tested for both rules).
- Mutation run: 21 planted errors. The first run let 2 survive, and both were test gaps:
  - `_ACTIVE` = 1e-12 survived because SLSQP usually lands exactly on a bound, so the polish had
    never been fed an asset slightly off one. Tests were added for that.
  - The cap * N = 1 equal-weight shortcut turned out to be redundant: SLSQP plus the polish
    return exactly 1/N even when the caps fall short of 1 by rounding (49 * (1/49) =
    0.9999999999999999). It was removed and that case is now tested.

  All 21 are caught now.
- TDD evidence: RED `3971e7f` (collection fails: `cannot import name 'weights'`), GREEN
  `22c8012` (105 weight tests), mutation fixes `ca219eb`. Suite: 480 passed, 99.79% coverage;
  ruff, ruff format, mypy clean.
- For session 3's dev-data checks: count polish fallbacks (expected 0) and time per call on
  the 12 industries.

## 2026-10-02: step 3, session 1 of 4 (covariance estimators)

- Plan approved and saved as `plans/step-3-plan.md`. Owner decisions: strategies declare
  their `lookback` (momentum gets its 252 days even with a 126-day covariance window; `window`
  moves from Step 4's engine into the strategy); minimum variance has no cap by default (the
  shortsale-constrained portfolio of DeMiguel et al.); PyPortfolioOpt's implementation is the
  independent check for the constant-correlation estimator. scikit-learn 1.9.1 is a dev
  dependency (test oracle only).
- `construct/covariance.py`: `sample_cov`, `ledoit_wolf_identity`,
  `ledoit_wolf_constant_correlation`, `ewma_cov` (decay 0.94, zero mean), `Shrunk(cov, target,
  intensity)`. One convention: demeaned, divided by T (scikit-learn's), so intensity 0 returns
  the sample matrix exactly. A target within float noise of the sample
  matrix (1e-12 relative) gets intensity 0; an asset with no variance raises for the
  constant-correlation target (its correlations are undefined). `_validate.window_values` checks
  every window: a DataFrame, complete, finite, unique sorted dates, unique assets.
- Constant correlation has no library oracle, so it is checked five ways: the docstring hand
  example redone in exact fractions (kappa = 651/88; delta = 1 at T = 4, 651/704 at T = 8); a
  loop-by-loop transcription of the paper's appendix; PyPortfolioOpt 1.6.0's function, vendored
  from the PyPI wheel (SHA-256 checked, MIT notice kept) and given its own sample matrix
  (divided by T - 1), matching to 1e-12; properties; and a mutation run. The identity target
  matches scikit-learn to 1e-10, incl. T < N and T = 2, and scikit-learn reproduces the hand
  value 177/352.
- Mutation run: 8 planted errors, all caught (rho's sqrt ratio swapped; rho's mirror term
  dropped; a theta sign; kappa / (T - 1); rbar averaging the diagonal; pi without s^2; the
  identity's b^2 divided by T instead of T^2; EWMA weighting the oldest period most). Each of
  the six constant-correlation mutations failed the hand example, the paper transcription and
  the PyPortfolioOpt match at once. On Windows, `Path.write_text` writes CRLF: a mutation script
  must restore files byte for byte (`read_bytes`/`write_bytes`) or with `git checkout --`.
- TDD evidence: RED `d0224bf` (collection fails: `cannot import name 'covariance'`), GREEN
  `0104255` (84 covariance tests). Suite 366 passed, 99.75% coverage; ruff and mypy clean.
- Dev-data diagnostics (12 industries, USD, 559 month-end windows of 252 days, 1964-06 to
  2010-12; estimates only, no performance):

  | | LW identity | LW constant correlation |
  |---|---|---|
  | Intensity 5% / median / 95% | 0.013 / 0.023 / 0.058 | 0.077 / 0.184 / 1.000 |
  | Median condition number (sample 167) | 113 | 121 |
  | Time per call | 0.5 ms | 1.3 ms |

  Constant correlation shrinks fully in 29 windows (1964, and 24 in 1987-90): the October
  1987 crash inflates pi-hat through fourth moments. Full shrinkage happens where the target
  is already close to the sample matrix (median ||F - S|| / ||S|| = 7%, vs 18% otherwise), and
  PyPortfolioOpt also gives delta = 1 on the same window, so this is the estimator, not a bug.
  The sample matrix's worst condition number is 749 (1988-09-30).
- Fresh-eyes review (2026-10-04; a separate agent working from the papers: both Ledoit-Wolf PDFs
  and RiskMetrics 1996 chapter 5). No formula errors: it re-derived every hand-example number
  in fractions and matched both estimators to 1e-14 on about 300 random cases each. Fixed:
  - Doc error: the docstring said the divisor (T vs T - 1) can't change a strategy. That holds
    for the sample matrix only; inside the shrinkage formulas T - 1 lowers the intensity by
    about 2/T in relative terms (checked: -0.8% at T = 252, -8% at T = 24). "Both Ledoit-Wolf
    papers divide by T" was overstated (2004a divides by T for zero-mean data; 1/T is
    scikit-learn's convention); the identity's scale is m (mu is the population value); EWMA
    is zero-mean, not demeaned.
  - Test gaps: kappa < 0 and the identity's clip at d^2 were reached only by chance (random
    Hypothesis draws), and nothing pinned the float-noise threshold. New tests: 7 integer days
    with kappa = -0.44 (intensity 0, not -0.06); 5 integer days with b-bar^2 = 38204/9375 >
    d^2 = 9068/5625 (intensity exactly 1, as in scikit-learn); correlations 1e-9 apart, a matrix
    1e-9 from the identity, an asset moving 1e-11 around 10%. Each fails under its mutation
    (dropped clip at 0, dropped clip at d^2, `_NOISE` = 1e-9 or 1e-6); the `_NOISE` mutations
    passed every earlier test.
  - EWMA decay stays 0.94 (owner, 2026-10-04): RiskMetrics' factor for 1-day forecasts; its
    1-month factor is 0.97. Revisit at 7a only if EWMA becomes a strategy variant.

## 2026-10-02: mypy with real pandas types (after step 2)

- mypy had been run ad hoc via `uvx` in an env without pandas, so every pandas
  type was `Any` and nothing pandas-related was checked. `mypy` and
  `pandas-stubs` are now dev dependencies, configured in `[tool.mypy]`
  (`files = src/portfolio_lab`; missing imports ignored only for `scipy.*` and
  `statsmodels.*`, which ship no stubs). CI runs `uv run mypy src/portfolio_lab`
  after ruff.
- Real types surfaced 32 errors (4 in `data/__init__.py`, 28 in the step-2
  `views.py`/`metrics.py`), all `Index[Any]` where a `DatetimeIndex` is
  needed, `Hashable` labels, or `float | Series` unions. Fixed by narrowing,
  no `# type: ignore`: `_validate.check_dates` now returns the checked
  `DatetimeIndex`; `data._datetime_index` does the same for
  `point_in_time_daily`, which therefore raises `TypeError` on a non-datetime
  index (date strings are not parsed silently). `Series.name` is set directly
  because the stubs reject a `Hashable` in `rename`; `make_views` finds the
  first usable CAD day by position instead of `idxmax`.
- `report/metrics.py` has `@overload` pairs again (removed in step 2 when
  mypy saw both as `Any`): Series in -> `float` (`int` for drawdown duration),
  DataFrame in -> `pd.Series`; `drawdowns` and `summary` map Series -> Series
  and DataFrame -> DataFrame. This also types `summary`'s rows without casts.
- Added `py.typed`: without it, code importing `portfolio_lab` saw every
  function as `Any`, overloads included (checked with `reveal_type`).

## 2026-10-02: step 2, session 2 of 2 (metrics)

- `report/metrics.py`: `periods_per_year` is a required keyword everywhere
  (`TRADING_DAYS = 252`, `MONTHS = 12` are named constants, never defaults).
  Series in -> float out; DataFrame in -> one value per column. Missing
  returns raise; so do returns below -100%.
- Conventions: CAGR geometric; vol, Sharpe, TE, IR arithmetic (ddof=1,
  scaled by ppy and sqrt(ppy)); Sharpe/Sortino always take rf (Series or a
  constant) so they never run on total returns by accident; Sortino's
  downside deviation averages over *all* periods (Sortino and Price 1994);
  drawdowns start from wealth 1, and an unrecovered drawdown counts toward
  duration; skew/kurtosis bias-corrected (G1/G2, Joanes and Gill 1998).
- Sanity, 12-industry 1/N, 1971-01-05 to 2010-12-31 (daily):

  | | USD | CAD | CAD hedged | US market |
  |---|---|---|---|---|
  | CAGR | 11.1% | 11.1% | 12.6% | 10.3% |
  | Vol | 15.8% | 16.1% | 15.8% | 16.5% |
  | Sharpe | 0.41 | 0.32 | 0.41 | 0.35 |
  | Max DD | -54.0% | -48.3% | -53.6% | -54.6% |

  Hedged Sharpe == USD Sharpe (CIP identity). 1/N beats the market's Sharpe
  in dev, as in DeMiguel et al. The CAD view's drawdown is shallower because
  the USD rallied in the 2008 crash; its Sharpe is lower because Canadian
  bills paid more (6.9% vs 5.5%/yr). Daily excess kurtosis ~16-19 (Oct 1987).
  Growth of $1: USD 68, CAD 68, CAD hedged 118.
- Review fixes: zero-risk ratios are NaN (a constant series once gave a
  Sharpe of 8.7e16 from float noise); drawdowns within 1e-12 of the peak
  are 0; metrics reject unsorted or repeated dates (reversed returns gave a
  wrong drawdown duration); index checks live in `portfolio_lab/_validate.py`.
- **End of the CAD views:** first version ended them at the last FX fix,
  which dropped 2010-12-31, a Fed holiday (New Year observed) when the NYSE
  traded. Now they end 7 days after the last observation: a lagging source
  (FRED posts USD/CAD with a delay) shortens them, a trailing holiday is
  carried like any other.
- Typing: `uvx mypy` sees pandas as `Any` (no pandas-stubs), so overloads
  (Series -> float, DataFrame -> Series) were dropped; with stubs there are
  12 existing errors in the data layer. Follow-up task, not step 2.

## 2026-10-02: step 2, session 1 of 2 (currency views)

- `data/views.py`: pure functions, no loading. US trading days (the French
  index) drive the calendar; FX and rate *levels* are carried onto them from
  the last observation at or before each day, then turned into returns. A
  level older than 7 calendar days raises (`MAX_AGE_DAYS`). On 1971-2010 the
  oldest carried USD/CAD fix is 3 days (Columbus Day: Fed closed, NYSE open).
- **Canadian rf = rolling 3-month bills (owner decision).** T-bill yields are
  simple act/365 (price 100 / (1 + y·T/365)), so the daily rate is
  `(1 + y·91/365)^(d/91) - 1`, with y known at the previous trading day and d
  calendar days (Monday earns 3). The blueprint's `(1 + y)^(d/365) - 1` would
  understate it: 8% quoted gives 8.24%/yr rolled, not 8.00%; 20% gives 21.6%.
  Dev 1971-2010: 6.91%/yr earned vs 6.76% average quote.
- **Day-count mismatch, harmless:** French RF spreads each month's T-bill
  return evenly over trading days; CAD rf accrues by calendar day (Mondays
  larger). Both sum to the right monthly return. Only daily `rf_cad - rf_usd`
  in the hedged view carries a weekly pattern.
- **CIP hedge duplicates USD Sharpe:** r_hedged = r_usd + rf_cad - rf_usd, so
  the hedged excess return equals the USD excess return *exactly* (property
  test). The CAD_HEDGED view adds information only through total-return
  metrics (CAGR, drawdown). To state in research/PLAN.md (step 7a).
- Sanity on dev data, 12 industries 1/N, 1971-01-05 to 2010-12-31: USD
  11.1%/yr, CAD unhedged 11.1% (USD/CAD ~flat, 1.01 -> 1.00), CAD hedged
  12.6% (= USD + ~1.4%/yr rate differential). No NaN in any view; CAD views
  start 1971-01-05.

## 2026-10-02: step 1, session 3 of 3 (guarded loaders, review)

- `load_assets/load_factors/load_fx/load_rates` stop at DEV_END; the holdout
  needs `allow_holdout` *and* an explicit later `end` (`end=None` = all).
- **Timing convention (data layer):** row t holds only what was observed by
  the close of t. Daily series stay on their own day (returns, noon USD/CAD,
  daily 10-year); the backtest adds the t+1 trade lag (step 4). Monthly
  StatCan rates enter the business day *after* their last-Wednesday stamp, a
  one-day margin that also covers "unless otherwise stated" exceptions. The
  10-year is deliberately not lagged here: step 5b regresses same-day yield
  changes.
- python-reviewer findings fixed: **holdout bypass via NaT** (`end=""`,
  `"NaT"`, `pd.NaT`, NaN all became NaT and `.loc[:NaT]` returns everything;
  the guard now fails closed); error pages no longer cached (`fetch_parsed`
  evicts what does not parse); `"nan"`/`"inf"` strings rejected; FF5 and
  momentum must share vintage and dates; a stray line inside a French block
  raises; `load_rates` with `end` before any data raises.
- **Test-suite politeness bug:** live tests had each used a fresh temporary
  cache, so one run downloaded the same StatCan vector 4+ times in seconds and
  StatCan's WDS stopped answering for a few minutes. Live tests now share the
  configured cache: a full live run is one download per file (9 files).
- 49 industries: 67,250 missing cells (industries that start later, coded
  -99.99). Step 7b's 49-industry robustness run must pick a start date or
  handle a changing universe.

## 2026-10-02: step 1, session 2 of 3 (FRED, Bank of Canada, Statistics Canada)

- `fred.py`, `boc.py`, `statcan.py` each refuse any series without an
  "allowed" verdict in DATA_SOURCES.md (`APPROVED` dicts, `PermissionError`).
- **Timing rule confirmed on live data.** StatCan monthly `v122531` re-stamped
  on the last Wednesday equals Valet's daily `TB.CDN.90D.MID` on that day in
  299 of 299 months where the Wednesday is a trading day (2001-02 to 2026-09,
  0.0 bp difference). Using the 1st instead is off by a median 4.5 bp, max
  126 bp. The live test `test_statcan_monthly_matches_valet_daily_on_last_wednesday`
  guards this.
- **Holiday months** (9 since 2001: Christmas/Boxing Day Wednesdays, and
  2026-09-30, National Day for Truth and Reconciliation) are observed on the
  *previous* business day, never later. Our Wednesday stamp is then 1-2 days
  late: slightly stale, never look-ahead. Kept as is.
- Coverage: `v122531` starts 1962-01 (4 missing months, none after 1971),
  `v122487` starts 1936-01; `DEXCAUS` 1971-01-04 to 2026-09-25, 558 missing
  days (US holidays); Valet daily series start 2001-01-02.
- Session 3 needs: forward-fill monthly rates onto the daily calendar from
  the business day *after* the stamp (so a rate is used only once observed).

## 2026-10-02: step 1, session 1 of 3 (sources, cache, French loader)

- Plan confirmed. The `allow_holdout=True` text scan exempts `tests/` as well
  as `cli.py` and `research/` (option A): guard tests must call the flag, and
  tests cannot leak the holdout into results.
- **Canadian rates come from Statistics Canada, not Valet.** Valet's daily
  3-month T-bill (`TB.CDN.90D.MID`) and 10-year benchmark (`BD.CDN.10YR.DQ.YLD`)
  start 2001-01-02; its monthly `V122531` copy starts 2005. The CAD view starts
  1971, so the history is table 10-10-0122-01: `v122531` (3-month T-bill) and
  `v122487` (GoC bonds over 10 years, average yield), monthly, Open Licence.
- **Timing of monthly rates (look-ahead trap):** table 10-10-0122-01 values are
  observed on the *last Wednesday* of the month but stamped with the 1st
  (`refPer` = YYYY-MM-01). Forward-filling from the 1st would use a rate up to
  four weeks before it was observed. Rule for session 2: stamp each value on
  that month's last Wednesday and forward-fill from the next business day.
- **USD/CAD:** FRED `DEXCAUS` runs from 1971-01-04 with no splice needed. It is a
  noon New York rate; US returns are at the 4 pm close, so same-day FX and
  returns are ~4 hours apart (documented; weekly regressions in step 5b).
- **FRED CSV quirk:** missing days are now an empty field (`1971-02-12,`), not
  `.` as documented in older guides. The parser must accept both.
- **French file quirks:** CRLF line endings, latin-1; daily files have no
  annual block; industry files have titled value-/equal-weighted blocks; FF5
  and momentum have one untitled block. The first line names the CRSP vintage
  (`202608` on 2026-10-02), which the manifest will record. The momentum
  series starts 1926-11-03; FF5 starts 1963-07-01, so FF5 dates drive the
  factor index.

## 2026-10-01: step 0, scaffold

- Plan approved (blueprint v2). G1: French library data fetched at runtime,
  cited, never committed; only derived analytics published. G5: MIT.
- Python 3.13 via uv; src layout; deps pandas, numpy, scipy, requests,
  statsmodels; dev: pytest, pytest-cov, ruff, hypothesis.
- Research split fixed in `src/portfolio_lab/config.py`: dev to 2010-12-31,
  holdout from 2011-01-01.
- Gotcha: `.gitignore` must anchor `/data/` to the root, or it also hides the
  `src/portfolio_lab/data/` package.
