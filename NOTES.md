# Notes

Running log of decisions, data quirks and findings. Newest first.

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
