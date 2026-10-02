# Notes

Running log of decisions, data quirks and findings. Newest first.

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
