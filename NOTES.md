# Notes

Running log of decisions, data quirks and findings. Newest first.

## 2026-10-12: dashboard design direction (after Step 8, owner asked)

- **Audit** (frontend-design-direction skill):
  - The registered answer sat at the bottom of the Performance tab, and fixed and live results
    looked alike.
  - An oversized title, raw strategy names in truncated chips, and a flat sidebar.
- **Direction, approved by the owner ("Apply all six"):** a research note. Quiet chrome, the
  numbers forward, serif headings, the validated palette unchanged.
  - **The memorable detail:** a "Registered · run once · fixed" strip above the tabs. A forest
    plot of the four registered Sharpe differences, each interval across zero, beside a verdict
    derived from the tests table ("No registered test rejects. ... every 95% interval contains
    zero, and every Holm-adjusted p is 1.00."). A gray "Descriptive · <view> · <months> · <bp>"
    badge then heads the live tabs.
  - `dashboard.display_name`: "Min variance · LW identity", "ERC · EWMA", "Momentum tilt ·
    0.5" (the 252-day window, common to all 17, is left out); unique over the registry.
  - **The sidebar** in four sections (Investor, Strategies, Period and costs, Risk), with
    segmented controls (`required=True`) for the view, benchmark, VaR level, VaR method and
    stress scenario.
  - **Theme tokens** in `.streamlit/config.toml`: `headingFont = "serif"` (Source Serif, bundled),
    smaller heading sizes, `baseRadius` 0.5rem, borders and table headers in the chart grid's
    tone.
- **Checked:** desktop light (1366 px) and phone dark (375 px). Every tab renders, there is no
  sideways scroll, and the forest plot sits within the phone's first screen. AppTest tests: 16;
  dashboard and scan: 78 with them.
- **AppTest:** a segmented control is set by its value (`segmented_control(key=...).set_value`).
  The metrics table is no longer `dataframe[0]`, because the registered table comes first.

## 2026-10-11: step 8, session 3 of 3 (reviews, verification)

- **python-reviewer** (no critical findings; it confirmed `check`, `realized`, the default
  metrics, the pyarrow read and the bundle hashes on the real data). Fixed:
  - **High:** the Factors tab crashed in the CAD view on any window ending before 2001 (the
    daily 10-year yield starts then) and stopped the Stress tab from drawing. A CAD window that
    starts before the yield now leaves it out (fx kept), and the caption says so and names the
    sample's dates.
  - Bounded caches (32 entries, 1 h); read-only bundle frames shared by every session; the
    17-strategy table only on request; defaults, VaR window, counts and the tests' verdict
    from the bundle; labelled R²; a clean error for a damaged bundle; `check` refuses other
    columns; copied holdout tables must match the manifest's hashes; `write` prunes an older
    bundle's files; `stress_realized` is no longer copied; the weights now feed a Holdings
    chart.
  - **Found while fixing:** French's 12 industries include one named "Other", so the folded
    band of `holdings` is "All others".
  - **Not done:** importing `engine._cost_rate` (an internal import, as robustness uses
    `study._performance`); mypy over `app/` (Altair keyword noise); gzip bytes that could
    differ under another zlib (the check is numeric, so only git churn).
- **`/code-review` (high): 10 findings, all fixed.** Holdings now start with the targets in
  force at the window's start; the stress tab handles a bundle with no registered window;
  the thinned VaR series is labelled; `read` requires 1/N; `net` computes only the
  strategies asked for; `performance` needs 12 months; the robustness series must match its
  recorded hash; `design.csv` is written last and lists every file's SHA-256, so a bundle
  written only in part is refused.
- **The "error" in the browser pane after the review fixes** was the dev server, not the code:
  `AttributeError: module 'portfolio_lab.dashboard' has no attribute 'MIN_MONTHS'`. Streamlit
  reran the edited app script but kept the old `dashboard` module, which `from portfolio_lab
  import dashboard` reads off the already-imported package. Restart the server after editing
  anything under `src/` (a deployed app always starts fresh). Replayed on the real bundle with
  AppTest: USD, CAD and CAD hedged, and CAD windows 1972-75, 1988-93 and 1990-2026, render
  with no error. It also showed the Factors caption printing `1972-02-01 00:00:00`; it now
  prints days.
- **Re-exports** (`83477bc`, `bc51d62`): every data file byte-identical to the first export;
  only `stress_realized.csv` left and `design.csv` gained the file list.

## 2026-10-11: step 8, session 2 of 3 (Risk, Factors, Stress; smoke tests; speed)

- **The tabs**, live at the selected view, months and cost:
  - **Risk:** one strategy at a time; rolling one-day VaR from the last 500 days, four methods
    (FHS by default), 95% or 99% from the sidebar; exceptions, Kupiec (exact), independence
    and conditional coverage, and the latest VaR and ES. `dashboard.var_backtest` equals the
    holdout's `var` table on the same data (tested on synthetic data). Example, USD 1/N at 99%
    FHS over the holdout: 43 exceptions for 39 expected, Kupiec p 0.52, independence p 0.00.
  - **Factors:** Step 5b's weekly partial betas over the selected months (the CAD view adds fx
    and dy), t-statistics, and rolling 252-day betas of a chosen factor. Holdout USD: R² 0.97,
    0.74, 0.95; latest market betas 0.77, 0.46, 0.66 (1/N, minimum variance, ERC).
  - **Stress:** each strategy's own path through the registered windows (live), then the
    registered replays and shocks (fixed). `dashboard.realized` equals the study's
    `stress_realized`.
- **The page asks for at least 12 months.** The AppTest found that fewer than 6 crashed the
  monthly Sharpe; annual figures from under a year would mislead anyway.
- **AppTest smoke tests** (`tests/test_app.py`, 7 tests, about 30 s): the whole app on a
  synthetic bundle whose registered tables come from `study.run`. AppTest quirks: a radio is
  set by its shown label; a select-slider by its values before `format_func`; widgets are
  found by key (main-area widgets come before the sidebar's in the lists).
- **Speed:** pyarrow (which comes with Streamlit) reads the bundle with exact floats in 0.5 s
  instead of 2.3 s. The first page after a server start: title 9.0 s, first chart 10.4 s,
  tables 11.7 s (about 4.4 s of it is importing Streamlit, scipy and statsmodels). A visit to
  a running server: title 0.4 s, chart 1.5 s, tables 2.6 s.
- **Browser checks:** every tab in light (desktop) and dark (375 px phone width): the palette
  follows the theme, and the page never scrolls sideways (tables scroll inside themselves).
  A chart in a tab clicked by a script before the page settles can stay at zero width; a
  user's click after load renders it.

## 2026-10-11: step 8, session 1 of 3 (plan, hosting, G1 removed)

- **Hosting, checked against the official docs on 2026-10-11** (details in
  `plans/step-8-plan.md`):
  - **Streamlit Community Cloud** (free): an app takes its repo's privacy, so a private repo
    gives one private app that only invited viewers can open. A public link needs a public
    repo (G3 ties to G4). Private repos need GitHub's broad `repo` OAuth scope. Apps sleep
    after 12 hours without traffic, and any visitor can wake them. Resources (dated
    2024-02): 0.078-2 cores, 690 MB-2.7 GB RAM.
  - **Render** (free): deploys from a private repo with a public URL. It spins down after 15
    idle minutes and takes about a minute to start; 750 hours a month.
  - **Hugging Face Spaces:** Gradio and Docker Spaces now need a paid plan. Out.
  - **Owner decision (D9):** build and verify locally in Step 8. Deploy on Community Cloud in
    Step 9 after the scrub, with G3 and G4 asked together.
- **G1 removed by the owner (2026-10-11).** The French data stay cited; the "runtime fetch,
  never committed, derived analytics only" restrictions are gone. Updated: the blueprint
  (header, G1 row, invariant 2, Step 8, risks, acceptance), `DATA_SOURCES.md`, the
  docstrings of `data/french.py` and `report/figures.py`, and REPORT §3's sentence that cited
  "the French library's terms".
- **Plan approved as written** (`plans/step-8-plan.md`): all nine decisions as recommended
  after the G1 change.
  - The dashboard reads a committed bundle in `app/data/`, written by `reproduce --dashboard`
    under the holdout lock, which checks its own 5 bp series against the holdout's tables.
  - All 17 strategies in all three views. The charts compare at most 3 (changed from the
    plan's 6, see below); the metrics table can show all 17.
  - Invariant 2 amended: `.gz` added to the check, `app/data/` allowed.
- **The bundle** (`reproduce --dashboard`, 2026-10-12 00:05 UTC on `40e3a69`, 2 min, exit 0,
  committed in `69095c1`):
  - 17 strategies x 3 views, daily from 1964-07-02 (USD) and 1972-01-04 (CAD, CAD hedged) to
    2026-08-31, plus factors, turnover, weights and the four registered holdout tables. 9.9 MB.
  - The check passed: every strategy's 5 bp holdout metrics in USD and CAD equal
    `performance.csv`, and the hedged monthly series equal the robustness run's, to 1e-12.
    `net_at` gives back the engine's net returns bit for bit (0, 5 and 20 bp in the tests).
  - The run warns that src changed since the tag (it names the 7b and 8 files) and records
    them; the dirty flag now leaves `app/data/` out, like the manifest.
  - Reading it takes about 2 s (exact float parsing); importing scipy and statsmodels on
    Windows adds about 4 s on a cold start.
- **The charts compare at most 3 strategies, not the plan's 6** (plan D2 and D7 amended).
  The dataviz validator on every pair (lines cross) passes only slots 1-3 in both modes; with 4
  or more the dark slots 1 and 7 sit at dE 9.8 and the light slots 2 and 4 at 13.7, under the
  hard floor of 15. The metrics table can show all 17 at the current settings.
- **The palette moved to `report/palette.py`** (no matplotlib), so the app can import it;
  `report.figures` re-exports `Theme` and `THEMES` unchanged.
- **The app's first version** (`app/streamlit_app.py`, `app/charts.py`, `.streamlit/config.toml`):
  sidebar (investor view, strategies, months, cost, benchmark), the Performance tab (growth of
  $1, drawdowns, metrics, the registered tests), an About tab with the citations. Checked in the
  browser: 1/N 12.72% CAGR and 16.7% volatility, minimum variance's registered difference
  −0.041 [−0.315, 0.234], as in the REPORT.
  - End labels that would collide are dropped (the legend and tooltip carry them), and the log
    scale is on by default only past 25 years.
  - Streamlit keeps imported modules from the app folder loaded: restart the server after
    editing `app/charts.py`.
  - Patch scripts on Windows: write with `newline="\n"`; never put backticks in a
    double-quoted `uv run python -c "..."` (bash runs them).

## 2026-10-11: step 7b, session 1 of 3 (plan, the one-shot holdout run)

- **Plan approved** by the owner as written (`plans/step-7b-plan.md`, commit `9ca40f4`): all
  nine decisions as recommended, including the one-shot holdout right after the plan commit.
  - Grounding (dates and metadata only): after 1969-07-01 the 49 industries miss one cell
    (Softw, 1971-03-11). The cached 49-industry file is not in the frozen vintage, so the plan
    records its SHA-256 (`8f394fe3…40de`) before any holdout number.
- **The holdout run, once:** `uv run portfolio-lab reproduce --holdout`.
  - Started 2026-10-11 19:30:00 UTC, finished 19:34:57 (298 s, 4 calibration processes),
    exit 0, no warnings. It wrote 11 tables to research/results/holdout/.
  - The manifest's `runs.holdout`: commit `9ca40f4`, clean, `tag_commit` `317d4c4`
    (prereg-v1), `changed_since_tag` empty, sources identical to the frozen vintage.
  - 188 months, 2011-01 → 2026-08, both views.
- **The four primary tests (PLAN.md §3): no rejection.** Every Holm-adjusted p is 1.0, so each
  reads "no detectable difference at this power", never "equal". Annual Sharpe difference
  against 1/N, its 95% CI, the calibrated block and the LW p:

  | Pair | USD | CAD |
  |---|---|---|
  | Minimum variance | −0.041 [−0.32, 0.23], b = 2, p = 0.77 | −0.026 [−0.36, 0.31], b = 1, p = 0.88 |
  | ERC | +0.012 [−0.071, 0.094], b = 1, p = 0.79 | +0.029 [−0.064, 0.122], b = 1, p = 0.54 |

  - Jobson-Korkie-Memmel agrees (p = 0.44-0.88).
  - The calibration's rejection rates are all at or below 4.4%, so the test is slightly
    conservative here, as on dev.
- **Performance** (holdout, net; monthly Sharpe × √12):
  - Sharpe: 1/N 0.80 (USD) and 1.14 (CAD); minimum variance 0.76 and 1.11; ERC 0.81 and 1.17.
  - Minimum variance in USD: volatility 13.8% against 16.7%, max drawdown −32.7% against
    −36.7%, CAGR 10.3% against 12.7%, annual turnover 0.88 against 0.15.
- **Exploratory:** the smallest raw p is 0.30 (ERC, EWMA, CAD). The smallest Holm-adjusted p
  over the 32 comparisons is 1.0. Nothing exploratory is a finding.
- **Luck:** best of 16 is p = 0.65 (USD) and 0.32 (CAD), not the dev's 0.000. Percentiles among
  each portfolio's own shadows: 1/N 0.68 and 0.79, minimum variance 0.16 and 0.49, ERC 0.95 and
  0.998.
- **VaR backtests** (500-day window, 6 series):
  - FHS passes Kupiec in all 6 at both levels, and conditional coverage in all 6 at 95%. At
    99% no method passes conditional coverage (FHS with 1000 days passes in 1).
  - Normal fails Kupiec at 99% in all 6. Every series is in the Basel green zone.
- **Stress at the HOLDOUT_END holdings** (replay; the realized path in brackets), USD:
  - COVID: 1/N −36.6% (−36.7%), minimum variance −36.1% (−32.6%), ERC −36.9% (−35.6%).
  - 2022 rates: 1/N −10.5% (−12.8%), minimum variance −0.4% (−3.8%), ERC −3.9% (−10.4%).
  - In CAD, the 2022 replay of minimum variance gains 4.0%.
- **Robustness code** (test-first; `study.py` untouched):
  - `robustness.py` (RED `80ac55a`, GREEN `f76924d`): the registered variants, the 49-industry
    universe rule, the `primary` cross-check against the holdout's tests table, and the
    `monthly` and `weights` series for the figures.
  - `reproduce --robustness` (RED `03d0a38`, GREEN `a1389f5`):
    - It runs under the holdout's lock, and only after the recorded one-shot holdout, on its
      tests table.
    - The 49-industry file must have the plan's SHA-256. It is recorded as the run's
      `extra_sources`, outside the frozen vintage.
    - A holdout `--rerun` is now recorded as `holdout_rerun`, never over the one-shot's record.
- **The robustness run** on `a1389f5` (clean; 95 s): 6 tables.
  - The manifest's `runs.robustness` names the three source files changed since the tag
    (`cli.py`, `manifest.py`, `robustness.py`), as planned.
  - The primary rows equal the holdout's tests table (the built-in check passed).
  - The 49-industry universe: all 49 industries from 1971-03-12, none dropped.
  - **The results** (annual Sharpe difference against 1/N; raw p at the holdout's block;
    descriptive, no Holm):

    | Variant | MV USD | MV CAD | ERC USD | ERC CAD |
    |---|---|---|---|---|
    | primary (252d, 5 bp) | −0.041 (0.77) | −0.026 (0.88) | +0.012 (0.79) | +0.029 (0.54) |
    | window 126d | −0.053 (0.71) | −0.005 (0.98) | +0.010 (0.82) | +0.040 (0.42) |
    | window 504d | −0.114 (0.40) | −0.168 (0.31) | +0.005 (0.91) | +0.010 (0.79) |
    | cost 0 bp | −0.034 (0.81) | −0.018 (0.91) | +0.012 (0.77) | +0.029 (0.53) |
    | cost 10 bp | −0.047 (0.74) | −0.033 (0.84) | +0.011 (0.78) | +0.029 (0.54) |
    | cost 20 bp | −0.059 (0.68) | −0.049 (0.76) | +0.011 (0.80) | +0.028 (0.53) |
    | 49 industries | +0.067 (0.65) | +0.133 (0.50) | +0.041 (0.15) | +0.080 (0.025) |

    - One raw p of 24 is below 0.05: ERC on the 49 industries in CAD, whose 95% CI is
      [0.011, 0.150]. With 24 descriptive tests about one is expected by chance, and PLAN.md
      §6 registers robustness as "no new hypotheses". It is reported, not claimed.
    - On the 49 industries, minimum variance turns over 1.4-1.6 a year (0.9 on 12), and its
      volatility is 13.1% against 1/N's 17.9% (USD).
  - **CAD hedged** (metrics only): 1/N's CAGR 12.7%, volatility 16.7%, max drawdown −36.7%;
    minimum variance 10.3%, 13.8%, −32.8%. These are about the USD view's, as CIP implies,
    with the CAD and US bill rates close on average over 2011-2026.
- **Figures** (RED `1f75641`, GREEN `108fea6`, PNGs `fc157d2`):
  - matplotlib is the optional extra `report`, and is also in the dev group.
  - `report/figures.py` and `portfolio-lab figures`: five figures, each as a light and a dark
    PNG.
  - The palette was checked with the dataviz validator: slots 1-3 on every pair and slots 1-7
    adjacent, in both modes. Aqua and yellow sit under 3:1 on the light surface, so the
    report's tables are the relief.
- **REPORT.md** (`8dbb3ab`): about 3850 words, 5 figures.
  - Its numbers come from a scratch script that prints the tables from the CSVs.
  - Two claims read off a chart were corrected against the tables before writing: health
    care peaks at 42% (not "30-60%"), and Business equipment is held in a fifth of the trades.
  - **New finding:** ERC's realized MDE is 0.12, twice the planned 0.06, because its holdout
    SE doubled. Minimum variance's is 0.39-0.47 against 0.48-0.51 planned.
- **Reproduction checks** (plan decision 8):
  - **The rerun:** `--holdout --rerun` after all 7b code (245 s), then `--robustness`. All 17
    tables match the one-shot's saved copy exactly (max difference 0), and 6 REPORT numbers
    were spot-checked. The one-shot's record is unchanged; the rerun is `holdout_rerun`.
  - **Drift, real data:** a fresh download of every source into a separate cache (2026-10-11
    20:31 UTC) was byte-identical to the frozen vintage. Nothing newer exists yet, so the
    dev run gave no warning, correctly.
  - **Drift, a made vintage:** the last USD/CAD row was dropped in the separate cache.
    - `--holdout --rerun` and `--robustness` refused before reading any data, naming `usdcad`.
    - A dev run warned, naming `usdcad`. Its 13 tables equal the recorded ones, since the row
      is after DEV_END.
    - research/results and the real cache were untouched.
- **python-reviewer** (no critical or high findings). Fixed (RED `7d3dd23`, GREEN `f19622d`):
  - Each run records the SHA-256 of every table it wrote and its own package versions.
    Robustness records the hash of the holdout tests table it extended. Before this, a rerun
    that differed would have overwritten the one-shot's untracked tables without a trace.
  - Robustness checks that every primary row is present and that every variant covers the
    holdout's months. The primary check now covers `months` too.
  - `universe()` refuses an empty universe, or less than 252 days of history. `role()` refuses
    unknown names.
  - `--rerun` is accepted only with `--holdout`. A manifest that is not a JSON object is a clean
    error. The figures name the extra only when matplotlib is the missing module.
  - The growth title comes from its table, and the primary row leads the forest plot.
  - `_reproduce` and `robustness.run` are split into one helper per mode or part.
  - **Not done** (cosmetic): moving `role()` to a lighter module, and the `ROLES` name used in
    two modules.
- **Final reruns** on `f19622d`: the holdout and robustness are again identical to the
  one-shot (17 tables, max difference 0), and the figures redraw to the same bytes.
- **`/code-review` (high): 9 findings, 7 fixed** (RED `81274bc` and `4004e7f`, GREEN
  `0428ea9`):
  - **A fresh clone could not run robustness.** `--robustness` reads only the cache, and
    nothing cached the 49-industry file. A dev `--refresh` now caches it, and REPORT's
    reproduce steps start with `--refresh`.
  - A first-ever `--holdout --rerun` is refused.
  - A missing matplotlib dependency (pillow, kiwisolver) names the report extra.
  - One month-end check for every variant and view.
  - `figures.weights` handles a single trade and gives a clear error without the view's
    minimum variance.
  - The help test goes through the CLI.
  - **Skipped:**
    - One `holdout_rerun` record is kept per manifest; git keeps every committed one.
    - The sibling `test_study` import works in pytest's default mode.
- The robustness rerun on the final code is identical to the one-shot again.
- **Mutation run** (scratchpad `mutate7b.py`): 21 of 23 caught at first.
  - The two survivors were test gaps: the "holdout recorded" refusal was masked by the next
    check's message, and nothing checked that the weights stack adds up to 1.
  - Both tests were tightened; now 23 of 23.
- **Verification loop:** 1424+ tests at 99.95% coverage. Ruff, format and mypy are clean;
  `uv build` and `uv lock --check` pass; no secrets in the diff.
  - Invariants 1-6 hold. `allow_holdout=True` appears only in cli.py.
  - `study.py` and PLAN.md are identical to `prereg-v1`, and the tag precedes the holdout
    record in the history.

## 2026-10-11: step 7a, extra-high review and re-freeze (owner asked)

- The owner asked for `/code-review` at xhigh effort over the whole branch: 11 findings.
  **All fixed** (RED `fd2c58f`, GREEN `9e2df18`).
- **The holdout lock's gaps:**
  - The one-shot rule could be bypassed by a first run to `--out elsewhere`. `--holdout`
    now refuses `--out`.
  - The code could change after the tag unnoticed. Uncommitted changes under src/ are now
    refused, and files changed since the tag are named in a warning and recorded in the
    manifest (`git.changed_since_tag`).
  - Every run after the freeze would have been "dirty", because the manifest is tracked and
    rewritten. The flag now ignores research/results/manifest.json.
- **Plan vs code:** PLAN.md §10 registers the 2001 → DEV_END betas as the comparison for the
  HOLDOUT_END shocks, but the holdout computed only the 2011 span. The shocks table now has
  a `span` level, with both spans on the holdout.
- **Also fixed:**
  - A corrupt manifest is a clean error.
  - Unreadable cache metadata warns once.
  - One git helper (`manifest.git_output`), and `inference` reuses `return_series`.
  - An explicit lookup for the cached primary test.
  - The drift warning no longer mangles the git ref on Windows.
- **The calibration runs in up to 4 processes** (`Settings.workers`, `cli._workers`), with
  identical results, which a test checks.
  - The dev reproduce took 453 s (it was 648 s).
  - Spawning works under the `portfolio-lab` console script on Windows.
- **Dev re-run on `9e2df18`:** all 13 tables are identical to the run on `9e87fdb`. The
  shocks table only gained the `span` level. The manifest records the commit as clean.
- **Mutation runs:** the session 2 script was re-pointed (the lock pattern had changed), and
  five mutations were added for this round's fixes.
- **Re-freeze:** PLAN.md §2 (the lock) and §13 (the commit) were updated, plus its change
  log. The tag `prereg-v1` moved from `9492cb5` to the new freeze commit. It was never
  pushed, and no holdout number exists.

## 2026-10-10/11: step 7a, session 3 of 3 (vintage, dev analysis, reviews, freeze)

- **Vintage refreshed 2026-10-10 (01:24 UTC)** (decision 6):
  - The French files' SHA-256 are unchanged from the 2026-10-02 cache, so no revision of
    history: ff5 `7881223b`, momentum `ecf82b11`, 12 industries `cde98642`. The 49-industry
    file also ends on 2026-08-31.
  - USD/CAD now reaches 2026-10-02 and the Canadian rates 2026-10-08.
  - **HOLDOUT_END = 2026-08-31** (`5b9144b`), 188 months.
- **Dev analysis** (`reproduce`; 648 s, then 13 tables): the numbers are in PLAN.md §13.
  - **Estimator rule:** lw_identity (12.71% against 12.77% sample and 12.82% LW constant
    correlation). It is nearly a tie.
  - **Primary pairs vs 1/N** (annual Sharpe difference, p):
    - Minimum variance: −0.015 (0.88) in USD, +0.060 (0.61) in CAD.
    - ERC: +0.011 (0.36) in USD, +0.014 (0.30) in CAD.
    - All four Holm-adjusted p-values are 1.0.
  - **Exploratory:** the smallest raw p is the momentum tilt at 0.5 (0.034 USD, 0.038 CAD).
    The smallest Holm-adjusted p over the 32 comparisons is 1.0.
  - **Calibration:** every rejection rate is at or below 5%, so the test is a little
    conservative on dev. The blocks chosen are USD MV 2, USD ERC 1, CAD MV 2, CAD ERC 4.
  - **MDE at 188 months:** minimum variance 0.48 / 0.57 (α 0.05 / Holm first step) in USD and
    0.51 / 0.61 in CAD; ERC 0.057 / 0.068 and 0.059 / 0.071.
  - **Luck: the null is degenerate for persistent tilts.**
    - Shadows' Sharpe ratios cluster near 1/N's: in USD, median 0.39 and 95th percentile
      0.41. Best of 16 is p = 0.000 in both views.
    - **Owner decision (2026-10-10): keep the registered method, and state that it measures
      "better than random rebalancing at the same turnover", not data snooping (Holm does
      that).**
    - The cap binds often only for EWMA minimum variance: shortfall 24-30%, turnover ratio
      0.84-0.87 (capped variant 0.94-0.95). My grounding estimate of about 3% was wrong.
  - VaR and stress match Steps 5a and 6:
    - FHS passes Kupiec in every series, and every method's misses cluster at 95%.
    - In the 2008 replay, 1/N loses 30.3% in USD and minimum variance (lw_identity) 20.4%.
- **python-reviewer** (no critical or high findings; it verified the HAC, the bootstrap, the
  calibration and the shadow accounting by derivation). Fixed (RED `d488950`, GREEN
  `e11e2fc`):
  - The holdout refuses `--refresh`. It checks the frozen vintage before any read and
    refuses a cache that changes while loading.
  - It reaches HOLDOUT_END's month, runs once (`--rerun` to repeat), and needs the tag as an
    ancestor of HEAD (`refs/tags/`), whose commit it records.
  - The root is the folder holding research/PLAN.md.
  - Also: a finite null, a unit-root guard, NaN-safe selection, a named Holm family, a
    manifest built from the cache without downloading, and an AST scan for every spelling
    of `allow_holdout=True`.
  - **Not done:** an external reference value from `bootTimeInference`. R is not installed,
    so the HAC rests on the loop reference, the hand cases and the size check.
- **`/code-review` (high): 8 findings, 7 fixed** (RED `3a5ed3b`, GREEN `aedd063`, plus
  `0cffd64`):
  - **The holdout now compares with the manifest committed at the tag.** Before, a dev
    `--refresh` after the freeze rewrote the working manifest, and the holdout would then
    see no drift.
  - The one-shot record no longer depends on `--out`, and each run keeps its own sources.
  - A dev run checks the cache before reading data, and `fetch` reuses `read_cached`.
  - **A primary test is run once per (pair, block).** Before, `tests` and
    `block_sensitivity` showed two Monte Carlo p-values for one test. This changed the dev
    p-values by up to 0.026; every Sharpe difference and every other table is identical
    between the two dev runs.
  - The luck table reports the turnover ratio. The degenerate null went to the owner (above).
  - `aedd063` went in with a test failing at random: the test zip carried a timestamp.
    `0cffd64` pins it.
- The manifest's dirty flag counts tracked changes only (`746a348`). The final dev run is on
  `9e87fdb`, clean.
- **Verification loop:** build, lock, mypy, ruff and format all pass. 1333 tests pass with
  99.91% coverage. There are no secrets, invariants 2-3 hold, and `allow_holdout=True` appears
  only in cli.py.
- **Freeze:** PLAN.md marked FROZEN, `study.PRIMARY_ESTIMATOR = "lw_identity"`, and
  `research/results/manifest.json` committed together in the last commit, which is tagged
  `prereg-v1` locally.

## 2026-10-09: step 7a, session 2 of 3 (luck benchmark, study, reproduce)

- **Built:**
  - `backtest/luck.py` (RED `c7b169d`, GREEN `c344831`):
    - Turnover-matched Dirichlet(1) shadows, vectorized over the shadows.
    - On their own targets they equal `engine.simulate` to 1e-12, costs included.
    - Shadows whose turnover is capped trade no more than the strategy, and the shortfall
      share is exact.
    - `percentile` and `best_of`.
  - **Bootstrap speed-up** (`7b60031`):
    - Resamples are summed from precomputed circular block sums, O(T/b) instead of O(T).
    - The draws are the same: a test compares it to resampling the rows, at blocks 1, 3
      and 7.
    - Calibration is about 42 s per pair at T = 188 and 2 min at T = 558 (it was 2 and 6).
  - `risk.backtest.report(start=...)` (`2b939c7`): exceptions are counted from an evaluation
    start, while the forecasts still use the earlier history.
  - `study.py` (RED `a61d955`, GREEN `8970045`). **The RED commit came after the code: this
    module and its tests were written in one pass.** It holds:
    - the registry: the 17 strategies, the seed and the candidates
    - the dev rule, the period months (the partial first month dropped) and seeded
      generators
    - `load` through the guarded loaders (RF taken on the asset days exactly)
    - `run`, which makes 13 tables
  - `cli.py reproduce`, `manifest.py`, and `config.HOLDOUT_END` / `PREREG_TAG` (RED
    `ab72d21`, GREEN `d66aa6c`):
    - **The holdout lock:** HOLDOUT_END and the estimator must be frozen, the tag must
      exist, and PLAN.md must equal its tagged copy (LF-normalized). It is tested in a
      temporary git repo.
    - **The manifest:** per source, URL, header line, SHA-256, size and retrieval time; per
      run, the command, time, estimator and git state; plus the packages and seed. It warns
      on drift.
    - The Step 0 stub-reproduce test was dropped (`7dfaaae`).
- **Mutation runs:**
  - Session 2: 19 planted bugs. 16 were caught at first.
    - Three survived because the small synthetic run could not tell the variants apart: the
      exploratory pairs on the ERC block, Holm over all for the primaries, and an unscaled
      MDE.
    - A test that every random part has its own seed label was added beforehand, for the
      "one RNG stream" mutation.
    - Targeted tests (`b397b62`): 19 of 19.
  - Session 1's run was re-pointed at the new bootstrap code: 21 of 21.
- **Timing, a full dev `reproduce` on the current cache** (written to the scratchpad):
  - **571 s (9.5 min)**: under the 10-minute bar, so no SLSQP warm start (decision 12).
  - 12 tables with the expected shapes:
    - 32 tests
    - 28 calibration rows
    - 34 performance and 34 luck rows
    - 60 VaR rows
    - 21 shock rows
    - 6 stress rows
  - MDE was skipped because HOLDOUT_END is not set yet. No warnings.
  - **Only the clock and the row counts were read; no number was looked at.** The recorded
    dev analysis is session 3's, on the refreshed vintage.
- Suite: 1310 passed, 99.8% coverage; ruff, format and mypy clean.

## 2026-10-09: step 7a, session 1 of 3 (draft pre-registration, Sharpe-ratio inference)

- **Plan approved** (`727ea80`): all twelve decisions as recommended, plus the
  post-registration addendum.
  - The owner asked whether data after 2026-08-31 could be used.
    - The French files (12 and 49 industries, factors) end on 2026-08-31. The 49-industry
      file downloaded on 2026-10-09 ends there too.
    - The library's page lists August 2026 as its latest month.
    - One more month would move the minimum variance MDE from about 0.450 to 0.449.
    - Splicing in other data was ruled out: invariant 3, and different return series.
    - Instead, months published after the tag can feed a descriptive addendum (PLAN.md §12).
- **Draft `research/PLAN.md` committed first** (`0f3c664`), before any strategy was compared
  on dev. The planning script printed no Sharpe ratio, mean, volatility or difference.
- **PLAN.md §5 amended before any dev comparison** (`d8d8efd`): the authors' own R code (CRAN
  `bootTimeInference`) settles what the paper leaves open.
  - It uses the prewhitened **Parzen** kernel, not QS (the paper's footnote 8 says they
    perform alike), with bandwidth 2.6614 (α n)^(1/5).
  - The singular values of the prewhitening VAR(1) are capped at 0.97. The small-sample
    factor is n/(n − 4) over the n = T − 1 residuals.
  - Bootstrap resamples have ⌊T/b⌋ blocks (l·b rows) and are centred at their own moments.
  - Calibration counts rejections of H0: Δ = Δ̂ on VAR(1) pseudo-series (intercept,
    stationary-bootstrap residuals with mean block 5, burn-in 50).
  - One difference is kept on purpose: the authors' Δ̂ uses the ddof = 1 standard
    deviation, and this code uses population moments throughout, as the paper's f does.
- **Built** (RED `bfb79ae`, GREEN `05a5f0a`, survivor tests `bedd6ce`):
  - `report/inference.py`:
    - `sharpe_ratio`, `sharpe_diff`, `diff_se`, `hac_psi`, `lw_test` and `calibrate_block`
    - `jobson_korkie_memmel`, `holm`, `min_detectable_difference`, `ceq` and
      `monthly_excess`
  - `risk/backtest.kupiec_exact`, plus a `kupiec_exact_p` column in `report`.
- **Tests** (58 for inference, 100% of the module):
  - Hand values for the Sharpe ratios, the gradient, Parzen weights, the kernel sum, the
    block Ψ*, the p-value, Holm, JK-Memmel, CEQ and the monthly compounding.
  - **A loop-based reference HAC**, written from the formulas, matches `hac_psi` to 1e-10.
  - **The size check:** iid normal pairs, T = 120, 300 tests at 5%, inside the 99% binomial
    band.
  - **Calibration:**
    - It prefers longer blocks for AR(1) data with φ = 0.8.
    - It tests the original estimate: H0 at 0 would reject nearly always.
    - It drops the burn-in.
  - **Mutation run:** 21 planted bugs.
    - 19 were caught at first. The burn-in and the calibration's null survived until the two
      targeted tests were added; then 21 of 21.
    - The script now retries its file writes. A Windows lock once left a mutation in place,
      and git restored it.
- **Timing at study size** (T = 188 months):
  - `lw_test` with M = 4999 takes 0.12 s.
  - `calibrate_block` takes about 2 minutes per pair (K = 1000, 7 block sizes, M = 499), so
    about 8 minutes for the four primary pairs on the holdout. The dev pairs are longer
    (T = 558): session 2 measures it and decides whether that needs speeding up.
- Suite: 1250 passed; ruff, format and mypy clean.

## 2026-10-09: step 6, session 2 of 2 (hypothetical shocks, dev check, review)

- **Built** (RED `c15ff95`, GREEN `a8dfec5`), `stress/hypothetical.py`:
  - `Shock` and the read-only `SHOCKS`: market (Mkt-RF −20%), momentum (Mom −35%), rates
    (dy +0.02), cad_up (fx = 1/1.10 − 1) and cad_down (fx = 1/0.90 − 1).
  - `cad_move(pct)` = 1/(1 + pct) − 1, for Step 2's CAD-per-USD fx.
  - `portfolio_excess`: the constant mix of the current weights, Σ w_i (r_i − rf), with
    zero-weight assets left out.
  - `apply`: per (asset, shock), P&L = β × move and SE = |move| × the HAC SE, with the fit's
    sample on every row.
- **Mutation runs:** 10 of 10 caught first; 34 of 34 on the final code of both modules.
- **Dev check** (model behaviour only, no ranking; DEV_END weights of 1/N and minimum
  variance (sample)):
  - **GFC replay** (63 days, 2008-09-02 → 2008-11-28; the whole table in 0.05 s):

    | Weights | USD view | CAD view | CAD weights, hedged view |
    |---|---|---|---|
    | 1/N | −30.3% | −19.0% | −30.1% |
    | Minimum variance | −20.2% | −7.3% | −20.0% |

    - The USD rose 16.3% against the CAD over the window, so the CAD view loses about 11 pp
      less. The hedged view is about the USD loss, as CIP implies.
    - The strategies' realized 2008 paths: 1/N −30.4% (USD) and −19.1% (CAD), about the
      replay since its weights barely move. Minimum variance −22.1% and −10.0%: its 2008
      holdings were not its DEV_END ones.
  - **Shocks** (weekly partial betas, 2001-01-03 → 2010-12-31, 523 weeks; P&L ± SE):

    | Weights | Market −20% | Momentum −35% | 10-year +200 bp | CAD +10% | CAD −10% |
    |---|---|---|---|---|---|
    | 1/N, CAD | −19.9% ± 0.2 | +0.4% ± 0.3 | −0.2% ± 0.3 | −8.8% ± 0.1 | +10.8% ± 0.2 |
    | Minimum variance, CAD | −15.0% ± 0.7 | −1.4% ± 0.8 | −3.0% ± 0.8 | −9.0% ± 0.3 | +11.0% ± 0.4 |
    | 1/N, USD | −19.9% ± 0.2 | +0.4% ± 0.3 | | | |
    | Minimum variance, USD | −14.4% ± 0.6 | −0.9% ± 0.9 | | | |

    - **Natural hedge, in numbers:** the partial market shock holds fx at 0. In the 2008
      window the market fell 30.6% in USD, but 1/N's CAD replay lost 19.0%.
    - **Sample sensitivity (USD, 1963+ instead of 2001+):** minimum variance's market shock is
      −17.5% instead of −14.4% (Mkt-RF 0.87 vs 0.72). 1/N's barely moves (−19.9%); its
      momentum beta is −0.045 (SE 0.011) instead of −0.010.
  - The shocks take 0.3 s in all.
- **`/code-review` (high): 8 findings, all fixed** (RED `0e54c7a`, GREEN `9516f2b`):
  - **Times of day:** 16:00 stamps shifted the window. The COVID window took in the peak day
    and dropped the 23 March trough. Windows now compare calendar dates.
  - **Weekend end:** it is reached on the Friday before (changed from session 1, which asked
    for a later day). A weekday end, even a holiday, still needs that day.
  - **Duplicates and mixed tables:** `table` refuses duplicate scenario names. `apply` checks
    each asset's own terms; a mixed USD/CAD table gave a KeyError.
  - **Shock moves:** `True` was accepted as a +100% move; a move must now be a real number.
  - **Shared check:** `data.views.check_view` replaces three copies of the View check, the
    engine's included.
- **python-reviewer: no critical or high findings.** The 4 mediums and the useful lows were
  fixed (RED `94e5a07`, GREEN `2b80a28`):
  - **Holes:** a hole inside a window raises: no close for more than 7 calendar days from
    the anchor to the last day. Three missing weeks were compounded over silently. The 9/11
    closure (7 days) passes.
  - **Zero weights:** `replay` skips assets at weight 0, like `portfolio_excess` (changed
    from session 1). An industry not yet listed no longer blocks a replay; a held asset's gap
    still raises.
  - **`table`:** it is tested with the holdout flag on a synthetic 2020 window, its dtypes
    are pinned, and a failing replay gets a note naming its row.
  - **Scenarios:** they keep calendar dates and check their name.
  - **`apply` and `Shock` checks:** `apply` refuses repeated rows, non-finite coef/se and a
    negative SE. A return shock cannot be −100% or worse.
- **Tests:** 1187 pass, 99.91% coverage (the stress modules 100%).
- **For 7b:**
  - Replay HOLDOUT_END weights through `table(..., allow_holdout=True)` from `reproduce`.
  - The realized paths come from `window(result.net, scenario, allow_holdout=True)`.
  - The 49-industry NaN policy now matters only for held assets.

## 2026-10-09: step 6, session 1 of 2 (historical replay)

- **Plan approved** (`3914321`): all eight decisions as recommended. The grounding numbers on
  dev data are in `plans/step-6-plan.md`.
- **Built** (RED `b0be0c2`, GREEN `cac9ef9`), `stress/historical.py`:
  - `Scenario` and the read-only `SCENARIOS` (gfc, covid, rates), pinned to the blueprint's
    dates by a test.
  - `window`: the days in (start, end]. The holdout rule (`resolve_end`) runs before the
    returns are read, so a 2020 or 2022 window raises `HoldoutError` even on synthetic data.
  - `replay`: buy-and-hold through `engine.simulate` with one target and no cost.
  - `outcome`: the P&L, the worst day and its date, the max drawdown (the anchor's value
    counts as the first peak) and the days covered.
  - `table`: one row per (scenario, view, portfolio). It checks every scenario against the
    holdout rule before replaying anything.
- **Conventions settled while building:**
  - The data must reach both edges: a day on or before `start`, and one on or after `end`.
    A weekend `end` therefore needs a later day, since data can't tell a weekend from a gap.
  - A gap inside the window raises even at weight 0, as in the engine. The 49-industry NaN
    policy is still 7b's.
- **Tests:** 54, with 100% coverage of the module. The full suite passes (1140).
- **Mutation run:** 12 of 12 planted bugs caught:
  - [start, end] windows, an exclusive end
  - the drawdown peak without the starting 1
  - constant weights instead of drift
  - the guard skipped, and the table's guard moved after the replays
  - the anchor taken after `start`
  - either coverage check skipped
  - the P&L summed instead of compounded
  - the worst date off by one
  - an empty window allowed

## 2026-10-09: step 5b, session 2 of 2 (rolling and macro betas, dev check, review)

- **Built** (RED `882c862`, GREEN `eb99c88`):
  - `rolling_betas`: numpy strided windows with batched normal equations, indexed by each
    window's last day; a collinear window raises naming its day.
  - `macro_betas`: CAD excess returns on the six factors, fx and `dy`, all partial; weekly fits
    sum `dy`.
  - `views.yield_changes`: carries levels like `fx_returns`, so a Canadian holiday is a change
    of 0.
- **Mutation runs:** 15 of 16 planted bugs caught. The survivor, a window as long as the data,
  is now tested: the single window equals the full-sample `regress` (`83c199f`). Session 1's
  survivor (Δy compounded) is caught.
- **Dev check** (scratch script, model behaviour only, no ranking; CAD 1972-2010, 10,093
  days, 2,088 weeks; lags 11 daily and 7 weekly):
  - **Market plumbing:**
    - USD: Mkt-RF beta 0.999999999999998, R² = 1.
    - CAD: Mkt-RF 1.0015, fx 0.9997, R² 0.9996.
  - **USD industry betas look right:**
    - Money HML +0.77.
    - BusEq HML −0.50, RMW −0.52, CMA −0.60.
    - Utils Mkt-RF 0.68, the lowest. Durbl Mkt-RF 1.19, the highest.
    - R² from 0.55 (Utils) to 0.90 (Manuf).
  - **CAD fx betas** run 0.72-1.10 daily and 0.62-1.21 weekly. Energy is the lowest (0.72 /
    0.62): US energy stocks rise with oil when the CAD does, a partial natural hedge.
  - **Strategies:**
    - 1/N: Mkt-RF 0.985, R² 0.985.
    - Minimum variance (sample): Mkt-RF 0.66, R² 0.74.
    - CAD fx betas 0.98-1.00, daily or weekly.
  - **Macro, 2001-2010 (2,514 days, 523 weeks)**, yield beta alone → partial daily (SE) →
    partial weekly (SE):

    | Portfolio | Alone | Partial daily (SE) | Partial weekly (SE) |
    |---|---|---|---|
    | 1/N | +8.43 | −0.02 (0.07) | −0.09 (0.15) |
    | Minimum variance | +4.76 | −0.63 (0.25) | −1.83 (0.39) |
    | Utils | +4.86 | −1.21 (0.47) | −3.14 (0.95) |
    | Money | +11.40 | −0.19 (0.31) | −0.22 (0.61) |
    | Market | +9.01 | +0.00 (0.01) | −0.02 (0.04) |

    - A +200 bp shock, partial weekly: Utils −6.3%, minimum variance −3.7%, 1/N −0.2%.
    - Weekly yield betas are 2.5-3 times the daily ones for the rate-sensitive portfolios.
      **For Step 6: use the weekly betas** (asynchronous closes; 523 weeks).
  - **Rolling (252 days):**
    - 1/N USD Mkt-RF beta 0.91-1.06 over 11,455 windows; CAD fx 0.91-1.04.
    - 0.2 s per series; matches lstsq to 2e-15 on dev data.
- **`/code-review` (high): 5 findings, all fixed** (RED `8e308cd`, GREEN `33efedd`):
  - **Weekly fits reported a holdout date.** On dev data ending Friday 2010-12-31, the end
    was 2011-01-05, that week's Wednesday label. Fits now report the daily days they used.
  - **Weekly fits were 30 times slower than daily.** pandas' resample ran per column per
    asset: 34 s on 49 industries. `to_weekly` now uses numpy `reduceat` over contiguous weeks:
    3.1 s on 49 industries, 0.8 s on 12. A hypothesis test pins it to pandas' W-WED
    resampling.
  - Asset labels 1 and "1" collided as text, silently dropping an asset.
  - `to_weekly` of no rows raised a numpy reshape error.
  - The whole-number check was copied 3 times; it is now `_validate.is_whole`, shared with
    `var`.
- **python-reviewer: no critical or high findings; 1 medium, 4 low, 6 test gaps.** Fixed
  (RED `c5bebeb`, GREEN `58d06f6`):
  - **Medium:** the `reduceat` rewrite split a week when stamps carried a time of day (a
    09:30 among 16:00s). Weeks now use calendar dates.
  - An inf now names its column and day.
  - A constant y raises instead of giving 0/0 R² and t-stats.
  - An excess frame with no columns raises clearly.
  - All six test gaps are closed: the full table vs `Fit`, inf in a regressor, lags = T − 1,
    text labels, the rolling window message, and `yield_changes`' `max_age_days`.
  - **Not changed, documented:** `rolling_betas` solves normal equations, which square the
    condition number. That only matters for a regressor SD below ~1e-7 within a window.
- **Final:**
  - 1086 tests pass, 99.90% coverage.
  - ruff, format and mypy are clean; `uv build` and `uv lock --check` are OK.
  - Invariants 1-6 hold.

## 2026-10-09: step 5b, session 1 of 2 (plan review, regression core, factor betas)

- **Plan review (the owner asked for one before approving).** Scratch scripts tested every
  claim on synthetic and dev data, looking at model behaviour only, never a ranking. Four
  decisions changed; the plan's "Review 2026-10-09" section lists them.
  - **Macro betas must be partial.** From 2001 to 2010 the CAD 1/N yield beta was +8.45 with
    no controls and −0.03 with the six factors held fixed. Market: +9.01 vs 0.00. Utilities:
    +4.86 vs −1.21 daily and −3.14 weekly.
    - The uncontrolled beta is the stock-bond correlation (corr(Δy, Mkt-RF) = +0.36), not
      duration. It would make Step 6's "+200 bp" stress a gain, and the sign flipped in 2022.
    - fx shows the same pattern: 0.53 alone (the USD rises when US stocks fall, a natural
      hedge for Canadians) against the mechanical 0.97-1.0 once the factors are held fixed.
  - Weekly yield changes must be summed, not compounded.
  - Alignment is strict: an inner join would accept a monthly y against daily factors.
  - Rolling betas use numpy windows: 0.15 s and 1e-14 from a lstsq loop, against `RollingOLS`
    at 0.75 s, whose add-and-remove updates give 1e-13.
- **Facts measured on dev data:**
  - statsmodels' HAC with default options is exactly the Bartlett sandwich with no 1/T factor
    and no correction (4e-16), with normal p-values (`use_t=False`).
  - Lags: 11 for both daily views (T = 11,959 USD, 10,094 CAD), about 7 weekly.
  - The small-sample correction would change SEs by 0.04-0.3%.
  - Compounding excess returns weekly instead of total and RF separately is off by at most
    1.3e-4 (the weekly SD is 1.9%).
  - Holidays: 91 Fridays and 46 Wednesdays from 1963 to 2010, 24 of those Wednesdays being the
    1968 paperwork-crisis closures.
  - Market plumbing: in the USD view, Mkt-RF beta 1 to 2e-15 and R² = 1; in the CAD view,
    fx 1.000, Mkt-RF 1.001, R² 0.9996.
  - Daily vs weekly fx betas with the factors in: 1/N 0.981 vs 0.991, so the asynchrony costs
    little at the portfolio level.
- **Built** (`risk/factors.py`, RED `55b6190`, GREEN `4533f62`):
  - `nw_lags`: computed exactly in integers (lag⁹·100² ≤ 4⁹·T²), because the float formula
    puts T = 51,200 at 15.999… instead of 16.
  - `regress` → `Fit` (coef, se, t, p by term with `alpha` first; r2, adj_r2, nobs, lags,
    start, end; `table()`).
  - `factor_betas`: a long table indexed by (asset, term).
  - `to_weekly`: `how="compound"` or `"sum"`, W-WED.
  - `_validate.check_aligned` now accepts a DataFrame as its second argument.
- **Tests:** 48 tests, 100% coverage of the module.
  - A plain-loop HAC sandwich at lag 1 (6 rows) and lag 3 (300 autocorrelated rows), plus
    lag 0 = White.
  - Hypothesis invariances: scaling y scales betas and SEs; adding c·X_k shifts beta_k only.
  - Synthetic USD and CAD betas recovered within 4 HAC SEs.
  - Alignment: span, monthly-vs-daily, edges, interior gaps; the weekly hand examples.
- **Mutation run:** 23 of 24 planted bugs caught. The survivor, yield changes compounded
  instead of summed in a weekly fit, has no caller until `macro_betas` (session 2). Its
  weekly test must catch it, so that mutant gets re-run then.

## 2026-10-09: step 5a, session 3 of 3 (dev data: VaR model behaviour)

- Series: `Result.net` of 1/N and minimum variance (sample, 252 d) from `compare`, plus the
  market benchmark, in the USD view (1964-07-02 to 2010-12-31, 11,706 days) and the CAD view
  (1972-01-04 to 2010-12-31, 9,841 days). All end at DEV_END. A scratch script ran them;
  nothing was committed.
- Per decision 1 (Step 4's decision 6), these checks show how the VaR models behave; they don't
  rank strategies.
- **Speed: the exit is met.**
  - All four methods at both levels take ~0.9 s on 11,706 days (best of 3); `report` takes
    1.6-2.3 s per series.
  - Cornish-Fisher first took 0.63 s because numpy's general power ran `dev**3` and `dev**4`.
    Products bring it to 0.13 s (commit `ddf5c5f`).
- **Window 500, 99% exception rates** (expected 1%; USD 1/N, USD market, CAD 1/N):
  - historical: 1.65%, 1.64%, 1.41%
  - normal: 2.20%, 2.08%, 1.95%
  - Cornish-Fisher: 1.02%, 1.12%, 0.92%
  - FHS: 1.10%, 1.13%, 1.17%
- **Who passes the tests:**
  - Historical and normal fail Kupiec at 99% on all 6 series: too many exceptions (the normal
    model has fat-tail misses; the historical 500-day window is slow to react).
  - Cornish-Fisher gets the 99% rate right but fails independence: its misses cluster.
  - FHS is the only method that passes conditional coverage at 99% (4 of 6 series; both
    minimum-variance series fail independence).
  - At 95%, every method fails independence on every series: daily misses cluster in all of
    them. FHS gets the 95% rate right (5.0-5.2%).
- **Basel zones over time** (share of trailing-250-day windows green/yellow/red at 99%):
  - FHS is never red (80-89% green).
  - Historical is red 3-11% of the time; normal 15-21%; Cornish-Fisher 0-5%.
  - On the last 250 dev days, every method is green.
- **Window sensitivity** (mean exception rate over the 6 series; windows 250/500/1000):
  - At 99%, FHS goes 1.35% → 1.15% → 1.06%: 250 days leave only ~2.5 residuals in the 1%
    tail. Historical stays at 1.5-1.6% and normal at 2.0-2.1%, whatever the window.
  - Conditional-coverage passes at 99% for FHS: 0 of 6 series at 250 days, 4 at 500, 4 at
    1000.
  - For 7a: 500 is a reasonable default; 1000 does slightly better for FHS. This is a
    pre-registration choice for 7a; nothing changes here.
- **Cornish-Fisher in practice:**
  - The rolling 500-day excess kurtosis reaches 91-104 and the skew −6.2 to −7.2. Those are
    the windows holding October 1987: 1/N USD fell −17.4% on 1987-10-19.
  - The CF quantile is non-monotone on 0-2.1% of days (slope ≤ 0 somewhere on z ∈ [−8, z_p]).
  - CF ES < VaR on no dev day: the turn sits deeper than the tail mass that matters.
- **Sanity:**
  - VaR > 0 and ES ≥ VaR on every day, for every method.
  - The 99% VaR for 1987-10-20, the day after the crash:
    - FHS 5.2% → 15.0%
    - CF 3.0% → 14.2%
    - normal 1.9% → 2.7%
    - historical 2.3% → 2.4% (one more value in a 500-day window)
  - `next_day` after 2010-12-31: FHS 1.53%, historical 4.36%. The 2008-09 crisis is still in
    the historical window, while the EWMA has forgotten it.
- **Review.** python-reviewer found no critical or high issues. It re-derived the CF ES
  bracket and checked every alignment by experiment, including block joins at sizes 1, 7 and
  64. `/code-review` (high) found 7 items. Fixed test-first (RED `d072432`, 29 failing;
  GREEN `6c85107`):
  - **ES dropped a tail observation** (reviewer, medium). When (n − 1)p should be a whole
    number k, 1 − 0.9 = 0.09999999999999998 put q a hair below the (k+1)-th smallest value,
    so that value left the tail. At n = 101 and 90%, this happened on 12 of 40 seeds (the
    reviewer saw 844 of 2000). Now an index within 1e-9 of a whole number takes that order
    statistic exactly (`np.partition`). Nothing changes at 95% or 99% for windows 250, 500
    and 1000 (indices 24.95, 4.99, 12.45, ...), so the dev numbers above stand.
  - **`forecast` raised on the row it drops** (reviewer, medium). It used to compute the
    day-after-the-data row too, so a series ending in 10 zero days raised "no variance".
    `forecast` now runs the windows of r[:-1], and `next_day` looks only at the last window
    (FHS keeps the whole history for its filter).
  - **Results that aren't finite now raise, naming the day.** A NaN VaR would compare False
    and silently count as "no exception".
    - A +1e200 return made the normal forecast inf and CF NaN.
    - In FHS, `lfilter` turns an overflowing r² into NaN (its padded zero coefficient times
      inf), which the σ check had reported as "volatility reached 0".
    - The point functions check their results too.
  - Numpy integers and reals are accepted for `window`, `days` and `level`.
  - Blocks now hold 2²⁰ values whatever the window (2048 rows meant 80 MB at window 5000).
  - `check_decay` is shared by `ewma_cov` and the filter.
  - `report` rejects empty `levels` or `methods`.
  - Test changes:
    - The FHS loop reference runs its own EWMA recursion instead of the filter under test.
    - The block-join test compares every row, with blocks of 3 windows. It used to compare
      only the last block, so it could not catch a bug at a join.
  - Left as is:
    - `report` recomputes each method per level (~1 s per series; revisit in 7a if the grid
      is slow).
    - `decay` is validated for every method, failing fast on a bad value.
    - `LikelihoodRatio.days` is T while the transition tests use T − 1; this is now
      documented.
- Verification: 1009 passed, 99.89% coverage (risk modules 100%); ruff, format and mypy
  clean; `uv build` and `uv lock --check` fine. Invariants 1-6 hold.

## 2026-10-09: step 5a, session 2 of 3 (rolling forecasts, backtests)

- `risk/var.py`:
  - `forecast(returns, *, method, level, window=WINDOW, decay=EWMA_DECAY)` returns
    `var` and `es` columns indexed by the day forecast; the first row is day `window` + 1.
  - `next_day(...)` gives today's VaR by the same rule: it equals the last row of
    `forecast` on the series with any return appended.
  - `METHODS = ("historical", "normal", "cornish_fisher", "fhs")`, `WINDOW = 500`.
  - One minimum window for every method, 4 days (excess kurtosis needs 4).
  - Rows are processed in blocks of 2048 windows (`sliding_window_view` builds no copy;
    `np.quantile` and the moments copy one block at a time).
  - Normal and CF compute each window's moments directly: the mean, sd with ddof = 1, and
    pandas' G1/G2 formulas from sums of powers of deviations. Pandas' online rolling moments
    aren't used.
  - A window with sd ≤ 1e-12 × its largest |return| raises, naming the forecast day.
  - FHS:
    - The seed is the mean square of the first window; a first window of zeros raises.
    - σ comes from `_ewma_var` (n + 1 values); a σ that underflows to 0 raises (decay 1e-10
      and 30 zero days do it).
- `risk/backtest.py`:
  - `exceptions(returns, var)` is strict: r < −VaR. Every forecast day needs a return.
  - `kupiec`, `transitions`, `christoffersen`, `conditional_coverage` return a
    `LikelihoodRatio(stat, pvalue, df, exceptions, days)`.
  - `basel_zone` takes exactly 250 days at 99%; `basel_zones` gives trailing 250-day windows.
  - `report(returns, *, levels, methods, window, decay)` has one row per (level, method);
    `basel` is the zone of the last 250 days at 99%, otherwise None.
  - `xlogy` handles 0·ln 0, and a statistic that rounds below 0 is 0. With only the last day
    an exception, π11 has no days behind it and drops out (LR_ind = 0 exactly).
- `_validate.tail_prob` and `_validate.return_series` are shared by both modules (they were
  private in var.py).
- **The Kupiec test's true size differs from 5%:**
  - From the exact binomial distribution, a 5% Kupiec test at 99% over 250 days rejects a
    correct model **9.5%** of the time; at 95% over 500 days, 5.4%; over 1000 days, 5.1%.
  - Zero exceptions in 250 days at 99% is itself rejected (LR = 5.03).
  - The Monte Carlo test (4000 × 500 days, iid normal, true VaR) matches 5.39% within 4
    standard errors. For 7a: report the exact size (or exact binomial p-values) next to the
    chi-square ones for 99% over short windows.
- Tests:
  - RED `507b999` (fails at collection), GREEN `020c45c`. 73 new tests.
  - Full suite 953 passed, 99.88% coverage, both risk modules at 100%; ruff, format and mypy
    clean.
  - The vectorized forecasts match a per-window loop of the point functions to 1e-12 (rtol),
    including across processing blocks (5200 days).
  - No look-ahead: perturbing returns from day k on never changes a forecast for a day ≤ k,
    for all four methods (hypothesis). A −25% day changes the next forecast, not its own.
- Test fixes in GREEN:
  - The FHS hand case used a 2-day window, below the minimum; it was redone with 4 days. The
    redone case also shows a 4% day lifting the next VaR from 2% to √1e-3 = 3.16%.
  - The Monte Carlo test built 4000 date ranges (46 s); it now builds one (2 s).
  - The "scale commutes" property hit a subnormal (5e-324 × 0.5 = 0 creates a tie); it now
    avoids subnormals, like the power-of-2 property.
- Mutation run: 22 of 22 planted bugs caught.
  - Windows that include the forecast day; rows shifted by one; an FHS seed from the whole
    series; residuals divided by the next day's σ; FHS scaled by the previous σ.
  - sd with ddof = 0; the G1 and G2 constants; no flat-window check; the wrong day in the
    message; an off-by-one length check; a block seam that drops rows.
  - `≤` for an exception; the Kupiec MLE term; π over T instead of T − 1; π11 taking π01's
    formula; df 1 for CC; the Basel 4/5 boundary; the rolling count off by one; no clamp at
    0; Basel on the first 250 days; a wrong n10 count.

## 2026-10-08: step 5a, session 1 of 3 (plan, point estimators)

- Plan `plans/step-5a-plan.md` approved as drafted. The seven owner decisions:
  - Input: any daily simple-return Series; the dev checks use total net returns (`Result.net`).
  - One `window` parameter, default 500 days; 250 and 1000 only as dev sensitivity.
  - Historical VaR uses numpy's linear quantile; ES is the mean of the returns at or below
    −VaR. Parametric ES uses closed forms.
  - FHS: zero-mean EWMA (λ = 0.94) in one causal pass, seeded with the first window's mean
    square; the residuals are rescaled by σ(t+1).
  - Basel zones only at 99% over exactly 250 days.
  - The risk functions are pure and unguarded, like `simulate`.
  - Vectorized windows, checked against a loop.
- `risk/var.py`:
  - `var_es(sample, level)`, `normal_var_es(mu, sigma, level)`,
    `cornish_fisher_var_es(mu, sigma, skew, exkurt, level)`,
    `ewma_vol(returns, *, seed_var, decay=EWMA_DECAY)`, `scale_horizon(x, days=10)`, `LEVELS`.
  - All return a `VarEs(var, es)` NamedTuple of positive losses.
  - The formulas live in private array cores (`_historical` on a 2-D array of windows,
    `_normal`, `_cornish_fisher`, `_ewma_var`) for session 2's rolling forecasts.
  - CF ES is the exact tail mean of the CF quantile:
    −μ + σ (φ(z)/p) [1 + zS/6 + (z² − 1)K/24 − (2z² − 1)S²/36], from E[Zᵏ | Z < z].
    Numerical integration agrees to 1e-9 (1e-11 in the plan's check).
    - It is not monotone for S = 1, K = 0 at 99%: VaR 1.215σ but ES 0.905σ. A test pins this
      as documented behaviour.
    - Also documented: excess kurtosis lowers the CF 95% VaR, because (z³ − 3z) > 0 at
      z = −1.645.
  - `ewma_vol` is `scipy.signal.lfilter` with initial state λ·seed, indexed by the day being
    forecast (σ(s) uses returns through s − 1). The forecast after the last day stays in
    `_ewma_var`'s extra element.
- Float details:
  - **The tail mean can round above q:** 7 copies of 0.045899312196799685 average 1 ulp above
    the value itself. ES = −min(tail mean, q) keeps ES ≥ VaR exact, and a test pins that
    sample.
  - **numpy's linear interpolation never rounds below its lower neighbour,** so the tail is
    never empty. A `max(q, min)` guard was dead code (the mutation run showed it) and was
    removed.
  - **Exact power-of-2 scaling fails for subnormals** (around 1e-311): halving drops bits. The
    property test keeps magnitudes above 1e-300.
- Tests:
  - RED `b9243a7` (fails at import), GREEN `082de7f`. 88 tests in `tests/test_var.py`.
  - Full suite 877 passed, 99.86% coverage; ruff, format and mypy clean.
- Mutation run: 13 of 13 planted bugs caught:
  - the quantile method
  - `<` instead of `≤` in the tail
  - no ES clamp
  - each of the three CF coefficients
  - the normal ES's `/ p`
  - the EWMA's (1 − λ) and seed state
  - a one-day look-ahead in `ewma_vol`
  - the level bound
  - √ in the horizon
  - p = level instead of 1 − level

## 2026-10-08: step 4, session 4 of 4 (review)

- python-reviewer: no critical or high findings. It ran its own differential fuzz (300
  random calendars with gaps vs a brute-force schedule and a daily loop, 0 mismatches to
  1e-13) and checked the guard on real dev data. `/code-review` (high) found 9 items, all
  low to medium.
- Fixed test-first (RED `c6d5b6f`, 34 failing; GREEN `0948748`):
  - **`reprice(result, view, *, cost_bps, allow_holdout=False)`** (the reviewer's medium):
    - The session-2 recipe for 7b's cost grid, `simulate(result.weights, returns)`, had no
      guard; on a view that runs past DEV_END it extended the ledger into the holdout.
    - `reprice` cuts the view at the result's last day by the holdout rule (a result ending
      past DEV_END needs the flag again) and requires exactly the result's days.
    - At 0/5/20 bp it equals `run` at that cost.
  - **`compare` checks an explicit `start` before anything runs.** It used to estimate the
    first strategies before failing on the long lookback. The check is shared with `targets`
    as `_schedule`.
  - **Dates:** `_validate.as_date` takes a str, date/datetime or datetime64, tz-naive and
    real. `rebalance_dates` and `data.resolve_end` use it.
    - An int `start=2001` was read as nanoseconds since 1970, which meant "from the
      beginning".
    - `end=20101231` truncated everything (it failed closed).
    - A tz-aware date raised a raw pandas TypeError.
  - **`cost_bps` in [0, 5000)**: at 10,000 bp a full switch gave a net return of -200%.
  - **`targets` puts the weights in the returns' column order by label.** Before, the first
    row's order decided.
  - **`summary`'s gross column now shows an annual cost of 0**; turnover stays the same in
    both columns.
  - **`Result` checks** that gross matches net's dates, and turnover matches cost's.
  - **New tests:**
    - a hand example of trades on consecutive days
    - a cost just under 5000 bp
    - a NaN benchmark
    - a note on why the trade-day shock test holds (a 10-day window)
- Left as is (for the owner):
  - `simulate` and `targets` stay unguarded pure functions, like `data.views`. `run`,
    `compare` and `reprice` apply the rule, and the inputs come from guarded loaders.
  - `summary` on fewer than 4 days fails with metrics' own "need at least 4 periods".
  - Targets are validated row by row (fast enough: `simulate` takes 0.1-0.5 s).
  - The NaN policy for 7b's 49 industries: `start=None` waits for `lookback` rows, not
    complete ones, and `simulate` rejects NaN even for zero-weight assets.
- Verification: 790 passed, 99.85% coverage; ruff, format and mypy clean; `uv build` and
  `uv lock --check` fine.
- Invariants 1-6 hold. Invariant 3's literal check (`git grep -il` of the personal-use data
  vendor's name outside `plans/`) had been failing since Step 3's NOTES entry, which named
  the vendor while saying it was absent. That line is reworded.

## 2026-10-08: step 4, session 3 of 4 (dev data: timing and sanity checks)

- The 12 strategies of Step 3's timing check go through `compare` on the dev views:
  - 1/N
  - minimum variance and ERC, each with the sample, LW identity, LW constant-correlation and
    EWMA estimators (252 d)
  - momentum and low-vol tilts (strength 1) on 1/N
  - momentum tilt on a 126-day LW-identity minimum variance
- The views are built from `load_*` and `make_views`, with assets from 1963-07-01. Asset and
  FF5 days match exactly: 11,959 days, no gaps.
  - USD: 558 trades, first 1964-07-01, last day 2010-12-31. Every strategy has the same first
    trade.
  - CAD: 468 trades from 1972-01-03.
  - Per decision 6, no strategy-vs-strategy performance table; the comparison waits for 7a's
    PLAN.md.
- **Exit met: under 30 s per view.**
  - USD 16.6-20.6 s and CAD 13.6-19.8 s over four runs each; this machine's timings vary
    with load.
  - Estimation dominates: `targets` for minimum variance and LW constant-correlation ERC
    takes 3-6 s each.
  - `simulate` takes 0.1-0.5 s per strategy.
  - Both views together take about 35 s, so a combined budget would need the SLSQP warm start
    (kept for 7a).
- Sanity checks, all passing:
  - **Monthly 1/N vs Step 2's daily-rebalanced 1/N** (same days):

    | | USD gross | USD net | USD daily | CAD gross | CAD net | CAD daily |
    |---|---|---|---|---|---|---|
    | CAGR | 10.58% | 10.56% | 10.49% | 11.03% | 11.01% | 10.95% |
    | Sharpe | 0.391 | 0.390 | 0.386 | 0.310 | 0.309 | 0.305 |

    The tracking error between monthly gross and daily is 0.17-0.18% a year.
  - **Costs:**
    - net ≤ gross on every day.
    - Net differs from gross on every trade day with nonzero turnover.
    - Minimum variance (sample, USD) had 53 of 557 trades with zero turnover: months held at
      100% in one asset, consistent with Step 3's Utilities finding.
    - The CAGR drag is about 1.1 x annual cost (cost x (1 + R), compounded).
  - **Turnover per month** (one-way, USD; CAD within 10%):

    | Strategy | Turnover / month | Turnover / year |
    |---|---|---|
    | 1/N | 1.1% | 0.13x |
    | ERC (sample, LW identity, LW constant correlation) | 1.5% | |
    | low-vol tilt | 3.2% | |
    | minimum variance (sample, LW identity, LW constant correlation) | 5.3-5.6% | 0.6-0.7x |
    | ERC (EWMA) | 6.4% | |
    | momentum tilt | 14-15% | 1.7-1.9x |
    | minimum variance (EWMA) | **28.7%** | 3.4x |

    For 7a: EWMA minimum variance costs 0.34% a year at 5 bp and about 1.4% at 20 bp.
  - **Traded weights** equal `strategy.weights(history through t)` exactly at the first,
    middle and last trade of every strategy in both views.
  - **Real-data reference loop** (USD, 11,706 days): the plain daily loop matches the engine to
    2.7e-16 per day, and wealth agrees to 8e-15, for 1/N and minimum variance.
  - **Real-data perturbation:** USD returns after 1990-06-15 were scaled by -1.3. The 312
    targets and every net return through the cut are identical, and later targets change.
  - **The CAD view estimates on CAD returns:** the CAD target equals `weights(CAD history)`, and
    it differs from the USD target on the same day by up to 0.061.
  - `summary(benchmark=market)` gives tracking error and IR. Final weights sum to 1 within
    2e-16.
- Mutation runs: 31 planted bugs over sessions 1-2, all caught (see those entries). The plan's
  "charge the cost before the return" is an equivalent mutant: a cost taken as a fraction of
  NAV multiplies the same either way. Trading at the open of t + 1 (the trade day earned by the
  new weights) was planted and caught.

## 2026-10-08: step 4, session 2 of 4 (strategies in the loop, guard, benchmarks)

- `backtest/engine.py` adds the following.
  - `targets(strategy, returns, *, start=None)`: at each estimation date t, the strategy gets
    `returns.iloc[: pos(t) + 1]`, every row through t and none after. The result is indexed by
    trade date ("traded").
    - Without `start`, the first t is the first month end with `lookback` rows. A `start` that
      would estimate earlier raises, naming the earliest date. Nothing is estimated before the
      check.
    - Rows are aligned by asset label, so a duck-typed strategy with wrong labels is caught by
      `simulate`.
  - `run(strategy, view, *, start, end=DEV_END, allow_holdout=False, cost_bps=5.0)`:
    - It checks `cost_bps` first, then drops rows after `end` by the loaders' rule.
      `data._resolve_end` is now the public `data.resolve_end`, one function for the holdout.
    - A non-`View` raises TypeError (the blueprint's old `returns` + `view="usd"` form).
  - `compare(strategies, view, ...)`: one common first rebalance, the first month end where
    the longest lookback is met (on the truncated view), and results keyed by name. It needs
    at least one strategy, with unique names.
  - `Result(Ledger)`: adds `name` and `rf`, the view's risk-free rate on the same days (checked
    aligned). `summary(benchmark=None)` gives:
    - `metrics.summary` of gross and net, one column each, daily (252)
    - `mean_turnover` per rebalance (NaN without a second trade), `annual_turnover` and
      `annual_cost` (sums per 252 days). These describe the trades, so both columns show them.
    - With a benchmark (cut to the result's days, which it must cover), tracking error and IR.
- `backtest/benchmarks.py`:
  - `market(factors)` = Mkt-RF + RF, named "Market".
  - `market_views(factors, usdcad, ca_tbill)` runs `make_views` on the one-column market
    frame, so the CAD and hedged markets use exactly the assets' conversion and calendar.
- No look-ahead, tested four ways:
  - A spy strategy sees every history end at its estimation date, with the trade the next row.
  - Hypothesis perturbation: new returns after day k change no target estimated by k and no
    gross, net or turnover through k. Run on a rule that reads its whole window (100 examples)
    and on minimum variance (25).
  - A +90% shock on every trade day leaves every target unchanged.
- Guard tests:
  - DEV_END is the last day by default, and the last estimate is 2010-11-30.
  - `end` past DEV_END, or `end=None`, raises `HoldoutError` before anything is estimated.
  - `allow_holdout` opens the later rows.
  - An earlier `end` truncates returns and rf.
- RED `a0397d6` (collection fails: no `Result`, `benchmarks`), GREEN `ffa1d01`. 752 passed,
  99.85% coverage, the backtest package 100%; ruff, format, mypy clean.
- Mutation run (16 planted bugs, all caught):
  - history through t + 1 (look-ahead), history to t - 1, lookback off by one
  - too-early start allowed, end day dropped, end ignored, holdout check bypassed
  - compare: shortest lookback, cost ignored, end ignored
  - start=None not lookback-aware, annualized by 365
  - benchmark ignored, benchmark gap not named, market without RF, cost checked late
- The full suite with coverage took 122-158 s on this machine (55 s once in session 1); the
  engine's three test files take about 25 s.
- For 7b's cost grid: `simulate(result.weights, returns, cost_bps=...)` re-prices the same
  targets, given the same truncated returns, without estimating again.

## 2026-10-08: step 4, session 1 of 4 (plan, calendar, accounting)

- Plan `plans/step-4-plan.md` approved as drafted. The six owner decisions:
  - `run` takes a step-2 `View`, monthly only.
  - The backtest starts invested, with no opening cost.
  - Start dates fail closed, and `compare` gives every strategy one common start.
  - The market benchmark is Mkt-RF + RF as an index.
  - Estimation (`targets`) is split from accounting (`simulate`); no SLSQP warm start.
  - Step 4 uses dev performance only for sanity checks.
- `backtest/engine.py`:
  - `rebalance_dates(days, *, start=None)` returns a Series from estimation date t (index
    "estimated") to trade date (the next day). The last day never estimates.
  - `simulate(targets, returns, *, cost_bps)` returns a `Ledger`: `gross`, `net`, `weights`,
    `turnover`, `cost` and `final_weights`.
  - Each holding period is priced from `w * cumprod(1 + r)`, and the daily return is
    holdings at the previous close times r, over their sum. A target summing to 1 ± 1e-9 is
    held as is.
  - Net is computed as `gross - c (1 + gross)`, the blueprint's (1 + R)(1 - c) - 1 rearranged
    so that zero cost gives gross exactly. Written as (1 + R)(1 - c) - 1, a 0.1 return came
    back as 0.10000000000000009.
  - Returns up to and including the first trade day are never read (the tests put NaN there).
  - A portfolio worth 0 raises; a single asset may lose 100%.
- `_validate.check_frame` is now shared (it was `_check_frame`).
- Tests:
  - Hand examples in exact fractions: the 3-day drift, turnover from drifted weights, t+1
    execution.
  - Zero cost = gross exactly; cost linear in bps.
  - A plain daily loop agrees to 1e-12 on random cases, with NaN before the first trade.
  - Buy and hold = sum w prod(1 + r).
  - RED `1763660` (fails at import), GREEN `d0d33c1`. 713 passed, 99.84% coverage; ruff, format,
    mypy clean.
- Mutation run (15 planted bugs, all caught): drift from end-of-day holdings, cost not scaled by
  1 + R, cost a day early, no drift, turnover not halved, trade date = estimation date, strict
  start, -100% rejected, wipeout missed, bps / 1e3, the new weights earning the trade day, final
  weights not drifted, net inexact at zero cost, month boundary shifted, first trade day's
  return read.
- Gotcha: `Path.write_text` on Windows writes CRLF. A script that rewrites a source file must
  restore it with `newline="\n"` (or `write_bytes`); otherwise git shows a whitespace-only change
  (`.gitattributes` is `eol=lf`).

## 2026-10-06: step 3, session 4 of 4 (review)

- python-reviewer: no critical or high findings. It re-derived the Ledoit-Wolf algebra, the ERC
  Newton steps and the KKT test, and ran ERC on 200 random problems (risk contributions equal to
  1e-6). `/code-review` (high) and the reviewer agreed on two real defects, both fixed test-first
  (RED bd7d0e4, GREEN a0d4d78, 19 new tests):
  - `tilt` returned all-NaN weights when `strength * z` overflowed (12 assets, strength 1e308:
    inf - inf). The z-scores are now shifted before scaling, so every exponent is <= 0 and a
    -inf exponent is a weight of 0.
  - Constructors accepted a cap outside (0, 1], a non-finite strength or a one-day window, and
    failed only at the first rebalance. They now check when the strategy is built.
  - Also fixed: names use every digit of their parameters (`:g` gave 0.1234567 and 0.1234568
    the same name, and names will key results); `ESTIMATORS` is read-only.
- Left as is, for the owner:
  - The z-score's "constant signal" threshold is relative (sd <= 1e-12 max|s|). Signals near 0
    with float noise can still tilt, but an absolute floor would break the scale invariance
    the tests pin (1e-20-sized signals tilt). Real 12-1 spreads are ~1e-2.
  - When polishing fails, `min_variance` returns SLSQP's clipped weights without a warning.
    There were 0 fallbacks in 5031 dev-data calls; recheck on 49 industries / 126 days (7b).
  - The window is validated several times per rebalance. That is minor next to SLSQP; revisit
    only if Step 4 needs speed.
- Verification: 663 passed, 99.82% coverage; ruff, format and mypy clean; `uv build` and
  `uv lock --check` fine; invariants 1-6 hold (no data files, no personal-use-data or
  sibling-repo references, no secrets, `allow_holdout=True` only in prose).
- Blueprint: Step 3 status, the interface superseded by `Strategy`, Step 4's `run(...)` without
  `window`, change log.

## 2026-10-05: step 3, session 3 of 4 (signals, tilts, strategies)

- `construct/weights.py` adds `momentum_12_1(returns)` (rows [-252, -21) of the history: close
  of t - 252 to close of t - 21, 231 returns; only the last 252 rows are read),
  `low_volatility(returns)` (minus the volatility, demeaned, divided by T) and
  `tilt(signal, *, strength, base=None)`: w_i proportional to b_i exp(strength z_i), with z the
  cross-sectional z-score. `strength` is required and keyword-only; the base defaults to 1/N.
- **Conventions to carry into 7a (the strength is pre-registered there):**
  - z uses the population sd across the N assets (divides by N). Centering z is cosmetic: it
    multiplies every weight by the same factor (the mutation run confirms it is equivalent).
  - Signals equal up to rounding (sd <= 1e-12 x max |s|) give the base, so float noise never
    tilts. A negative strength tilts away from the signal. An asset with base weight 0 stays
    at 0.
  - Exponents are shifted so the largest among the assets held is 0: no overflow at any
    strength (tested up to 1e6).
- `construct/strategy.py`: `Strategy(name, lookback, rule)`. `weights(history)` checks that the
  whole history is in date order and that the last `lookback` rows are complete; rows before
  them may have gaps and are never read (this matters for the 49 industries in 7b). It passes
  those rows to the rule and checks the result: a Series labeled by the history's columns, in
  order, finite, >= 0, summing to 1 within 1e-9. Constructors: `equal_weight()` (lookback 1),
  `min_variance(estimator, *, window=252, cap=1.0)`, `erc(estimator, *, window=252)`,
  `momentum_tilt(*, strength, base=None)` and `low_volatility_tilt(*, strength, window=252,
  base=None)`. A tilt's base is any strategy, and its lookback is max(signal, base): momentum
  on a 126-day minimum variance looks back 252 days (owner decision 1, now tested).
- **Choices made here (for the owner at review):**
  - Estimators are named by a key of `ESTIMATORS` (`sample`, `lw_identity`,
    `lw_constant_correlation`, `ewma`), not passed as functions. This keeps strategy names
    readable for 7a and the dashboard ("min variance (lw_identity, 252d)"), and it is still
    explicit with no default.
  - 1/N has lookback 1, so on its own it could start trading earlier than the others. The
    Step 4 / 7a comparison needs one common start date.
  - Windows are checked up front in every constructor: a tilt's lookback also covers its base,
    so a bad window would not show in the lookback.
- Shared checks in `_validate.py`: `last_rows` (the window of a history) and
  `portfolio_values` (a long-only portfolio over given assets), used by the tilt's base and by
  every strategy's output.
- Dev data (12 industries, USD, 559 month-end dates 1964-06 to 2010-12, 252-day windows;
  weights and estimator diagnostics only, never performance):
  - **Minimum variance is concentrated in Utilities.** Average weight 0.61 (sample matrix) and
    0.57 (LW identity). It is the largest weight in 73% of months, then NoDur (21%). The
    portfolio is 100% Utilities in 59 of 559 months with the sample matrix and 15 with LW
    identity. Median 3-4 assets held, effective N (1 / sum w^2) about 1.7-1.9. A 0.25 cap
    gives a median of 6 held and effective N 4.6. Capped variants are a 7a pre-registration
    choice (owner decision 2).
  - **Polish: 0 fallbacks in 5031 calls**, worst KKT gap 7e-16. An independent KKT check on
    the returned weights agrees (<= 7.2e-16 for every estimator).
  - ERC is close to 1/sigma: half the L1 distance has a median of 0.03-0.05 and a max of 0.09
    (EWMA 0.21). Risk contributions are equal to 9e-16. The largest ERC weight is 0.21 (EWMA
    0.31).
  - Correlation-matrix condition numbers, median / max: sample 138 / 756, LW identity 100 /
    256, LW constant correlation 87 / 195, EWMA 235 / 3521 (1987-10-30; EWMA's effective
    sample is about 32 days). All far from the 1e10 singularity threshold.
  - Tilts of 1/N, effective N (median [5%]): strength 0.5 gives 9.4 [7.7] for momentum and
    9.8 [8.0] for low vol; strength 1 gives 5.5 [2.6] and 6.3 [3.1], max weight up to 0.70;
    strength 2 gives about 2.5, close to a single asset. With N = 12, |z| <= sqrt(11), so
    strengths above 1 concentrate quickly. Context for 7a.
  - Time through `Strategy.weights` (the engine's path), per call: 1/N 0.3 ms, minimum
    variance 3.2-3.8 ms, ERC 1.0-1.7 ms, tilts of 1/N 1.2-1.3 ms, momentum on a 126-day
    minimum variance 5.3 ms. **12 strategies over 559 dates: 15.1 s.** Step 4's budget is
    30 s for everything. If 7a adds capped or 126-day variants, minimum variance dominates;
    warm-starting SLSQP from the previous weights is the first speed-up to try.
- Mutation run: 40 planted errors in the new code (momentum window ends, skip length,
  compounding, the -100% check; low-vol sign and divisor; z-score sd, threshold and centering;
  tilt shift, base and sign; every input check; the strategy lookbacks, window slicing,
  output check and estimator registry). 39 caught. The survivor (z-score without centering)
  is equivalent, as noted above.
- TDD evidence: RED `90ecdb7` (46 signal tests fail with AttributeError; the strategy tests
  fail at collection: `cannot import name 'strategy'`), GREEN `c639fe2` (164 new tests). One
  test tolerance went from 1e-16 to 1e-15 (a 1-ulp difference at 1/3). Suite: 644 passed,
  99.82% coverage; ruff, ruff format, mypy clean.

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
