# Step 8 plan: Streamlit dashboard + deploy

**Status: APPROVED by the owner on 2026-10-11, as written** (all nine decisions as
recommended, after the owner removed G1 the same day). Read it with the blueprint's Step 8
section. Branch `step-8-dashboard` (first commit `ae71557`: the blueprint marks 7b merged).
Default model, 3 sessions, cut-able.

**G1 removed by the owner on 2026-10-11.**
- The French data no longer have to be "derived analytics only", and raw files are no longer
  forbidden in the repo. Only the citation stays.
- Invariant 2 still keeps data files out of git as hygiene, except the paths it names (D3).
  Nothing in this step needs a raw file committed.
- The places that state G1 are updated in this step:
  - the blueprint: the header, the G1 row, invariant 2's note, Step 8's tasks, the risks, the
    acceptance criteria and the change log;
  - `DATA_SOURCES.md`;
  - the docstrings of `data/french.py` and `report/figures.py`;
  - one sentence of `research/REPORT.md` §3.

## Grounding 2026-10-11

**Hosting terms, checked today against the official docs:**
- **Streamlit Community Cloud** (free):
  - **An app takes its privacy from its repo.** In the docs' words, "if your repo is private
    your app will be private". The limit is "one private app at a time", seen only by people
    invited as viewers.
  - **A public demo link therefore needs a public repo.** That ties G3 to G4.
  - **Private repos need GitHub's `repo` OAuth scope.** Streamlit's own docs say it gives
    them "extra permissions that we do not really need".
  - **Sleep:** an app sleeps after 12 hours without traffic, and any visitor can wake it.
  - **Resources** (dated February 2024, may change): 0.078-2 CPU cores, 690 MB-2.7 GB RAM,
    50 GB storage.
  - **Dependencies:** it reads the first dependency file it finds, in the entrypoint's folder
    and then the repo root, in the order `uv.lock`, Pipfile, environment.yml,
    requirements.txt, then pyproject.toml (Poetry).
- **Render** (free web service), the alternative:
  - It deploys from a private repo and still gives a public URL.
  - It spins down after 15 minutes without traffic and takes "about one minute" to start.
  - 750 free hours a month; the disk is ephemeral; it can restart at any time.
  - The RAM is not on the free-tier page (not verified).
- **Hugging Face Spaces is out.** Gradio and Docker Spaces "require a paid plan to create";
  only static Spaces are free.

**The code today:**
- **No daily strategy series are saved.** The study's tables are summaries, and
  `robustness/monthly.csv` and `weights.csv` are monthly. They are also gitignored, so a
  deployed app cannot read them.
- **`tests/test_holdout_scan.py` scans every .py file outside `tests/` and `research/`,
  `app/` included.** So the app may not call a guarded function on holdout dates:
  - the guarded ones are the loaders, `engine.run`/`compare`/`reprice`, and
    `stress.historical.window`/`replay`/`table`;
  - `report.metrics`, `risk.var` and `risk.backtest` are pure, and the app may call them.
- **The net return at any cost is exact from gross returns and turnover** (engine
  docstring). After a trade the weights are exactly the target whatever the cost, so drift
  does not depend on the cost:
  - net = (1 + R)(1 − c) − 1, with c = cost_bps / 1e4 × 2 × turnover on trade days;
  - net = R on every other day.
  - **A cost slider therefore needs no backtest,** only the daily gross returns and turnover.
- **The versions:** Streamlit 1.65.0 (2026-10-02) depends on altair (≥5, <7) and pyarrow.
  `st.context.theme.type` tells the app whether it is drawing in light or dark.

## What gets built

- **`src/portfolio_lab/dashboard.py`** (new; pure pandas, no Streamlit):
  - the bundle's schema, `write`, and `read` (which validates it);
  - `net_at(gross, turnover, cost_bps)`;
  - date slicing and the metric tables.
  It is tested and inside the coverage like the rest of src/.
- **`cli.py`: `reproduce --dashboard`.**
  - It runs under the holdout's lock: the tag, PLAN.md, the frozen vintage, committed src/, and
    a recorded holdout.
  - It writes the bundle to `app/data/` and records `runs.dashboard` (table SHA-256s) in the
    manifest.
  - It refuses to write unless its 5 bp series reproduce the holdout's own tables (see D1).
- **`app/streamlit_app.py`** (thin UI) and **`app/charts.py`** (Altair charts).
  - **The sidebar:** strategies, dates, view (USD, CAD unhedged, CAD hedged), the cost
    slider and the VaR level.
  - **The tabs:** Performance, Risk, Factors, Stress and About.
- **`pyproject.toml`:** a `streamlit` dependency group `app`. CI already syncs
  `--all-groups`.
- **`tests/test_dashboard.py`** (the pure functions) and **`tests/test_app.py`** (an
  `AppTest` smoke test on a synthetic bundle, with no network).
- NOTES.md (the hosting research above), the blueprint's invariant 2 (D3), its status line and
  its change log.

## Decisions for the owner (recommendation first)

1. **D1. The data: a precomputed bundle of daily series, committed in `app/data/`, made by
   `reproduce --dashboard`.**
   - The app reads it at start-up, in well under a second. It then recomputes metrics, VaR
     and backtests live with the pure functions.
   - **The bundle:**
     - daily gross returns and turnover per strategy and view;
     - each view's risk-free rate;
     - the daily Fama-French market, the six factors, the USD/CAD returns and the Canadian
       10-year yield changes (D4, D5);
     - the monthly target weights;
     - copies of the holdout's small result tables (tests, stress, shocks).
   - **Why:** a fast cold load, no network, numbers identical to the REPORT, and no holdout
     flag in `app/`.
   - **Its built-in check:** at 5 bp the bundle's holdout months must give back the holdout's
     `performance.csv` metrics and the robustness monthly series (all three views) to 1e-12.
     Otherwise it writes nothing.
   - Alternative: the app downloads the raw data and backtests on start-up.
     - The cold load would be about 1-2 minutes.
     - `app/` would need the holdout flag (a scan exemption).
     - New data months would change its numbers away from the REPORT.
2. **D2. Which series: all 17 strategies (1/N and the 16 of PLAN.md §6), in all three
   views.**
   - The study ran the 17 in USD and CAD unhedged. The CAD-hedged runs of the 13
     non-primary strategies are new and descriptive.
   - The export check covers what has a recorded reference:
     - all 17 strategies in USD and CAD, against `performance.csv`;
     - the primary three in CAD hedged, against the robustness series.
   - **The sidebar shows at most 6 strategies at once,** so each gets its own palette color
     (D7). The default is the primary three.
   - Alternative: only the primary three. The bundle is smaller and the dashboard simpler.
3. **D3. Invariant 2 is amended to allow `app/data/` and to catch `.gz`.**
   - Today's check misses `.csv.gz`.
   - The new check: `git ls-files | grep -Ei '\.(csv|zip|parquet|xlsx|json|gz)$' | grep -v
     -e '^tests/fixtures/' -e '^research/results/manifest.json$' -e '^app/data/'` is empty.
   - The bundle is gzip CSV written with `mtime=0`, so the same data give the same bytes.
     With 17 strategies it should be about 9 MB at full precision: about 3 MB per view
     (measured on synthetic data of the same shape), one file per view, plus the factors.
   - Alternative: parquet. It is about a third of the size, but it needs pyarrow in the core
     package, which writes the bundle.
4. **D4. Tracking error and information ratio have a benchmark selector,** defaulting to the
   Fama-French market.
   - The market default matches the REPORT's `performance.csv`, so the dashboard can be
     checked against it.
   - The other choice is 1/N, the study's own benchmark.
5. **D5. What changes live, and what is fixed:**
   - **Live** (cost, dates, view, strategies, VaR level and method):
     - Performance: growth of $1, drawdown, and the metrics table, tracking error and
       information ratio included.
     - Risk: rolling VaR/ES with a 500-day window, exceptions, Kupiec and Christoffersen.
     - Factors: the betas over the selected dates, with `risk.factors` (pure functions),
       weekly and partial as in Step 5b. Also rolling 252-day betas, and the FX and rate
       betas in the CAD views.
   - **Fixed:**
     - Stress: the holdout's scenario replays and the rate and FX shocks, at the
       HOLDOUT_END weights as registered. Replaying needs `stress.historical`, a guarded
       function.
     - The realized path of each selected strategy through each scenario is drawn live from
       its own returns.
6. **D6. The registered tests stay fixed. Everything live is labelled descriptive.**
   - Performance shows the four primary tests from the holdout's `tests.csv`, as registered.
   - No p-value is recomputed for another cost or date range: each bootstrap takes minutes,
     and those numbers would read as new tests.
   - **The dates:** they open on the holdout (2011-01 to 2026-08). They can go back to the
     first rebalance (1964 for USD, 1972 for CAD), and the earlier years are marked as
     the design period.
7. **D7. The charts use Altair, in the palette from `report.figures.THEMES`.**
   - Altair ships with Streamlit, so there is no new dependency. That palette was already
     validated in both modes.
   - The palette follows `st.context.theme.type`. The charts use the full container width
     and work at phone width.
   - The palette has 7 validated slots: 6 for the selected strategies (D2's cap) and 1 for
     the benchmark's reference line. No two lines share a color.
   - Alternative: Plotly. Better hover, but one more dependency of about 15 MB.
8. **D8. The layout: the logic lives in `src/portfolio_lab/dashboard.py`; `app/` is a thin
   UI.**
   - The logic is then tested, type-checked and inside the 80% coverage gate.
   - Streamlit goes in an `app` dependency group, as the blueprint says, so the core install
     stays lean.
9. **D9. Hosting (G3, G4): build and verify locally in Step 8. Deploy on Streamlit
   Community Cloud in Step 9, right after the scrub, with G3 and G4 asked together then.**
   - A free public link needs a public repo (grounding).
   - Step 9 already scrubs the repo before it goes public. Deploying earlier would mean the
     scrub, or a second host, now.
   - You'll still see the dashboard in this step: I run it locally in the app's browser pane
     at the end of session 1.
   - Alternatives:
     - **(b) Render now, from the private repo.** A public link today, but you create a
       Render account. Each visit after 15 idle minutes waits about a minute, against the
       10-second target, and the RAM is unverified.
     - **(c) The Step 9 scrub now,** then G3 and G4 at the end of Step 8.

## Sessions

1. **Session 1:**
   - `dashboard.py` and its tests, written first.
   - `reproduce --dashboard`: the bundle, with the D1 check.
   - Invariant 2.
   - The app's skeleton with the Performance tab, run locally so you can see it.
2. **Session 2:**
   - The Risk, Factors, Stress and About tabs.
   - Theming and phone width.
   - The `AppTest` smoke test, and the cold-load time.
3. **Session 3:**
   - python-reviewer, `/code-review`, the verification loop, and the invariants.
   - NOTES and the blueprint.
   - The PR (asking first).
   - Deployment, if D9 (b) or (c) is chosen.

## Tests first (examples)

- **`net_at`:**
  - It equals `engine.reprice`'s net, to 1e-15, on a synthetic view at 0, 5 and 20 bp.
  - It is the gross return when no trade happens.
  - A hand-computed day: R = 1%, turnover 0.3, 10 bp gives (1.01)(1 − 0.0006) − 1.
- **`read`:**
  - It refuses a bundle with a column that is not in the schema, a strategy outside the
    registry, a missing view, or non-finite values.
  - It returns read-only frames.
- **The export check:** it raises when one gross value is perturbed by 1e-9 (on a synthetic
  study), and passes on the unperturbed one.
- **`reproduce --dashboard`:**
  - It is refused before a recorded holdout.
  - It is refused with `--refresh`, uncommitted src/, a drifted vintage, or `--out`.
  - It records `runs.dashboard` with SHA-256s.
- **The app:**
  - With `AppTest` on a synthetic bundle (`PORTFOLIO_LAB_APP_DATA` points to a temp folder),
    every tab renders without an exception.
  - The cost slider changes the CAGR shown.
  - A seventh strategy cannot be selected.

## Files to change

| File | Change |
|---|---|
| `src/portfolio_lab/dashboard.py` | new |
| `src/portfolio_lab/cli.py` | `reproduce --dashboard` |
| `src/portfolio_lab/manifest.py` | the run record if needed (`tables` already exists) |
| `app/streamlit_app.py`, `app/charts.py` | new |
| `app/data/*` | the bundle (committed, D3) |
| `tests/test_dashboard.py`, `tests/test_app.py`, `tests/test_reproduce.py` | new and extended |
| `pyproject.toml`, `uv.lock` | the `app` group (streamlit) |
| `plans/portfolio-lab-blueprint.md` | G1 removed; invariant 2, the status line, the change log |
| `DATA_SOURCES.md`, `data/french.py`, `report/figures.py`, `research/REPORT.md` | the G1 wording only |
| `NOTES.md` | the hosting research, G1's removal, and the session log |

`study.py`, `robustness.py` and `research/PLAN.md` stay unchanged.

## Risks

| Risk | Mitigation |
|---|---|
| Streamlit 1.65.0 is 9 days old | Pin `>=1.6x,<2` and fall back one minor version on a regression |
| Community Cloud may not install the src-layout package from `uv.lock` | Checked at deploy time. Fallback: `app/requirements.txt` with the package path, which takes precedence in the entrypoint's folder |
| Live VaR is slow over 15,900 days × 4 methods | Compute only the selected series, method and level; `st.cache_data` |
| The bundle grows the git history with each re-export | Re-export only when the data change; about 9 MB a time |
| Live metrics are read as new tests | D6: fixed registered tests, "descriptive" labels, no live p-values |

**Complexity:** medium. About 300-400 lines in src/ and 300-400 in app/, plus tests.

## Acceptance (from the blueprint)

- `uv run --group app streamlit run app/streamlit_app.py` shows every tab.
- An `AppTest` smoke test passes on fixture data with no network, in CI.
- Each tab is checked in a browser, at desktop and phone width, in light and dark. The
  cold-load time is noted.
- The French data are cited in the About tab and in DATA_SOURCES.md (G1 removed; the
  citation stays).
- Invariants 1-6 hold, with invariant 2 as amended (D3).
