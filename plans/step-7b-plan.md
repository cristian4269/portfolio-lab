# Step 7b plan: holdout run, robustness, report

**Status: APPROVED by the owner on 2026-10-11, as written** (all nine decisions as
recommended, including the one-shot holdout run right after this commit). Read it with the
blueprint's Step 7b section and `research/PLAN.md`, which is FROZEN at the tag `prereg-v1`
(→ `317d4c4`, pushed). Branch `step-7b-holdout`. Strongest model, 3 sessions, MVP.

**Approving this plan also approves running the one-shot holdout** (order step 3, below)
right after this file is committed. No holdout number exists before that run.

## Grounding 2026-10-11 (dates and file metadata only, no returns)

- `git diff prereg-v1 HEAD` touches only `plans/portfolio-lab-blueprint.md`. src/ is exactly
  the tagged code, so the holdout run will record `changed_since_tag = []`.
- **The 49 industries on dev** (`load_assets(49)` through DEV_END):
  - Every industry exists from 1926-07, except Softw (from 1965-07-01) and Hlth (from
    1969-07-01).
  - After 1969-07-01 exactly one cell is missing: Softw on 1971-03-11. Every other hole is
    before 1946.
- **The cached 49-industry file is not in the frozen vintage.** `manifest.SOURCES` holds 7
  files, and `ind49_daily` is not one of them. Its cache record, fixed here before any
  holdout number exists:
  - retrieved 2026-10-10T01:24:39Z (the same refresh as the frozen vintage)
  - header "This file was created using the 202608 CRSP database."
  - **SHA-256 `8f394fe34bea54d41b9aafed410425ee8f8e252ede3c71a7c1cd20bab83040de`**,
    4,173,817 bytes
- `manifest.merge` replaces a period's run record, so a `--holdout --rerun` today would
  overwrite the one-shot's record (time, commit) in the working manifest.
- No plotting library is installed. The dependencies are numpy, pandas, requests, scipy and
  statsmodels.
- The holdout stress scenarios (COVID and rates) and both shock spans are already in
  `study.run(period="holdout")`, so the blueprint's "2020 and 2022 stress on HOLDOUT_END
  weights" comes from the one-shot run itself.

## The order

1. The owner approves this plan.
2. Commit this plan, so the tree is clean.
3. **Run the holdout once:** `uv run portfolio-lab reproduce --holdout`.
   - It runs on the tagged code, with no 7b code written yet.
   - The lock checks the tag, PLAN.md, the frozen vintage, a committed src/ and no earlier
     run.
   - It takes about 8 minutes (4 calibration processes) and writes
     `research/results/holdout/*.csv` (13 tables) and `runs.holdout` in the manifest.
4. Log the run in NOTES.md: the time, commit, duration, any warnings, and the four primary
   results read by PLAN.md §3's rules. Commit the manifest with NOTES. The owner gets the
   primary results.
5. Only then write the 7b code, test-first: robustness, then figures.
6. REPORT.md, the reviews, verification, the blueprint status, then the PR (ask first).

## What gets built

- **`src/portfolio_lab/robustness.py`** (new; `study.py` stays as tagged):
  - the registered robustness of PLAN.md §6, for the primary pair only, with no new
    hypotheses
  - the series that the figures need
- **`cli.py`:**
  - `reproduce --robustness`: the same lock as the holdout, plus a recorded holdout run.
  - `portfolio-lab figures`: reads research/results and writes research/figures/*.png.
  - A holdout `--rerun` gets its own record (D8).
- **`report/figures.py`** (new): pure functions from result tables to matplotlib figures.
- **`pyproject.toml`:** matplotlib in a new optional extra `report`, and in the dev group
  (tests, mypy, CI).
- **`research/REPORT.md`** (4-8 pages) and **`research/figures/*.png`** (committed).
- NOTES.md, the blueprint's status line and change log.

## Decisions for the owner (recommendation first)

1. **The one-shot holdout runs first, before any 7b code.**
   - The holdout's numbers then come from exactly the tagged code (`changed_since_tag`
     empty), and nothing written for 7b can steer them.
   - Alternative: write the robustness code first and run everything together. One fewer
     context switch, but the holdout's code would differ from the tag, and the manifest
     would say so.
2. **Robustness lives in a new module, `robustness.py`, run by `reproduce --robustness`.**
   - `study.py`, the registered procedure, stays byte-identical to the tag.
   - The mode refuses to run before the one-shot holdout is recorded, so robustness can never
     come first.
   - It uses the holdout's lock: tag, PLAN.md, frozen vintage, committed src/.
   - It writes `research/results/robustness/*.csv`, recorded as `runs.robustness` with its
     own `changed_since_tag`. That list will name the new 7b files, which is honest.
   - It reads the calibrated blocks from `research/results/holdout/tests.csv`. It recomputes
     the primary rows and refuses to go on unless they equal that file's (a stale or edited
     CSV fails).
   - Alternatives:
     - `study.robustness`: one module, but it edits the tagged file.
     - A `research/` script: outside the package, the lock and the coverage.
3. **What robustness reports: one change at a time from the primary configuration, for
   minimum variance and ERC (lw_identity) against 1/N, in USD and CAD unhedged.**
   - **The variants:**
     - windows of 126 and 504 days: `compare([1/N, min variance (lw_identity, Wd), ERC
       (lw_identity, Wd)])`
     - costs of 0, 10 and 20 bp: `engine.reprice` of the primary results, 1/N included (the
       same trades)
     - the 49 industries (D5)
     - the CAD-hedged view (D4, metrics only)
   - **Per row:** the Sharpe difference Δ̂ (monthly, × √12), the LW standard error and 95%
     CI, and the raw LW p-value.
     - The p-value uses the holdout's calibrated block for the same view and strategy.
     - M = 4999, with seeds `rng_for("robustness", variant, view, strategy, "test", b)`.
   - **Also per row:** both Sharpe ratios, the volatility, max drawdown, CVaR 95%, CEQ,
     annual turnover and cost.
   - **Labelled descriptive.** No Holm, no rejection column, no claim, as PLAN.md §6 says.
   - **A `primary` row** recomputes the registered test with the holdout's own seed labels.
     It must equal `tests.csv` exactly: a built-in check that robustness and the holdout
     agree.
   - Alternative: recalibrate the block for every variant. That is 24 more calibrations
     (about 12 minutes with 4 processes), and each variant then changes two things at once.
4. **The CAD-hedged view: the strategies run on the `CAD_HEDGED` view itself.**
   - This applies the fixed design's rule: each view estimates in its own returns.
   - It reports total-return metrics (CAGR, volatility, max drawdown, CVaR, growth of $1) for
     1/N, minimum variance and ERC. It has no test, since its excess return is the USD one
     (PLAN.md §2).
   - Alternative: the USD view's targets simulated on hedged returns, an overlay on the USD
     portfolio. The two differ by little, because the hedge adds a nearly constant daily
     term, the same for every asset.
5. **The 49-industry NaN policy: all 49 industries, starting the day after the last missing
   cell before the holdout.**
   - On dev that is 1971-03-12. The 252-day lookback then puts the first rebalance about
     1972-03, and the evaluation is the same 188 holdout months.
   - **If an industry has a missing holdout day** (the code checks it; nobody looks now), that
     industry is dropped from the whole run and the table names it.
   - **The input is locked:** the code refuses a cached 49-industry file whose SHA-256 is not
     the one under Grounding, unless `--accept-drift`. A test checks that the hash in the
     code is the one in this file.
   - Alternatives:
     - Start 1969-07 and drop Softw (48 industries).
     - Cut the 49-industry check, the blueprint's first cut-able item.
6. **Figures: matplotlib in an optional `report` extra, `report/figures.py` and
   `portfolio-lab figures`.**
   - It reads the CSVs only, so it never opens data, and writes PNGs at 150 dpi, without a
     timestamp in their metadata.
   - **Five figures,** all derived analytics (G1: no raw industry series):
     1. Growth of $1 over the holdout, net, for 1/N, minimum variance and ERC (USD and CAD
        panels).
     2. Their drawdowns.
     3. The Sharpe differences with 95% CIs: the four primary tests, then each robustness
        variant (a forest plot).
     4. Minimum variance's month-end weights over the holdout (USD; stacked area).
     5. Stress: the scenario replays at HOLDOUT_END holdings, next to the realized paths.
   - The robustness mode writes the two tables the figures need beyond the holdout's:
     - `monthly`: the monthly net and excess returns of the three portfolios per view
     - `weights`: the pair's month-end targets
   - Alternatives:
     - A `research/figures.py` script run with `uv run --with matplotlib`: lighter, but
       untested.
     - Plotly: interactive, but the report is static markdown.
7. **REPORT.md (4-8 pages), written in session 3 from the CSVs.**
   - **The outline:**
     1. The answer in five lines
     2. Question and pre-registration timeline (draft, freeze, tag, holdout commits)
     3. Data
     4. Method
     5. Primary results with CIs and Holm
     6. Performance and risk
     7. Exploratory results and the luck benchmark, with its stated meaning
     8. VaR backtests and stress
     9. Robustness
     10. What failed or changed (the kernel amendment, the luck null, the review fixes)
     11. Limitations (power, untradable industry portfolios, the cost proxy, known history,
         one vintage)
     12. Next steps
     13. How to reproduce
   - The tables' numbers come from a scratch script that prints them from the CSVs, rounded.
     The prose is written by hand.
8. **Verification of reproduction:**
   - **The rerun:**
     - `reproduce --holdout --rerun`, after all 7b code, on the same vintage. Every number of
       every holdout CSV must match the first run's saved copy to 1e-6 (0 is expected: the
       seeds are fixed).
     - Then spot-check at least 5 REPORT numbers against the rerun.
     - **A rerun is recorded as `runs.holdout_rerun`, never over the one-shot's record** (a
       small `cli`/`manifest` change, tested).
     - Alternative: restore the committed manifest from git after the rerun, and log it in
       NOTES only.
   - **Drift:**
     - A dev run against a fresh download of every source, in a separate cache folder
       (`PORTFOLIO_LAB_CACHE=<scratchpad>`), with a copy of the manifest in
       `--out <scratchpad>`.
     - It must warn and name the changed files. If French revised history, NOTES records how
       far the dev numbers moved.
     - It never touches the real cache or research/results.
   - **History:** `git log` shows the `prereg-v1` commit before the commit that adds
     `runs.holdout`.
9. **The post-registration addendum (PLAN.md §12) is deferred out of 7b.**
   - September 2026 would add a month or two: too short to say anything, even
     descriptively.
   - Refreshing the shared cache now would also replace the frozen vintage that the 1e-6
     check needs.
   - Later, it runs from a separate cache folder, reported apart.
   - Alternative: build the mode now (half a session).

## Sessions

- **Session 1 (today):**
  - Approval, plan commit, the holdout run, NOTES and manifest commit, results to the owner.
  - Then `robustness.py` test-first on synthetic data (RED, then GREEN): the variants, the
    `primary` cross-check, the 49-industry start rule and drop rule, the hedged run, and the
    `monthly` and `weights` tables.
- **Session 2:**
  - `reproduce --robustness` and its lock (recorded holdout, the 49-industry hash), then run
    it.
  - `runs.holdout_rerun`.
  - `report/figures.py`, `portfolio-lab figures`, and the PNGs.
- **Session 3:**
  - REPORT.md.
  - python-reviewer, `/code-review` (high), mutation runs on the new code, the
    verification-loop.
  - The rerun and drift checks, the invariants, the blueprint status and change log.
  - The PR, after asking.

## Tests first (examples)

- **Robustness:**
  - The `primary` row equals a hand call of `inf.lw_test` with the holdout's labels.
  - Each window variant's strategy names end in `126d`/`504d`.
  - A 0 bp reprice has net equal to gross.
  - A 49-industry frame with a hole: before the holdout it moves the start; inside the
    holdout it drops that column and names it.
  - The hedged view's monthly excess equals the USD one when the weights are the same.
  - `monthly` and `weights` cover exactly the holdout months.
- **The CLI:**
  - `--robustness` refuses with no recorded holdout, refuses a 49-industry hash mismatch
    (and runs with `--accept-drift`, warning), and keeps the holdout lock.
  - `--rerun` writes `holdout_rerun` and leaves `runs.holdout` byte-identical.
  - `figures` without matplotlib is a clean error.
- **Figures:** each function returns a figure from a synthetic table, and the command writes
  five PNGs to a temporary folder.
- **The holdout scan:** `allow_holdout=True` still appears only in cli.py. `robustness.py`
  takes the flag as a parameter, like `study.py`.

## Files to change

| File | Action | Why |
|---|---|---|
| `plans/step-7b-plan.md` | CREATE | This plan, committed before the holdout run |
| `research/results/manifest.json` | UPDATE (by the run) | `runs.holdout`, then `runs.robustness` and `runs.holdout_rerun` |
| `NOTES.md` | UPDATE | The holdout log, each session |
| `src/portfolio_lab/robustness.py` | CREATE | D2-D5 |
| `src/portfolio_lab/report/figures.py` | CREATE | D6 |
| `src/portfolio_lab/cli.py` | UPDATE | `--robustness`, `figures`, the rerun record |
| `src/portfolio_lab/manifest.py` | UPDATE | The rerun key; the 49-industry record in the robustness run only |
| `pyproject.toml`, `uv.lock` | UPDATE | The `report` extra, matplotlib in dev |
| `tests/test_robustness.py`, `tests/test_figures.py`, `tests/test_reproduce.py` | CREATE / UPDATE | Tests first |
| `research/REPORT.md`, `research/figures/*.png` | CREATE | The report |
| `plans/portfolio-lab-blueprint.md` | UPDATE | Status line and change log at the end |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| The holdout run fails partway | Low | It computes every table before writing any. On a failure before the write, fix it in a commit (which `changed_since_tag` then records) and log it in NOTES. |
| The results are unwelcome (a strategy loses to 1/N) | Medium | PLAN.md §3's reading rules are fixed. REPORT states the result as it is, and a null is a valid finding (§8). |
| 7b code changes a holdout number | Low | `study.py` untouched, the `primary` cross-check, and the 1e-6 rerun |
| A missing 49-industry day in the holdout | Low | The drop rule of D5, fixed now |
| matplotlib changes the PNG bytes between versions | Medium | Figures are regenerated, never compared byte for byte; the CSVs carry the numbers |
| The drift download hits a rate limit (StatCan WDS) | Low | One download per file, into the separate cache |

**Complexity:** high in method, medium in code (about 400-500 lines of src plus tests).

## Acceptance (from the blueprint)

- [ ] The holdout was run once, to HOLDOUT_END, logged in NOTES.md, and its manifest record
  committed after `prereg-v1`.
- [ ] The 2020 and 2022 stress scenarios on HOLDOUT_END weights are reported.
- [ ] Robustness: costs 0/5/10/20 bp, windows 126/252/504, the hedged CAD view, the 49
  industries (or cut, with the owner's yes).
- [ ] `research/REPORT.md`, 4-8 pages, with CIs, robustness, what failed, limitations and next
  steps; figures in `research/figures/`.
- [ ] The rerun matches to 1e-6 (5 REPORT numbers spot-checked); a newer vintage warns and
  reports drift; the history shows `prereg-v1` before the holdout results.
- [ ] Invariants 1-6 hold; coverage stays at or above 80% (it is at 99.94%).
