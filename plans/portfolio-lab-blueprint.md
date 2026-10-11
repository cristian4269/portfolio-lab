# Blueprint: Risk-aware portfolio lab

Status: **APPROVED v2 (2026-10-01)** · Created 2026-10-01 · Owner: cristian4269
Decided: G1 = fetch French data at runtime, cite, derived analytics only · G5 = MIT.
Mode: git + gh available → one branch per step; PRs once the GitHub repo exists (gate G2).

## How to use this file (cold start)

Every session executes **one step**. Before starting, read these sections in
full: **Owner decisions**, **Fixed research design**, **Invariants**,
**Per-step workflow**, then **your step**. Other steps' sections are optional
background. Update the step's status line and the change log when done.

## Objective

A public Python library + live dashboard that builds portfolios, backtests them
honestly, and measures their risk, for a HEC Montréal finance student targeting
asset management, quant, and risk internships. A recruiter should get it in 30
seconds (README top + demo link); a quant interviewer should be able to probe
it for 30 minutes (tested math, documented method, pre-registered research).

Framing: **a replication and extension of DeMiguel, Garlappi & Uppal (2009),
"Optimal versus naive diversification", for a Canadian investor**, with
risk-model validation on top. This reads as research, not coursework.

## Owner decisions

| Gate | Decision | Default proposed | When |
|---|---|---|---|
| **G1** | Kenneth French Data Library shows a copyright notice but no explicit reuse terms. Use it by fetching at runtime, never committing raw files, citing it, and publishing only derived analytics (no per-industry raw series displays or downloads)? Alternative: email Prof. French for permission first. | Fetch at runtime + cite; derived results only | Before step 1 |
| **G2** | Create the GitHub repo `cristian4269/portfolio-lab` so CI runs early. | **Private** at end of step 0 | End of step 0 |
| **Push** | Every `git push` (branches, tags) needs the owner's yes **in that session**. A yes given in an earlier session does not carry over. | Ask each time | Every push |
| **G3** | Deploy the dashboard. If the chosen free host requires a public repo, G3 and G4 are asked together. | Ask in step 8 | Step 8 |
| **G4** | Make the repo public (after the step-9 scrub). | Ask in step 9 | Step 9 |
| G5 | Code license. | MIT | Step 0 |

Claude never creates the remote, pushes, deploys, or changes visibility without
the owner's explicit yes for that action in that session.

## Fixed research design (set now so development can't leak into it)

- **Dev period:** USD view from 1963-07 (FF5 start); CAD view from 1971-01
  (daily USD/CAD start). **Dev ends 2010-12-31 (`DEV_END`). Holdout starts
  2011-01-01.** `HOLDOUT_END` is frozen in step 7a as the last full month of
  the data vintage used.
- **Guard in the data layer, not just the engine:** `load_assets`,
  `load_factors`, `load_fx`, `load_rates` take `end=DEV_END,
  allow_holdout=False` and truncate by default. A test fails if the text
  `allow_holdout=True` appears anywhere except `src/portfolio_lab/cli.py`
  (the `reproduce` command) and `research/`.
- Steps 3-6 build and test on synthetic data and dev data only. Stress
  scenarios in the holdout (March 2020, 2022) are registered in step 6 and
  first executed in step 7b.
- **Prior knowledge** (to declare in PLAN.md): 1/N being hard to beat
  (DeMiguel et al. 2009) and the low-volatility anomaly are well known to the
  owner and to Claude. The holdout guards against data snooping, not against
  this prior.
- **Universe:** Fama-French **12 industry portfolios**, daily, value-weighted,
  as the investable assets; 49 industries as a robustness check. Factors:
  FF5 + momentum, daily.
- **Currency of optimization:** USD-view strategies estimate on USD returns;
  CAD-view strategies estimate on CAD (unhedged) returns. Pre-registered.
- **Execution:** estimate through close t, trade at close t+1 (no same-close
  look-ahead). Costs default 5 bp per side, 0-20 bp sensitivity (industry
  portfolios aren't directly tradable; the cost is a proxy for ETF or basket trading).

## Invariants (checked at the end of every step)

1. `uv run pytest` passes (network tests excluded by default); `uv run ruff check .` and `uv run ruff format --check .` clean; coverage ≥ 80% on `src/` from step 2 on.
2. No data committed except synthetic fixtures and allowed result files:
   `git ls-files | grep -Ei '\.(csv|zip|parquet|xlsx|json)$' | grep -v -e '^tests/fixtures/' -e '^research/results/manifest.json$'` is empty.
   `research/results/` is gitignored except `manifest.json`; figures (PNG) under `research/figures/` are committed (derived analytics, per G1).
3. No personal-use data or sibling-repo material: `git grep -il tiingo -- ':!plans/'` is empty; nothing copied from `../stock-prices/` data, cache or `.env`.
4. No secrets: all planned sources are keyless. If a key is ever needed it lives in `.env` (gitignored).
5. Holdout rule above holds until step 7b.
6. Every financial formula has a test against a hand-computed or textbook value.

## Dependency graph (serial)

```
0 → 1 → 2 → 3 → 4 → 5a → 5b → 6 → 7a → 7b → 8 → 9
```
Serial on purpose: steps share `uv.lock` and NOTES.md, step 5a's filtered
historical simulation uses step 4 return series, and 6 needs 5b's macro betas.

| Step | Name | Model tier | Sessions | MVP? |
|---|---|---|---|---|
| 0 | Scaffold, tooling, CI | default | 1 | yes |
| 1 | Data sources, licenses, loaders | default | 3 | yes |
| 2 | Returns, CAD/hedged views, metrics | default | 2 | yes |
| 3 | Covariance + portfolio construction | **strongest** | 4 | yes |
| 4 | Walk-forward backtest engine | **strongest** | 4 | yes |
| 5a | VaR/ES + VaR backtesting | **strongest** | 3 | yes |
| 5b | Factor and macro regressions | default | 2 | cut-able |
| 6 | Stress testing | default | 2 | cut-able |
| 7a | Pre-registration, dev analysis, luck benchmark, reproduce CLI | **strongest** | 3 | yes |
| 7b | Holdout run, robustness, REPORT | **strongest** | 3 | yes |
| 8 | Streamlit dashboard + deploy | default | 3 | cut-able |
| 9 | README, docs, final audit, publish | default | 2 | yes |

Total ≈ 32 sessions of ~3-4 h. At 4-5 sessions a week that is **6-8 weeks**.
**Minimum viable project** (0-4, 5a, 7a, 7b, 9) ≈ 25 sessions ≈ 5-6 weeks.
If time runs short, cut in this order: EWMA covariance (step 3), hypothetical
stresses (step 6), 49-industry robustness (7b), dashboard (8: ship a static
HTML report instead), then 5b and 6 entirely.

## Per-step workflow (every step)

1. `/resume-session` (if not the first). Read the sections listed in "How to use this file".
2. `git switch main && git pull` (pull only once a remote exists) then `git switch -c step-N-<slug>`.
3. `/plan` the step and **wait for the owner to confirm the plan**.
4. Tests first (tdd-workflow skill) for every formula, then implement.
5. python-reviewer agent → `/code-review` → verification-loop skill. Fix findings.
6. Check all invariants. Update this file: step status line + change log.
7. Before G2: `git switch main && git merge --no-ff step-N-<slug>`.
   After G2: **ask the owner**, then push the branch and open a PR with `gh pr create`; merge after CI is green.
8. `/save-session`.

Rollback: abandon an unmerged step with `git switch main && git branch -D step-N-<slug>`;
revert a merged step with `git revert -m 1 <merge-commit>`.

---

## Step 0: Scaffold, tooling, CI
Status: **done 2026-10-01** (commit 967e8db; private repo created, first CI run green)

**Context.** Repo at `C:\Users\crist\Documents\portfolio-lab` (only `plans/` exists).
uv 0.12.20 and CPython 3.13.15 are installed; gh is logged in as cristian4269.

**Tasks**
- `git init -b main`; `uv init --package --name portfolio-lab --python 3.13` (src layout `src/portfolio_lab/`).
- Deps: pandas, numpy, scipy, requests, statsmodels (HAC standard errors). Dev group: pytest, pytest-cov, ruff, hypothesis. Optional group `app`: streamlit (added in step 8).
- Packages: `data`, `construct`, `backtest`, `risk`, `stress`, `report` (each with `__init__.py`); `config.py` with `DEV_END = "2010-12-31"`, `HOLDOUT_START = "2011-01-01"`, cache dir `~/.cache/portfolio-lab` (env var override `PORTFOLIO_LAB_CACHE`).
- `cli.py` with `main()` registered as `[project.scripts] portfolio-lab = "portfolio_lab.cli:main"` (stub subcommands).
- `pyproject.toml`: ruff (line length 100); pytest `addopts = "-m 'not network' --strict-markers"`, markers `network`; coverage `source = ["src/portfolio_lab"]`, `fail_under = 80` (enforced from step 2).
- `.gitignore`: `.venv/`, `__pycache__/`, `.env`, `data/`, `.cache/`, `*.parquet`, `.coverage`, `htmlcov/`, `research/results/*` with `!research/results/manifest.json`, `docs/career/`.
- `.github/workflows/ci.yml`: push + PR → `astral-sh/setup-uv` → `uv sync --locked --all-groups` → ruff check + format check → `uv run pytest --cov`.
- `LICENSE` (MIT), `README.md` stub, `NOTES.md` (running log), `DATA_SOURCES.md` stub; smoke test `import portfolio_lab`.
- **Commit `uv.lock`** (CI uses `--locked`).

**Verify:** `uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run portfolio-lab --help`
**Exit:** all pass; first commit on `main`; ask the owner **G2**. If yes: `gh repo create cristian4269/portfolio-lab --private --source . ` and, after a second yes for the push, push `main` and confirm CI is green.

---

## Step 1: Data sources, licenses, loaders
Status: **done and merged 2026-10-02** (PR #1, merge commit 91e6245; 3 sessions; 176 unit + 10 live tests, 99.5% coverage; python-reviewer and `/code-review` findings fixed, incl. a holdout bypass via NaT)

**Context.** Pre-flight (2026-10-01): French library = copyright notice, no
explicit reuse terms (handled by G1); daily FF5, momentum and 5-49 industry
files exist. Bank of Canada = reuse allowed with attribution and noting
changes; don't circumvent Valet limits. FRED = per-series: public domain (cite
"Source: X via FRED"), citation-required, or pre-approval-required (personal
use only → **not allowed**). SEC EDGAR: considered, not needed for this scope.

**Tasks**
- search-first skill: re-check each terms page; record URL, a quoted key sentence, retrieval date, citation and verdict in `DATA_SOURCES.md`.
- Pick series and check each one's FRED copyright line (shown under each graph):
  - French: FF5 (2x3) daily, momentum daily, 12 and 49 industry portfolios daily (value-weighted).
  - USD/CAD daily, CAD per USD: FRED `DEXCAUS` (H.10, noon New York rates) for history; BoC Valet `FXUSDCAD` (daily average since 2017) only if needed. Document any splice and the timing mismatch with US closes.
  - Canadian short rate: daily if Valet has it back to 1971; otherwise the monthly series forward-filled, documented. Canadian 10-year yield (for step 5b) from Valet. US risk-free = French `RF`.
  - Drop any pre-approval-only series and record it.
- `data/french.py`: download the zip; parse CSV blocks (skip header text, split daily/annual blocks, missing codes `-99.99`/`-999`, percent → decimal); tidy DataFrames.
- `data/boc.py`, `data/fred.py`: fetch, parse dates, missing values (`.` in FRED), annualized % → decimal.
- `data/cache.py`: on-disk cache keyed by URL with retrieval date and SHA-256; `refresh=False` default; polite delay between requests.
- `data/__init__.py`: `load_assets/load_factors/load_fx/load_rates(end=DEV_END, allow_holdout=False)` that truncate at `end` unless `allow_holdout=True`; `allow_holdout` with `end > DEV_END` and without the flag raises.
- Tests: hand-written synthetic fixtures mimicking each format (never copies of real files); HTTP mocked via `monkeypatch` on `requests` so fetch code is covered; guard tests (truncation, raise); the `allow_holdout=True` text-scan test. One `@pytest.mark.network` smoke test per source (run locally only: `uv run pytest -m network`).

**Exit:** DATA_SOURCES.md has a verdict per source and series; loaders tested; data quirks logged in NOTES.md.

---

## Step 2: Returns, CAD/hedged views, metrics
Status: **done and merged 2026-10-02** (PR #2, merge commit b0913fc; 2 sessions; `data/views.py`, `report/metrics.py`, shared `_validate.py`; 281 unit tests, 99.7% coverage; python-reviewer x2 and `/code-review` findings fixed; typing follow-up merged as PR #3, merge commit 5114847)

**Context.** `fx_t = S_t / S_{t-1} - 1` with `S` = CAD per USD. Unhedged CAD
return of a USD asset: `r_cad = (1 + r_usd)(1 + fx) - 1` (USD up → CAD investor
gains). Hedged approximation by covered interest parity:
`r_hedged ≈ r_usd + (rf_cad - rf_usd)` per period. Excess returns use the
matching risk-free rate (US for USD view, Canadian for CAD views).
Canadian T-bill yields are simple act/365 yields, so the daily rate is the
return from rolling 3-month bills: `(1 + y·91/365)^(days/91) - 1`, days = calendar
days since the previous trading day (amended 2026-10-02 by the owner; v2 had
`(1 + y)^(days/365) - 1`, which treats y as an effective annual rate).

**Tasks**
- `data/views.py`: `to_cad_unhedged`, `to_cad_hedged`, `excess`; US/Canada holiday alignment rule (inner join on US trading days, carry last FX/rate, documented and tested).
- `report/metrics.py`: CAGR, annualized vol, Sharpe, Sortino, max drawdown and duration, Calmar, skew, excess kurtosis, tracking error and information ratio vs a benchmark. Frequency is an explicit argument (no silent √252).

**Verify:** hand-computed tests (a 4-day series with CAGR, max DD, Sharpe worked out in the docstring); CAD conversion sign tests both directions; CIP hedge on a toy example.
**Exit:** coverage ≥ 80% enforced from here on.

---

## Step 3: Covariance + portfolio construction  *(strongest model)*
Status: **done and merged 2026-10-06** (PR #4, merge commit cf6d45e; 4 sessions; plan `plans/step-3-plan.md`; `construct/covariance.py`, `weights.py`, `strategy.py`; 663 unit tests, 99.8% coverage; python-reviewer and `/code-review` findings fixed)

**Context.** Long-only portfolios over N industry assets, estimated on a
trailing window (default 252 days), using only data up to the estimation date.

**Tasks**
- `construct/covariance.py`: sample; **Ledoit-Wolf** shrinkage to (a) scaled identity (Ledoit & Wolf 2004, *J. Multivariate Analysis*) and (b) constant correlation (Ledoit & Wolf 2004, *J. Portfolio Management*); intensity clipped to [0, 1]; optional EWMA (cut-able).
- `construct/weights.py`: equal weight; minimum variance (long-only, sum 1, max-weight cap, `scipy.optimize` SLSQP with analytic gradient); **equal risk contribution** (convex log-barrier formulation, then normalize); tilts: momentum (12-1 = cumulative return from t−252 to t−21 trading days, own lookback) and low-volatility, applied to a base portfolio with a tilt strength.
- Interface: ~~`weights(returns_window, **params) -> pd.Series`~~ superseded (2026-10-02, owner decision 1 of the step-3 plan) by `Strategy(name, lookback, rule)` in `construct/strategy.py`: `strategy.weights(history)` reads only the last `lookback` rows of the history ending at t and returns a long-only portfolio (≥ 0, sums to 1) labeled by its columns.

**Verify (tests first)**
- LW identity target matches scikit-learn's `LedoitWolf` (test-only dev dependency; same 1/n normalization) to 1e-10; constant-correlation target matches a hand-built 3-asset example; intensity in [0, 1]; condition number improves.
- Min variance: 2-asset closed form `w1 = (σ2² − σ12)/(σ1² + σ2² − 2σ12)` with parameters where the answer lies in [0, 1]; constraints hold (hypothesis property tests).
- ERC: risk contributions equal within 1e-6; uncorrelated assets → weights ∝ 1/σ.
**Exit:** valid weights on random positive-definite inputs; no data beyond the window used.

---

## Step 4: Walk-forward backtest engine  *(strongest model)*
Status: **done and merged 2026-10-08** (PR #5, merge commit ad76e79; 4 sessions; plan `plans/step-4-plan.md`; `backtest/engine.py`, `benchmarks.py`; 790 unit tests, 99.85% coverage; exit met: 12 strategies in 13.6-20.6 s per view; python-reviewer and `/code-review` findings fixed)

**Context.** Monthly rebalance on daily data. At the last trading day t of
each month: estimate on the window ending at close t, trade at close t+1, then
weights drift with returns until the next trade.

**Definitions (tested):** `turnover = ½ Σ|w_target − w_drifted|`;
`cost = cost_bps/1e4 × Σ|w_target − w_drifted|`, deducted from NAV on the
trade day after that day's return.

**Tasks**
- `backtest/engine.py`: ~~`run(strategy, returns, start, end=DEV_END, rebalance="M", cost_bps=5, view="usd")`~~ superseded (2026-10-08, owner decision 1 of the step-4 plan) by `run(strategy, view, *, start=None, end=DEV_END, allow_holdout=False, cost_bps=5.0)`, where `view` is a step-2 `View` (returns + their risk-free rate) and rebalancing is monthly only → daily gross/net returns, weights history, turnover, costs. Inherits the data-layer guard. No `window` argument (amended 2026-10-06): each `Strategy` declares its `lookback`; the engine passes `strategy.weights` the history ending at close t, and the first rebalance waits until `lookback` rows exist. Comparisons across strategies need one common first rebalance date (1/N has lookback 1).
- Benchmarks: equal weight, market (Mkt-RF + RF).
- `Result.summary()` using step-2 metrics.

**Verify (tests first):** no look-ahead (property test: perturbing returns after t never changes weights traded at or before t+1); 2-asset 3-day drift hand example exact; zero cost = gross; turnover/cost hand example; t+1 execution test.
**Exit:** all step-3 strategies run on dev data in under 30 s.

---

## Step 5a: VaR/ES + VaR backtesting  *(strongest model)*
Status: **done and merged 2026-10-09** (PR #6, merge commit f736aac; 3 sessions; plan `plans/step-5a-plan.md`; `risk/var.py`, `risk/backtest.py`; 1009 unit tests, 99.89% coverage; exit met: every method runs on step-4 dev return series, all four at both levels in ~0.9 s per series; python-reviewer and `/code-review` findings fixed)

**Conventions:** VaR and ES reported as **positive losses**; 1-day horizon at
95% and 99%; historical quantile uses linear interpolation (stated); 10-day
by √10 labelled as an approximation.

**Tasks**
- `risk/var.py`: historical VaR/ES; parametric normal and Cornish-Fisher; **filtered historical simulation** (EWMA volatility, λ = 0.94, rescaled standardized residuals). Rolling one-day-ahead forecasts.
- `risk/backtest.py`: exception series; **Kupiec POF**; **Christoffersen independence** (handle n11 = 0) and conditional coverage; Basel traffic-light zones **only for 99% over 250 observations** (stated).

**Verify (tests first):** historical VaR/ES on a 20-value hand list; normal VaR = `−(μ + zσ)` sign-checked; Cornish-Fisher = normal when skew = excess kurtosis = 0; Kupiec LR for T = 250, x = 8, p = 0.01 computed in the test; Christoffersen on a hand-built sequence, including the n11 = 0 case.
**Exit:** works on any step-4 return series (dev only).

---

## Step 5b: Factor and macro regressions
Status: **done and merged 2026-10-09** (PR #7, merge commit dc12652; 2 sessions; plan `plans/step-5b-plan.md`; `risk/factors.py`, `views.yield_changes`; 1086 unit tests, 99.90% coverage; verified: synthetic betas recovered within 4 HAC SEs, HAC SEs equal a hand-computed sandwich to 1e-12; python-reviewer and `/code-review` findings fixed)

**Tasks**
- `risk/factors.py`: USD excess returns on FF5 + momentum; CAD excess returns on the same factors **plus the USD/CAD return**; Newey-West (HAC) errors via statsmodels with lag `floor(4(T/100)^(2/9))`; rolling betas.
- `macro_betas(returns, d_cad10y, fx)`: sensitivity to Canadian 10-year yield changes and FX (inputs for step 6).
- Caveat in docstring: daily FX (noon NY / BoC average) is not synchronous with US closes; also report weekly regressions.

**Verify:** recovers known betas on synthetic data (y = 0.5·MKT + 0.2·SMB + noise) within tolerance; HAC SE cross-checked against a hand-computed lag-1 example.

---

## Step 6: Stress testing
Status: **done and merged 2026-10-09** (PR #8, merge commit b1468b8; 2 sessions; plan `plans/step-6-plan.md`; `stress/historical.py`, `stress/hypothetical.py`; 1187 unit tests, 99.91% coverage; verified: replay = hand compounding to 1e-15, the FX sign through `make_views`, synthetic shock P&L within 4 SEs of the truth; 2020 and 2022 run only on synthetic data; python-reviewer and `/code-review` findings fixed)

**Tasks**
- `stress/historical.py`: registry (GFC 2008-09-01 → 2008-11-30; COVID 2020-02-19 → 2020-03-23; rates 2022-01-01 → 2022-12-31). Replays on given weights in USD and CAD views: P&L, worst day, drawdown. "Current weights" = weights at `DEV_END` in this step; at `HOLDOUT_END` in 7b.
- `stress/hypothetical.py` (cut-able): shocks via 5b betas: MKT −20%; momentum crash UMD −35% over 3 months (cf. 2009); Canadian 10-year +200 bp; CAD ±10%.
- **Run only synthetic data and the 2008 window here.** 2020 and 2022 first run in 7b.

**Verify:** replay = hand compounding on synthetic returns; FX sign convention test.

---

## Step 7a: Pre-registration, dev analysis, luck benchmark, reproduce CLI  *(strongest model)*
Status: **done and merged 2026-10-11** (PR #9, merge commit e673afb; tag `prereg-v1` → 317d4c4, pushed; 3 sessions; plan `plans/step-7a-plan.md`; `research/PLAN.md` FROZEN; `report/inference.py`, `backtest/luck.py`, `study.py`, `cli.py reproduce`, `manifest.py`; 1341 unit tests, 99.94% coverage; HOLDOUT_END 2026-08-31; python-reviewer, `/code-review` (high, then xhigh) findings fixed, then re-frozen)

**Context.** Same discipline as the owner's earlier study: write the plan
before looking, fixed holdout, counted trials, one-shot holdout.

**Tasks**
- Commit **draft** `research/PLAN.md`: question ("Do minimum variance and equal-risk-contribution beat 1/N after costs, risk-adjusted, for USD and CAD-unhedged investors, out of sample?"); hypotheses; all variants (= trial count, with multiple-testing bar); metrics: Sharpe difference tested by the **Ledoit-Wolf (2008) studentized circular block bootstrap** (block length pre-registered), Jobson-Korkie with Memmel correction as secondary (iid-normal caveat), CVaR, max drawdown, turnover; **minimum detectable Sharpe difference** for the holdout length; prior-knowledge statement; `HOLDOUT_END` frozen.
- `backtest/luck.py`: random long-only strategies with matched turnover → Sharpe distribution and "best of N".
- `cli.py reproduce`: runs everything, writes `research/results/` and `research/results/manifest.json` (source URL, file header line, SHA-256, retrieval date, package versions, seed).
- Run dev analysis; record dev decisions in PLAN.md; commit as **frozen** and tag `prereg-v1`. **Ask the owner to push the tag** (a pushed tag is the credible timestamp).

**Exit:** PLAN.md frozen and tagged before any holdout number exists.

---

## Step 7b: Holdout run, robustness, report  *(strongest model)*
Status: **done and merged 2026-10-11** (PR #10, merge commit `a4c9161`; 1 long session; plan `plans/step-7b-plan.md`; the one-shot holdout ran on `9ca40f4` with src identical to `prereg-v1`, recorded in `adccda8`; `robustness.py`, `report/figures.py`, `reproduce --robustness`, `portfolio-lab figures`; `research/REPORT.md` and 5 figures; reruns bit-identical; python-reviewer and `/code-review` (high) findings fixed)

**Tasks**
- Run the holdout **once** (`allow_holdout=True` in `reproduce` only) to `HOLDOUT_END`; log the run in NOTES.md.
- Run 2020 and 2022 stress scenarios on `HOLDOUT_END` weights.
- Robustness: 49 industries (cut-able), costs 0/5/10/20 bp, windows 126/252/504, hedged CAD view.
- `research/REPORT.md` (4-8 pages): question, data, method, results with confidence intervals, robustness, what failed, limitations, next steps; figures generated into `research/figures/`.

**Verify:** `uv run portfolio-lab reproduce` on the **same vintage** matches REPORT numbers to 1e-6 (spot-check 5); on a newer vintage it warns and reports drift vs the manifest. Git history shows `prereg-v1` before holdout results.

---

## Step 8: Streamlit dashboard + deploy
Status: in progress (started 2026-10-11, branch `step-8-dashboard`)

**Tasks**
- First: check the current free-tier terms of Streamlit Community Cloud (does it need a public repo? sleep behaviour? resources) vs one alternative; record in NOTES.md. If a public repo is required, ask **G3 and G4 together**.
- `app/streamlit_app.py` (deps in the `app` group): sidebar (strategies, dates, USD / CAD / CAD-hedged view, cost slider, VaR level); tabs: Performance (equity curve, drawdown, metrics incl. tracking error and information ratio), Risk (rolling VaR/ES, exceptions, Kupiec/Christoffersen), Factors, Stress, About (method, data sources and licenses, "not investment advice").
- Shows derived analytics only: no raw per-industry series tables or download buttons (G1).
- Speed: `st.cache_data`, precomputed default runs; target ~10 s cold load.
- Chart colors per the dataviz skill; works at phone width.

**Verify:** `uv run --group app streamlit run app/streamlit_app.py`; `AppTest` smoke test with injected fixture data (no network in CI); browser check of each tab; cold-load time noted.

---

## Step 9: README, docs, final audit, publish
Status: not started

**Tasks**
- `README.md`, recruiter first: one-paragraph summary, screenshot, demo link, 3 headline results, then method, results, limitations, how to run, data sources and licenses, "not investment advice".
- `docs/career/` (gitignored, stays local): CV bullet in French and English, 60-second pitch, likely interview questions with answers (why Ledoit-Wolf, why a holdout, what Kupiec tests, t+1 execution, CAD hedging).
- **Scrub before going public:** remove local Windows paths and `../stock-prices` references from `plans/` (or gitignore `plans/`), check git history for anything private.
- Fresh-clone audit: `uv sync --all-groups && uv run pytest --cov` (≥ 80%), `uv run portfolio-lab reproduce`, invariants 1-4, README links.
- **Ask G4**, make the repo public, confirm the CI badge is green.

---

## Acceptance criteria

- [ ] `uv sync && uv run pytest` passes from a fresh clone; coverage ≥ 80% on `src/`.
- [ ] GitHub Actions CI (tests + lint) green on `main`.
- [ ] `uv run portfolio-lab reproduce` regenerates every report number on the same data vintage (manifest), and flags drift on a newer one.
- [ ] Live dashboard (or static report if step 8 is cut): strategies, dates, performance, drawdowns, VaR/ES, stress.
- [ ] README recruiter-first with summary, screenshot, demo link.
- [ ] `research/REPORT.md` 4-8 pages with confidence intervals, robustness, limitations; `prereg-v1` tag precedes holdout results.
- [ ] `DATA_SOURCES.md` lists every source and series with terms and citation.
- [ ] CV bullet + pitch exist locally in `docs/career/`.
- [ ] No raw data, no Tiingo, no secrets, no local paths in the public repo.

## Out of scope

Investment advice or recommendations; real-money trading, broker APIs, order
code; ML price prediction; paid data; scraping against terms; intraday data.

## Risks

| Risk | Mitigation |
|---|---|
| French library terms unclear | G1; runtime fetch, no raw files, citation, derived analytics only; optional permission email |
| Data revisions break reproducibility | Frozen `HOLDOUT_END`, manifest with SHA-256, drift warning |
| Holdout contamination | Data-layer guard + text-scan test; holdout stresses deferred; tagged pre-registration |
| Underpowered Sharpe tests | Minimum detectable difference stated in advance; a null result is a valid finding |
| Asynchronous FX timing | Documented; weekly regressions alongside daily |
| Free host needs a public repo | Checked first in step 8; G3 + G4 asked together |
| Scope creep | Serial steps, MVP path, explicit cut order |

## Change log

- 2026-10-11: step 7b done (plan `plans/step-7b-plan.md`, all nine decisions approved;
  details in NOTES.md).
  - **The one-shot holdout:** `reproduce --holdout` ran once on `9ca40f4`, with src identical
    to the tag and the frozen vintage. Its record was committed in `adccda8`.
  - **No primary test rejects.** Every Holm-adjusted p is 1.00. Minimum variance against 1/N
    is −0.041 (USD) and −0.026 (CAD); ERC is +0.012 and +0.029.
  - **Power:** the realized MDE is 0.39-0.47 for minimum variance and 0.12-0.13 for ERC. ERC's
    is twice the planned 0.06, because its holdout SE doubled.
  - **Robustness** (registered, descriptive): windows 126/504, costs 0/10/20 bp, 49 industries
    (all 49 from 1971-03-12), CAD hedged. One raw p of 24 is below 0.05 (ERC, 49 industries,
    CAD).
  - **Interface:**
    - `robustness`: `run`, `universe`, `load_industries`, `cache_industries`, `role`,
      `Industries`, `IND49_SHA256`.
    - `report.figures`: `growth`, `drawdowns`, `sharpe_differences`, `weights`, `stress`,
      `write`, `THEMES`.
    - `cli`: `reproduce --robustness`, `portfolio-lab figures [--results] [--out]`.
    - `manifest`: `ROBUSTNESS_SOURCES`, and `sources(table)`. `build` gains `extra_sources`,
      `tables` and `holdout_tests`.
    - matplotlib is the optional extra `report`.
  - **The lock grew:**
    - `--robustness` needs the recorded holdout, its tests table, and the 49-industry file
      whose SHA-256 the plan fixed.
    - A `--rerun` is recorded as `holdout_rerun`. A first-ever `--rerun` is refused.
    - Each run records the SHA-256 of every table it wrote, and its own package versions.
    - `study.py` is byte-identical to the tag.
  - **Verified:**
    - `--holdout --rerun` matches the one-shot exactly (17 tables, max difference 0).
    - A changed vintage is refused by the holdout and robustness, and warned by a dev run.
    - The tag precedes the holdout record in the history.
  - **For Step 8:** the dashboard can read `research/results/robustness/monthly.csv` and
    `weights.csv`. A fresh clone starts with `reproduce --refresh`, which also caches the
    49-industry file.

- 2026-10-11: step 7a done (3 sessions, plan `plans/step-7a-plan.md`, details in NOTES.md).
  - **Pre-registration:** `research/PLAN.md`.
    - The draft (`0f3c664`) came before any dev comparison.
    - It is frozen in the commit tagged `prereg-v1` (317d4c4; pushed 2026-10-11 after PR #9
      merged as e673afb).
    - Four primary tests (minimum variance and ERC against 1/N, in USD and CAD), Holm at a
      5% family-wise error rate.
    - The Ledoit-Wolf (2008) bootstrap on monthly net excess returns, with Algorithm 3.1
      block calibration.
    - 16 strategies and 32 comparisons, with the trial disclosure.
  - **Amendments to the blueprint's wording:**
    - The block length is pre-registered as a rule (LW's calibration), not as a number.
    - The HAC kernel is the authors' prewhitened Parzen (their `bootTimeInference` code), not
      the paper's QS.
  - **Interface:**
    - `report.inference`: `sharpe_ratio`, `sharpe_diff`, `diff_se`, `hac_psi`, `lw_test`,
      `calibrate_block`, `jobson_korkie_memmel`, `holm`, `min_detectable_difference`, `ceq`,
      `monthly_excess`.
    - `backtest.luck`: `shadows`, `percentile`, `best_of`.
    - `study`: the registry (17 strategies, SEED 20261009), `load`, `run` (13 tables),
      `PRIMARY_ESTIMATOR`.
    - `cli reproduce [--holdout] [--refresh] [--accept-drift] [--rerun] [--out]`, and
      `manifest`.
    - `config.HOLDOUT_END` = 2026-08-31 and `PREREG_TAG`.
    - `risk.backtest.kupiec_exact` and `report(start=)`; `cache.read_cached`.
  - **The holdout lock:** frozen HOLDOUT_END and estimator; the tag an ancestor of HEAD;
    PLAN.md identical to its tagged copy; the cache equal to the manifest at the tag; one shot.
  - **Owner decisions:**
    - 2026-10-09: all twelve as recommended, plus the post-registration addendum.
    - 2026-10-10: keep the luck benchmark and state what it measures.
  - **Dev results** (descriptive):
    - The estimator rule chose lw_identity (a near tie).
    - No primary difference with 1/N: every Holm-adjusted p is 1.0.
    - The MDE at 188 months is about 0.5 for minimum variance and 0.06 for ERC.
  - **Reviews fixed:**
    - The holdout compared against a manifest that dev runs overwrite.
    - Two p-values were reported for one test.
    - Plus the vintage, coverage, one-shot and root checks, the null and unit-root guards,
      and an AST holdout scan.
  - **For 7b:**
    - Run `reproduce --holdout` once, after the tag is pushed and the PR merged.
    - Robustness: windows 126 and 504, costs 0, 10 and 20 bp, the hedged view, 49
      industries.
    - The 49-industry NaN policy is still open.
  - **Extra-high review (owner asked, 2026-10-11): 11 findings, all fixed, then the freeze
    was redone** (the tag was never pushed; no holdout number exists).
    - The one-shot rule could be bypassed with `--out`.
    - Code could change after the tag unnoticed; now uncommitted src/ is refused and
      changes since the tag are recorded.
    - The dirty flag counted the tracked results manifest.
    - The holdout lacked the 2001 → DEV_END shock comparison that PLAN.md §10 registers.
    - Plus smaller items, and parallel calibration with identical numbers.
    - Every dev table was reproduced.

- 2026-10-09: step 6 done (2 sessions, plan `plans/step-6-plan.md`, details in NOTES.md).
  - **Interface (`stress.historical`):**
    - `Scenario` and the read-only `SCENARIOS` (gfc, covid, rates, the dates above).
    - `window(returns, scenario, *, allow_holdout=False)`: the days in (start, end], by
      calendar date. The data must reach both edges (a weekend end on the Friday) and have
      no hole of more than 7 calendar days.
    - `replay(weights, view, scenario, *, allow_holdout=False)`: buy-and-hold through
      `engine.simulate`, assets at weight 0 not read.
    - `outcome(returns)` → `Outcome`: pnl, worst_day, worst_date, max_drawdown, days, first,
      last.
    - `table(weights_by_view, views, scenarios, *, allow_holdout=False)`.
  - **Interface (`stress.hypothetical`):**
    - `Shock` and the read-only `SHOCKS` (market, momentum, rates, cad_up, cad_down).
    - `cad_move(pct)` = 1/(1 + pct) − 1.
    - `portfolio_excess(weights, view)`: the constant mix.
    - `apply(betas, shocks)`: P&L = β × move, SE = |move| × SE.
    - Also `data.views.check_view`, shared with the engine.
  - **Owner decisions (2026-10-09, all eight as recommended):**
    - The DEV_END `final_weights` of each view's own run.
    - Replay by buy-and-hold; the realized path is shown separately.
    - Windows are (start, end].
    - `window`/`replay`/`table` follow `resolve_end`'s holdout rule; 2020 and 2022 ran only
      on synthetic data.
    - Shock betas: weekly, partial, 2001-01-03 → DEV_END for both views.
    - The betas are those of the constant mix of the current weights, not the strategy's
      history.
    - One variable per shock, with linear P&L.
  - **Dev data** (model behaviour only):
    - GFC replay of 1/N: −30.3% USD, −19.0% CAD, −30.1% hedged. The USD rose 16.3% against
      the CAD, a natural hedge the partial market shock leaves out.
    - Minimum variance: the 10-year +200 bp shock costs −3.0% ± 0.8.
    - USD market shock: −14.4% from 2001 vs −17.5% from 1963 (sample sensitivity).
  - **For 7b:**
    - Replay HOLDOUT_END weights with `table(..., allow_holdout=True)` from `reproduce`.
    - The realized paths come from `window(result.net, ...)`.
    - The 49-industry NaN policy matters only for held assets.
  - **Reviews fixed:**
    - 16:00 stamps shifted the window by a day.
    - Holes inside a window were compounded over.
    - Duplicate scenario names and mixed beta tables were accepted.
    - `True` was accepted as a shock move.
    - Non-finite or negative SEs were accepted.
    - There were three copies of the View check.

- 2026-10-09: step 5b done (2 sessions, plan `plans/step-5b-plan.md`, details in NOTES.md).
  - **Interface (`risk.factors`):**
    - `regress(y, X, *, lags=None)` → `Fit`: coef, se, tstat and pvalue by term, `alpha`
      first; r2, adj_r2, nobs and lags; start and end are the daily days used; `table()`.
    - `factor_betas(excess, factors, *, fx=None, frequency="daily"|"weekly")`: a long table
      indexed by (asset, term).
    - `macro_betas(excess, d_cad10y, fx, factors, *, frequency)`. **This deviates from the
      blueprint's `macro_betas(returns, d_cad10y, fx)`:** it adds `factors` so every beta is
      partial.
    - `rolling_betas(excess, factors, *, fx=None, window=252)`, indexed by each window's last
      day.
    - `to_weekly(..., how="compound"|"sum")`, W-WED.
    - `nw_lags`, exact in integers.
    - `data.views.yield_changes`.
  - **Owner decisions (2026-10-09, after a review the owner asked for):**
    - Regress each view's own excess returns. The CAD-hedged view needs no model, being equal
      to USD (CIP).
    - fx is Step 2's `fx_returns`. For Step 6, "CAD +10%" means fx = 1/1.10 − 1.
    - HAC: Bartlett kernel, lag floor(4(T/100)^(2/9)), no small-sample correction, normal
      p-values.
    - Weekly: Wednesday-to-Wednesday weeks; returns compounded, yield changes summed.
    - Rolling betas: 252 days, point estimates only.
    - Macro betas are partial, with the six factors held fixed.
    - Alignment is strict: shared span, matching dates, NaN only at the edges.
    - Pure and unguarded.
  - **Why partial:** on dev data, the yield betas without controls are +4.8 to +11.4 for every
    portfolio. That is the stock-bond correlation, which would make a +200 bp stress a gain.
    Partial weekly betas: Utils −3.14, minimum variance −1.83, 1/N −0.09.
  - **For Step 6:**
    - Use the weekly betas: the weekly yield betas are 2.5-3 times the daily ones.
    - The daily 10-year starts in 2001 (523 dev weeks).
  - **Reviews fixed:**
    - Weekly fits reported their Wednesday label (2011-01-05) as their end.
    - Weekly was 30 times slower than daily; `to_weekly` now uses numpy `reduceat`.
    - Times of day split weeks.
    - 1 and "1" asset labels collided.
    - A constant y gave 0/0.
    - The inf and no-column errors are clearer.
    - `is_whole` moves to `_validate`.

- 2026-10-09: step 5a done (3 sessions, plan `plans/step-5a-plan.md`, details in NOTES.md). **Interface:** `risk.var`: `var_es`, `normal_var_es`, `cornish_fisher_var_es` (point estimates, `VarEs(var, es)` as positive losses), `ewma_vol`, `forecast(returns, *, method, level, window=500, decay=0.94)` indexed by the day forecast, `next_day` (today's VaR), `scale_horizon` (√days, labelled an approximation); `METHODS = historical, normal, cornish_fisher, fhs`. `risk.backtest`: `exceptions` (strict r < −VaR), `kupiec`, `transitions`, `christoffersen`, `conditional_coverage` (LR_uc on the T − 1 transition days + LR_ind, exact), `basel_zone`/`basel_zones` (99% over 250 days only), `report`. Owner decisions (2026-10-08): any daily return Series (dev checks on total net returns); one `window`, default 500 (7a pre-registers it); historical ES = mean of returns at or below −VaR; CF ES = exact tail mean of the CF quantile (non-monotone cases reported, not corrected); FHS = one causal zero-mean EWMA pass seeded with the first window's mean square; the risk functions are pure and unguarded, like `simulate`. Conventions: the minimum window is 4 for every method; a window with no variance, or a result that isn't finite, raises naming the day; a whole-number interpolation index takes the order statistic exactly. Dev data (model behaviour only, no ranking): at 99%, historical and normal give too many exceptions on all 6 series; CF gets the rate but its misses cluster; FHS alone passes conditional coverage (4 of 6); at 95% every method's misses cluster. **For 7a:** the chi-square Kupiec test's exact size at 99% over 250 days is 9.5%, not 5%; FHS does slightly better with a 1000-day window; `report` recomputes each method per level (~1 s per series). Reviews fixed: ES dropping a tail value when 1 − level rounds down (90%), `forecast` raising on its dropped next-day row, silent NaN/inf results, numpy scalars rejected, block memory growing with the window.
- 2026-10-08: step 4 done (4 sessions, plan `plans/step-4-plan.md`, details in NOTES.md). **Interface:** `run(strategy, view, *, start=None, end=DEV_END, allow_holdout=False, cost_bps=5.0)` takes a step-2 `View`, so a CAD view estimates on CAD returns; monthly only. Also `compare` (one common first rebalance, the longest lookback; an explicit `start` checked before anything runs), `reprice` (a result's targets at another cost, for 7b's grid, guarded and on the result's own days), and `benchmarks.market`/`market_views` (Mkt-RF + RF as an index, converted by `make_views`). Owner decisions (2026-10-08): start invested with no opening cost; start dates fail closed; no SLSQP warm start; dev performance used only for sanity checks. Conventions: net = gross − c(1 + gross), exact at zero cost; weights after a trade are exactly the target; a portfolio worth 0 raises; `cost_bps` in [0, 5000); `start`/`end` must be tz-naive dates (`_validate.as_date`, now also in `data.resolve_end`: an int was read as nanoseconds since 1970); `summary`'s gross column pays no cost. Dev data: 12 strategies take 13.6-20.6 s per view (both views ~35 s, so 7a may need the warm start); minimum variance (EWMA) turns over 29% a month. For 7b: `start=None` waits for `lookback` rows, not complete ones, and `simulate` rejects NaN returns even for zero-weight assets, so the 49-industry run must choose its NaN policy. Reviews fixed: the guarded `reprice`, compare's late start check, non-date `start`/`end`, the cost bound, label alignment of targets, `Result` index checks. Also reworded a Step 3 NOTES line that broke invariant 3's literal check.
- 2026-10-06: step 3 done (4 sessions, plan `plans/step-3-plan.md`, details in NOTES.md). **Interface change:** the strategy contract `Strategy(name, lookback, rule)` replaces `weights(returns_window, **params)`, and Step 4's `run(...)` drops `window` (strategies declare their lookback; a tilt's lookback covers its base: momentum on a 126-day minimum variance looks back 252). Owner decisions (2026-10-02): minimum-variance cap default none (DeMiguel et al.'s shortsale-constrained portfolio), capped variants a 7a choice; PyPortfolioOpt's constant-correlation code vendored into `tests/` as an independent check. Conventions: covariances divide by T (scikit-learn's); SLSQP's active set is polished to the exact optimum (kept only if it passes KKT); minimum variance rejects singular matrices (use shrinkage); ERC runs Spinu's Newton method on the correlation matrix; the tilt's z-score uses the population sd; estimators are named by `ESTIMATORS` keys; 1/N has lookback 1. Dev data (12 industries, weights only): minimum variance is mostly Utilities (100% in 59 of 559 months with the sample matrix); 12 strategies take 15 s over 559 rebalances. Reviews fixed: tilt overflow at strength ~1e308, constructor inputs checked up front, names keep every digit, read-only `ESTIMATORS`.
- 2026-10-02: tooling: `mypy` + `pandas-stubs` in the dev group, `[tool.mypy]` config, mypy step in CI after ruff (pandas types were `Any` under the old ad hoc `uvx mypy`). 32 errors fixed by narrowing, no ignores (`check_dates` returns the `DatetimeIndex`; `point_in_time_daily` now requires one). `metrics.py` overloads restored (Series -> float, DataFrame -> Series). `py.typed` added so importers get the types.
- 2026-10-02: step 2 done. Metrics: `periods_per_year` required; zero-risk ratios (Sharpe, Sortino, IR, Calmar) are NaN, never inf, so they cannot rank first; drawdowns within 1e-12 of the peak count as 0; returns must be finite, >= -100%, in date order with unique dates. CAD views end 7 days after the last FX/rate observation (a lagged source shortens them, a trailing holiday is carried). Hypothesis runs without a deadline. Follow-up: pandas-stubs + mypy in CI.
- 2026-10-02: step 2 session 1. Owner chose the rolling-bill rate conversion `(1 + y·91/365)^(d/91) - 1` over v2's `(1 + y)^(d/365) - 1` (T-bills are quoted as simple act/365 yields; v2 understated CAD rf by ~24 bp/yr on average, ~160 bp/yr at 1981 rates). Carried FX/rate levels older than 7 calendar days raise.
- 2026-10-02: step 1 done. Data-layer timing convention: row t = observed by close of t; monthly StatCan rates enter one business day after their last-Wednesday stamp (verified 299/299 months against Valet daily). Reviews found and fixed a holdout bypass (`end` that parses to NaT) and an unsorted-index truncation leak; truncation now uses a mask. Live tests share one cache (one download per file per run).
- 2026-10-02: step 1 session 1. Canadian rate history moved from Valet to Statistics Canada table 10-10-0122-01 (Valet daily series start 2001, the CAD view starts 1971); adds `data/statcan.py` to step 1. The `allow_holdout=True` text scan also exempts `tests/` (owner chose option A). FRED missing values are empty fields, not `.`.

- 2026-10-01: draft v1 (steps 0-9).
- 2026-10-01: step 0 done. G2 approved: private repo github.com/cristian4269/portfolio-lab, `main` pushed, CI green (checkout@v4, setup-uv@v6). Formal python-reviewer pass skipped for this scaffold-only step (no logic); required from step 1.
- 2026-10-01: owner approved v2; G1 = runtime fetch + cite; G5 = MIT.
- 2026-10-01: v2 after adversarial review (deep-opus): data-layer holdout guard, frozen vintage + manifest, t+1 execution, serial graph, steps 5 and 7 split, CAD design specified (optimizer currency, FX definition, CIP hedge, rate conversion), FHS replaces normal Monte Carlo, Ledoit-Wolf 2008 bootstrap test + minimum detectable difference, push approval per session, hosting/G4 coupling, CI lockfile and markers, `docs/career/` kept local, plan scrub before publishing.
