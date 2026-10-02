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
Status: **done 2026-10-02, pending PR merge** (3 sessions; 176 unit + 10 live tests, 99.5% coverage; python-reviewer and `/code-review` findings fixed, incl. a holdout bypass via NaT)

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
Status: not started

**Context.** `fx_t = S_t / S_{t-1} - 1` with `S` = CAD per USD. Unhedged CAD
return of a USD asset: `r_cad = (1 + r_usd)(1 + fx) - 1` (USD up → CAD investor
gains). Hedged approximation by covered interest parity:
`r_hedged ≈ r_usd + (rf_cad - rf_usd)` per period. Excess returns use the
matching risk-free rate (US for USD view, Canadian for CAD views).
Annualized % rates → daily: `(1 + y)^(days/365) - 1` per calendar day between
observations (act/365), documented.

**Tasks**
- `data/views.py`: `to_cad_unhedged`, `to_cad_hedged`, `excess`; US/Canada holiday alignment rule (inner join on US trading days, carry last FX/rate, documented and tested).
- `report/metrics.py`: CAGR, annualized vol, Sharpe, Sortino, max drawdown and duration, Calmar, skew, excess kurtosis, tracking error and information ratio vs a benchmark. Frequency is an explicit argument (no silent √252).

**Verify:** hand-computed tests (a 4-day series with CAGR, max DD, Sharpe worked out in the docstring); CAD conversion sign tests both directions; CIP hedge on a toy example.
**Exit:** coverage ≥ 80% enforced from here on.

---

## Step 3: Covariance + portfolio construction  *(strongest model)*
Status: not started

**Context.** Long-only portfolios over N industry assets, estimated on a
trailing window (default 252 days), using only data up to the estimation date.

**Tasks**
- `construct/covariance.py`: sample; **Ledoit-Wolf** shrinkage to (a) scaled identity (Ledoit & Wolf 2004, *J. Multivariate Analysis*) and (b) constant correlation (Ledoit & Wolf 2004, *J. Portfolio Management*); intensity clipped to [0, 1]; optional EWMA (cut-able).
- `construct/weights.py`: equal weight; minimum variance (long-only, sum 1, max-weight cap, `scipy.optimize` SLSQP with analytic gradient); **equal risk contribution** (convex log-barrier formulation, then normalize); tilts: momentum (12-1 = cumulative return from t−252 to t−21 trading days, own lookback) and low-volatility, applied to a base portfolio with a tilt strength.
- Interface: `weights(returns_window, **params) -> pd.Series` (≥ 0, sums to 1).

**Verify (tests first)**
- LW identity target matches scikit-learn's `LedoitWolf` (test-only dev dependency; same 1/n normalization) to 1e-10; constant-correlation target matches a hand-built 3-asset example; intensity in [0, 1]; condition number improves.
- Min variance: 2-asset closed form `w1 = (σ2² − σ12)/(σ1² + σ2² − 2σ12)` with parameters where the answer lies in [0, 1]; constraints hold (hypothesis property tests).
- ERC: risk contributions equal within 1e-6; uncorrelated assets → weights ∝ 1/σ.
**Exit:** valid weights on random positive-definite inputs; no data beyond the window used.

---

## Step 4: Walk-forward backtest engine  *(strongest model)*
Status: not started

**Context.** Monthly rebalance on daily data. At the last trading day t of
each month: estimate on the window ending at close t, trade at close t+1, then
weights drift with returns until the next trade.

**Definitions (tested):** `turnover = ½ Σ|w_target − w_drifted|`;
`cost = cost_bps/1e4 × Σ|w_target − w_drifted|`, deducted from NAV on the
trade day after that day's return.

**Tasks**
- `backtest/engine.py`: `run(strategy, returns, start, end=DEV_END, window=252, rebalance="M", cost_bps=5, view="usd")` → daily gross/net returns, weights history, turnover, costs. Inherits the data-layer guard.
- Benchmarks: equal weight, market (Mkt-RF + RF).
- `Result.summary()` using step-2 metrics.

**Verify (tests first):** no look-ahead (property test: perturbing returns after t never changes weights traded at or before t+1); 2-asset 3-day drift hand example exact; zero cost = gross; turnover/cost hand example; t+1 execution test.
**Exit:** all step-3 strategies run on dev data in under 30 s.

---

## Step 5a: VaR/ES + VaR backtesting  *(strongest model)*
Status: not started

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
Status: not started

**Tasks**
- `risk/factors.py`: USD excess returns on FF5 + momentum; CAD excess returns on the same factors **plus the USD/CAD return**; Newey-West (HAC) errors via statsmodels with lag `floor(4(T/100)^(2/9))`; rolling betas.
- `macro_betas(returns, d_cad10y, fx)`: sensitivity to Canadian 10-year yield changes and FX (inputs for step 6).
- Caveat in docstring: daily FX (noon NY / BoC average) is not synchronous with US closes; also report weekly regressions.

**Verify:** recovers known betas on synthetic data (y = 0.5·MKT + 0.2·SMB + noise) within tolerance; HAC SE cross-checked against a hand-computed lag-1 example.

---

## Step 6: Stress testing
Status: not started

**Tasks**
- `stress/historical.py`: registry (GFC 2008-09-01 → 2008-11-30; COVID 2020-02-19 → 2020-03-23; rates 2022-01-01 → 2022-12-31). Replays on given weights in USD and CAD views: P&L, worst day, drawdown. "Current weights" = weights at `DEV_END` in this step; at `HOLDOUT_END` in 7b.
- `stress/hypothetical.py` (cut-able): shocks via 5b betas: MKT −20%; momentum crash UMD −35% over 3 months (cf. 2009); Canadian 10-year +200 bp; CAD ±10%.
- **Run only synthetic data and the 2008 window here.** 2020 and 2022 first run in 7b.

**Verify:** replay = hand compounding on synthetic returns; FX sign convention test.

---

## Step 7a: Pre-registration, dev analysis, luck benchmark, reproduce CLI  *(strongest model)*
Status: not started

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
Status: not started

**Tasks**
- Run the holdout **once** (`allow_holdout=True` in `reproduce` only) to `HOLDOUT_END`; log the run in NOTES.md.
- Run 2020 and 2022 stress scenarios on `HOLDOUT_END` weights.
- Robustness: 49 industries (cut-able), costs 0/5/10/20 bp, windows 126/252/504, hedged CAD view.
- `research/REPORT.md` (4-8 pages): question, data, method, results with confidence intervals, robustness, what failed, limitations, next steps; figures generated into `research/figures/`.

**Verify:** `uv run portfolio-lab reproduce` on the **same vintage** matches REPORT numbers to 1e-6 (spot-check 5); on a newer vintage it warns and reports drift vs the manifest. Git history shows `prereg-v1` before holdout results.

---

## Step 8: Streamlit dashboard + deploy
Status: not started

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

- 2026-10-02: step 1 done. Data-layer timing convention: row t = observed by close of t; monthly StatCan rates enter one business day after their last-Wednesday stamp (verified 299/299 months against Valet daily). Reviews found and fixed a holdout bypass (`end` that parses to NaT) and an unsorted-index truncation leak; truncation now uses a mask. Live tests share one cache (one download per file per run).
- 2026-10-02: step 1 session 1. Canadian rate history moved from Valet to Statistics Canada table 10-10-0122-01 (Valet daily series start 2001, the CAD view starts 1971); adds `data/statcan.py` to step 1. The `allow_holdout=True` text scan also exempts `tests/` (owner chose option A). FRED missing values are empty fields, not `.`.

- 2026-10-01: draft v1 (steps 0-9).
- 2026-10-01: step 0 done. G2 approved: private repo github.com/cristian4269/portfolio-lab, `main` pushed, CI green (checkout@v4, setup-uv@v6). Formal python-reviewer pass skipped for this scaffold-only step (no logic); required from step 1.
- 2026-10-01: owner approved v2; G1 = runtime fetch + cite; G5 = MIT.
- 2026-10-01: v2 after adversarial review (deep-opus): data-layer holdout guard, frozen vintage + manifest, t+1 execution, serial graph, steps 5 and 7 split, CAD design specified (optimizer currency, FX definition, CIP hedge, rate conversion), FHS replaces normal Monte Carlo, Ledoit-Wolf 2008 bootstrap test + minimum detectable difference, push approval per session, hosting/G4 coupling, CI lockfile and markers, `docs/career/` kept local, plan scrub before publishing.
