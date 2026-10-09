# Step 6 plan: stress testing

Approved by the owner on 2026-10-09 (all eight decisions as recommended). Read with the
blueprint's Step 6 section. Branch `step-6-stress`.

Before drafting, a scratch script on dev data measured the choices below. It looked at method
sensitivity only and ranked no strategies. The numbers are under "Grounding 2026-10-09".

## What gets built

- `stress/historical.py`:
  - `Scenario(name, start, end, note)` and `SCENARIOS`: the blueprint's three windows, pinned by
    a test.
    - `gfc`: 2008-09-01 → 2008-11-30.
    - `covid`: 2020-02-19 → 2020-03-23.
    - `rates`: 2022-01-01 → 2022-12-31.
  - `window(returns, scenario, *, allow_holdout=False)`: the rows of a return Series or
    DataFrame inside the scenario (decision 3). It is guarded (decision 5).
  - `replay(weights, view, scenario, *, allow_holdout=False)`: the daily returns of `weights`
    bought at the close before the window and held through it, drifting with no trades
    (decision 2). It reuses `engine.simulate` with a single target and no cost.
  - `outcome(returns)`: P&L, worst day and its date, max drawdown, the number of days, and the
    first and last day (decision 4). It works on any daily return Series, whether a replay or
    a strategy's realized `Result.net` cut by `window`.
  - `table(...)`: one row per (scenario, view, portfolio), for 7b and the dashboard.
- `stress/hypothetical.py` (cut-able in the blueprint; I recommend building it, since 5b was
  built for it):
  - `Shock(name, moves, note)` and `SHOCKS`: one variable moves and the others stay at 0
    (decision 8).
    - `market`: Mkt-RF −20%.
    - `momentum`: Mom −35%.
    - `rates`: dy +200 bp.
    - `cad_up`: fx = 1/1.10 − 1.
    - `cad_down`: fx = 1/0.90 − 1.
  - `cad_move(pct)`: the fx return for a CAD move of `pct`, which is 1/(1 + pct) − 1.
  - `portfolio_excess(weights, view)`: the excess return of `weights` held as a constant mix,
    Σ w_i·excess_i (decision 7).
  - `apply(betas, shocks)`: takes the long beta table from `factor_betas` or `macro_betas`.
    It returns, per (portfolio, shock): the beta, the P&L = Σ beta_k × move_k, and its
    SE = |move| × SE(beta). A shock that names a term missing from the table raises (USD
    fits have no fx or dy).
- Every formula gets a hand-computed test (invariant 6). Dev data is used only for the 2008
  window and for sanity checks; 2020 and 2022 run only on synthetic data here.

## Decisions for the owner (recommendation first)

1. **The weights to stress: each view's own "current weights", `Result.final_weights` of a run
   through DEV_END.**
   - These are the holdings at the close of 2010-12-31, after December's drift. The last
     target (the December trade) is the alternative; it ignores one month of drift.
   - USD-view weights are replayed on USD returns and CAD-view weights on CAD returns. That
     follows the pre-registered rule: each view estimates on its own currency.
   - The functions take any long-only weights. The dev check uses 1/N and minimum variance
     (sample), as 5b did. Step 7a picks the strategies, and 7b swaps in HOLDOUT_END weights.
2. **Replay: buy-and-hold.** The weights are bought at the close before the window and then
   drift with returns, with no trades and no costs.
   - This is what a portfolio held through the crisis does, and it is the engine's own
     accounting (`simulate` with one target).
   - The alternative is constant weights, rebalanced daily. In the 2008 window it changes the
     P&L by 0.04-0.19 pp and the max drawdown by at most 0.4 pp, and the worst day is the
     same day.
   - The strategy's **realized** path in the window is a separate number: `Result.net` cut by
     `window`. Over a dev window both can be shown. The replay is "today's portfolio in that
     crisis" and the realized path is "what the strategy did then".
3. **Window convention: returns of days in (start, end].** The weights are held from the
   close of the last day on or before `start`.
   - For COVID that measures peak to trough: from the S&P close of 2020-02-19 to the close of
     2020-03-23. The peak day's own return is left out.
   - For the GFC (2008-09-01 is Labor Day) and for 2022 (2022-01-01 is a Saturday), the
     result is the same as [start, end]: 2008-09-02 → 2008-11-28, 63 days.
   - The registry keeps the blueprint's dates exactly; they are pre-registered.
   - The data must cover the window: a day on or before `start`, and a last day on or after
     `end`. Otherwise it raises, naming the gap. A shortened window never runs silently.
     (Amended in review: a weekend `end` is reached on the Friday before it, and dates are
     compared as calendar days, whatever the time of the stamps.)
4. **Outcome metrics, in the view's currency, as fractions of the value at the anchor
   close:**
   - P&L: the NAV at the end minus 1, a compounded total return (not excess).
   - Worst day: the lowest daily return in the window, with its date.
   - Max drawdown: `metrics.max_drawdown`, which already counts the starting NAV of 1 as the
     first peak. So a loss on day one is a drawdown.
   - The number of days and the first and last day, so a short window is visible.
5. **Holdout guard: `window` and `replay` take `allow_holdout=False` and apply
   `data.resolve_end(scenario.end, allow_holdout)`, the one holdout rule.**
   - So the 2020 and 2022 scenarios raise `HoldoutError` without the flag, even on a view that
     somehow holds those dates. The flag is the engine's, and only `cli.py`, `research/` and
     `tests/` may pass it (the text scan already enforces that).
   - In this step, 2020 and 2022 run only in tests, on synthetic data. Real data is used only
     for the GFC window.
   - `outcome` and the hypothetical functions are pure and unguarded, like the 5a/5b risk
     functions: they see only what they are given.
6. **Betas for the hypothetical shocks: weekly, partial, 2001-01-03 → DEV_END, the same
   sample for both views.**
   - CAD view: `macro_betas(..., frequency="weekly")`, which regresses on the six factors, fx
     and dy. Every CAD shock reads its beta from this one regression, so the betas are
     consistent partials.
   - USD view: `factor_betas(..., frequency="weekly")` on the same 2001-2010 span. Only the
     market and momentum shocks apply in USD.
   - **Why 2001-2010 for USD too:** the dy and fx betas can only be estimated from 2001 (the
     daily 10-year starts then). A recent window also describes the current portfolio better.
   - **The sample matters:** for minimum-variance DEV_END weights in USD, Mkt-RF is 0.72
     (SE 0.03) from 2001 but 0.87 (0.02) from 1963. A −20% market shock gives −14.4% or
     −17.5%. 1/N moves by 0.001.
   - The full-sample number goes into NOTES as a sensitivity. The alternative is the full
     sample for USD's market and momentum betas.
7. **Portfolio betas: regress the constant mix of the current weights, not the strategy's own
   return history.**
   - The constant mix is `portfolio_excess(w, view)`: today's weights over 2001-2010. Its
     regression gives each beta and its HAC SE directly.
   - The strategy's realized series (which 5b's dev check regressed) averages the exposures of
     past portfolios. For minimum variance, its partial yield beta is −1.83; the DEV_END
     weights' beta is −1.49 (SE 0.39).
   - The weight-sum of asset betas, Σ w_i β_i, gives the same point estimates within 0.01 (by
     OLS linearity, up to weekly compounding), but no SE. It is kept only as a test
     cross-check.
8. **Shock definitions: each shock moves one variable and holds the rest at 0. The P&L is
   linear, β × move, an excess return as a fraction of value.**
   - Market −20%: Mkt-RF = −0.20.
   - Momentum crash: Mom = −0.35, the cumulative move over 3 months (cf. 2009). The linear
     beta is applied to the cumulative move. The caveat is stated: betas assumed stable and
     linear over the 3 months, no compounding, no path.
   - Canadian 10-year +200 bp: dy = +0.02 (yields in decimals, 5b's units).
   - CAD ±10%: fx = 1/1.10 − 1 = −9.09% (CAD up, a loss for the unhedged view) and
     fx = 1/0.90 − 1 = +11.1% (CAD down, a gain). This is 5b's decision 2.
   - **The natural-hedge caveat** goes in the docstring and the notes. A partial market shock
     holds fx at 0, but in 2008 the USD rose 16.3% against the CAD over the GFC window.
     - In that window the CAD-view replay of 1/N lost 19.0% against 30.3% in USD.
     - So the partial market shock overstates a CAD investor's crisis loss, as history goes.
     - The historical replay shows the combined effect. A correlated "crash with USD rally"
       scenario is out of scope.
   - No combined scenarios: 5b's fit keeps no covariance between betas, so a sum would have
     no SE. Adding single shocks by hand stays possible, since the betas are partial.

## Conventions

- **Units:** returns are decimals; P&L is a fraction of the portfolio's value at the anchor
  close, in the view's currency.
- **FX sign** (Step 2): fx = S_t/S_{t−1} − 1 with S in CAD per USD. A positive fx (USD up) is a
  gain for the unhedged CAD investor. CAD unhedged = (1 + r_usd)(1 + fx) − 1.
- **Replay accounting:** NAV_t = Σ_i w_i Π_{s ≤ t}(1 + r_i,s). The daily return is
  NAV_t/NAV_{t−1} − 1, and the P&L is NAV_end − 1. This is the same as `engine.simulate`'s
  gross return with one target.
- **Hypothetical P&L** is linear in the moves. Its SE assumes the beta's HAC SE (the shock
  itself is fixed).

## Sessions

**Session 1: historical replay.** Tests first, then code.
- `SCENARIOS`: the three windows pinned to the blueprint's dates.
- `window`:
  - (start, end] on a synthetic calendar: a start on a holiday, a start on a trading day
    (its return excluded), an end on a weekend.
  - The data must cover the window, or it raises, naming the gap.
  - Guard: `covid` and `rates` raise `HoldoutError` on synthetic data without the flag, and
    run with `allow_holdout=True` (in tests only). `gfc` runs without it.
- `replay`:
  - It equals hand compounding: 2 assets over 3 days, Σ w_i Π(1 + r_i) − 1, to 1e-12.
  - Weights are checked as the engine checks them: labels, order, ≥ 0, sum 1.
  - **FX sign:**
    - A USD asset with r_usd = 0 and fx = +5% gives the CAD replay +5% (USD up, CAD
      investor gains).
    - The CAD view built by `make_views` equals (1 + r_usd)(1 + fx) − 1 day by day.
- `outcome`:
  - A hand example: the P&L, the worst day and its date, and the max drawdown from the
    anchor, including a loss on day one.
  - It agrees with `metrics.max_drawdown`.
- `table`: one row per (scenario, view, portfolio). A 2020 scenario passed without the flag
  raises before anything runs.
- RED and GREEN commits. Then a mutation run:
  - the window [start, end] instead of (start, end]
  - the drawdown peak without the starting 1
  - constant weights instead of drift
  - the guard skipped
  - the anchor taken after `start`
  - the coverage check skipped
- As built (2026-10-09, see NOTES.md):
  - RED `b0be0c2`, GREEN `cac9ef9`.
  - 54 tests, 100% coverage of the module.
  - The mutation run caught 12 of 12 planted bugs: the plan's six plus an exclusive end, the
    table's guard after the replays, a summed P&L, the worst date off by one and an empty
    window.

**Session 2: hypothetical shocks, dev check, review.**
- `SHOCKS` values are exact: fx = −1/11 and +1/9, dy = 0.02 (decimals).
- `cad_move(0.10)` gives −1/11.
- `apply`:
  - A hand table: P&L = β × move, SE = |move| × SE.
  - A shock that names a missing term raises.
  - USD fits accept only the market and momentum shocks.
- `portfolio_excess`:
  - It equals Σ w_i·excess_i exactly.
  - Zero-weight columns are dropped first, so an unlisted (NaN) asset with weight 0 does
    not spoil the series (the 49-industry case).
- **Synthetic end-to-end check:**
  - y = 0.8·MKT − 0.3·UMD + 1·fx − 7·dy + ε, weekly.
  - Each shock's P&L lands within 4 SEs of β_true × move.
  - CAD +10% gives about −9.09%.
- **Mutation run:**
  - fx = −0.10 instead of 1/1.10 − 1
  - dy in bp (2.0) instead of decimals
  - SE without |move|
  - the strategy's realized series regressed instead of the constant mix
- **Dev check** (model behaviour only, no ranking):
  - **2008 replay:** DEV_END weights in USD, CAD, and the CAD weights in the hedged view (to
    separate the currency effect), next to the realized path.
  - **Hypothetical shocks:** with their SEs, plus the full-sample USD sensitivity
    (decision 6).
  - **Timing.**
- python-reviewer, then `/code-review` (high), then verification-loop. Fix the findings.
- Check invariants 1-6. Update NOTES.md, the blueprint's status line and the change log.
- Ask the owner before pushing and opening the PR.
- As built (2026-10-09, see NOTES.md):
  - RED `c15ff95`, GREEN `a8dfec5`. The plan's mutation "the strategy's realized series
    regressed" lives in the caller, so the run planted "excess without rf" instead.
  - Mutation runs: 10 of 10 caught; 34 of 34 on the final code of both modules.
  - Dev check: the 2008 replay and the shocks match the grounding numbers. The partial market
    shock (fx held at 0) costs 1/N −19.9% in CAD, where the 2008 replay lost 19.0% on a
    30.6% market fall.
  - Reviews:
    - `/code-review` found 8 issues, fixed: windows now compare calendar dates, a weekend
      end is reached on the Friday, duplicate scenario names and mixed beta tables are
      refused, a shock's move must be a number, and there is one `check_view`.
    - python-reviewer found no critical or high issues. Fixed: holes inside a window raise,
      `replay` skips zero weights, `table` was tested with the flag, and the table and name
      checks were added.
  - 1187 tests pass, 99.91% coverage.

## Checks every session

`uv run pytest --cov`; `uv run ruff check .` and `uv run ruff format --check .`; `uv run mypy`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Holdout data used through a stress window | Medium, irreversible | `resolve_end` guard on `window`/`replay`; real data only for 2008; the text scan; 2020/2022 tests on synthetic data |
| FX sign inverted in the CAD shocks | Medium, silent | `cad_move` with exact tests; synthetic and replay sign tests; a mutation |
| Partial shocks read as forecasts | Medium | Docstring caveats: linear, one variable, fx held at 0; the 2008 natural-hedge numbers in NOTES |
| Beta sample drives the answer | Certain (USD minimum variance 0.72 vs 0.87) | One stated window; full-sample sensitivity reported |
| Stale weights read as a realized loss | Medium | Replay and realized path labelled separately (decision 2) |
| A window silently shortened by missing data | Low | Coverage check; days, first and last day reported |
| NaN assets (49 industries) in a replay | Certain in 7b | `simulate` rejects NaN; 7b's NaN policy (carried over); `portfolio_excess` drops zero weights |

## Considered and rejected

- **Constant weights in the replay:** it is less realistic and differs by ≤ 0.19 pp in 2008
  (decision 2).
- **[start, end] windows:** they put the COVID peak day's gain into the loss (decision 3).
- **Regressing the strategy's realized returns for the shocks:** it describes past
  portfolios, not the current one (decision 7).
- **Σ w_i β_i from asset betas:** the same estimate with no SE (decision 7).
- **Total macro betas:** 5b's decision 6. They give a rates shock the wrong sign.
- **Correlated or combined scenarios:** there is no beta covariance to give them an SE, and
  the historical replay covers joint moves (decision 8).

## Grounding 2026-10-09 (dev data, method sensitivity only)

- **DEV_END weights:**
  - 1/N is about 8.3% each.
  - Minimum variance (sample) is 76% NoDur and 24% Hlth in USD, and 70% NoDur, 22% Hlth and
    7% Utils in CAD.
- **2008 window (63 days, 2008-09-02 → 2008-11-28):**
  - **Buy-and-hold vs constant weights:** the P&L differs by 0.04-0.19 pp and the max
    drawdown by ≤ 0.4 pp. The worst day is the same.
  - **Currency:**
    - The USD rose 16.3% against the CAD.
    - Market: −30.6% in USD, −19.3% in CAD.
    - 1/N replay: −30.3% in USD, −19.0% in CAD, −30.1% in the hedged view. The hedged view is
      about the USD loss, as CIP implies.
- **CAD weekly partial betas of the DEV_END weights (2001-01-03 → 2010-12-31, 523 weeks):**

  | Weights | Mkt-RF | Mom | fx | dy (SE) |
  |---|---|---|---|---|
  | 1/N | 0.995 | −0.011 | 0.971 | −0.10 (0.15) |
  | Minimum variance | 0.751 | +0.040 | 0.992 | −1.49 (0.39) |

  - The constant-mix regression and Σ w_i β_i agree within 0.01 on every term.
- **USD weekly Mkt-RF, 2001+ vs 1963+:**
  - 1/N: 0.994 vs 0.993.
  - Minimum variance: 0.719 (0.029) vs 0.874 (0.016).
  - The Mom betas are −0.01 to +0.04 in every case.
