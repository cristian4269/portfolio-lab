# Step 4 plan: walk-forward backtest engine

Approved by the owner as drafted on 2026-10-08 (session 1 of 4). Read with the blueprint's Step 4
section.
Branch `step-4-backtest`.

## What gets built

- `backtest/engine.py`: monthly walk-forward simulation of one `Strategy` on one currency view.
  - It estimates at the last trading day t of each month, trades at close t+1, and lets weights
    drift with returns between trades.
  - It produces daily gross and net returns, target weights by trade date, turnover and costs,
    plus `Result.summary()` built on the step-2 metrics.
- `backtest/benchmarks.py`: the market (Mkt-RF + RF) in each view. The 1/N benchmark is
  `equal_weight()` run through the engine like any other strategy.
- Every formula has a hand-computed test (invariant 6). Tests use synthetic data, and dev data
  is used only for timing and sanity checks.

## Decisions for the owner (recommendation first)

1. **`run` takes a step-2 `View`, not `returns` + `view="usd"`.**
   - Signature: `run(strategy, view, *, start=None, end=DEV_END, allow_holdout=False,
     cost_bps=5.0) -> Result`.
   - Why a `View`:
     - It carries the matching risk-free rate, which Sharpe needs.
     - Strategies estimate on the view's own returns, so the pre-registered rule "CAD-view
       strategies estimate on CAD returns" holds by construction.
     - The engine never loads data itself.
   - Monthly only, so `rebalance` leaves the signature (the study fixes monthly; adding
     quarterly later would be one line).
2. **The backtest starts invested in the first target portfolio, with no opening cost.**
   - Turnover and costs count from the second trade on.
   - The opening trade would cost every long-only strategy exactly the same `cost_bps`
     (Σ|w − 0| = 1). On a common start it cannot change a comparison, and it would make the first
     row a cost-only day.
   - Alternative: charge it as a day-τ1 return of −cost.
3. **Start dates fail closed.**
   - `start=None` means the first month-end where the strategy has `lookback` rows.
   - A given `start` too early for the lookback raises an error naming the earliest valid date.
     The engine never silently moves it.
   - `compare(strategies, view, *, start=None, ...)` runs every strategy from one common first
     rebalance: the latest of their earliest dates (the largest lookback). 1/N has lookback 1, so
     on its own it would start sooner.
4. **The market benchmark is Mkt-RF + RF, with no costs and no rebalancing (an index).**
   - It is converted to the CAD views by the step-2 functions, by calling `make_views` on a
     one-column frame, so no new FX code.
   - The 1/N benchmark pays costs and rebalances monthly like the strategies.
5. **No SLSQP warm start in Step 4.**
   - Instead, estimation and accounting are split: `targets(strategy, returns, start)` computes
     the weights once, and `simulate(targets, returns, cost_bps)` prices them. Step 7b's cost
     grid (0/5/10/20 bp) then reuses the weights.
   - Weights alone took 15 s for 12 strategies in Step 3, inside the 30 s budget.
   - A warm start would make weights depend on the previous rebalance and change Step 3's API.
     Revisit in 7a only if the pre-registered variant list breaks the budget.
   - The exit's "under 30 s" is read per view: the 12 strategies of Step 3's timing check on the
     USD view. The CAD view's time is reported too.
6. **Step 4 looks at dev performance only for sanity checks.**
   - Checks: the accounting reconciles; costs reduce returns; monthly 1/N is close to Step 2's
     daily-rebalanced 1/N.
   - No strategy-vs-strategy table yet: the dev analysis is recorded in 7a's draft PLAN.md.

## Conventions

- **Calendar.**
  - Rebalance dates t are the last row of each calendar month in the view's index that has a
    next row. The trade date τ is that next row.
  - The strategy gets `returns.iloc[: pos(t) + 1]`, every row up to and including t.
  - The trading calendar counts as known in advance; returns do not.
- **Accounting.** On each day s, with w the weights at the close of s − 1:
  - The day's return is R_s = Σ w_i r_{i,s}.
  - Weights drift to w_i (1 + r_{i,s}) / (1 + R_s).
  - On a trade day τ, the old drifted weights earn that day's return. Then, at the close:
    - c = cost_bps / 1e4 × Σ|w* − w_τ|
    - turnover = ½ Σ|w* − w_τ|
    - net_τ = (1 + R_τ)(1 − c) − 1
  - After the trade, the weights are exactly w*. The cost is paid pro rata from the new
    holdings. The cost of selling to pay the cost is ignored, a second-order term (cost × cost
    rate, about 1e-7 for a typical month).
  - Gross returns follow the same weight path with c = 0, so net = gross except on trade days,
    and net ≤ gross there.
- **Vectorized, then checked.**
  - Each holding period uses cumulative products: holdings w_i ∏(1 + r_i), with portfolio value
    their sum. Drift error therefore never accumulates past one month.
  - A plain day-by-day loop in the tests must agree to 1e-12 on random data.
- **Inputs.**
  - `cost_bps` must be finite and ≥ 0.
  - Returns on the simulated days must be complete, finite and ≥ −100%. Rows before the first
    trade may have gaps, since the strategy checks its own window.
  - Rows after `end` are dropped before anything else, by the loaders' rule: `_resolve_end`
    becomes the public `data.resolve_end`, one function for the holdout. The `allow_holdout=True`
    scan already covers the engine.
- **`Result`** (frozen, like `View`):
  - Fields: `name`, `gross`, `net`, `rf` (the view's, on the same days), `weights` (targets by
    trade date), `turnover`, `cost` (fraction of NAV, by trade date), `final_weights` (drifted,
    at the last close; Step 6's "current weights").
  - `summary(benchmark=None)`: `metrics.summary` over gross and net (daily, 252), plus mean
    turnover per rebalance, annual turnover and annual cost. A longer benchmark is cut to the
    result's days and must cover them.
- **Hand examples, in exact fractions.** Two assets start at (½, ½).
  - Day 1: returns (+10%, −10%) give R = 0 and weights (0.55, 0.45).
  - Day 2: returns (0, +10%) give R = 4.5% and weights (10/19, 9/19).
  - Day 3 is a trade day: returns (−10%, 0) give R = −1/19, and the drifted weights are (½, ½).
  - The day-3 trade to (1, 0) gives Σ|Δ| = 1, turnover ½ and cost 5e-4 at 5 bp, so
    net = (18/19)(1 − 5e-4) − 1.

## Sessions

**Session 1 (max effort): plan, then the accounting core.** Tests first, then code:
- `rebalance_dates`:
  - month ends, holidays at month end, a month with one trading day
  - the last row is never a rebalance
  - a given `start` maps to the first month-end on or after it
- `simulate` with hand-built targets:
  - the 3-day example above, exact
  - the turnover and cost example
  - zero cost = gross, exactly
  - net ≤ gross on trade days, equal elsewhere
  - t+1 execution: asset A gains +50% on τ while the target moves into A. The old weights earn
    the jump; trading at t would capture it.
  - Hypothesis: the plain loop and the vectorized engine agree on random returns and targets
- RED and GREEN commits.
- As built (2026-10-08, see NOTES.md):
  - `rebalance_dates` returns a Series from estimation date to trade date.
  - `simulate` returns a `Ledger`, and net = gross − c(1 + gross), so zero cost gives gross
    exactly.
  - A portfolio worth 0 raises.
  - The mutation run caught all 15 planted bugs.

**Session 2 (max effort): strategies in the loop.** `targets`, `run`, `Result`, `compare`,
benchmarks and the guard.
- No look-ahead, three ways:
  - A recording strategy checks that every history ends at a month-end t and that its trade is
    the next row.
  - Hypothesis perturbation (blueprint): changing any returns after day k never changes a
    target estimated at or before k, nor any net return through k.
  - The same perturbation through a real minimum-variance strategy.
- Start rules:
  - too-early `start` raises
  - `start=None` waits for `lookback` rows
  - `compare` gives every strategy the same first trade
- Guard:
  - `end` past DEV_END raises `HoldoutError`
  - rows after `end` never reach a strategy
- Benchmarks:
  - market = Mkt-RF + RF, by hand
  - the CAD market equals `to_cad_unhedged` of the USD market
- `summary` rows by hand on a short series.

**Session 3 (extra effort): dev data, timing, sanity.**
- The 12 strategies of Step 3's timing check on the USD view (1963-07 on) and the CAD view
  (1971 on). Time per view: exit < 30 s.
- Sanity checks (no comparison table, decision 6):
  - monthly 1/N vs Step 2's daily 1/N
  - gross − net = the cost drag
  - 1/N turnover per month
  - min-variance turnover vs 1/N's
  - weights match `strategy.weights` on spot dates
- Mutation run on the engine: shift t by one row, earn τ with the new weights, drop the drift,
  charge the cost before the return, use `≤ t + 1` in the history slice, and so on.
- NOTES.md entry.

**Session 4 (extra effort): review and merge.**
- python-reviewer → `/code-review` → verification-loop, then fix the findings.
- Check invariants 1-6. Update the blueprint status and change log (incl. the signature
  change), then `/save-session`.
- Ask the owner before pushing and opening the PR.

## Checks every session

`uv run pytest --cov`; `uv run ruff check .` and `uv run ruff format --check .`; `uv run mypy`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Off-by-one in timing (trade at t, or τ's return earned by the new weights) | Medium, high impact | t+1 hand test, recording strategy, loop reference, mutation run |
| Look-ahead through the history slice | Low, fatal | Perturbation property tests; the recording strategy sees no row after t |
| Strategies compared from different first dates | Medium | `compare` uses one start; a too-early `start` raises |
| 7a's variants break the 30 s budget | Medium | `targets`/`simulate` split; warm start kept for 7a |
| Holdout reaches the engine | Low | The loaders' `resolve_end` in `run`; the scan test |
| Float drift over 12,000 days | Low | Drift restarts at each trade; loop agrees to 1e-12 |
