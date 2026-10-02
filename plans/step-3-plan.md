# Step 3 plan: covariance + portfolio construction

Approved by the owner on 2026-10-02 (session 1 of 4). Read with the blueprint's Step 3 section.
Branch `step-3-construct`.

## What gets built

- `construct/covariance.py`: covariance matrices from a returns window: sample; Ledoit-Wolf
  shrinkage toward a scaled identity (2004, *JMVA*) and toward constant correlation (2004, *JPM*),
  intensity clipped to [0, 1]; EWMA (cut-able).
- `construct/weights.py`: long-only weight rules: equal weight; minimum variance (sum 1, optional
  max-weight cap, SLSQP with analytic gradient); equal risk contribution (convex log-barrier, then
  normalize); momentum 12-1 and low-volatility signals; a tilt of a base portfolio toward a signal.
- `construct/strategy.py`: the contract Step 4's engine calls.
- Each function uses only the window it is given; every formula has a hand-computed or textbook
  test (invariant 6); synthetic and dev data only.

## Decisions (owner, 2026-10-02)

1. **Strategies declare their history.** `Strategy(name, lookback, rule)`: the engine passes the
   returns history ending at the estimation date t, at least `lookback` rows; the strategy uses only
   its last `lookback` rows. Momentum with a 126-day covariance window has
   `lookback = max(126, 252) = 252`. Step 4: `window` moves out of `run(...)` into the strategy, and
   the first rebalance waits for `lookback` rows (update Step 4's `run(...)` line and the change log).
2. **Minimum-variance cap default: none (1.0)**, i.e. the shortsale-constrained minimum-variance
   portfolio of DeMiguel et al. (2009). The cap stays a parameter; capped variants are a 7a
   pre-registration choice.
3. **Independent check for the constant-correlation estimator:** PyPortfolioOpt's implementation
   (MIT), vendored into `tests/` with its notice from a pinned PyPI release (one download, asked
   first). Its sample matrix uses 1/(T-1) while its pi and theta average with 1/T; the internal
   function takes the sample matrix as an argument so the test can feed it that matrix and require
   an exact match.

## Conventions

- scikit-learn is a test-only dev dependency (oracle for the identity target, to 1e-10).
- Covariance = demeaned cross-products / T everywhere (both Ledoit-Wolf papers, scikit-learn), so
  intensity 0 returns the sample matrix exactly. Minimum-variance and ERC weights are invariant to
  scaling the matrix (tested), so this cannot change a strategy.
- No silent estimator default: minimum-variance and ERC strategies take the estimator explicitly.
- Tilt: `w_i ∝ base_i * exp(strength * z_i)`, z = cross-sectional z-score of the signal (momentum:
  12-1 return; low-vol: -sigma). Strength 0 or equal signals return the base. `strength` is
  required (value pre-registered in 7a); base defaults to equal weight.
- Momentum 12-1 = prod(1 + r) over rows `[-252, -21)` of the history, minus 1 (close of t-252 to
  close of t-21: 231 returns).
- EWMA kept: lambda = 0.94, zero mean (RiskMetrics), weights normalized over the window; first to
  cut if session 1 runs long.
- Windows must be complete (no NaN); the 49-industry universe stays in 7b.
- Dev-data checks look at weights and estimator diagnostics only, never performance (Steps 4/7a).

## Sessions

**Session 1 (max effort): covariance.** Plan file, scikit-learn. Tests first, then code:
- `sample_cov`: hand example; equals `np.cov(ddof=0)`.
- `ledoit_wolf_identity` -> `Shrunk(cov, target, intensity)`: matches scikit-learn `LedoitWolf` to
  1e-10, incl. T < N; N = 1 and S ∝ I give intensity 0; condition number never worse.
- `ledoit_wolf_constant_correlation`, five independent checks: (a) docstring hand example with
  rational square roots, redone in exact fractions; (b) exact match with the vendored PyPortfolioOpt
  function under its conventions; (c) a loop-by-loop transcription of the paper vs the vectorized
  code; (d) properties: intensity in [0, 1], equal sample correlations -> intensity 0 and S, target
  diagonal = variances and target correlations = rbar, symmetric PSD, permutation equivariance,
  common rescaling leaves the intensity unchanged; (e) a planted wrong rho term must fail the tests.
- `ewma_cov`: hand example (3 periods, lambda = 0.5).
- Fresh-eyes review: a separate subagent checks the code against the papers' formulas.

**Session 2 (extra effort): optimizers.**
- `min_variance(cov, *, cap=1.0)`: SLSQP on the matrix scaled to unit mean variance, analytic
  gradient, bounds [0, cap], sum 1; raises if cap * N < 1 or the solver fails. Tests: the 2-asset
  closed form; a corner case outside [0, 1]; a binding cap; KKT conditions on random PD matrices
  (assets strictly inside the bounds share one marginal variance, zero-weight ones are higher,
  capped ones lower); `check_grad`; scale invariance.
- `equal_risk_contribution(cov)`: minimize 0.5 y'Σy - (1/N) Σ ln y_i by damped Newton (Spinu
  2013), w = y / Σy; raises for a non-PD matrix. Tests: risk contributions equal within 1e-10;
  uncorrelated -> ∝ 1/sigma; 2 assets -> w1 = sigma2 / (sigma1 + sigma2) for any correlation; scale
  invariance; permutations.

**Session 3 (extra effort): signals, tilts, strategies, dev-data checks.**
- `momentum_12_1`, `low_volatility`, `tilt` with hand examples (incl. a big move in the last 21
  days that momentum must ignore).
- `Strategy` and constructors (equal weight, min variance, ERC, momentum tilt, low-vol tilt). Tests:
  weights depend only on the last `lookback` rows; too-short history raises; output >= 0, sums to
  1, labeled by the history's columns.
- Dev data, 12 industries, weights only: LW intensities over time, condition numbers, min-var
  concentration, ERC vs 1/sigma, time per call (Step 4 runs all strategies in < 30 s). NOTES.md.

**Session 4 (extra effort): review and merge.** python-reviewer -> `/code-review` ->
verification-loop; fix findings. Invariants 1-6; blueprint status + change log (incl. the Step 4
`window` change); `/save-session`; ask the owner before pushing and opening the PR.

## Checks every session

`uv run pytest --cov`; `uv run ruff check .` and `uv run ruff format --check .`; `uv run mypy`.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| A misread constant-correlation formula passes its own test | Medium | Five checks, incl. an implementation written by someone else |
| SLSQP stops early on tiny daily variances | High without scaling | Scale the matrix; tight tolerance; KKT tests |
| The strategy contract changes Step 4's `run(...)` | Certain (by design) | Change log; contract tested here |
| ERC fails on a near-singular matrix (49 industries, 126 days) | Low | Clear error; shrinkage estimators are the fix |
| Convention mismatches with references | Medium | One convention in the docstring; tests adapt references explicitly |
