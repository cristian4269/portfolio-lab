"""Long-only weight rules (pure functions): equal weight, minimum variance, equal risk contribution.

Each rule returns weights as a Series labeled by the assets, >= 0 and summing to 1. The two
covariance rules take a covariance matrix from any estimator; only its shape matters, not its
scale (daily or annual, divided by T or T - 1: the same weights). The matrix must be numerically
positive definite, so a sample matrix from fewer periods than assets raises: shrinkage is the fix.

Minimum variance: min w'Sw subject to sum_i w_i = 1 and 0 <= w_i <= cap. With cap = 1 this is
the shortsale-constrained minimum-variance portfolio of DeMiguel, Garlappi and Uppal (2009).
SLSQP with the analytic gradient works on S divided by its mean variance: daily variances near
1e-4 would otherwise meet SLSQP's absolute tolerance after a step or two. SLSQP stops about 1e-7
from the optimum but finds its active set (the assets at 0 and at the cap), which is then
polished: with those assets fixed, the free ones solve the equality-constrained problem exactly,
a linear system. The polished weights are kept if they are feasible and satisfy the KKT
conditions, which prove them optimal for this convex problem: every asset that could be sold
has a marginal variance (Sw)_i no higher than every asset that could be bought. Otherwise
SLSQP's weights are returned.

Equal risk contribution (ERC; Maillard, Roncalli and Teiletche 2010): the risk contributions
w_i (Sw)_i are all equal. Spinu (2013): with C the correlation matrix, the minimizer y of the
convex function f(y) = y'Cy / 2 - sum_i ln y_i satisfies y_i (Cy)_i = 1, so w_i proportional
to y_i / sigma_i has equal risk contributions, and the long-only ERC portfolio is unique.
(Spinu's budgets b_i sum to 1; b_i = 1 gives the same weights and makes f standard
self-concordant.) Newton's method for self-concordant functions (Nesterov 2004, ch. 4), as Spinu
runs it: from the best multiple of equal weights, damped steps (Newton step) / (1 + lambda),
lambda the Newton decrement, while lambda >= 0.95 (3 - sqrt 5) / 2; full steps after.

Hand examples (percent units, as in the tests):
- 2 assets, sd 2% and 3%, covariance 1: w1 = (s2^2 - s12) / (s1^2 + s2^2 - 2 s12) = 8/11.
- Uncorrelated, variances 1, 4, 9: minimum variance is proportional to 1/sigma^2, (36, 9, 4) /
  49; with cap = 1/2 it is (1/2, 9/26, 4/26), marginal variances 1/2 for the capped asset and
  18/13 for the free ones. ERC is proportional to 1/sigma: (6, 3, 2) / 11.
- sd 1%, 2%, 4%, correlations 1/2 (a with b and with c) and -1/16 (b with c): ERC weights
  (1/2, 1/3, 1/6), with Sw = (7/6, 7/4, 7/2) and every risk contribution 7/12.
"""

import math
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_lab._validate import as_floats

__all__ = ["equal_risk_contribution", "equal_weight", "min_variance"]

# Rounding: N caps may fall short of 1 by this much (49 * (1/49) < 1); polished weights may miss
# their bounds and their sum by this much; cov and cov' may differ by this much (relative).
_ROUNDING = 1e-12
# A correlation matrix whose smallest eigenvalue is below this fraction of its largest is
# singular for our purposes: the weights would be dominated by rounding.
_SINGULAR = 1e-10
# Polishing: weights this close to a bound count as at it (SLSQP stops about 1e-7 away); the
# KKT conditions must hold to this tolerance (marginal variances over the mean variance).
_ACTIVE = 1e-6
_KKT = 1e-9
# Newton for ERC: damped steps while the decrement is at least _QUADRATIC; once it is below
# _DONE, one more full step squares it (quadratic convergence) and the search stops.
_QUADRATIC = 0.95 * (3 - math.sqrt(5)) / 2
_DONE = 1e-8
_NEWTON_STEPS = 100


def equal_weight(assets: Sequence[Hashable] | pd.Index) -> pd.Series:
    """1/N in each asset."""
    index = pd.Index(assets)
    if len(index) == 0:
        raise ValueError("equal weight needs assets, got no assets")
    if not index.is_unique:
        raise ValueError("assets must be unique")
    return pd.Series(1 / len(index), index=index)


def min_variance(cov: pd.DataFrame, *, cap: float = 1.0) -> pd.Series:
    """Long-only minimum variance, each weight at most `cap` (1 = no cap)."""
    s = _covariance(cov)
    n = len(s)
    if not 0 < cap <= 1:
        raise ValueError(f"cap must be in (0, 1], got {cap}")
    if cap * n < 1 - _ROUNDING:
        raise ValueError(f"a cap of {cap} cannot hold {n} assets: cap * N must be at least 1")
    a = s / np.mean(np.diag(s))
    result = minimize(
        _objective,
        np.full(n, 1 / n),
        args=(a,),
        jac=_gradient,
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"minimum variance: SLSQP failed ({result.message})")
    return pd.Series(_polish(a, result.x, cap), index=cov.index)


def equal_risk_contribution(cov: pd.DataFrame) -> pd.Series:
    """Long-only weights whose risk contributions w_i (Sw)_i are equal (Spinu's Newton method)."""
    s = _covariance(cov)
    sd = np.sqrt(np.diag(s))
    c = s / np.outer(sd, sd)
    y = np.full(len(c), math.sqrt(len(c) / c.sum()))  # minimizes f along the equal-weight ray
    for _ in range(_NEWTON_STEPS):
        gradient = c @ y - 1 / y
        step = np.linalg.solve(c + np.diag(1 / y**2), gradient)
        decrement = math.sqrt(max(float(gradient @ step), 0.0))
        if decrement < _DONE:
            y = y - step
            break
        y = y - step / (1 + decrement) if decrement >= _QUADRATIC else y - step
    else:
        raise RuntimeError(
            f"equal risk contribution: Newton's method did not converge in {_NEWTON_STEPS} steps"
        )
    w = y / sd
    return pd.Series(w / w.sum(), index=cov.index)


def _objective(w: np.ndarray, a: np.ndarray) -> float:
    return 0.5 * float(w @ a @ w)


def _gradient(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    return a @ w


def _polish(a: np.ndarray, w: np.ndarray, cap: float) -> np.ndarray:
    """The exact optimum on the active set of approximate weights `w`, if it is one.

    Assets within _ACTIVE of a bound stay at it; the others minimize x'ax / 2 subject to the
    budget, a linear system. That result is returned if it is feasible and satisfies the KKT
    conditions; otherwise `w`, clipped to the bounds and rescaled to sum to 1.
    """
    capped = w >= cap - _ACTIVE
    free = (w > _ACTIVE) & ~capped
    x = np.where(capped, cap, 0.0)
    if free.any():
        f, c = np.flatnonzero(free), np.flatnonzero(capped)
        k = len(f)
        system = np.block([[a[np.ix_(f, f)], np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
        x[f] = np.linalg.solve(system, np.append(-a[np.ix_(f, c)] @ x[c], 1 - x.sum()))[:k]
    feasible = (
        x.min() >= -_ROUNDING and x.max() <= cap + _ROUNDING and abs(x.sum() - 1) <= _ROUNDING
    )
    marginal = a @ x
    sell = marginal[x > 0].max(initial=-np.inf)
    buy = marginal[x < cap].min(initial=np.inf)
    if feasible and sell <= buy + _KKT:
        return np.clip(x, 0.0, cap)
    w = np.clip(w, 0.0, cap)
    return w / w.sum()


def _covariance(cov: pd.DataFrame) -> np.ndarray:
    """The matrix as floats, symmetrized, after checking that it is a usable covariance matrix.

    Square, labeled by the same unique assets in the same order on both axes, finite,
    symmetric, with positive variances and a numerically positive definite correlation matrix.
    """
    if not isinstance(cov, pd.DataFrame):
        raise TypeError(f"cov must be a DataFrame labeled by the assets, got {type(cov).__name__}")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    if not cov.index.equals(cov.columns):
        raise ValueError("cov must have the same assets, in the same order, as rows and columns")
    if len(cov) == 0:
        raise ValueError("cov has no assets")
    if not cov.index.is_unique:
        raise ValueError("cov needs unique asset names")
    s = as_floats(cov, "cov")
    if not np.isfinite(s).all():
        raise ValueError("cov must be finite")
    if np.abs(s - s.T).max() > _ROUNDING * np.abs(s).max():
        raise ValueError("cov must be symmetric")
    s = (s + s.T) / 2
    variances = np.diag(s)
    if (variances <= 0).any():
        asset = cov.index[int(np.argmax(variances <= 0))]
        raise ValueError(f"{asset} has no positive variance")
    sd = np.sqrt(variances)
    eigenvalues = np.linalg.eigvalsh(s / np.outer(sd, sd))
    if eigenvalues[0] <= _SINGULAR * eigenvalues[-1]:
        raise ValueError(
            "cov is not positive definite (singular or nearly so, e.g. a sample matrix from "
            "fewer periods than assets): use a shrinkage estimator"
        )
    return s
