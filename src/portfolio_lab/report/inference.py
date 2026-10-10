"""Sharpe-ratio inference for the pre-registered study (blueprint Step 7a, research/PLAN.md §5).

Ledoit and Wolf (2008), "Robust performance hypothesis testing with the Sharpe ratio", Journal
of Empirical Finance 15(5), test H0: SR1 = SR2 for two return series observed on the same
periods. Where the paper is silent, this follows the authors' own R code (CRAN package
`bootTimeInference`).

- **Statistic:** Δ = f(v) with v = (μ1, μ2, γ1, γ2), γ = E[r²] and
  f = μ1/√(γ1 − μ1²) − μ2/√(γ2 − μ2²). The Sharpe ratios use population moments (divide by T),
  as f does, so they are √(T/(T − 1)) times the ddof = 1 ratios of `report.metrics`.
- **Standard error:** s(Δ̂) = √(∇f' Ψ ∇f / T), with Ψ the long-run covariance of
  y_t = (r1 − μ1, r2 − μ2, r1² − γ1, r2² − γ2), estimated by `hac_psi`:
  - VAR(1) prewhitening without intercept (Andrews and Monahan 1992), its matrix A refit
    with singular values capped at 0.97;
  - a Parzen kernel on the n = T − 1 residuals, bandwidth S = 2.6614 (α n)^(1/5), with α
    Andrews' (1991) AR(1) plug-in pooled over the columns with equal weights;
  - Ψ_e = n/(n − 4) Σ_{|j| < S} k(j/S) Γ(j), Γ(j) = (1/n) Σ e_t e'_{t−j}, recoloured to
    Ψ = (I − A)⁻¹ Ψ_e (I − A)⁻¹'.
- **Bootstrap** (`lw_test`): a circular block bootstrap of the return pairs. Each resample has
  l = ⌊T/b⌋ blocks of b rows from uniform starts, wrapping around. It is studentized by its
  own block standard error: y* centred at the resample's moments,
  ζ_j = b^(−1/2) Σ_{block j} y*, Ψ* = (1/l) Σ ζ_j ζ_j', and the gradient at the resample's
  moments, over the original T. With d = |Δ̂ − Δ0|/s(Δ̂) and d*_m = |Δ̂*_m − Δ̂|/s(Δ̂*_m), the
  p-value is (#{d*_m ≥ d} + 1)/(M + 1); the symmetric interval is Δ̂ ± q s(Δ̂), with q the
  `level` quantile of the d*.
- **Block size** (`calibrate_block`, the paper's Algorithm 3.1): fit a VAR(1) with intercept
  to the pair, build pseudo-series from its residuals resampled by a stationary bootstrap
  (geometric blocks of mean 5, a burn-in of 50, started at the first observation), test
  H0: Δ = Δ̂ on each with every block size, and keep the size whose rejection rate at `alpha`
  is closest to `alpha` (ties to the smaller block).

Also the Jobson-Korkie test with Memmel's (2003) correction (iid normal returns), Holm's step-
down adjustment, the minimum detectable difference, DeMiguel et al.'s certainty equivalent and
the monthly excess returns the tests take.

Every random draw comes from the `rng` passed in, so a seeded generator makes a result
repeatable. The functions are pure: guarding the holdout is the caller's job.
"""

import math
from collections.abc import Sequence
from numbers import Real
from typing import NamedTuple, overload

import numpy as np
import pandas as pd
from scipy import stats

from portfolio_lab._validate import as_floats, check_aligned, check_dates, is_whole

__all__ = [
    "BLOCKS",
    "Calibration",
    "SharpeTest",
    "ZTest",
    "calibrate_block",
    "ceq",
    "diff_se",
    "hac_psi",
    "holm",
    "jobson_korkie_memmel",
    "lw_test",
    "min_detectable_difference",
    "monthly_excess",
    "sharpe_diff",
    "sharpe_ratio",
]

BLOCKS = (1, 2, 4, 6, 8, 10, 12)  # the calibration grid (research/PLAN.md §5)
RESAMPLES = 4999
PSEUDO = 1000
CALIBRATION_RESAMPLES = 499
MEAN_BLOCK = 5
BURN_IN = 50

_PARZEN = 2.6614  # Andrews (1991): the Parzen kernel's bandwidth constant
_MAX_SINGULAR = 0.97  # Andrews and Monahan (1992): keep I - A invertible
_PARAMS = 4  # the small-sample factor n / (n - 4): v has 4 parameters
_MIN_PERIODS = 6  # n = T - 1 residuals must exceed the 4 parameters
_CHUNK = 1000  # resamples per vectorized batch (about 6 MB at T = 200)
_DEGENERATE = 1e-12  # theta of an O(1) quantity below this is rounding error


class SharpeTest(NamedTuple):
    """Ledoit-Wolf's test of SR1 = SR2 (+ the null value), with its symmetric interval."""

    diff: float
    se: float
    pvalue: float
    ci_low: float
    ci_high: float
    block: int
    resamples: int


class Calibration(NamedTuple):
    """The chosen block size and the rejection rate of each block size tried."""

    block: int
    rejection: pd.Series


class ZTest(NamedTuple):
    z: float
    pvalue: float


# --- Sharpe ratios ----------------------------------------------------------------------------


def sharpe_ratio(r: pd.Series) -> float:
    """μ/σ with population moments, per period (no annualization)."""
    x = _series(r, "returns")
    return float(x.mean() / _sd(x, "returns"))


def sharpe_diff(r1: pd.Series, r2: pd.Series) -> float:
    """SR1 − SR2 on the same periods (Ledoit-Wolf's Δ̂)."""
    x1, x2 = _pair(r1, r2)
    return float(x1.mean() / _sd(x1, "r1") - x2.mean() / _sd(x2, "r2"))


def diff_se(r1: pd.Series, r2: pd.Series) -> float:
    """The HAC standard error of SR1 − SR2 (prewhitened Parzen, the delta method)."""
    x1, x2 = _pair(r1, r2)
    return _se(x1, x2)


def hac_psi(y: np.ndarray) -> np.ndarray:
    """The prewhitened Parzen HAC estimate of the long-run covariance of the rows of `y`.

    `y` is a T x p matrix of centred moment conditions, one row per period (the module doc
    has the steps). Returns a symmetric p x p matrix.
    """
    x, z = y[:-1], y[1:]
    a = _cap_singular(np.linalg.lstsq(x, z, rcond=None)[0].T)
    e = z - x @ a.T
    bandwidth = _PARZEN * (_andrews_alpha(e) * len(e)) ** 0.2
    recolour = np.linalg.inv(np.eye(y.shape[1]) - a)
    psi = recolour @ _kernel_psi(e, bandwidth) @ recolour.T
    return (psi + psi.T) / 2


# --- the Ledoit-Wolf bootstrap ----------------------------------------------------------------


def lw_test(
    r1: pd.Series,
    r2: pd.Series,
    *,
    block: int,
    rng: np.random.Generator,
    resamples: int = RESAMPLES,
    null: float = 0.0,
    level: float = 0.95,
) -> SharpeTest:
    """Ledoit and Wolf's studentized circular block bootstrap test of SR1 − SR2 = `null`."""
    x1, x2 = _pair(r1, r2)
    _check_block(block, len(x1))
    _check_count(resamples, "resamples")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if np.array_equal(x1, x2):
        raise ValueError("the two series are identical: their Sharpe difference has no variance")
    diff = float(x1.mean() / _sd(x1, "r1") - x2.mean() / _sd(x2, "r2"))
    se = _se(x1, x2)
    d_star = _bootstrap_d(x1, x2, block, resamples, rng, diff)
    q = float(np.quantile(d_star, level))
    return SharpeTest(
        diff=diff,
        se=se,
        pvalue=_p_value(abs(diff - null) / se, d_star),
        ci_low=diff - q * se,
        ci_high=diff + q * se,
        block=int(block),
        resamples=int(resamples),
    )


def calibrate_block(
    r1: pd.Series,
    r2: pd.Series,
    *,
    rng: np.random.Generator,
    blocks: Sequence[int] = BLOCKS,
    pseudo: int = PSEUDO,
    resamples: int = CALIBRATION_RESAMPLES,
    alpha: float = 0.05,
    mean_block: float = MEAN_BLOCK,
    burn_in: int = BURN_IN,
) -> Calibration:
    """Ledoit and Wolf's Algorithm 3.1: the block size whose test has the right size here."""
    x1, x2 = _pair(r1, r2)
    t = len(x1)
    grid = _check_grid(blocks, t)
    _check_count(pseudo, "pseudo")
    _check_count(resamples, "resamples")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    target = float(x1.mean() / _sd(x1, "r1") - x2.mean() / _sd(x2, "r2"))
    series = _pseudo_series(
        np.column_stack([x1, x2]), rng, pseudo=pseudo, mean_block=mean_block, burn_in=burn_in
    )
    rejected = np.zeros(len(grid))
    for k in range(pseudo):
        s1, s2 = series[k, :, 0], series[k, :, 1]
        diff = float(_sr(s1.mean(), (s1 * s1).mean()) - _sr(s2.mean(), (s2 * s2).mean()))
        d = abs(diff - target) / _se(s1, s2)
        for i, size in enumerate(grid):
            d_star = _bootstrap_d(s1, s2, size, resamples, rng, diff)
            rejected[i] += _p_value(d, d_star) <= alpha
    rates = pd.Series(rejected / pseudo, index=pd.Index(grid, name="block"), name="rejection")
    return Calibration(_pick_block(rates, alpha), rates)


# --- secondary measures -----------------------------------------------------------------------


def jobson_korkie_memmel(r1: pd.Series, r2: pd.Series) -> ZTest:
    """Jobson and Korkie's test of SR1 = SR2 with Memmel's (2003) correction.

    z = (SR1 − SR2)/√(θ/T), θ = 2 − 2ρ + ½(SR1² + SR2² − 2 SR1 SR2 ρ²), with population
    moments. It assumes iid normal returns; the p-value is two-sided.
    """
    x1, x2 = _pair(r1, r2)
    s1, s2 = _sd(x1, "r1"), _sd(x2, "r2")
    sr1, sr2 = x1.mean() / s1, x2.mean() / s2
    rho = np.mean((x1 - x1.mean()) * (x2 - x2.mean())) / (s1 * s2)
    theta = 2 - 2 * rho + 0.5 * (sr1**2 + sr2**2 - 2 * sr1 * sr2 * rho**2)
    if not theta > _DEGENERATE:  # rho = 1 with equal Sharpe ratios leaves rounding error only
        raise ValueError("the two series are perfectly correlated: the test is undefined")
    z = float((sr1 - sr2) / math.sqrt(theta / len(x1)))
    return ZTest(z, float(2 * stats.norm.sf(abs(z))))


@overload
def holm(pvalues: pd.Series) -> pd.Series: ...
@overload
def holm(pvalues: Sequence[float] | np.ndarray) -> np.ndarray: ...
def holm(pvalues: pd.Series | Sequence[float] | np.ndarray) -> pd.Series | np.ndarray:
    """Holm's (1979) step-down adjusted p-values, in the order given (labels kept).

    Sorted ascending, p_(k) becomes max_{i <= k} min(1, (m − i + 1) p_(i)); a test rejects at
    α when its adjusted p-value is <= α.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("holm needs one or more p-values in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    if isinstance(pvalues, pd.Series):
        return pd.Series(adjusted, index=pvalues.index, name=pvalues.name)
    return adjusted


def min_detectable_difference(se: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """The smallest true difference a two-sided level-`alpha` test detects with `power`:
    (z_{1−α/2} + z_power) × se, in the units of se."""
    if not (math.isfinite(se) and se >= 0):
        raise ValueError(f"se must be a finite number >= 0, got {se}")
    for value, what in ((alpha, "alpha"), (power, "power")):
        if not 0 < value < 1:
            raise ValueError(f"{what} must be in (0, 1), got {value}")
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def ceq(r: pd.Series, gamma: float = 1.0) -> float:
    """DeMiguel et al.'s certainty-equivalent return per period: mean − γ/2 × variance (ddof 1)."""
    if isinstance(gamma, bool) or not isinstance(gamma, Real) or not math.isfinite(gamma):
        raise ValueError(f"gamma must be a finite number, got {gamma!r}")
    x = _series(r, "returns")
    return float(x.mean() - gamma / 2 * x.var(ddof=1))


def monthly_excess(returns: pd.Series, rf: pd.Series) -> pd.Series:
    """Daily returns compounded within each calendar month, minus the risk-free rate
    compounded the same way: Π(1 + r) − Π(1 + rf). Labelled by each month's last day."""
    for s, what in ((returns, "returns"), (rf, "rf")):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{what} must be a Series of daily returns, got {type(s).__name__}")
    days = check_dates(returns.index, "returns")
    check_aligned(returns, rf, "returns and rf dates")
    months = days.to_period("M")
    growth = (1 + returns.astype(float)).groupby(months).prod()
    carry = (1 + rf.astype(float)).groupby(months).prod()
    last = pd.Series(days, index=days).groupby(months).max()
    return pd.Series(
        (growth - carry).to_numpy(), index=pd.DatetimeIndex(last.to_numpy()), name=returns.name
    )


# --- internals: moments and the HAC ----------------------------------------------------------


def _sr[A: (float, np.ndarray)](mu: A, gamma: A) -> A:
    return mu / np.sqrt(gamma - mu * mu)


def _gradient[A: (float, np.ndarray)](mu1: A, mu2: A, g1: A, g2: A) -> np.ndarray:
    """∇f at v = (μ1, μ2, γ1, γ2): one row per entry when the moments are arrays."""
    v1 = (g1 - mu1 * mu1) ** 1.5
    v2 = (g2 - mu2 * mu2) ** 1.5
    return np.stack([g1 / v1, -g2 / v2, -mu1 / (2 * v1), mu2 / (2 * v2)], axis=-1)


def _moment_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """y_t = (r1 − μ1, r2 − μ2, r1² − γ1, r2² − γ2), one row per period."""
    q1, q2 = x1 * x1, x2 * x2
    return np.column_stack([x1 - x1.mean(), x2 - x2.mean(), q1 - q1.mean(), q2 - q2.mean()])


def _se(x1: np.ndarray, x2: np.ndarray) -> float:
    g = _gradient(x1.mean(), x2.mean(), (x1 * x1).mean(), (x2 * x2).mean())
    quad = float(g @ hac_psi(_moment_matrix(x1, x2)) @ g)
    if not quad > 0:
        raise ValueError("the Sharpe difference has no estimated variance: check the series")
    return math.sqrt(quad / len(x1))


def _parzen(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= 0.5, 1 - 6 * a**2 + 6 * a**3, np.where(a <= 1, 2 * (1 - a) ** 3, 0.0))


def _andrews_alpha(e: np.ndarray) -> float:
    """Andrews' (1991) α(2) from an AR(1) fit (with intercept, after demeaning) per column."""
    num = den = 0.0
    for col in e.T:
        c = col - col.mean()
        design = np.column_stack([np.ones(len(c) - 1), c[:-1]])
        coef, *_ = np.linalg.lstsq(design, c[1:], rcond=None)
        rho = float(coef[1])
        s2 = float(np.mean((c[1:] - design @ coef) ** 2))
        num += 4 * rho**2 * s2**2 / (1 - rho) ** 8
        den += s2**2 / (1 - rho) ** 4
    alpha = num / den if den > 0 else math.nan
    if not math.isfinite(alpha):
        raise ValueError("the bandwidth is undefined: a moment series is a unit root or constant")
    return alpha


def _kernel_psi(e: np.ndarray, bandwidth: float) -> np.ndarray:
    """n/(n − 4) Σ_{|j| < S} k(j/S) Γ(j), Γ(j) = (1/n) Σ_t e_t e'_{t−j}, Parzen weights."""
    n = len(e)
    psi = e.T @ e / n
    j = 1
    while j < bandwidth and j < n:
        g = e[j:].T @ e[:-j] / n
        psi = psi + _parzen(np.array([j / bandwidth]))[0] * (g + g.T)
        j += 1
    return n / (n - _PARAMS) * psi


def _cap_singular(a: np.ndarray) -> np.ndarray:
    u, d, vt = np.linalg.svd(a)
    return u @ np.diag(np.minimum(d, _MAX_SINGULAR)) @ vt


# --- internals: the bootstrap ----------------------------------------------------------------


def _cbb_rows(starts: np.ndarray, block: int, n: int) -> np.ndarray:
    """Row numbers of circular blocks: each start's `block` rows, wrapping past n − 1."""
    rows = (starts[..., None] + np.arange(block)) % n
    return rows.reshape(*starts.shape[:-1], starts.shape[-1] * block)


def _block_psi(y: np.ndarray, block: int) -> np.ndarray:
    """(1/l) Σ_j ζ_j ζ_j', ζ_j = b^(−1/2) Σ_{block j} y, for each resample (first axis)."""
    m, n, p = y.shape
    blocks = n // block
    z = y[:, : blocks * block].reshape(m, blocks, block, p).sum(axis=2) / math.sqrt(block)
    return np.einsum("mlp,mlq->mpq", z, z) / blocks


def _bootstrap_d(
    x1: np.ndarray,
    x2: np.ndarray,
    block: int,
    resamples: int,
    rng: np.random.Generator,
    diff: float,
) -> np.ndarray:
    """|Δ̂* − Δ̂| / s(Δ̂*) for each circular block resample."""
    t = len(x1)
    blocks = t // block
    out = np.empty(resamples)
    for first in range(0, resamples, _CHUNK):
        m = min(_CHUNK, resamples - first)
        rows = _cbb_rows(rng.integers(0, t, size=(m, blocks)), block, t)
        a, b = x1[rows], x2[rows]
        mu1, mu2 = a.mean(axis=1), b.mean(axis=1)
        q1, q2 = a * a, b * b
        g1, g2 = q1.mean(axis=1), q2.mean(axis=1)
        y = np.stack(
            [a - mu1[:, None], b - mu2[:, None], q1 - g1[:, None], q2 - g2[:, None]], axis=-1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = _gradient(mu1, mu2, g1, g2)
            se = np.sqrt(np.einsum("mi,mij,mj->m", grad, _block_psi(y, block), grad) / t)
            out[first : first + m] = np.abs(_sr(mu1, g1) - _sr(mu2, g2) - diff) / se
    if not np.isfinite(out).all():
        raise ValueError(
            "a bootstrap resample has no variance: the series are too short for this block"
        )
    return out


def _p_value(d: float, d_star: np.ndarray) -> float:
    return float((1 + np.count_nonzero(d_star >= d)) / (len(d_star) + 1))


# --- internals: calibration ------------------------------------------------------------------


def _pseudo_series(
    x: np.ndarray, rng: np.random.Generator, *, pseudo: int, mean_block: float, burn_in: int
) -> np.ndarray:
    """`pseudo` series as long as x from its fitted VAR(1): started at x's first row, driven by
    residuals resampled by the stationary bootstrap, the first `burn_in` steps dropped."""
    t = len(x)
    c, b, resid = _var1_fit(x)
    rows = _stationary_rows(
        rng, n=t - 1, mean_block=mean_block, length=burn_in + t - 1, paths=pseudo
    )
    return _var1_paths(c, b, x[0], resid[rows])[:, burn_in:]


def _var1_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS of x_t on (1, x_{t−1}): the intercept c, the matrix B (x_t = c + B x_{t−1} + e_t)
    and the residuals."""
    design = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    coef, *_ = np.linalg.lstsq(design, x[1:], rcond=None)
    return coef[0], coef[1:].T, x[1:] - design @ coef


def _var1_paths(c: np.ndarray, b: np.ndarray, start: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Paths of x_{t+1} = c + B x_t + eps_t from `start`, one per first axis of `eps`."""
    paths, steps, p = eps.shape
    out = np.empty((paths, steps + 1, p))
    out[:, 0] = start
    for t in range(steps):
        out[:, t + 1] = c + out[:, t] @ b.T + eps[:, t]
    return out


def _stationary_rows(
    rng: np.random.Generator, *, n: int, mean_block: float, length: int, paths: int
) -> np.ndarray:
    """Politis and Romano's (1994) stationary bootstrap: blocks of geometric length (mean
    `mean_block`) from uniform starts on a circle of n rows, `length` rows per path."""
    out = np.empty((paths, length), dtype=int)
    for path in range(paths):
        parts, filled = [], 0
        while filled < length:
            k = int((length - filled) // mean_block) + 1
            starts = np.asarray(rng.integers(0, n, size=k))
            sizes = np.asarray(rng.geometric(1 / mean_block, size=k))
            offsets = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
            parts.append((np.repeat(starts, sizes) + offsets) % n)
            filled += int(sizes.sum())
        out[path] = np.concatenate(parts)[:length]
    return out


def _pick_block(rates: pd.Series, alpha: float) -> int:
    """The block whose rejection rate is closest to alpha; ties (to 1e-12) to the smaller."""
    distance = (rates - alpha).abs().round(12)
    return int(rates.index[int(np.argmin(distance.to_numpy()))])


# --- internals: checks -----------------------------------------------------------------------


def _series(r: object, what: str) -> np.ndarray:
    if not isinstance(r, pd.Series):
        raise TypeError(f"{what} must be a Series of returns by date, got {type(r).__name__}")
    check_dates(r.index, what)
    x = as_floats(r, what)
    if not np.isfinite(x).all():
        raise ValueError(f"{what} must be finite, with no gaps")
    if len(x) < _MIN_PERIODS:
        raise ValueError(f"{what} need at least {_MIN_PERIODS} periods, got {len(x)}")
    return x


def _pair(r1: object, r2: object) -> tuple[np.ndarray, np.ndarray]:
    for r, what in ((r1, "r1"), (r2, "r2")):
        if not isinstance(r, pd.Series):
            raise TypeError(f"{what} must be a Series of returns by date, got {type(r).__name__}")
    assert isinstance(r1, pd.Series) and isinstance(r2, pd.Series)  # for the type checker
    check_aligned(r1, r2, "the two series' dates")
    return _series(r1, "r1"), _series(r2, "r2")


def _sd(x: np.ndarray, what: str) -> float:
    sd = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
    if not sd > 0:
        raise ValueError(f"{what} have no variance: the Sharpe ratio is undefined")
    return sd


def _check_block(block: object, t: int) -> None:
    if not is_whole(block) or not 1 <= int(block) <= t // 2:  # type: ignore[call-overload]
        raise ValueError(f"block must be a whole number in [1, T/2 = {t // 2}], got {block!r}")


def _check_grid(blocks: Sequence[int], t: int) -> tuple[int, ...]:
    grid = tuple(blocks)
    if not grid or any(not is_whole(b) for b in grid) or list(grid) != sorted(set(grid)):
        raise ValueError(f"blocks must be distinct whole numbers in increasing order, got {grid}")
    if grid[0] < 1 or grid[-1] > t // 2:
        raise ValueError(f"blocks must lie in [1, T/2 = {t // 2}], got {grid}")
    return tuple(int(b) for b in grid)


def _check_count(value: object, what: str) -> None:
    if not is_whole(value) or int(value) < 1:  # type: ignore[call-overload]
        raise ValueError(f"{what} must be a whole number >= 1, got {value!r}")
