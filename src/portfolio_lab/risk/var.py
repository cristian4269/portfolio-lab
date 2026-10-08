"""One-day value at risk (VaR) and expected shortfall (ES) (blueprint Step 5a).

Signs: VaR and ES are positive losses, as fractions of portfolio value. VaR = -q, where q is
the return quantile at p = 1 - level; ES is minus the mean return beyond it. `level` is the
confidence (0.95, 0.99), in (0.5, 1).

Methods, as point estimates on one sample (rolling forecasts build on them):
- Historical (`var_es`): q = numpy's `quantile(..., method="linear")`, Hyndman-Fan type 7 (the
  numpy and pandas default). ES = minus the mean of the values at or below q (the tail mean,
  Hull's definition): "at or below" keeps the tail non-empty, and ES >= VaR always.
- Normal (`normal_var_es`): with z = Phi^-1(p) < 0, VaR = -(mu + z sigma) and
  ES = -mu + sigma phi(z) / p.
- Cornish-Fisher (`cornish_fisher_var_es`), with skew S and excess kurtosis K:
  z_cf = z + (z^2 - 1) S/6 + (z^3 - 3z) K/24 - (2z^3 - 5z) S^2/36 and VaR = -(mu + z_cf sigma).
  ES is the exact tail mean of the CF quantile, from the truncated normal moments
  E[Z^k | Z < z]: ES = -mu + sigma (phi(z) / p) [1 + z S/6 + (z^2 - 1) K/24 - (2z^2 - 1) S^2/36].
  S = K = 0 gives the normal method. For some (S, K), e.g. S = 1 and K = 0, the CF quantile
  is not monotone in the tail and ES falls below VaR: reported, not corrected. This is not the
  Boudt-Peterson-Croux "modified ES" (an Edgeworth expansion).
- Filtered historical simulation (FHS) uses `ewma_vol`, a zero-mean EWMA (RiskMetrics, as in
  Step 3's `ewma_cov`): sigma^2(s + 1) = decay sigma^2(s) + (1 - decay) r(s)^2. The residuals
  r(s) / sigma(s) scaled by tomorrow's sigma are a sample for `var_es`; since sigma > 0, that
  is sigma times the residuals' VaR and ES.

`scale_horizon` turns a 1-day figure into a 10-day one by sqrt(10): an approximation that
assumes independent, zero-mean daily returns.
"""

import math
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import ndtri

from portfolio_lab._validate import as_floats, check_dates, check_finite
from portfolio_lab.construct.covariance import EWMA_DECAY

__all__ = [
    "LEVELS",
    "VarEs",
    "cornish_fisher_var_es",
    "ewma_vol",
    "normal_var_es",
    "scale_horizon",
    "var_es",
]

LEVELS = (0.95, 0.99)

_SQRT_2PI = math.sqrt(2 * math.pi)


class VarEs(NamedTuple):
    """A one-day VaR and ES, as positive losses."""

    var: float
    es: float


def var_es(sample: Iterable[float], level: float) -> VarEs:
    """Historical VaR and ES of one sample: returns, or FHS's rescaled residuals.

    The sample may be in any order; it must be one-dimensional, finite and non-empty.
    """
    p = _tail_prob(level)
    var, es = _historical(_sample(sample)[None, :], p)
    return VarEs(float(var[0]), float(es[0]))


def normal_var_es(mu: float, sigma: float, level: float) -> VarEs:
    """VaR and ES of a normal distribution with mean `mu` and standard deviation `sigma`."""
    p = _tail_prob(level)
    _check_moments(mu, sigma)
    var, es = _normal(mu, sigma, p)
    return VarEs(float(var), float(es))


def cornish_fisher_var_es(
    mu: float, sigma: float, skew: float, exkurt: float, level: float
) -> VarEs:
    """VaR and ES from the Cornish-Fisher expansion of the quantile (see the module docstring)."""
    p = _tail_prob(level)
    _check_moments(mu, sigma)
    check_finite(skew, "skew")
    check_finite(exkurt, "exkurt")
    var, es = _cornish_fisher(mu, sigma, skew, exkurt, p)
    return VarEs(float(var), float(es))


def ewma_vol(returns: pd.Series, *, seed_var: float, decay: float = EWMA_DECAY) -> pd.Series:
    """Zero-mean EWMA volatility forecasts, indexed by the day they are for.

    The value on day s uses returns through s - 1 only: sigma^2 on the first day is `seed_var`,
    then sigma^2(s + 1) = decay sigma^2(s) + (1 - decay) r(s)^2.
    """
    r = _returns(returns, "returns")
    if not 0 < decay <= 1:
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    if not (math.isfinite(seed_var) and seed_var > 0):
        raise ValueError(f"seed_var must be a positive, finite variance, got {seed_var}")
    var = _ewma_var(r, seed_var, decay)
    return pd.Series(np.sqrt(var[:-1]), index=returns.index, name=returns.name)


def scale_horizon[T: (float, pd.Series, pd.DataFrame)](x: T, days: int = 10) -> T:
    """A 1-day VaR or ES scaled to `days` days by sqrt(days): an approximation (iid, zero mean)."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError(f"days must be a whole number >= 1, got {days!r}")
    return x * math.sqrt(days)


# --- the formulas, on arrays (one estimate per row or element) --------------------------------


def _historical(windows: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    """VaR and ES of each row of a 2-D array (one sample per row)."""
    # Linear interpolation never rounds below the lower neighbour, so the tail is never empty.
    q = np.quantile(windows, p, axis=1, method="linear")
    tail = windows <= q[:, None]
    tail_mean = np.where(tail, windows, 0.0).sum(axis=1) / tail.sum(axis=1)
    # A mean of values <= q is <= q, but its rounding is not (7 copies of 0.045899312196799685
    # average 1 ulp above it): the minimum keeps ES >= VaR exact.
    return -q, -np.minimum(tail_mean, q)


def _normal[A: (float, np.ndarray)](mu: A, sigma: A, p: float) -> tuple[A, A]:
    z = float(ndtri(p))
    return -(mu + z * sigma), -mu + sigma * _pdf(z) / p


def _cornish_fisher[A: (float, np.ndarray)](mu: A, sigma: A, s: A, k: A, p: float) -> tuple[A, A]:
    z = float(ndtri(p))
    z_cf = z + (z * z - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s * s / 36
    tail = 1 + z * s / 6 + (z * z - 1) * k / 24 - (2 * z * z - 1) * s * s / 36
    return -(mu + z_cf * sigma), -mu + sigma * _pdf(z) / p * tail


def _ewma_var(r: np.ndarray, seed_var: float, decay: float) -> np.ndarray:
    """sigma^2 for days 1..n + 1: the seed, then one forecast after each return."""
    after, _ = lfilter([1 - decay], [1, -decay], r * r, zi=[decay * seed_var])
    return np.concatenate([[seed_var], after])


def _pdf(z: float) -> float:
    return math.exp(-z * z / 2) / _SQRT_2PI


# --- checks -----------------------------------------------------------------------------------


def _tail_prob(level: float) -> float:
    """p = 1 - level, for a confidence level in (0.5, 1)."""
    if isinstance(level, bool) or not isinstance(level, int | float) or not 0.5 < level < 1:
        raise ValueError(f"level must be a confidence in (0.5, 1) such as 0.99, got {level!r}")
    return 1.0 - float(level)


def _check_moments(mu: float, sigma: float) -> None:
    check_finite(mu, "mu")
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be positive and finite, got {sigma}")


def _sample(sample: Iterable[float]) -> np.ndarray:
    try:
        x = np.asarray(sample, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("the sample must be numeric") from err
    if x.ndim != 1:
        raise ValueError(f"the sample must be one-dimensional, got {x.ndim} dimensions")
    if x.size < 1:
        raise ValueError("the sample needs at least 1 value")
    if not np.isfinite(x).all():
        raise ValueError("the sample must be finite")
    return x


def _returns(returns: pd.Series, what: str) -> np.ndarray:
    """Daily returns as floats: a Series on sorted, unique dates, complete, finite, >= -100%."""
    if not isinstance(returns, pd.Series):
        raise TypeError(f"{what} must be a Series of one portfolio, got {type(returns).__name__}")
    check_dates(returns.index, what)
    if len(returns) < 1:
        raise ValueError(f"{what} need at least 1 day")
    r = as_floats(returns, what)
    if not np.isfinite(r).all():
        raise ValueError(f"{what} must be finite, with no gaps")
    if (r < -1).any():
        raise ValueError(f"{what} below -100% are impossible without leverage")
    return r
