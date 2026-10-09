"""One-day value at risk (VaR) and expected shortfall (ES) (blueprint Step 5a).

Signs: VaR and ES are positive losses, as fractions of portfolio value. VaR = -q, where q is
the return quantile at p = 1 - level; ES is minus the mean return beyond it. `level` is the
confidence (0.95, 0.99), in (0.5, 1).

Methods, as point estimates on one sample:
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

Rolling forecasts (`forecast`, `next_day`): the forecast for day t uses the `window` returns
before it, t - window, ..., t - 1, and is indexed by t, the day it is tested against. Every
method sees the same window. Normal and CF use the window's mean, sample standard deviation
(ddof = 1) and Step 2's bias-corrected skew and excess kurtosis (`metrics.skew`,
`metrics.excess_kurtosis`); a window with no variance raises. FHS runs one causal filter over
the whole series, seeded with the mean square of the first window (so the seed uses only data
the first forecast has seen), and rescales the window's residuals by sigma(t).

`scale_horizon` turns a 1-day figure into a 10-day one by sqrt(10): an approximation that
assumes independent, zero-mean daily returns.
"""

import math
from collections.abc import Callable, Iterable
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from scipy.special import ndtri

from portfolio_lab._validate import check_finite, return_series, tail_prob
from portfolio_lab.construct.covariance import EWMA_DECAY

__all__ = [
    "LEVELS",
    "METHODS",
    "WINDOW",
    "VarEs",
    "cornish_fisher_var_es",
    "ewma_vol",
    "forecast",
    "next_day",
    "normal_var_es",
    "scale_horizon",
    "var_es",
]

LEVELS = (0.95, 0.99)
METHODS = ("historical", "normal", "cornish_fisher", "fhs")
WINDOW = 500  # trading days (~2 years); 7a pre-registers the window it reports

_SQRT_2PI = math.sqrt(2 * math.pi)
_MIN_WINDOW = 4  # excess kurtosis needs 4 values; one rule for every method
_BLOCK = 2048  # windows per block: bounds the copies np.quantile and the moments make
# A window whose standard deviation is this small next to its largest return has no variance.
_NOISE = 1e-12

type _Rows = tuple[np.ndarray, ...]


class VarEs(NamedTuple):
    """A one-day VaR and ES, as positive losses."""

    var: float
    es: float


def var_es(sample: Iterable[float], level: float) -> VarEs:
    """Historical VaR and ES of one sample: returns, or FHS's rescaled residuals.

    The sample may be in any order; it must be one-dimensional, finite and non-empty.
    """
    p = tail_prob(level)
    var, es = _historical(_sample(sample)[None, :], p)
    return VarEs(float(var[0]), float(es[0]))


def normal_var_es(mu: float, sigma: float, level: float) -> VarEs:
    """VaR and ES of a normal distribution with mean `mu` and standard deviation `sigma`."""
    p = tail_prob(level)
    _check_moments(mu, sigma)
    var, es = _normal(mu, sigma, p)
    return VarEs(float(var), float(es))


def cornish_fisher_var_es(
    mu: float, sigma: float, skew: float, exkurt: float, level: float
) -> VarEs:
    """VaR and ES from the Cornish-Fisher expansion of the quantile (see the module docstring)."""
    p = tail_prob(level)
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
    r = return_series(returns, "returns")
    _check_decay(decay)
    if not (math.isfinite(seed_var) and seed_var > 0):
        raise ValueError(f"seed_var must be a positive, finite variance, got {seed_var}")
    var = _ewma_var(r, seed_var, decay)
    return pd.Series(np.sqrt(var[:-1]), index=returns.index, name=returns.name)


def forecast(
    returns: pd.Series,
    *,
    method: str,
    level: float,
    window: int = WINDOW,
    decay: float = EWMA_DECAY,
) -> pd.DataFrame:
    """Rolling one-day-ahead VaR and ES (columns `var`, `es`), indexed by the day forecast.

    The first forecast is for the day after the first `window` returns; `decay` is FHS's.
    """
    r, p = _rolling_inputs(returns, method, level, window, decay)
    if len(r) <= window:
        raise ValueError(
            f"returns need more than {window} days to forecast with window={window}, got {len(r)}"
        )
    var, es = _rolling(r, returns.index, method, p, window, decay)
    return pd.DataFrame({"var": var[:-1], "es": es[:-1]}, index=returns.index[window:])


def next_day(
    returns: pd.Series,
    *,
    method: str,
    level: float,
    window: int = WINDOW,
    decay: float = EWMA_DECAY,
) -> VarEs:
    """The forecast for the day after the last return ("today's VaR"), by `forecast`'s rule."""
    r, p = _rolling_inputs(returns, method, level, window, decay)
    if len(r) < window:
        raise ValueError(f"returns need at least {window} days for window={window}, got {len(r)}")
    var, es = _rolling(r, returns.index, method, p, window, decay)
    return VarEs(float(var[-1]), float(es[-1]))


def scale_horizon[T: (float, pd.Series, pd.DataFrame)](x: T, days: int = 10) -> T:
    """A 1-day VaR or ES scaled to `days` days by sqrt(days): an approximation (iid, zero mean)."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError(f"days must be a whole number >= 1, got {days!r}")
    return x * math.sqrt(days)


# --- rolling windows --------------------------------------------------------------------------


def _rolling(
    r: np.ndarray, days: pd.Index, method: str, p: float, window: int, decay: float
) -> tuple[np.ndarray, np.ndarray]:
    """VaR and ES for days window + 1, ..., n + 1 (the last is the day after the data)."""
    windows = sliding_window_view(r, window)  # row i: days i .. i + window - 1, for day i + window
    if method == "historical":
        var, es = _by_block(lambda w: _historical(w, p), windows)
        return var, es
    if method == "fhs":
        seed_var = float(np.mean(r[:window] ** 2))
        if seed_var == 0:
            raise ValueError("FHS needs a first window that moves: its returns are all 0")
        sigma = np.sqrt(_ewma_var(r, seed_var, decay))  # days 1 .. n + 1
        if not (sigma > 0).all():
            day = _day(days, int(np.argmin(sigma > 0)))
            raise ValueError(f"the EWMA volatility reached 0 by {day}: FHS can't standardize")
        residuals = sliding_window_view(r / sigma[:-1], window)
        var, es = _by_block(lambda w: _historical(w, p), residuals)
        return sigma[window:] * var, sigma[window:] * es
    var, es, flat = _by_block(lambda w: _parametric(w, p, method), windows)
    if flat.any():
        day = _day(days, window + int(np.argmax(flat)))
        raise ValueError(
            f"returns have no variance in the {window} days before {day}: "
            f"the {method} method needs some"
        )
    return var, es


def _parametric(w: np.ndarray, p: float, method: str) -> _Rows:
    """Normal or CF VaR and ES of each row, and which rows have no variance."""
    n = w.shape[1]
    mu = w.mean(axis=1)
    dev = w - mu[:, None]
    dev2 = dev * dev  # products, not **: numpy's general power made CF 5x slower
    m2 = dev2.sum(axis=1)
    sd = np.sqrt(m2 / (n - 1))
    flat = sd <= _NOISE * np.abs(w).max(axis=1)
    if method == "normal":
        return (*_normal(mu, sd, p), flat)
    # Step 2's bias-corrected G1 and G2 (pandas' formulas), from sums of powers of deviations.
    with np.errstate(divide="ignore", invalid="ignore"):  # flat rows raise in the caller
        skew = n * math.sqrt(n - 1) / (n - 2) * (dev2 * dev).sum(axis=1) / m2**1.5
        kurt = (n + 1) * n * (n - 1) * (dev2 * dev2).sum(axis=1) / (
            (n - 2) * (n - 3) * m2**2
        ) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return (*_cornish_fisher(mu, sd, skew, kurt, p), flat)


def _by_block(func: Callable[[np.ndarray], _Rows], windows: np.ndarray) -> _Rows:
    """`func` on blocks of rows, joined: one array per output, one value per row."""
    parts = [func(windows[i : i + _BLOCK]) for i in range(0, len(windows), _BLOCK)]
    return tuple(np.concatenate(column) for column in zip(*parts, strict=True))


def _day(days: pd.Index, position: int) -> str:
    """The date of a day by position; position n is the day after the data."""
    if position < len(days):
        return str(days[position].date())
    return f"the day after {days[-1].date()}"


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


def _rolling_inputs(
    returns: pd.Series, method: str, level: float, window: int, decay: float
) -> tuple[np.ndarray, float]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    p = tail_prob(level)
    if isinstance(window, bool) or not isinstance(window, int) or window < _MIN_WINDOW:
        raise ValueError(f"window must be a whole number of days >= {_MIN_WINDOW}, got {window!r}")
    _check_decay(decay)
    return return_series(returns, "returns"), p


def _check_decay(decay: float) -> None:
    if not 0 < decay <= 1:
        raise ValueError(f"decay must be in (0, 1], got {decay}")


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
