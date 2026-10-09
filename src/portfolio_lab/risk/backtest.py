"""VaR backtests (blueprint Step 5a): do exceptions come as often as promised, and independently?

An exception is a day whose loss is strictly larger than that day's VaR forecast:
r(t) < -VaR(t) (`exceptions`). With T days, x exceptions and p = 1 - level:
- Kupiec (1995) proportion of failures:
  LR_uc = -2 ln[(1 - p)^(T - x) p^x] + 2 ln[(1 - x/T)^(T - x) (x/T)^x], chi-square(1).
- Christoffersen (1998) independence, from the transitions between consecutive days (n_ij =
  days in state j after a day in state i, 1 = exception): with pi01 = n01 / (n00 + n01),
  pi11 = n11 / (n10 + n11) and pi = (n01 + n11) / (T - 1),
  LR_ind = -2 ln[(1 - pi)^(n00 + n10) pi^(n01 + n11)]
           + 2 ln[(1 - pi01)^n00 pi01^n01 (1 - pi11)^n10 pi11^n11], chi-square(1).
- Conditional coverage: LR_cc = LR_uc + LR_ind, chi-square(2), with LR_uc computed on the
  T - 1 days that follow another day, so the sum is exact. The stand-alone Kupiec test uses all
  T days.
0 ln 0 = 0 throughout (scipy's xlogy), so x = 0, x = T, n11 = 0 and a transition probability
with no days behind it (pi11 when no exception is followed by a day) are ordinary cases. A
statistic that rounds below 0 is 0.
- Basel traffic light (BCBS 1996), defined only for 99% VaR over the last 250 days: green for
  0-4 exceptions, yellow 5-9, red 10 or more (`basel_zone`, `basel_zones`).

The chi-square p-values are asymptotic. With few expected exceptions the true size differs:
a 5% Kupiec test rejects a correct 99% model over 250 days 9.5% of the time (5.4% at 95% over
500 days), from the exact binomial distribution of x.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from portfolio_lab._validate import as_floats, check_dates, return_series, tail_prob
from portfolio_lab.construct.covariance import EWMA_DECAY
from portfolio_lab.risk.var import LEVELS, METHODS, WINDOW, forecast

__all__ = [
    "BASEL_DAYS",
    "BASEL_LEVEL",
    "LikelihoodRatio",
    "basel_zone",
    "basel_zones",
    "christoffersen",
    "conditional_coverage",
    "exceptions",
    "kupiec",
    "report",
    "transitions",
]

BASEL_LEVEL = 0.99
BASEL_DAYS = 250

_COLUMNS = (
    "days",
    "exceptions",
    "expected",
    "rate",
    "kupiec_stat",
    "kupiec_p",
    "independence_stat",
    "independence_p",
    "cc_stat",
    "cc_p",
    "basel",
)


class LikelihoodRatio(NamedTuple):
    """A likelihood-ratio test: statistic, chi-square p-value and degrees of freedom, and the
    exceptions and days of the series tested. Independence and conditional coverage use the
    series' T - 1 transitions between consecutive days; `days` is still T."""

    stat: float
    pvalue: float
    df: int
    exceptions: int
    days: int


def exceptions(returns: pd.Series, var: pd.Series) -> pd.Series:
    """True on the days whose loss is strictly larger than the VaR forecast for that day.

    `var` is indexed by the day each forecast is for (as `forecast` gives it); every one of its
    days needs a return. The result has `var`'s days.
    """
    return_series(returns, "returns")
    if not isinstance(var, pd.Series):
        raise TypeError(f"VaR forecasts must be a Series by day, got {type(var).__name__}")
    check_dates(var.index, "VaR forecasts")
    forecasts = as_floats(var, "VaR forecasts")
    if not np.isfinite(forecasts).all():
        raise ValueError("VaR forecasts must be finite")
    missing = var.index.difference(returns.index)
    if len(missing):
        raise ValueError(f"VaR forecasts for days with no return, first {missing[0].date()}")
    on_days = returns.reindex(var.index).to_numpy(dtype=float)
    return pd.Series(on_days < -forecasts, index=var.index, name="exceptions")


def kupiec(exc: pd.Series, level: float) -> LikelihoodRatio:
    """Kupiec's proportion-of-failures test: is the exception rate 1 - level?"""
    p = tail_prob(level)
    flags = _flags(exc)
    return _kupiec(int(flags.sum()), len(flags), p)


def transitions(exc: pd.Series) -> tuple[int, int, int, int]:
    """(n00, n01, n10, n11): consecutive-day pairs by state (1 = exception), in date order."""
    return _transitions(_flags(exc))


def christoffersen(exc: pd.Series) -> LikelihoodRatio:
    """Christoffersen's independence test: is an exception more likely after an exception?"""
    flags = _flags(exc, min_days=2)
    n00, n01, n10, n11 = _transitions(flags)
    pi = (n01 + n11) / (len(flags) - 1)
    stat = -2 * (_bernoulli(n00 + n10, n01 + n11, pi) - _markov(n00, n01, n10, n11))
    return _lr(stat, 1, flags)


def conditional_coverage(exc: pd.Series, level: float) -> LikelihoodRatio:
    """Christoffersen's conditional coverage: the right rate and independence, jointly."""
    p = tail_prob(level)
    flags = _flags(exc, min_days=2)
    n00, n01, n10, n11 = _transitions(flags)
    stat = -2 * (_bernoulli(n00 + n10, n01 + n11, p) - _markov(n00, n01, n10, n11))
    return _lr(stat, 2, flags)


def basel_zone(exc: pd.Series, level: float) -> str:
    """The Basel traffic-light zone of exactly 250 days of 99% VaR exceptions."""
    flags = _basel_flags(exc, level)
    if len(flags) != BASEL_DAYS:
        raise ValueError(f"Basel zones need exactly {BASEL_DAYS} days, got {len(flags)}")
    return _zone(int(flags.sum()))


def basel_zones(exc: pd.Series, level: float) -> pd.Series:
    """The Basel zone of every trailing 250-day window, indexed by the window's last day."""
    flags = _basel_flags(exc, level)
    if len(flags) < BASEL_DAYS:
        raise ValueError(f"Basel zones need at least {BASEL_DAYS} days, got {len(flags)}")
    total = np.concatenate([[0], np.cumsum(flags, dtype=int)])
    counts = total[BASEL_DAYS:] - total[:-BASEL_DAYS]
    zones = [_zone(int(x)) for x in counts]
    return pd.Series(zones, index=exc.index[BASEL_DAYS - 1 :], name="basel", dtype=object)


def report(
    returns: pd.Series,
    *,
    levels: Sequence[float] = LEVELS,
    methods: Sequence[str] = METHODS,
    window: int = WINDOW,
    decay: float = EWMA_DECAY,
) -> pd.DataFrame:
    """Every backtest of every method's rolling forecasts, one row per (level, method).

    `basel` is the zone of the last 250 days at 99%, and None at other levels or with fewer
    days.
    """
    if not levels:
        raise ValueError("levels must name at least one confidence level")
    if not methods:
        raise ValueError("methods must name at least one method")
    keys, rows = [], []
    for level in levels:
        p = tail_prob(level)
        for method in methods:
            fc = forecast(returns, method=method, level=level, window=window, decay=decay)
            exc = exceptions(returns, fc["var"])
            flags = exc.to_numpy()
            days, x = len(flags), int(flags.sum())
            uc, ind = kupiec(exc, level), christoffersen(exc)
            cc = conditional_coverage(exc, level)
            basel = (
                basel_zone(exc.iloc[-BASEL_DAYS:], level)
                if level == BASEL_LEVEL and days >= BASEL_DAYS
                else None
            )
            keys.append((level, method))
            rows.append([days, x, p * days, x / days, *uc[:2], *ind[:2], *cc[:2], basel])
    index = pd.MultiIndex.from_tuples(keys, names=["level", "method"])
    table = pd.DataFrame(rows, index=index, columns=list(_COLUMNS))
    table["basel"] = pd.Series([row[-1] for row in rows], index=index, dtype=object)
    return table


# --- the statistics ---------------------------------------------------------------------------


def _kupiec(x: int, t: int, p: float) -> LikelihoodRatio:
    rate = x / t
    stat = -2 * _bernoulli(t - x, x, p) + 2 * _bernoulli(t - x, x, rate)
    return LikelihoodRatio(_positive(stat), float(chi2.sf(_positive(stat), 1)), 1, x, t)


def _bernoulli(n0: int, n1: int, pi: float) -> float:
    """Log-likelihood of n0 non-exceptions and n1 exceptions with probability pi each."""
    return float(xlogy(n0, 1 - pi) + xlogy(n1, pi))


def _markov(n00: int, n01: int, n10: int, n11: int) -> float:
    """Log-likelihood under first-order Markov exceptions, at the fitted probabilities."""
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    return _bernoulli(n00, n01, pi01) + _bernoulli(n10, n11, pi11)


def _transitions(flags: np.ndarray) -> tuple[int, int, int, int]:
    before, after = flags[:-1], flags[1:]
    n01 = int((~before & after).sum())
    n10 = int((before & ~after).sum())
    n11 = int((before & after).sum())
    return len(after) - n01 - n10 - n11, n01, n10, n11


def _lr(stat: float, df: int, flags: np.ndarray) -> LikelihoodRatio:
    stat = _positive(stat)
    return LikelihoodRatio(stat, float(chi2.sf(stat, df)), df, int(flags.sum()), len(flags))


def _positive(stat: float) -> float:
    """A likelihood ratio is >= 0; rounding can leave it a hair below."""
    return stat if stat > 0 else 0.0


def _zone(x: int) -> str:
    return "green" if x <= 4 else "yellow" if x <= 9 else "red"


# --- checks -----------------------------------------------------------------------------------


def _flags(exc: pd.Series, *, min_days: int = 1) -> np.ndarray:
    """Exceptions as a bool array: a Series of True/False (or 1/0) on sorted, unique dates."""
    if not isinstance(exc, pd.Series):
        raise TypeError(
            f"exceptions must be a Series of True/False by day, got {type(exc).__name__}"
        )
    if len(exc) < min_days:
        raise ValueError(f"exceptions need at least {min_days} days, got {len(exc)}")
    check_dates(exc.index, "exceptions")
    if exc.dtype == bool:
        return exc.to_numpy(dtype=bool)
    try:
        values = exc.to_numpy(dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("exceptions must be True/False (or 1/0)") from err
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("exceptions must be True/False (or 1/0), with no gaps")
    return values == 1.0


def _basel_flags(exc: pd.Series, level: float) -> np.ndarray:
    if level != BASEL_LEVEL:
        raise ValueError(f"Basel zones are defined for 99% VaR only (level=0.99), got {level!r}")
    return _flags(exc)
