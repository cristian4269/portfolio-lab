"""Checks shared across the library: one rule for order, alignment, windows and portfolios."""

import datetime as dt
import math
import numbers

import numpy as np
import pandas as pd

# A portfolio's weights must sum to 1 within this (a rule's own rounding is far smaller).
_BUDGET = 1e-9


def check_order(index: pd.Index, what: str) -> None:
    """Rows must be in increasing order with no repeats (anything path-dependent needs it)."""
    if not index.is_monotonic_increasing:
        raise ValueError(f"{what} must be sorted by date")
    if not index.is_unique:
        raise ValueError(f"{what} must have unique dates")


def as_date(value: object, what: str) -> pd.Timestamp:
    """A calendar date given as a str, a date or datetime, or a numpy datetime64.

    Raises unless it parses to a real, tz-naive timestamp. "", "NaT" and NaN would become NaT
    (and `.loc[:NaT]` keeps everything); numbers are refused because pandas reads them as
    nanoseconds since 1970; a time zone can't compare with a calendar of dates.
    """
    if not isinstance(value, str | dt.date | np.datetime64):
        raise ValueError(f"{what}={value!r} is not a date")
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"{what}={value!r} is not a date")
    if ts.tz is not None:
        raise ValueError(f"{what}={value!r} is not a date: it has a time zone")
    return ts


def check_dates(index: pd.Index, what: str) -> pd.DatetimeIndex:
    """A sorted, unique, tz-naive DatetimeIndex (calendar dates); returns it, typed as such."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{what} must have a DatetimeIndex, got {type(index).__name__}")
    if index.tz is not None:
        raise ValueError(f"{what} must be tz-naive (dates), got tz={index.tz}")
    check_order(index, what)
    return index


def check_aligned(a: pd.Series | pd.DataFrame, b: pd.Series, what: str = "dates") -> None:
    """`b` must have exactly `a`'s index, in the same order: no silent pandas alignment."""
    if not a.index.equals(b.index):
        diff = a.index.symmetric_difference(b.index)
        where = f"first difference {diff[0]}" if len(diff) else "same dates, different order"
        raise ValueError(f"{what} don't align ({len(a)} vs {len(b)} rows; {where})")


def as_floats(s: pd.Series | pd.DataFrame, what: str) -> np.ndarray:
    """The values as floats; raises a clear error for non-numeric data."""
    try:
        return s.to_numpy(dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{what} must be numeric") from err


def window_values(returns: pd.DataFrame, what: str, *, min_rows: int) -> np.ndarray:
    """A window of returns as floats: rows = periods in date order, columns = assets.

    Every value must be present and finite (an estimator can't guess what a gap means), with
    unique asset names, unique sorted dates and at least `min_rows` rows.
    """
    check_frame(returns, what)
    if returns.shape[1] == 0:
        raise ValueError(f"{what} have no assets")
    if not returns.columns.is_unique:
        raise ValueError(f"{what} need unique asset names")
    if len(returns) < min_rows:
        raise ValueError(f"{what} need at least {min_rows} periods, got {len(returns)}")
    check_order(returns.index, what)
    values = as_floats(returns, what)
    missing = np.isnan(values)
    if missing.any():
        row, col = (int(k) for k in np.argwhere(missing)[0])
        raise ValueError(f"{what} are missing for {returns.columns[col]} on {returns.index[row]}")
    if not np.isfinite(values).all():
        raise ValueError(f"{what} must be finite")
    return values


def last_rows(history: pd.DataFrame, what: str, rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    """The last `rows` rows of a history (the window) and their values, checked.

    The whole history must be in date order (otherwise its last rows are not the latest); only
    the window must be complete, as in `window_values`. Older rows are never read.
    """
    check_frame(history, what)
    if len(history) < rows:
        raise ValueError(f"{what} need at least {rows} periods, got {len(history)}")
    check_order(history.index, what)
    window = history.iloc[len(history) - rows :]
    return window, window_values(window, what, min_rows=rows)


def portfolio_values(w: pd.Series, assets: pd.Index, what: str, other: str) -> np.ndarray:
    """Long-only weights as floats: a Series labeled by exactly `assets` (in their order, like
    `other`), finite, >= 0 and summing to 1 within _BUDGET."""
    if not isinstance(w, pd.Series):
        raise TypeError(f"{what} must be a Series labeled by the assets, got {type(w).__name__}")
    if not w.index.equals(assets):
        raise ValueError(f"{what} must have the same assets, in the same order, as {other}")
    x = as_floats(w, what)
    if not np.isfinite(x).all():
        raise ValueError(f"{what} must be finite")
    if (x < 0).any():
        asset = assets[int(np.argmax(x < 0))]
        raise ValueError(f"{what} must be >= 0 (negative weight for {asset})")
    if abs(x.sum() - 1) > _BUDGET:
        raise ValueError(f"{what} must sum to 1, got {x.sum():.12g}")
    return x


def check_cap(cap: float) -> None:
    """A maximum weight per asset must be in (0, 1] (1 = no cap)."""
    if not 0 < cap <= 1:
        raise ValueError(f"cap must be in (0, 1], got {cap}")


def check_finite(x: float, what: str) -> None:
    if not math.isfinite(x):
        raise ValueError(f"{what} must be finite, got {x}")


def tail_prob(level: float) -> float:
    """p = 1 - level for a VaR confidence level in (0.5, 1): 0.99 gives 0.01.

    Any real number counts (numpy's too), but not a bool."""
    if (
        isinstance(level, bool | np.bool_)
        or not isinstance(level, numbers.Real)
        or not 0.5 < level < 1
    ):
        raise ValueError(f"level must be a confidence in (0.5, 1) such as 0.99, got {level!r}")
    return 1.0 - float(level)


def check_decay(decay: float) -> None:
    """An exponential decay per period must be in (0, 1] (1 = equal weights)."""
    if not 0 < decay <= 1:
        raise ValueError(f"decay must be in (0, 1], got {decay}")


def return_series(returns: pd.Series, what: str) -> np.ndarray:
    """One portfolio's daily returns as floats: a Series on sorted, unique dates, at least one
    day, complete, finite and >= -100%."""
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


def check_frame(returns: pd.DataFrame, what: str) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            f"{what} must be a DataFrame (rows = periods, columns = assets), "
            f"got {type(returns).__name__}"
        )
