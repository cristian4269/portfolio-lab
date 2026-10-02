"""Index checks shared by the views and the metrics: one rule for order and alignment."""

import numpy as np
import pandas as pd


def check_order(index: pd.Index, what: str) -> None:
    """Rows must be in increasing order with no repeats (anything path-dependent needs it)."""
    if not index.is_monotonic_increasing:
        raise ValueError(f"{what} must be sorted by date")
    if not index.is_unique:
        raise ValueError(f"{what} must have unique dates")


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
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            f"{what} must be a DataFrame (rows = periods, columns = assets), "
            f"got {type(returns).__name__}"
        )
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
