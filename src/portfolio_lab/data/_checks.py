"""Checks shared by the single-series loaders (FRED, Bank of Canada, Statistics Canada)."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def require_approved(series: str, approved: Mapping[str, str], source: str) -> None:
    """Only series whose terms were checked in DATA_SOURCES.md may be downloaded."""
    if series not in approved:
        raise PermissionError(
            f"{source} series {series!r} has no 'allowed' verdict in DATA_SOURCES.md; "
            f"approved: {sorted(approved)}"
        )


def to_float(raw: object, missing: Sequence[object], where: str) -> float:
    """Parse one observation; values in `missing` become NaN, anything unparseable raises."""
    if raw in missing:
        return np.nan
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{where}: cannot parse value {raw!r}") from None


def build_series(index: pd.DatetimeIndex, values: list[float], name: str) -> pd.Series:
    """A float Series on a strictly increasing date index named `date`."""
    if len(index) == 0:
        raise ValueError(f"{name}: no observations")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(f"{name}: dates must be strictly increasing")
    return pd.Series(np.asarray(values, dtype=np.float64), index=index.rename("date"), name=name)
