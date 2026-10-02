"""Currency views of USD asset returns: USD, CAD unhedged, CAD hedged (pure functions).

Calendar: US trading days (the French return index) drive everything. FX and
rate *levels* are carried onto those days from their last observation at or
before each day (`align_levels`), then turned into returns. So a Canadian
holiday keeps the last Canadian rate, and a missing USD/CAD fix makes that day
flat and puts the whole move on the next fix. A level older than
`MAX_AGE_DAYS` calendar days raises instead of being carried silently.

Formulas, per trading day t (S = CAD per 1 USD):
- FX return: fx_t = S_t / S_{t-1} - 1 (USD up -> CAD investor gains).
- CAD unhedged: r_cad = (1 + r_usd)(1 + fx) - 1.
- Canadian risk-free: 3-month T-bill yields are simple act/365 yields (price
  100 / (1 + y T/365)), so rolling bills earns (1 + y T/365)^(d/T) - 1 over d
  calendar days, T = 91. The yield is the one known at the previous trading
  day (start of the period); a Monday earns 3 days.
- CAD hedged, covered interest parity: r_hedged = r_usd + rf_cad - rf_usd.
  Hence hedged excess return == USD excess return exactly: the hedged view
  only differs in total-return metrics (CAGR, drawdown), not in Sharpe.
- Excess returns use the view's own risk-free rate: US (French RF) for USD,
  Canadian for both CAD views.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "MAX_AGE_DAYS",
    "TBILL_TERM_DAYS",
    "View",
    "align_levels",
    "daily_rate",
    "excess",
    "fx_returns",
    "make_views",
    "to_cad_hedged",
    "to_cad_unhedged",
]

MAX_AGE_DAYS = 7  # longest calendar gap a carried level may cover (holidays + weekend)
TBILL_TERM_DAYS = 91  # 3-month Government of Canada T-bill


@dataclass(frozen=True)
class View:
    """Daily returns in one currency view, with the matching risk-free rate.

    `returns` may hold NaN (an asset not yet listed, as in the 49 industries);
    `rf` may not.
    """

    returns: pd.DataFrame
    rf: pd.Series

    def __post_init__(self) -> None:
        _check_calendar(self.returns.index, "returns")
        _check_aligned(self.returns, self.rf)
        if self.rf.isna().any():
            raise ValueError(f"rf is missing on {self.rf.index[self.rf.isna()][0].date()}")

    @property
    def excess(self) -> pd.DataFrame:
        return excess(self.returns, self.rf)


def align_levels(
    levels: pd.Series, days: pd.DatetimeIndex, *, max_age_days: int = MAX_AGE_DAYS
) -> pd.Series:
    """The last non-missing level at or before each day; NaN before the first one.

    Raises if either index is not a sorted, unique, tz-naive DatetimeIndex, or
    a carried level is more than `max_age_days` calendar days old.
    """
    _check_calendar(days, "days")
    _check_calendar(levels.index, f"levels {levels.name!r}")

    observed = levels.dropna()
    pos = observed.index.searchsorted(days, side="right") - 1
    found = pos >= 0
    values = np.full(len(days), np.nan)
    values[found] = observed.to_numpy(dtype=float)[pos[found]]

    age = days[found] - observed.index[pos[found]]
    stale = age > pd.Timedelta(days=max_age_days)
    if stale.any():
        day = days[found][stale][0]
        raise ValueError(
            f"levels {levels.name!r} are stale on {day.date()}: last observed "
            f"{age[stale][0].days} days earlier (max {max_age_days})"
        )
    return pd.Series(values, index=days, name=levels.name)


def fx_returns(
    usdcad: pd.Series, days: pd.DatetimeIndex, *, max_age_days: int = MAX_AGE_DAYS
) -> pd.Series:
    """Daily change in CAD per USD on `days`; NaN on the first day with no prior fix."""
    observed = usdcad.dropna()
    bad = ~np.isfinite(observed) | (observed <= 0)
    if bad.any():
        raise ValueError(f"USD/CAD must be positive and finite, got {observed[bad].iloc[0]!r}")
    s = align_levels(usdcad, days, max_age_days=max_age_days)
    return (s / s.shift(1) - 1).rename("fx")


def daily_rate(
    yields: pd.Series,
    days: pd.DatetimeIndex,
    *,
    term_days: int = TBILL_TERM_DAYS,
    max_age_days: int = MAX_AGE_DAYS,
) -> pd.Series:
    """Per-period return from rolling `term_days` bills quoted as simple act/365 yields.

    The return for day t is (1 + y * term/365) ** (d / term) - 1, where y is the
    yield known at the previous day in `days` and d the calendar days since it.
    The first day is NaN.
    """
    if term_days <= 0:
        raise ValueError(f"term_days must be positive, got {term_days}")
    y = align_levels(yields, days, max_age_days=max_age_days).shift(1)
    gap = np.r_[np.nan, np.diff(days.to_numpy()) / np.timedelta64(1, "D")]
    return (1 + y * term_days / 365) ** (gap / term_days) - 1


def to_cad_unhedged[T: (pd.Series, pd.DataFrame)](r_usd: T, fx: pd.Series) -> T:
    """CAD return of USD assets held unhedged: (1 + r_usd)(1 + fx) - 1."""
    _check_aligned(r_usd, fx)
    return _keep_name(r_usd, (1 + r_usd).mul(1 + fx, axis=0) - 1)


def to_cad_hedged[T: (pd.Series, pd.DataFrame)](
    r_usd: T, rf_cad: pd.Series, rf_usd: pd.Series
) -> T:
    """CAD return of USD assets with the currency hedged by CIP: r_usd + rf_cad - rf_usd."""
    _check_aligned(r_usd, rf_cad)
    _check_aligned(r_usd, rf_usd)
    return _keep_name(r_usd, r_usd.add(rf_cad - rf_usd, axis=0))


def excess[T: (pd.Series, pd.DataFrame)](r: T, rf: pd.Series) -> T:
    """Returns minus the risk-free rate of the same day."""
    _check_aligned(r, rf)
    return _keep_name(r, r.sub(rf, axis=0))


def make_views(
    assets: pd.DataFrame,
    rf_usd: pd.Series,
    usdcad: pd.Series,
    ca_tbill: pd.Series,
    *,
    term_days: int = TBILL_TERM_DAYS,
) -> dict[str, View]:
    """The USD, CAD and CAD_HEDGED views of daily USD asset returns.

    `assets` and `rf_usd` (French RF, already a daily return) share the US
    trading-day index. `usdcad` is CAD per USD; `ca_tbill` the annualized
    3-month T-bill yield (decimal) on a daily calendar, as `load_rates` gives
    it. The CAD views start on the first day with both an FX return and a
    Canadian rate (1971-01-05 with the real sources).
    """
    usd = View(assets, rf_usd)
    days = assets.index
    fx = fx_returns(usdcad, days)
    rf_cad = daily_rate(ca_tbill, days, term_days=term_days)

    usable = fx.notna() & rf_cad.notna()
    if not usable.any():
        raise ValueError("no day has both a USD/CAD return and a Canadian rate")
    cad_days = days[days >= usable.idxmax()]

    r, fx, rf_cad = assets.loc[cad_days], fx.loc[cad_days], rf_cad.loc[cad_days]
    return {
        "USD": usd,
        "CAD": View(to_cad_unhedged(r, fx), rf_cad),
        "CAD_HEDGED": View(to_cad_hedged(r, rf_cad, rf_usd.loc[cad_days]), rf_cad),
    }


def _check_calendar(index: pd.Index, what: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{what} must have a DatetimeIndex, got {type(index).__name__}")
    if index.tz is not None:
        raise ValueError(f"{what} must be tz-naive (dates), got tz={index.tz}")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{what} must be sorted by date")
    if not index.is_unique:
        raise ValueError(f"{what} must have unique dates")


def _check_aligned(a: pd.Series | pd.DataFrame, b: pd.Series) -> None:
    if not a.index.equals(b.index):
        diff = a.index.symmetric_difference(b.index)
        where = f"first difference {diff[0]}" if len(diff) else "same dates, different order"
        raise ValueError(f"dates don't align ({len(a)} vs {len(b)} rows; {where})")


def _keep_name[T: (pd.Series, pd.DataFrame)](like: T, out: T) -> T:
    return out.rename(like.name) if isinstance(out, pd.Series) else out
