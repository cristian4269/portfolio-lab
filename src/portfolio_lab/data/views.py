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

from portfolio_lab._validate import as_floats, check_aligned, check_dates

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
    "yield_changes",
]

MAX_AGE_DAYS = 7  # longest calendar gap a carried level may cover (holidays + weekend)
TBILL_TERM_DAYS = 91  # 3-month Government of Canada T-bill


@dataclass(frozen=True, eq=False)  # DataFrame fields: no value equality or hashing
class View:
    """Daily returns in one currency view, with the matching risk-free rate.

    `returns` may hold NaN (an asset not yet listed, as in the 49 industries);
    `rf` may not.
    """

    returns: pd.DataFrame
    rf: pd.Series

    def __post_init__(self) -> None:
        check_dates(self.returns.index, "returns")
        check_aligned(self.returns, self.rf)
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
    check_dates(days, "days")
    known = levels.notna().to_numpy()
    observed = check_dates(levels.index, f"levels {levels.name!r}")[known]
    pos = observed.searchsorted(days, side="right") - 1
    found = pos >= 0
    values = np.full(len(days), np.nan)
    values[found] = levels.to_numpy(dtype=float)[known][pos[found]]

    age = days[found] - observed[pos[found]]
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
    observed = as_floats(usdcad.dropna(), "USD/CAD")
    bad = ~np.isfinite(observed) | (observed <= 0)
    if bad.any():
        raise ValueError(f"USD/CAD must be positive and finite, got {observed[bad][0]!r}")
    s = align_levels(usdcad, days, max_age_days=max_age_days)
    return (s / s.shift(1) - 1).rename("fx")


def yield_changes(
    levels: pd.Series, days: pd.DatetimeIndex, *, max_age_days: int = MAX_AGE_DAYS
) -> pd.Series:
    raise NotImplementedError


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
    check_aligned(r_usd, fx)
    return _keep_name(r_usd, (1 + r_usd).mul(1 + fx, axis=0) - 1)


def to_cad_hedged[T: (pd.Series, pd.DataFrame)](
    r_usd: T, rf_cad: pd.Series, rf_usd: pd.Series
) -> T:
    """CAD return of USD assets with the currency hedged by CIP: r_usd + rf_cad - rf_usd."""
    check_aligned(r_usd, rf_cad)
    check_aligned(r_usd, rf_usd)
    return _keep_name(r_usd, r_usd.add(rf_cad - rf_usd, axis=0))


def excess[T: (pd.Series, pd.DataFrame)](r: T, rf: pd.Series) -> T:
    """Returns minus the risk-free rate of the same day."""
    check_aligned(r, rf)
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
    Canadian rate (1971-01-05 with the real sources). They end MAX_AGE_DAYS
    after the last observation of either source at the latest: a source that
    is published later (FRED posts USD/CAD with a lag) shortens the CAD views
    instead of failing, while a holiday at the end is carried like any other
    holiday. A gap of more than MAX_AGE_DAYS inside the period still raises.
    """
    usd = View(assets, rf_usd)
    fx_end = _last_observed(usdcad, "USD/CAD")
    rate_end = _last_observed(ca_tbill, "Canadian T-bill yields")
    if fx_end is None or rate_end is None:
        raise ValueError("no day has both a USD/CAD return and a Canadian rate")
    fresh_until = min(fx_end, rate_end) + pd.Timedelta(days=MAX_AGE_DAYS)
    us_days = check_dates(assets.index, "returns")
    days = us_days[us_days <= fresh_until]
    fx = fx_returns(usdcad, days)
    rf_cad = daily_rate(ca_tbill, days, term_days=term_days)

    usable = (fx.notna() & rf_cad.notna()).to_numpy()  # on `days`, position by position
    if not usable.any():
        raise ValueError("no day has both a USD/CAD return and a Canadian rate")
    cad_days = days[int(usable.argmax()) :]  # from the first usable day on

    r, fx, rf_cad = assets.loc[cad_days], fx.loc[cad_days], rf_cad.loc[cad_days]
    return {
        "USD": usd,
        "CAD": View(to_cad_unhedged(r, fx), rf_cad),
        "CAD_HEDGED": View(to_cad_hedged(r, rf_cad, rf_usd.loc[cad_days]), rf_cad),
    }


def _last_observed(s: pd.Series, what: str) -> pd.Timestamp | None:
    observed = check_dates(s.index, what)[s.notna().to_numpy()]
    return observed[-1] if len(observed) else None


def _keep_name[T: (pd.Series, pd.DataFrame)](like: T, out: T) -> T:
    if isinstance(out, pd.Series):  # `out` is always a new object here
        out.name = like.name
    return out
