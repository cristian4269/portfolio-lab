"""Data loaders for freely shareable sources (see DATA_SOURCES.md).

The four `load_*` functions are the only way the rest of the library gets
data. They stop at DEV_END by default; reading the holdout takes both
`allow_holdout` and an explicit later `end` (or `end=None` for everything),
which only the `reproduce` command and research/ may pass (tests enforce it).

Timing convention: row t holds only values observed by the close of day t.
Daily series sit on their own day (returns, the noon USD/CAD fix, the daily
10-year yield); the backtest adds the t+1 trade lag (blueprint: estimate
through close t, trade at close t+1). Monthly rates get a further one
business day margin (see `point_in_time_daily`).

Units: returns are simple daily decimal returns; USD/CAD is CAD per 1 USD;
rates are annualized decimal yields as quoted by the source (converted to
daily rates in step 2).
"""

import pandas as pd

from portfolio_lab.config import DEV_END
from portfolio_lab.data import boc, fred, french, statcan

__all__ = [
    "HoldoutError",
    "load_assets",
    "load_factors",
    "load_fx",
    "load_rates",
    "point_in_time_daily",
]

type End = str | pd.Timestamp | None


class HoldoutError(ValueError):
    """Raised when data after DEV_END is requested without allow_holdout."""


def _resolve_end(end: End, allow_holdout: bool) -> pd.Timestamp | None:
    end_ts = None if end is None else pd.Timestamp(end)
    if end_ts is not None and pd.isna(end_ts):
        # "", "NaT" and NaN all become NaT, and `.loc[:NaT]` would return everything.
        raise ValueError(f"end={end!r} is not a date")
    within_dev = end_ts is not None and end_ts <= pd.Timestamp(DEV_END)
    if not allow_holdout and not within_dev:  # fails closed: only a real date <= DEV_END passes
        raise HoldoutError(
            f"end={end!r} reaches past DEV_END={DEV_END}, into the holdout. The holdout opens "
            "only after the pre-registration in research/PLAN.md is committed (blueprint "
            "step 7a), from the `reproduce` command or research/."
        )
    return end_ts


def _truncate[T: (pd.Series, pd.DataFrame)](obj: T, end: pd.Timestamp | None) -> T:
    # A mask, not `.loc[:end]`: label slicing keeps later rows if the index is unsorted.
    return obj if end is None else obj.loc[obj.index <= end]


def load_assets(
    universe: int = 12, *, end: End = DEV_END, allow_holdout: bool = False, refresh: bool = False
) -> pd.DataFrame:
    """Daily value-weighted returns of the 12 (or 49) French industry portfolios."""
    end_ts = _resolve_end(end, allow_holdout)
    return _truncate(french.industry_returns(universe, refresh=refresh), end_ts)


def load_factors(
    *, end: End = DEV_END, allow_holdout: bool = False, refresh: bool = False
) -> pd.DataFrame:
    """Daily FF5 + momentum factor returns and the US risk-free rate RF."""
    end_ts = _resolve_end(end, allow_holdout)
    return _truncate(french.factors(refresh=refresh), end_ts)


def load_fx(*, end: End = DEV_END, allow_holdout: bool = False, refresh: bool = False) -> pd.Series:
    """Daily USD/CAD, CAD per 1 USD (noon New York; NaN on US holidays)."""
    end_ts = _resolve_end(end, allow_holdout)
    return _truncate(fred.usdcad(refresh=refresh), end_ts)


def load_rates(
    *, end: End = DEV_END, allow_holdout: bool = False, refresh: bool = False
) -> pd.DataFrame:
    """Canadian yields on a business-day calendar (timing convention in the module doc).

    CA_TBILL_3M and CA_GOVT_LONG are monthly (last Wednesday) and enter the
    calendar the business day after they are observed. CA_GOVT_10Y is the daily
    Valet benchmark close, on its own day, NaN before 2001.
    """
    end_ts = _resolve_end(end, allow_holdout)
    monthly = {
        "CA_TBILL_3M": statcan.fetch_vector("v122531", refresh=refresh),
        "CA_GOVT_LONG": statcan.fetch_vector("v122487", refresh=refresh),
    }
    daily_10y = boc.fetch_valet("BD.CDN.10YR.DQ.YLD", refresh=refresh)

    usable_from = [_next_business_day(s.dropna().index) for s in monthly.values()]
    first = min(min(ix[0] for ix in usable_from), daily_10y.index[0])
    last = max(max(ix[-1] for ix in usable_from), daily_10y.index[-1])
    if end_ts is not None:
        last = min(last, end_ts)
    if last < first:
        raise ValueError(
            f"end={end!r} is before the first Canadian rate observation ({first.date()})"
        )
    calendar = pd.bdate_range(first, last, name="date")

    out = pd.DataFrame({name: point_in_time_daily(s, calendar) for name, s in monthly.items()})
    out["CA_GOVT_10Y"] = daily_10y.reindex(calendar)
    return out


def point_in_time_daily(monthly: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Spread values stamped on their observation day over `calendar`, without look-ahead.

    A value observed on day t is first used on the next business day; it then
    holds until the next observed value. Missing observations keep the last one.
    The one-day margin also covers a source observing later than its stamp
    ("last Wednesday unless otherwise stated").

    `BDay` knows weekends but no holidays: if the next weekday is not in
    `calendar`, the value simply starts on the following calendar date.
    """
    observed = monthly.dropna()
    shifted = pd.Series(observed.to_numpy(), index=_next_business_day(observed.index))
    union = calendar.union(shifted.index)
    out = shifted.reindex(union).ffill().reindex(calendar)
    return out.rename(monthly.name)


def _next_business_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index + pd.offsets.BDay(1)
