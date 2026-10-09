"""Historical stress tests: replay given weights through registered crisis windows (Step 6).

Window (`window`): a scenario covers the returns of the trading days d with start < d <= end.
The portfolio is bought at the close of the last day on or before `start` (the anchor) and held
to the close of the last day on or before `end`. For COVID that is peak to trough, from the
close of 2020-02-19; when `start` or `end` falls on a holiday or a weekend it is the same as
[start, end]. Days are compared as calendar dates, whatever time the stamps carry. The returns
must reach both edges, a day on or before `start` and one on or after the last weekday on or
before `end` (weekends are known to be closed; a holiday is not, so data ending before a
weekday `end` is refused), so a window is never cut short silently.

Replay (`replay`): buy and hold. The weights are bought at the anchor close and drift with the
returns, with no trades and no costs: NAV_t = sum_i w_i prod_{s <= t} (1 + r_i,s), which is the
engine's own accounting (`engine.simulate` with one target). The weights are "current" ones
(at DEV_END in Step 6, at HOLDOUT_END in 7b), so a replay is today's portfolio in that crisis.
What a strategy did at the time is its realized `Result.net`, cut by `window`.

Outcome (`outcome`): in the view's currency, as fractions of the value at the anchor close: the
P&L (the compounded total return), the worst day and its date, the max drawdown counting the
starting value as the first peak (`metrics.max_drawdown`), and the days covered.

Holdout: `window`, `replay` and `table` apply the loaders' rule (`data.resolve_end`) to the
scenario's end, so the 2020 and 2022 windows need `allow_holdout`, which only the `reproduce`
command and research/ may pass. `outcome` is a pure function of the returns it is given.
"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType

import numpy as np
import pandas as pd

from portfolio_lab._validate import as_date, check_dates, portfolio_values, return_series
from portfolio_lab.backtest.engine import simulate
from portfolio_lab.data import resolve_end
from portfolio_lab.data.views import View, check_view
from portfolio_lab.report import metrics

__all__ = ["SCENARIOS", "Outcome", "Scenario", "outcome", "replay", "table", "window"]


@dataclass(frozen=True)
class Scenario:
    """A named crisis window: the returns of the days in (start, end] (see the module doc)."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    note: str

    def __post_init__(self) -> None:
        start, end = as_date(self.start, "start"), as_date(self.end, "end")
        if end <= start:
            raise ValueError(
                f"scenario {self.name!r} must end after it starts ({start.date()} to {end.date()})"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


SCENARIOS: Mapping[str, Scenario] = MappingProxyType(  # read-only: pre-registered (blueprint)
    {
        s.name: s
        for s in (
            Scenario(
                "gfc",
                pd.Timestamp("2008-09-01"),
                pd.Timestamp("2008-11-30"),
                "Global financial crisis: September to November 2008, from the failure of "
                "Lehman Brothers (15 September) through the autumn crash.",
            ),
            Scenario(
                "covid",
                pd.Timestamp("2020-02-19"),
                pd.Timestamp("2020-03-23"),
                "COVID-19 crash: from the S&P 500's closing peak (19 February 2020) to its "
                "closing trough (23 March 2020).",
            ),
            Scenario(
                "rates",
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2022-12-31"),
                "2022 rate shock: calendar 2022, when stocks and bonds fell together as "
                "central banks raised rates.",
            ),
        )
    }
)


@dataclass(frozen=True)
class Outcome:
    """A window's result, as fractions of the value at the anchor close (see the module doc)."""

    pnl: float
    worst_day: float
    worst_date: pd.Timestamp
    max_drawdown: float
    days: int
    first: pd.Timestamp
    last: pd.Timestamp


def window[T: (pd.Series, pd.DataFrame)](
    returns: T, scenario: Scenario, *, allow_holdout: bool = False
) -> T:
    """The rows of `returns` in the scenario's window (start, end], by the holdout rule.

    Raises `HoldoutError` for a scenario that ends after DEV_END without `allow_holdout`, before
    looking at the returns; ValueError if the returns don't reach both edges of the window or
    have no day inside it.
    """
    _check_scenario(scenario, allow_holdout)
    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError(f"returns must be a Series or DataFrame, got {type(returns).__name__}")
    first, stop = _span(check_dates(returns.index, "returns"), scenario)
    return returns.iloc[first:stop]


def replay(
    weights: pd.Series, view: View, scenario: Scenario, *, allow_holdout: bool = False
) -> pd.Series:
    """The daily returns of `weights` bought at the close before the window and held through it.

    `weights`: long-only (>= 0, summing to 1), labeled by the view's assets in their order. Every
    return inside the window must be present and finite, even at weight 0, as in the engine.
    The result is indexed by the window's days.
    """
    _check_scenario(scenario, allow_holdout)
    check_view(view)
    returns = view.returns
    w = portfolio_values(weights, returns.columns, "weights", "the view's returns")
    first, stop = _span(check_dates(returns.index, "returns"), scenario)
    held = returns.iloc[first - 1 : stop]  # the anchor (bought at its close) through the window
    target = pd.DataFrame([w], index=held.index[:1], columns=returns.columns)
    return simulate(target, held, cost_bps=0.0).gross.rename("replay")


def outcome(returns: pd.Series) -> Outcome:
    """P&L, worst day, max drawdown and days covered of one portfolio's daily returns."""
    r = return_series(returns, "returns")  # checks the dates too
    days = pd.DatetimeIndex(returns.index)
    worst = int(np.argmin(r))
    return Outcome(
        pnl=float(np.prod(1 + r) - 1),
        worst_day=float(r[worst]),
        worst_date=days[worst],
        max_drawdown=float(metrics.max_drawdown(returns)),
        days=len(r),
        first=days[0],
        last=days[-1],
    )


def table(
    weights: Mapping[str, Mapping[str, pd.Series]],
    views: Mapping[str, View],
    scenarios: Iterable[Scenario],
    *,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """The `outcome` of every replay: one row per (scenario, view, portfolio).

    `weights` maps a view's name to its portfolios, e.g. {"USD": {"1/N": w}, "CAD": {...}}, and
    every such name must be a key of `views`. Every scenario is checked against the holdout rule
    before anything is replayed.
    """
    chosen = list(scenarios)
    if not chosen:
        raise ValueError("table needs at least one scenario")
    for s in chosen:
        _check_scenario(s, allow_holdout)
    names = [s.name for s in chosen]
    if len(set(names)) < len(names):
        raise ValueError(f"scenario names must be unique (rows are keyed by them): {names}")
    missing = [name for name in weights if name not in views]
    if missing:
        raise ValueError(f"no view for the weights of {missing}: the views are {list(views)}")
    portfolios = [(v, name, w) for v, group in weights.items() for name, w in group.items()]
    if not portfolios:
        raise ValueError("table needs at least one portfolio")
    keys, rows = [], []
    for s in chosen:
        for v, name, w in portfolios:
            keys.append((s.name, v, name))
            rows.append(asdict(outcome(replay(w, views[v], s, allow_holdout=allow_holdout))))
    index = pd.MultiIndex.from_tuples(keys, names=["scenario", "view", "portfolio"])
    return pd.DataFrame(rows, index=index, columns=[f.name for f in fields(Outcome)])


def _check_scenario(scenario: object, allow_holdout: bool) -> None:
    """A Scenario whose end passes the holdout rule (`HoldoutError` otherwise)."""
    if not isinstance(scenario, Scenario):
        raise TypeError(
            f"scenario must be a Scenario (see SCENARIOS), got {type(scenario).__name__}"
        )
    resolve_end(scenario.end, allow_holdout)


def _span(days: pd.DatetimeIndex, scenario: Scenario) -> tuple[int, int]:
    """Positions of the window's first day and one past its last: the days in (start, end].

    Calendar dates are compared, not times of day (a 16:00 close is that day's). Raises unless
    `days` reach both edges, a day on or before `start` to buy at and one on or after the last
    weekday on or before `end` (a weekend is known to be closed, a holiday is not), and hold at
    least one day inside.
    """
    name, start, end = scenario.name, scenario.start.normalize(), scenario.end.normalize()
    dates = days.normalize()
    if len(dates) == 0 or dates[0] > start:
        begins = "have no days" if len(dates) == 0 else f"start on {dates[0].date()}"
        raise ValueError(
            f"the returns {begins}, after {name}'s start ({start.date()}): no close to buy at"
        )
    if dates[-1] < pd.offsets.BDay().rollback(end):
        raise ValueError(
            f"the returns end on {dates[-1].date()}, before {name}'s end ({end.date()}): "
            "the window would be cut short"
        )
    first = int(dates.searchsorted(start, side="right"))
    stop = int(dates.searchsorted(end, side="right"))
    if first == stop:
        raise ValueError(f"no day of the returns falls in {name} ({start.date()}, {end.date()}]")
    return first, stop
