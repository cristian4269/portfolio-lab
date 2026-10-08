"""Walk-forward backtest engine: monthly rebalancing on daily returns (blueprint Step 4).

Calendar (`rebalance_dates`): at the last trading day t of each month the strategy estimates
on the returns history ending at the close of t, and the portfolio trades at the close of the
next trading day (the trade date). The calendar of trading days is known in advance; returns
are not.

Accounting (`simulate`), per day s, with w the weights at the close of s - 1:
- The gross return is R_s = sum_i w_i r_i,s; the weights then drift to w_i (1 + r_i,s) / (1 + R_s).
- On a trade day the drifted weights earn that day's return, then the trade happens at the
  close: turnover = 1/2 sum|w* - w|, and the cost c = cost_bps / 1e4 x sum|w* - w| is deducted
  from NAV after the return, so net = (1 + R)(1 - c) - 1. After the trade the weights are
  exactly the target w*: the cost is paid pro rata, and the cost of selling to pay it is
  ignored (a term of order c x cost_bps / 1e4).
- The backtest starts invested in the first target at the close of its trade date, with no
  opening cost (it would be the same cost_bps for every long-only portfolio): returns start the
  next day, turnover and costs at the second trade.
- Gross returns follow the same weights with c = 0, so net equals gross except on trade days.
Each holding period is priced from cumulative products of (1 + r), so drift error never
outlives a month.

Strategies (`targets`, `run`, `compare`): at each estimation date the strategy gets every row
of returns up to and including t, never a later one. Its first rebalance waits for `lookback`
rows; a `start` that would estimate earlier raises instead of moving. `run` backtests one
strategy on one currency view, so a CAD view estimates on CAD returns; rows after `end` are
dropped first by the loaders' holdout rule. `compare` starts every strategy on one date, and
`reprice` prices a result's targets at another cost without estimating again.

`run`, `compare` and `reprice` apply the holdout rule. `rebalance_dates`, `simulate` and
`targets` are pure functions of the frames they are given, like `data.views`: their inputs
must already come from the guarded loaders.
"""

from collections.abc import Iterable
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from portfolio_lab._validate import (
    as_date,
    check_aligned,
    check_dates,
    check_finite,
    check_frame,
    portfolio_values,
    window_values,
)
from portfolio_lab.config import DEV_END
from portfolio_lab.construct.strategy import Strategy
from portfolio_lab.data import resolve_end
from portfolio_lab.data.views import View
from portfolio_lab.report import metrics

__all__ = [
    "Ledger",
    "Result",
    "compare",
    "rebalance_dates",
    "reprice",
    "run",
    "simulate",
    "targets",
]

type Date = str | pd.Timestamp | None

# A full switch trades 2 units (sell 1, buy 1): at 5000 bp the cost would be the whole NAV.
_MAX_COST_BPS = 5000


@dataclass(frozen=True, eq=False)  # pandas fields: no value equality or hashing
class Ledger:
    """What a sequence of trades earned.

    - `gross`, `net`: daily returns, from the day after the first trade to the last day.
    - `weights`: the targets, by trade date.
    - `turnover`, `cost` (a fraction of NAV): by trade date, from the second trade on.
    - `final_weights`: the drifted weights at the last close.
    """

    gross: pd.Series
    net: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    cost: pd.Series
    final_weights: pd.Series


@dataclass(frozen=True, eq=False)
class Result(Ledger):
    """A strategy's backtest on one view: its ledger, its name, and the view's risk-free rate
    on the same days (Sharpe and Sortino need it)."""

    name: str
    rf: pd.Series

    def __post_init__(self) -> None:
        check_aligned(self.net, self.gross, "gross and net dates")
        check_aligned(self.net, self.rf, "rf and returns dates")
        check_aligned(self.turnover, self.cost, "turnover and cost trade dates")

    def summary(self, benchmark: pd.Series | None = None) -> pd.DataFrame:
        """The step-2 metrics of the daily gross and net returns, one column each, plus the trades.

        Trade rows: `mean_turnover` per rebalance after the first (NaN if none) and
        `annual_turnover` (per 252 days) are the same in both columns, since both follow the
        same trades; `annual_cost` is 0 for gross, which pays none by definition. A benchmark
        adds tracking error and information ratio; it may cover more days than the result but
        must cover all of them.
        """
        days = check_dates(self.net.index, "returns")
        bench = None if benchmark is None else _on_days(benchmark, days)
        both = pd.DataFrame({"gross": self.gross, "net": self.net})
        table = metrics.summary(
            both, self.rf, periods_per_year=metrics.TRADING_DAYS, benchmark=bench
        )
        years = len(days) / metrics.TRADING_DAYS
        table.loc["mean_turnover"] = self.turnover.mean()
        table.loc["annual_turnover"] = self.turnover.sum() / years
        table.loc["annual_cost"] = [0.0, self.cost.sum() / years]  # gross, net
        return table


def rebalance_dates(days: pd.Index, *, start: Date = None) -> pd.Series:
    """Estimation date t -> trade date, for each month of `days` that has a later day.

    t is the month's last day in `days` and the trade date is the next day in `days`, so the
    last day never estimates (nothing trades after it). With `start`, only t on or after it.
    """
    d = check_dates(days, "days")
    months = d.to_period("M")
    last = np.flatnonzero(months[1:] != months[:-1])  # positions whose next day is a new month
    if start is not None:
        last = last[d[last] >= as_date(start, "start")]
    estimated = d[last].rename("estimated")
    return pd.Series(d[last + 1].to_numpy(), index=estimated, name="traded")


def simulate(targets: pd.DataFrame, returns: pd.DataFrame, *, cost_bps: float) -> Ledger:
    """Hold each target from the close of its trade date to the close of the next one.

    `targets`: one long-only portfolio (>= 0, summing to 1) per trade date, labeled by the
    columns of `returns` in their order; the trade dates are days of `returns`, sorted and
    unique. `returns`: daily simple returns, one row per day in date order. Every day after the
    first trade must be complete, finite and >= -100%; earlier rows, and the first trade day's
    own return, are never read. Raises if the portfolio loses everything.
    """
    rate = _cost_rate(cost_bps)
    check_frame(targets, "targets")
    check_frame(returns, "returns")
    days = check_dates(returns.index, "returns")
    if len(targets) == 0:
        raise ValueError("targets need at least one trade date")
    trade_dates = check_dates(targets.index, "targets")
    pos = days.get_indexer(trade_dates)
    if (pos < 0).any():
        raise ValueError(f"trade date {trade_dates[pos < 0][0].date()} is not a day of the returns")
    w = np.vstack(
        [
            portfolio_values(
                targets.iloc[k], returns.columns, f"targets on {day.date()}", "returns"
            )
            for k, day in enumerate(trade_dates)
        ]
    )
    first = int(pos[0])
    if first == len(days) - 1:
        raise ValueError(f"no day after the first trade ({trade_dates[0].date()}) to simulate")
    r = window_values(returns.iloc[first + 1 :], "returns", min_rows=1)
    if (r < -1).any():
        row, col = (int(k) for k in np.argwhere(r < -1)[0])
        raise ValueError(
            f"returns below -100% for {returns.columns[col]} on {days[first + 1 + row].date()}"
        )

    # Row j of r is day first + 1 + j. Target k is held over rows starts[k]:stops[k], i.e. from
    # the day after its trade date through the next trade date (whose return it earns).
    starts = pos - first
    stops = np.r_[starts[1:], len(r)]
    gross = np.empty(len(r))
    traded = np.empty(len(pos) - 1)  # sum |target - drifted| at each trade after the first
    final = w[-1]
    for k, (a, b) in enumerate(zip(starts.tolist(), stops.tolist(), strict=True)):
        if a == b:  # the last trade is on the last day
            break
        held = w[k] * np.cumprod(1 + r[a:b], axis=0)  # holdings at each close, per unit of NAV
        value = held.sum(axis=1)
        if (value <= 0).any():
            day = days[first + 1 + a + int(np.argmax(value <= 0))]
            raise ValueError(f"the portfolio lost everything on {day.date()}: nothing to hold")
        before = np.vstack([w[k], held[:-1]])  # holdings at the previous close
        gross[a:b] = (before * r[a:b]).sum(axis=1) / before.sum(axis=1)
        drifted = held[-1] / value[-1]
        if k + 1 < len(pos):
            traded[k] = np.abs(w[k + 1] - drifted).sum()
        else:
            final = drifted

    cost = rate * traded
    net = gross.copy()
    on_trade = starts[1:] - 1  # the rows of the trade days after the first
    net[on_trade] -= cost * (1 + gross[on_trade])  # (1 + R)(1 - c) - 1, exact when c = 0
    simulated = days[first + 1 :]
    later = trade_dates[1:]
    return Ledger(
        gross=pd.Series(gross, index=simulated, name="gross"),
        net=pd.Series(net, index=simulated, name="net"),
        weights=targets.astype(float),
        turnover=pd.Series(traded / 2, index=later, name="turnover"),
        cost=pd.Series(cost, index=later, name="cost"),
        final_weights=pd.Series(final, index=returns.columns, name="final weights"),
    )


def targets(strategy: Strategy, returns: pd.DataFrame, *, start: Date = None) -> pd.DataFrame:
    """The strategy's portfolio from each month-end estimate, labeled by its trade date.

    At each estimation date t (`rebalance_dates`), `strategy.weights` gets every row of
    `returns` up to and including t. Without `start`, the first t is the first month end with
    `strategy.lookback` rows; a `start` that would estimate earlier than that raises.
    """
    check_frame(returns, "returns")
    days = check_dates(returns.index, "returns")
    schedule = _schedule(days, strategy.lookback, start, strategy.name)
    pos = days.get_indexer(schedule.index)
    rows = [strategy.weights(returns.iloc[: p + 1]) for p in pos]
    # Put in the returns' column order by asset label: a missing asset becomes NaN and an
    # unknown one is dropped, so `simulate` rejects either (not finite, or not summing to 1).
    traded = pd.DatetimeIndex(schedule.to_numpy(), name="traded")
    return pd.DataFrame(rows, index=traded, columns=returns.columns)


def run(
    strategy: Strategy,
    view: View,
    *,
    start: Date = None,
    end: Date = DEV_END,
    allow_holdout: bool = False,
    cost_bps: float = 5.0,
) -> Result:
    """Backtest `strategy` on one currency view: monthly estimates, trades at t + 1, costs.

    The strategy estimates on the view's own returns. Rows after `end` are dropped before
    anything else; an `end` past DEV_END (or None) needs `allow_holdout`, which only the
    `reproduce` command and research/ may pass (as for the loaders).
    """
    _cost_rate(cost_bps)  # fail before estimating
    returns, rf = _until(view, end, allow_holdout)
    ledger = simulate(targets(strategy, returns, start=start), returns, cost_bps=cost_bps)
    return _result(ledger, strategy.name, rf)


def reprice(result: Result, view: View, *, cost_bps: float, allow_holdout: bool = False) -> Result:
    """`result`'s targets at another cost, without estimating again (Step 7b's cost grid).

    `view` must be the one the result was run on. Its rows are cut at the result's last day by
    the holdout rule (a result that ends past DEV_END needs `allow_holdout` again), and it must
    still hold every day of the result.
    """
    if not isinstance(result, Result):
        raise TypeError(f"result must be a Result from run or compare, got {type(result).__name__}")
    _cost_rate(cost_bps)
    days = result.net.index
    returns, rf = _until(view, days[-1], allow_holdout)
    ledger = simulate(result.weights, returns, cost_bps=cost_bps)
    if not ledger.net.index.equals(days):
        raise ValueError(
            f"the view does not hold the days {result.name!r} was run on "
            f"({days[0].date()} to {days[-1].date()}): pass the view it was run on"
        )
    return _result(ledger, result.name, rf)


def compare(
    strategies: Iterable[Strategy],
    view: View,
    *,
    start: Date = None,
    end: Date = DEV_END,
    allow_holdout: bool = False,
    cost_bps: float = 5.0,
) -> dict[str, Result]:
    """Backtest every strategy from one common first rebalance; results keyed by name.

    Without `start`, the first month end where every strategy has its lookback (the longest
    decides). 1/N, with a lookback of 1, would otherwise start sooner and cover other months.
    A `start` too early for any strategy raises.
    """
    group = list(strategies)
    if not group:
        raise ValueError("compare needs at least one strategy")
    names = [s.name for s in group]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"strategy names must be unique (results are keyed by name): {repeated}")
    # Check the start against the longest lookback before estimating anything.
    returns, _ = _until(view, end, allow_holdout)
    days = check_dates(returns.index, "returns")
    longest = max(group, key=lambda s: s.lookback)
    start = _schedule(days, longest.lookback, start, longest.name).index[0]
    return {
        s.name: run(s, view, start=start, end=end, allow_holdout=allow_holdout, cost_bps=cost_bps)
        for s in group
    }


def _result(ledger: Ledger, name: str, rf: pd.Series) -> Result:
    parts = {f.name: getattr(ledger, f.name) for f in fields(Ledger)}
    return Result(**parts, name=name, rf=rf.loc[ledger.net.index])


def _schedule(days: pd.DatetimeIndex, lookback: int, start: Date, name: str) -> pd.Series:
    """The rebalance dates from `start`, by default the first month end with `lookback` rows.

    Raises if `start` would estimate on fewer rows, or leaves no month end to trade after.
    """
    earliest = _earliest(days, lookback)
    schedule = rebalance_dates(days, start=earliest if start is None else start)
    if schedule.empty:
        raise ValueError(f"no month end on or after start={start!r} has a day after it to trade")
    if schedule.index[0] < earliest:
        raise ValueError(
            f"{name!r} needs {lookback} days of history: its first rebalance can be "
            f"{earliest.date()}, not {schedule.index[0].date()} (start={start!r})"
        )
    return schedule


def _earliest(days: pd.DatetimeIndex, lookback: int) -> pd.Timestamp:
    """The first estimation date with at least `lookback` rows of history, t included."""
    estimated = rebalance_dates(days).index
    enough = days.get_indexer(estimated) + 1 >= lookback
    if not enough.any():
        raise ValueError(
            f"no month end has {lookback} days of history and a day after it to trade "
            f"(the returns have {len(days)} days)"
        )
    return pd.Timestamp(estimated[enough][0])


def _until(view: View, end: Date, allow_holdout: bool) -> tuple[pd.DataFrame, pd.Series]:
    """The view's returns and risk-free rate through `end`, by the loaders' holdout rule."""
    if not isinstance(view, View):
        raise TypeError(
            f"view must be a View (returns and their risk-free rate, from make_views), "
            f"got {type(view).__name__}"
        )
    end_ts = resolve_end(end, allow_holdout)
    if end_ts is None:
        return view.returns, view.rf
    keep = view.returns.index <= end_ts
    return view.returns.loc[keep], view.rf.loc[keep]


def _on_days(benchmark: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """The benchmark's returns on exactly `days` (it may cover more)."""
    if not isinstance(benchmark, pd.Series):
        raise TypeError(f"benchmark must be a Series of returns, got {type(benchmark).__name__}")
    missing = days.difference(check_dates(benchmark.index, "benchmark"))
    if len(missing):
        raise ValueError(f"the benchmark has no return on {missing[0].date()}")
    return benchmark.reindex(days)


def _cost_rate(cost_bps: float) -> float:
    """The cost per unit traded, from basis points in [0, 5000)."""
    check_finite(cost_bps, "cost_bps")
    if not 0 <= cost_bps < _MAX_COST_BPS:
        raise ValueError(
            f"cost_bps must be in [0, {_MAX_COST_BPS}), got {cost_bps}: a full switch trades 2 "
            "units, so 5000 bp would cost the whole NAV"
        )
    return cost_bps / 1e4
