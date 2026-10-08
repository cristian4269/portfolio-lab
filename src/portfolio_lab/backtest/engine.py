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
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_lab._validate import check_dates, check_frame, portfolio_values, window_values

__all__ = ["Ledger", "rebalance_dates", "simulate"]


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


def rebalance_dates(days: pd.Index, *, start: str | pd.Timestamp | None = None) -> pd.Series:
    """Estimation date t -> trade date, for each month of `days` that has a later day.

    t is the month's last day in `days` and the trade date is the next day in `days`, so the
    last day never estimates (nothing trades after it). With `start`, only t on or after it.
    """
    d = check_dates(days, "days")
    months = d.to_period("M")
    last = np.flatnonzero(months[1:] != months[:-1])  # positions whose next day is a new month
    if start is not None:
        start_ts = pd.Timestamp(start)
        if pd.isna(start_ts):
            raise ValueError(f"start={start!r} is not a date")
        last = last[d[last] >= start_ts]
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


def _cost_rate(cost_bps: float) -> float:
    """The cost per unit traded, from basis points; must be finite and >= 0."""
    if not (math.isfinite(cost_bps) and cost_bps >= 0):
        raise ValueError(f"cost_bps must be finite and >= 0, got {cost_bps}")
    return cost_bps / 1e4
