"""Luck benchmark (blueprint Step 7a, research/PLAN.md §7): turnover-matched random shadows.

A shadow of a backtest trades on the backtest's own trade dates, at its cost, with its
turnover, but toward random portfolios:
- It starts at a Dirichlet(1) portfolio (uniform on the simplex) at the first trade date.
- At each later trade date it draws u ~ Dirichlet(1) and trades from its drifted weights w to
  w + λ(u − w), with λ = min(1, 2τ / Σ|u − w|): its turnover ½ Σ|target − w| is the strategy's
  τ at that trade, unless reaching it would take more than the whole step to u (λ capped at
  1). The capped trades are counted (`Shadows.shortfall`).
- Between trades the weights drift with returns, and the cost is charged on the trade day
  after that day's return, as in `engine.simulate` (a test checks they agree).

Many shadows run at once (vectorized over them). `percentile` places a strategy among its
shadows; `best_of` asks how often the best of K shadows (one per strategy) does as well as
the best of the K strategies, the data-snooping question when K strategies were tried.

Pure, like `engine.simulate`: the returns passed in must come from the guarded loaders.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_lab._validate import check_dates, check_frame, is_whole, window_values
from portfolio_lab.backtest.engine import Result, _cost_rate

__all__ = ["SHADOWS", "Shadows", "best_of", "percentile", "shadows"]

SHADOWS = 1000  # per strategy (research/PLAN.md §7)


@dataclass(frozen=True, eq=False)
class Shadows:
    """Daily net returns of each shadow (columns 0 .. count − 1) on the result's days, and the
    share of (later trade, shadow) pairs whose turnover fell short of the strategy's."""

    net: pd.DataFrame
    shortfall: float


@dataclass(frozen=True, eq=False)
class _Paths:
    net: np.ndarray  # days x shadows
    shortfall: float
    targets: np.ndarray | None  # shadows x trades x assets, when kept


def shadows(
    result: Result,
    returns: pd.DataFrame,
    *,
    cost_bps: float,
    rng: np.random.Generator,
    count: int = SHADOWS,
) -> Shadows:
    """`count` turnover-matched random shadows of `result`, on the asset `returns` it ran on."""
    paths = _shadow_paths(result, returns, cost_bps=cost_bps, rng=rng, count=count)
    net = pd.DataFrame(paths.net, index=result.net.index, columns=range(count))
    return Shadows(net=net, shortfall=paths.shortfall)


def percentile(value: float, sample: Sequence[float] | np.ndarray | pd.Series) -> float:
    """The share of the sample strictly below `value`."""
    x = np.asarray(sample, dtype=float)
    return float(np.mean(x < value))


def best_of(actual: pd.Series, shadow: pd.DataFrame) -> float:
    """The share of replicates (rows) whose best shadow (over the strategies, columns) reaches
    the best actual value. Row r holds shadow r of every strategy."""
    if set(actual.index) != set(shadow.columns):
        raise ValueError(
            f"the shadows' strategies {list(shadow.columns)} differ from {list(actual.index)}"
        )
    return float(np.mean(shadow[actual.index].max(axis=1).to_numpy() >= actual.max()))


def _shadow_paths(
    result: Result,
    returns: pd.DataFrame,
    *,
    cost_bps: float,
    rng: np.random.Generator,
    count: int,
    keep_targets: bool = False,
) -> _Paths:
    if not isinstance(result, Result):
        raise TypeError(f"result must be a Result from run or compare, got {type(result).__name__}")
    if not is_whole(count) or count < 1:
        raise ValueError(f"count must be a whole number >= 1, got {count!r}")
    rate = _cost_rate(cost_bps)
    check_frame(returns, "returns")
    days = check_dates(returns.index, "returns")
    if not returns.columns.equals(result.final_weights.index):
        raise ValueError(
            f"the returns' assets {list(returns.columns)} are not the result's "
            f"{list(result.final_weights.index)}"
        )
    trade_dates = result.weights.index
    pos = days.get_indexer(trade_dates)
    first = int(pos[0])
    held_days = result.net.index
    if (pos < 0).any() or not days[first + 1 : first + 1 + len(held_days)].equals(held_days):
        raise ValueError(
            f"the returns do not hold all the days of {result.name!r} "
            f"({trade_dates[0].date()} to {held_days[-1].date()})"
        )
    r = window_values(returns.iloc[first + 1 : first + 1 + len(held_days)], "returns", min_rows=1)
    tau = result.turnover.to_numpy(dtype=float)
    n_assets = r.shape[1]

    starts = pos - first  # row of r where target k is first held
    stops = np.r_[starts[1:], len(r)]
    net = np.empty((len(r), count))
    w = rng.dirichlet(np.ones(n_assets), size=count)
    kept = [w] if keep_targets else []
    short = 0
    for k, (a, b) in enumerate(zip(starts.tolist(), stops.tolist(), strict=True)):
        if a == b:  # the last trade is on the last day
            break
        held = w[:, None, :] * np.cumprod(1 + r[a:b], axis=0)[None]  # shadows x days x assets
        value = held.sum(axis=2)
        before = np.concatenate([np.ones((count, 1)), value[:, :-1]], axis=1)
        net[a:b] = (value / before - 1).T
        if k + 1 == len(pos):  # the last target is held to the end: no trade follows
            continue
        drifted = held[:, -1] / value[:, -1:]
        u = rng.dirichlet(np.ones(n_assets), size=count)
        gap = np.abs(u - drifted).sum(axis=1)
        need = 2 * tau[k]
        step = np.minimum(1.0, need / gap)
        short += int(np.count_nonzero(need > gap))
        w = drifted + step[:, None] * (u - drifted)
        if keep_targets:
            kept.append(w)
        cost = rate * step * gap  # rate x sum |target - drifted|
        row = b - 1  # the trade day: its return is earned, then the trade pays
        net[row] = (1 + net[row]) * (1 - cost) - 1
    trades = (len(pos) - 1) * count
    return _Paths(
        net=net,
        shortfall=short / trades if trades else 0.0,
        targets=np.stack(kept, axis=1) if keep_targets else None,
    )
