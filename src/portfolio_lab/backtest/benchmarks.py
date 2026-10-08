"""Benchmarks for the backtests (blueprint Step 4).

- 1/N is a strategy like the others (`construct.strategy.equal_weight()`): it pays costs and
  rebalances monthly through the engine.
- The market is an index: the US market's total return from the French factors, Mkt-RF + RF,
  with no costs and no rebalancing. In the CAD views it is converted exactly as the assets are,
  by `data.views.make_views` on a one-column frame, so it shares their calendar.
"""

import pandas as pd

from portfolio_lab.data.views import make_views

__all__ = ["MARKET", "market", "market_views"]

MARKET = "Market"


def market(factors: pd.DataFrame) -> pd.Series:
    """The US market's daily total return, Mkt-RF + RF, from factors as `load_factors` gives."""
    missing = [col for col in ("Mkt-RF", "RF") if col not in factors.columns]
    if missing:
        raise ValueError(f"factors need the columns {', '.join(missing)}")
    out = factors["Mkt-RF"] + factors["RF"]
    out.name = MARKET
    return out


def market_views(
    factors: pd.DataFrame, usdcad: pd.Series, ca_tbill: pd.Series
) -> dict[str, pd.Series]:
    """The market in each currency view (USD, CAD, CAD_HEDGED), keyed as `make_views` keys them.

    `usdcad` (CAD per USD) and `ca_tbill` (the annualized 3-month yield on a daily calendar)
    are what `make_views` takes for the assets.
    """
    views = make_views(market(factors).to_frame(), factors["RF"], usdcad, ca_tbill)
    return {name: view.returns[MARKET] for name, view in views.items()}
