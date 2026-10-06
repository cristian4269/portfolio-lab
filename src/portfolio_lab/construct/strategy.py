"""Strategies: the contract that Step 4's backtest engine calls at each rebalance.

A strategy has a name, a lookback and a rule. The engine passes `weights` the returns history
ending at the estimation date t (rows = periods, oldest first). The strategy reads only the
last `lookback` rows, which must be complete (older rows may have gaps), gives them to the rule
and checks the result: a long-only portfolio (>= 0, summing to 1) labeled by the history's
columns. The engine's first rebalance therefore waits until `lookback` rows exist.

Constructors for the strategies of the study:
- `equal_weight()`: 1/N, lookback 1 (it needs only the assets).
- `min_variance(estimator, *, window, cap)` and `erc(estimator, *, window)`: the weight rule on
  a covariance estimate of the last `window` rows. The estimator is always named, never
  defaulted: one of ESTIMATORS.
- `momentum_tilt(*, strength, base)` and `low_volatility_tilt(*, strength, window, base)`: a
  base strategy (default 1/N) tilted toward momentum 12-1 or low volatility. The lookback
  covers the signal and the base: momentum on a 126-day minimum variance looks back 252 days.
"""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import pandas as pd

from portfolio_lab._validate import last_rows, portfolio_values
from portfolio_lab.construct import covariance as cv
from portfolio_lab.construct import weights as wt

__all__ = [
    "ESTIMATORS",
    "WINDOW",
    "Strategy",
    "equal_weight",
    "erc",
    "low_volatility_tilt",
    "min_variance",
    "momentum_tilt",
]

type Rule = Callable[[pd.DataFrame], pd.Series]
type Estimator = Callable[[pd.DataFrame], pd.DataFrame]

WINDOW = 252  # default estimation window in trading days (about a year)

ESTIMATORS: dict[str, Estimator] = {
    "sample": cv.sample_cov,
    "lw_identity": lambda r: cv.ledoit_wolf_identity(r).cov,
    "lw_constant_correlation": lambda r: cv.ledoit_wolf_constant_correlation(r).cov,
    "ewma": cv.ewma_cov,
}


@dataclass(frozen=True)
class Strategy:
    """A named rule mapping the last `lookback` periods of returns to long-only weights."""

    name: str
    lookback: int
    rule: Rule

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"a strategy needs a name, got {self.name!r}")
        object.__setattr__(self, "lookback", _positive_int(self.lookback, "lookback"))
        if not callable(self.rule):
            raise TypeError(f"rule must be callable, got {type(self.rule).__name__}")

    def weights(self, history: pd.DataFrame) -> pd.Series:
        """The portfolio for the estimation date t, the last row of `history`."""
        window, _ = last_rows(history, f"returns for {self.name!r}", self.lookback)
        w = self.rule(window)
        portfolio_values(w, history.columns, f"weights from {self.name!r}", "the history")
        return w


def equal_weight() -> Strategy:
    """1/N in every asset of the history."""
    return Strategy("1/N", 1, lambda window: wt.equal_weight(window.columns))


def min_variance(estimator: str, *, window: int = WINDOW, cap: float = 1.0) -> Strategy:
    """Long-only minimum variance on the `estimator` covariance of the last `window` rows."""
    estimate, window = _estimator(estimator), _positive_int(window, "window")
    capped = f", cap {cap:g}" if cap != 1 else ""
    return Strategy(
        f"min variance ({estimator}, {window}d{capped})",
        window,
        lambda history: wt.min_variance(estimate(history), cap=cap),
    )


def erc(estimator: str, *, window: int = WINDOW) -> Strategy:
    """Equal risk contribution on the `estimator` covariance of the last `window` rows."""
    estimate, window = _estimator(estimator), _positive_int(window, "window")
    return Strategy(
        f"ERC ({estimator}, {window}d)",
        window,
        lambda history: wt.equal_risk_contribution(estimate(history)),
    )


def momentum_tilt(*, strength: float, base: Strategy | None = None) -> Strategy:
    """`base` (default 1/N) tilted toward momentum 12-1 with the given strength."""
    core = equal_weight() if base is None else base

    def rule(history: pd.DataFrame) -> pd.Series:
        signal = wt.momentum_12_1(history)
        return wt.tilt(signal, strength=strength, base=core.weights(history))

    lookback = max(wt.MOMENTUM_LOOKBACK, core.lookback)
    return Strategy(f"momentum tilt (strength {strength:g}) on {core.name}", lookback, rule)


def low_volatility_tilt(
    *, strength: float, window: int = WINDOW, base: Strategy | None = None
) -> Strategy:
    """`base` (default 1/N) tilted toward low volatility over the last `window` rows."""
    core = equal_weight() if base is None else base
    window = _positive_int(window, "window")

    def rule(history: pd.DataFrame) -> pd.Series:
        signal = wt.low_volatility(history.iloc[len(history) - window :])
        return wt.tilt(signal, strength=strength, base=core.weights(history))

    name = f"low-vol tilt (strength {strength:g}, {window}d) on {core.name}"
    return Strategy(name, max(window, core.lookback), rule)


def _estimator(name: str) -> Estimator:
    if name not in ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}: use one of {', '.join(ESTIMATORS)}")
    return ESTIMATORS[name]


def _positive_int(value: int, what: str) -> int:
    """`value` as an int if it is a positive integer (bools excluded), else ValueError."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return int(value)
