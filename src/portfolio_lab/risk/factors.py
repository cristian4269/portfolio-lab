"""Factor and macro regressions with Newey-West (HAC) errors (blueprint Step 5b)."""

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "ALPHA",
    "FACTORS",
    "FREQUENCIES",
    "Fit",
    "factor_betas",
    "nw_lags",
    "regress",
    "to_weekly",
]

FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
FREQUENCIES = ("daily", "weekly")
ALPHA = "alpha"


@dataclass(frozen=True, eq=False)
class Fit:
    coef: pd.Series
    se: pd.Series
    tstat: pd.Series
    pvalue: pd.Series
    r2: float
    adj_r2: float
    nobs: int
    lags: int
    start: pd.Timestamp
    end: pd.Timestamp

    def table(self) -> pd.DataFrame:
        raise NotImplementedError


def nw_lags(nobs: int) -> int:
    raise NotImplementedError


def regress(y: pd.Series, X: pd.DataFrame, *, lags: int | None = None) -> Fit:
    raise NotImplementedError


def factor_betas(
    excess: pd.Series | pd.DataFrame,
    factors: pd.DataFrame,
    *,
    fx: pd.Series | None = None,
    frequency: str = "daily",
) -> pd.DataFrame:
    raise NotImplementedError


def to_weekly[T: (pd.Series, pd.DataFrame)](returns: T, *, how: str = "compound") -> T:
    raise NotImplementedError
