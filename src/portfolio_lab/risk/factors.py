"""Factor and macro regressions with Newey-West (HAC) errors (blueprint Step 5b).

Model: OLS of excess returns on factor returns, with an intercept `alpha` per period (daily
or weekly, not annualized).
- USD view (`factor_betas`): excess returns (over French RF) on FF5 + momentum (`FACTORS`).
- CAD view (unhedged, `factor_betas(..., fx=)`): excess returns (over the Canadian T-bill) on
  the same factors plus the USD/CAD return fx = S_t / S_{t-1} - 1, S = CAD per USD (Step 2's
  `fx_returns`). A positive fx beta means the CAD investor gains when the USD strengthens;
  unhedged USD assets have a beta near 1. The CAD alpha also absorbs the US-Canada rate gap
  (CAD excess = USD excess + fx + r_usd fx + rf_usd - rf_cad), so compare alphas within a
  view, not across views. For a stress, "CAD +10%" (the CAD appreciates) is fx = 1/1.10 - 1.
- The CAD-hedged view needs no model of its own: its excess return is the USD one (Step 2, CIP).
- Macro (`macro_betas`): the CAD model plus dy, the daily change in the Canadian 10-year yield
  (`views.yield_changes`, in yield units: a +200 bp shock moves P&L by 0.02 times the dy
  beta). Every beta is partial, with the other regressors held fixed, which is how Step 6
  applies its shocks. Without the factors, the yield beta measures the stock-bond correlation,
  not rate risk: on dev data (2001-2010) yields rose on good equity days, and the 1/N yield
  beta was +8.5 alone against -0.03 with the factors held fixed. That correlation turned in
  2022.
- Rolling (`rolling_betas`): the daily model over a trailing window, each row indexed by the
  window's last day. It describes the past (for the dashboard) and is no forecast, unlike
  `var.forecast`, whose rows are indexed by the day they forecast. Point estimates only.

Errors: Newey-West (HAC) with a Bartlett kernel and the lag floor(4 (T/100)^(2/9))
(`nw_lags`), no small-sample correction, normal p-values (statsmodels' defaults):
V = (X'X)^-1 Omega (X'X)^-1, with Omega = sum_t e_t^2 x_t x_t'
+ sum_{j=1..L} (1 - j/(L+1)) sum_t e_t e_{t-j} (x_t x_{t-j}' + x_{t-j} x_t').

Timing caveat: the daily USD/CAD rate is the noon New York fix, while US returns run to the
4 pm close. The fx move of the afternoon lands in the next day's fix, so a daily fx beta is
blurred by that asynchrony. Hence weekly regressions too (`frequency="weekly"`):
Wednesday-to-Wednesday weeks (a Wednesday is a market holiday far less often than a Friday),
returns compounded and yield changes summed over the trading days of the week, each week
labelled by its Wednesday.

Alignment: the inputs are cut to the dates they share, and within that span their dates must
match exactly (a monthly series against daily factors raises). A block of missing values at
the start or end (a portfolio listed later, a series that stops earlier) is dropped; a gap
between observed values raises, because HAC lags count rows and a gap would make days that
are not adjacent look adjacent. Weeks are formed after that, so the returns and the
regressors always cover the same days, partial weeks at either end included. Each fit reports
its number of observations and its first and last dates (week labels for weekly fits).

The functions are pure and unguarded, like `var`: they regress whatever they are given.
"""

import math
import numbers
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.regression.linear_model import OLS

from portfolio_lab._validate import as_floats, check_aligned, check_dates

__all__ = [
    "ALPHA",
    "DY",
    "FACTORS",
    "FREQUENCIES",
    "FX",
    "ROLLING_WINDOW",
    "WEEK",
    "Fit",
    "factor_betas",
    "macro_betas",
    "nw_lags",
    "regress",
    "rolling_betas",
    "to_weekly",
]

FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
FREQUENCIES = ("daily", "weekly")
WEEK = "W-WED"  # weeks ending on Wednesday, labelled by it
ALPHA = "alpha"
FX = "fx"
DY = "dy"
ROLLING_WINDOW = 252  # trading days (~1 year)
_UNNAMED = "returns"  # the asset label of an unnamed Series

type _Part = tuple[str, pd.Series | pd.DataFrame]


@dataclass(frozen=True, eq=False)  # Series fields: no value equality or hashing
class Fit:
    """One regression: coefficients and HAC errors by term (`alpha` first), and its sample."""

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
        """Coefficient, HAC standard error, t-stat and p-value per term."""
        return pd.DataFrame({"coef": self.coef, "se": self.se, "t": self.tstat, "p": self.pvalue})


def nw_lags(nobs: int) -> int:
    """Newey-West lag for T observations: floor(4 (T/100)^(2/9)), the whole number when the
    value is one (floats put 4 (512)^(2/9) a hair under 16)."""
    if isinstance(nobs, bool | np.bool_) or not isinstance(nobs, numbers.Integral) or nobs < 1:
        raise ValueError(f"the number of observations must be a positive integer, got {nobs!r}")
    t = int(nobs)
    # The float value is off by far less than 1, so start just below it and step up exactly,
    # in integers: lag <= 4 (T/100)^(2/9)  <=>  lag^9 * 100^2 <= 4^9 * T^2.
    lag = max(math.floor(4 * (t / 100) ** (2 / 9)) - 1, 0)
    while (lag + 1) ** 9 * 10_000 <= 4**9 * t * t:
        lag += 1
    return lag


def regress(y: pd.Series, X: pd.DataFrame, *, lags: int | None = None) -> Fit:
    """OLS of `y` on `X` plus an intercept, with Newey-West errors (lag `nw_lags(T)` unless
    `lags` is given). The inputs are aligned as the module doc says."""
    if not isinstance(y, pd.Series):
        raise TypeError(f"y must be a Series, got {type(y).__name__}")
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame of regressors, got {type(X).__name__}")
    _check_names(X)
    return _fit(*_sample(("y", y), [("regressors", X)]), lags)


def factor_betas(
    excess: pd.Series | pd.DataFrame,
    factors: pd.DataFrame,
    *,
    fx: pd.Series | None = None,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Factor regressions of each excess-return column on FF5 + momentum (plus `fx`, the
    USD/CAD return, for the CAD view), daily or weekly.

    `factors` is a frame as `load_factors` gives it (other columns, like RF, are ignored).
    Returns one block per asset, indexed by (asset, term): coef, se, t, p, then the fit's r2,
    adj_r2, nobs, lags, start and end on every row of its block. An unnamed Series is labelled
    "returns".
    """
    return _each_asset(excess, _factor_parts(factors, fx), frequency)


def macro_betas(
    excess: pd.Series | pd.DataFrame,
    d_cad10y: pd.Series,
    fx: pd.Series,
    factors: pd.DataFrame,
    *,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Partial sensitivities to the Canadian 10-year yield and to USD/CAD, the inputs of
    Step 6's hypothetical shocks: CAD excess returns on FF5 + momentum, `fx` and `d_cad10y`
    (the daily yield change, as `views.yield_changes` gives it, named `dy` in the table).

    The factors are held fixed (see the module doc for why). Days before the first yield
    change (the daily 10-year starts in 2001) drop out, and the table reports the sample.
    Weekly: yield changes are summed, returns compounded. Same table as `factor_betas`.
    """
    if not isinstance(fx, pd.Series):
        raise TypeError(f"fx must be a Series of USD/CAD returns, got {type(fx).__name__}")
    if not isinstance(d_cad10y, pd.Series):
        raise TypeError(
            f"d_cad10y must be a Series of yield changes, got {type(d_cad10y).__name__}"
        )
    regressors = [*_factor_parts(factors, fx), (DY, d_cad10y.rename(DY))]
    return _each_asset(excess, regressors, frequency, summed={DY})


def rolling_betas(
    excess: pd.Series,
    factors: pd.DataFrame,
    *,
    fx: pd.Series | None = None,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Daily `factor_betas` coefficients over each trailing `window` of days, one row per
    window, indexed by its last day (a description, not a forecast). Point estimates only.

    Same model and alignment as `factor_betas`; the first row is the `window`-th day of the
    aligned sample. Raises naming the day if a window's regressors are collinear (a factor
    flat over the whole window).
    """
    if not isinstance(excess, pd.Series):
        raise TypeError(f"excess must be a Series of one portfolio, got {type(excess).__name__}")
    name = _UNNAMED if excess.name is None else str(excess.name)
    y, X = _sample((name, excess), _factor_parts(factors, fx))
    _check_names(X)
    k = X.shape[1] + 1
    if (
        isinstance(window, bool | np.bool_)
        or not isinstance(window, numbers.Integral)
        or not k < window <= len(y)
    ):
        raise ValueError(
            f"window must be a whole number of days, more than the {k} parameters and at most "
            f"the {len(y)} days of data, got {window!r}"
        )
    # Each window's normal equations, all at once: windows are strided views, never copies.
    design = np.column_stack([np.ones(len(y)), as_floats(X, "regressors")])
    xw = sliding_window_view(design, int(window), axis=0)  # (windows, k, window)
    yw = sliding_window_view(as_floats(y, name), int(window))  # (windows, window)
    xtx = np.einsum("nkw,nlw->nkl", xw, xw)
    xty = np.einsum("nkw,nw->nk", xw, yw)
    singular = np.linalg.matrix_rank(xtx) < k
    if singular.any():
        day = y.index[int(window) - 1 + int(np.argmax(singular))]
        raise ValueError(
            f"the regressors are collinear in the window ending {day.date()} "
            "(a factor flat over the whole window?)"
        )
    betas = np.linalg.solve(xtx, xty[..., None])[..., 0]
    terms = pd.Index([ALPHA, *X.columns], name="term")
    return pd.DataFrame(betas, index=y.index[int(window) - 1 :], columns=terms)


def to_weekly[T: (pd.Series, pd.DataFrame)](returns: T, *, how: str = "compound") -> T:
    """Daily values to Wednesday-to-Wednesday weeks, labelled by the Wednesday.

    how="compound" compounds returns, prod(1 + r) - 1; how="sum" adds changes, like yield
    changes, which do not compound. Weeks with no rows do not appear. Every value must be
    present and finite.
    """
    if how not in ("compound", "sum"):
        raise ValueError(f'how must be "compound" or "sum", got {how!r}')
    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError(f"returns must be a Series or a DataFrame, got {type(returns).__name__}")
    check_dates(returns.index, "returns")
    values = as_floats(returns, "returns").reshape(len(returns), -1)
    missing = np.isnan(values).any(axis=1)
    if missing.any():
        day = returns.index[int(np.argmax(missing))]
        raise ValueError(f"returns are missing on {day.date()}: drop or fill them first")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    weeks = returns.resample(WEEK)
    out = (1 + returns).resample(WEEK).prod() - 1 if how == "compound" else weeks.sum()
    return out[weeks.size().to_numpy() > 0]


# --- internals --------------------------------------------------------------------------------


def _check_frequency(frequency: str) -> None:
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}, got {frequency!r}")


def _check_names(X: pd.DataFrame) -> None:
    if X.shape[1] == 0:
        raise ValueError("a regression needs at least one regressor")
    if not X.columns.is_unique:
        raise ValueError("regressors need unique names")
    if ALPHA in X.columns:
        raise ValueError(f"{ALPHA!r} names the intercept; rename that regressor")


def _each_asset(
    excess: pd.Series | pd.DataFrame,
    regressors: Sequence[_Part],
    frequency: str,
    summed: Collection[str] = (),
) -> pd.DataFrame:
    """One regression per column of `excess` (an unnamed Series is "returns"), as a long table."""
    if isinstance(excess, pd.Series):
        assets = {_UNNAMED if excess.name is None else str(excess.name): excess}
    elif isinstance(excess, pd.DataFrame):
        if not excess.columns.is_unique:
            raise ValueError("excess returns need unique column names")
        assets = {str(col): excess[col] for col in excess.columns}
    else:
        raise TypeError(f"excess must be a Series or a DataFrame, got {type(excess).__name__}")
    _check_frequency(frequency)
    fits = {name: _model(name, y, regressors, frequency, summed) for name, y in assets.items()}
    return _long_table(fits)


def _factor_parts(factors: pd.DataFrame, fx: pd.Series | None) -> list[_Part]:
    if not isinstance(factors, pd.DataFrame):
        raise TypeError(f"factors must be a DataFrame, got {type(factors).__name__}")
    missing = [name for name in FACTORS if name not in factors.columns]
    if missing:
        raise ValueError(f"factors need the columns {', '.join(missing)}")
    parts: list[_Part] = [("factors", factors[list(FACTORS)])]
    if fx is not None:
        if not isinstance(fx, pd.Series):
            raise TypeError(f"fx must be a Series of USD/CAD returns, got {type(fx).__name__}")
        parts.append((FX, fx.rename(FX)))
    return parts


def _model(
    name: str,
    y: pd.Series,
    regressors: Sequence[_Part],
    frequency: str,
    summed: Collection[str] = (),
) -> Fit:
    """Align, then (weekly) make weeks, then fit. `summed` columns are changes, not returns."""
    y, X = _sample((name, y), regressors)
    _check_names(X)
    if frequency == "weekly":
        y = to_weekly(y)
        X = pd.concat(
            [to_weekly(X[[c]], how="sum" if c in summed else "compound") for c in X.columns],
            axis=1,
        )
    return _fit(y, X, None)


def _sample(
    y: tuple[str, pd.Series], regressors: Sequence[_Part]
) -> tuple[pd.Series, pd.DataFrame]:
    """`y` and the regressors (side by side) on their shared span, with matching dates and the
    missing edges dropped."""
    parts: list[_Part] = [y, *regressors]
    for what, obj in parts:
        check_dates(obj.index, what)
        if len(obj) == 0:
            raise ValueError(f"{what} have no rows")
    start = max(obj.index[0] for _, obj in parts)
    end = min(obj.index[-1] for _, obj in parts)
    names = " and ".join(what for what, _ in parts)
    if start > end:
        raise ValueError(f"{names} have no dates in common")
    y_cut = y[1].loc[(y[1].index >= start) & (y[1].index <= end)]
    cut: list[_Part] = [(y[0], y_cut)]
    cut += [(what, obj.loc[(obj.index >= start) & (obj.index <= end)]) for what, obj in regressors]
    first_what, first = cut[0]
    for what, obj in cut[1:]:
        check_aligned(first, obj, f"{first_what} and {what} dates")

    labels: list[str] = []
    blocks = []
    for what, obj in cut:
        values = as_floats(obj, what).reshape(len(obj), -1)
        cols = [what] if isinstance(obj, pd.Series) else [f"{what} {c}" for c in obj.columns]
        if np.isinf(values).any():
            raise ValueError(f"{what} must be finite")
        labels += cols
        blocks.append(values)
    values = np.column_stack(blocks)
    complete = ~np.isnan(values).any(axis=1)
    if not complete.any():
        raise ValueError(f"{names} have no complete rows on their shared dates")
    lo = int(np.argmax(complete))
    hi = len(complete) - int(np.argmax(complete[::-1]))
    gaps = ~complete[lo:hi]
    if gaps.any():
        row = lo + int(np.argmax(gaps))
        col = int(np.argmax(np.isnan(values[row])))
        raise ValueError(
            f"{labels[col]} is missing on {first.index[row].date()}, between observed values: "
            "fill or cut it (HAC lags count rows, so a gap would join days that aren't adjacent)"
        )
    return y_cut.iloc[lo:hi], pd.concat([obj.iloc[lo:hi] for _, obj in cut[1:]], axis=1)


def _fit(y: pd.Series, X: pd.DataFrame, lags: int | None) -> Fit:
    nobs, k = len(y), X.shape[1] + 1
    if nobs <= k:
        raise ValueError(
            f"a regression on {k} parameters needs more than {k} observations, got {nobs}"
        )
    design = np.column_stack([np.ones(nobs), as_floats(X, "regressors")])
    if np.linalg.matrix_rank(design) < k:
        raise ValueError(
            "the regressors are collinear (or one is constant, which duplicates the intercept)"
        )
    if lags is None:
        lags = nw_lags(nobs)
    elif isinstance(lags, bool | np.bool_) or not isinstance(lags, numbers.Integral):
        raise ValueError(f"lags must be a whole number, got {lags!r}")
    elif not 0 <= lags < nobs:
        raise ValueError(f"lags must be in [0, {nobs}), got {lags}")
    res = OLS(as_floats(y, "y"), design).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})

    terms = pd.Index([ALPHA, *X.columns], name="term")
    return Fit(
        coef=pd.Series(res.params, index=terms, name="coef"),
        se=pd.Series(res.bse, index=terms, name="se"),
        tstat=pd.Series(res.tvalues, index=terms, name="t"),
        pvalue=pd.Series(res.pvalues, index=terms, name="p"),
        r2=float(res.rsquared),
        adj_r2=float(res.rsquared_adj),
        nobs=nobs,
        lags=int(lags),
        start=y.index[0],
        end=y.index[-1],
    )


def _long_table(fits: dict[str, Fit]) -> pd.DataFrame:
    blocks = []
    for fit in fits.values():
        block = fit.table()
        block["r2"] = fit.r2
        block["adj_r2"] = fit.adj_r2
        block["nobs"] = fit.nobs
        block["lags"] = fit.lags
        block["start"] = fit.start
        block["end"] = fit.end
        blocks.append(block)
    return pd.concat(blocks, keys=list(fits), names=["asset", "term"])
