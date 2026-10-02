"""Performance metrics of periodic simple returns (decimal, e.g. 0.01 = 1%).

Every annualized metric takes `periods_per_year` explicitly (252 for daily,
12 for monthly): nothing assumes daily data. A Series gives a number, a
DataFrame gives one value per column. Returns must be finite numbers, no
lower than -100%, with no missing values (drop or fill them first,
knowingly). Drawdowns work from 1 period; everything else needs at least 2
(skew 3, kurtosis 4). A risk-free rate or benchmark Series must have exactly
the returns' index, in the same order.

Conventions:
- CAGR is geometric: (prod(1 + r)) ** (periods_per_year / n) - 1.
- Volatility, Sharpe, tracking error and IR are arithmetic: mean and sample
  std (ddof=1) per period, scaled by periods_per_year and its square root.
- Sharpe and Sortino take the risk-free rate per period (a Series aligned to
  the returns, or a constant), so they always work on excess returns.
- Sortino's downside deviation is sqrt(mean(min(excess, 0)^2)) over *all*
  periods (Sortino and Price 1994), so it has no ddof.
- Drawdowns are measured from the running peak of wealth, starting at 1
  before the first return. Duration = the longest run of periods below an
  earlier peak; a drawdown not recovered by the last period still counts.
  Drawdowns within float noise of the peak (`_RECOVERED`) count as 0.
- A ratio whose risk is zero (Sharpe, Sortino, IR with a constant series;
  Sortino with no period below rf; Calmar with no drawdown) is NaN, not inf:
  undefined, and it can never rank first.
- Skew and excess kurtosis are the bias-corrected sample statistics (pandas
  `skew` and `kurt`; G1 and G2 in Joanes and Gill 1998).

Hand example (periods_per_year=4, so the 4 periods are one year). Returns of
+10%, -20%, +5%, +10% grow $1 to 1.10, 0.88, 0.924, 1.0164:
- CAGR = 1.0164 ** (4 / 4) - 1 = 1.64%.
- Max drawdown = 0.88 / 1.10 - 1 = -20%. Duration 3 periods, not recovered.
- Sharpe (rf = 0) = mean / std * sqrt(4) = 0.0125 / 0.1436 * 2 = 0.174.
- Sortino = 0.0125 / sqrt(0.2^2 / 4) * 2 = 0.25.
- Calmar = 1.64% / 20% = 0.082.
"""

import math

import numpy as np
import pandas as pd

__all__ = [
    "METRICS",
    "MONTHS",
    "TRADING_DAYS",
    "annualized_vol",
    "cagr",
    "calmar",
    "drawdowns",
    "excess_kurtosis",
    "information_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "sharpe",
    "skew",
    "sortino",
    "summary",
    "tracking_error",
]

TRADING_DAYS = 252
MONTHS = 12

METRICS = (
    "cagr",
    "annualized_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "max_drawdown_duration",
    "calmar",
    "skew",
    "excess_kurtosis",
)

# Wealth within this fraction of its peak counts as at the peak (float noise).
_RECOVERED = 1e-12
# A risk measure this small relative to the size of the returns is zero (float noise).
_NOISE = 1e-12

type Returns = pd.Series | pd.DataFrame
type Rate = pd.Series | float


def cagr(r: Returns, *, periods_per_year: float) -> float | pd.Series:
    """Compound annual growth rate."""
    f, ppy = _frame(r), _ppy(periods_per_year)
    return _out(r, (1 + f).prod() ** (ppy / len(f)) - 1)


def annualized_vol(r: Returns, *, periods_per_year: float) -> float | pd.Series:
    """Sample standard deviation (ddof=1) times sqrt(periods_per_year)."""
    f, ppy = _frame(r), _ppy(periods_per_year)
    return _out(r, f.std(ddof=1) * math.sqrt(ppy))


def sharpe(r: Returns, rf: Rate, *, periods_per_year: float) -> float | pd.Series:
    """Annualized Sharpe ratio: mean / std of excess returns, times sqrt(periods_per_year)."""
    f, ppy = _frame(r), _ppy(periods_per_year)
    ex = _minus(f, rf, "rf")
    return _out(r, _ratio(ex.mean(), ex.std(ddof=1), _scale(f, rf)) * math.sqrt(ppy))


def sortino(r: Returns, rf: Rate, *, periods_per_year: float) -> float | pd.Series:
    """Annualized Sortino ratio: mean excess return / downside deviation (target = rf)."""
    f, ppy = _frame(r), _ppy(periods_per_year)
    ex = _minus(f, rf, "rf")
    downside = (ex.clip(upper=0) ** 2).mean() ** 0.5
    return _out(r, _ratio(ex.mean(), downside, _scale(f, rf)) * math.sqrt(ppy))


def drawdowns(r: Returns) -> Returns:
    """Wealth / running peak - 1 each period (0 at a new high, -0.2 = 20% below the peak)."""
    dd = _drawdowns(_frame(r, min_periods=1))
    return dd.iloc[:, 0].rename(r.name) if isinstance(r, pd.Series) else dd


def max_drawdown(r: Returns) -> float | pd.Series:
    """The deepest drawdown, as a negative fraction (0 if wealth never fell)."""
    return _out(r, _drawdowns(_frame(r, min_periods=1)).min())


def max_drawdown_duration(r: Returns) -> int | pd.Series:
    """The longest run of periods spent below an earlier peak, recovered or not."""
    under = _drawdowns(_frame(r, min_periods=1)) < 0

    def longest(col: pd.Series) -> int:
        return int(col.groupby((~col).cumsum()).sum().max())

    out = under.apply(longest)
    return int(out.iloc[0]) if isinstance(r, pd.Series) else out


def calmar(r: Returns, *, periods_per_year: float) -> float | pd.Series:
    """CAGR / |max drawdown|; NaN when wealth never fell."""
    f = _frame(r)
    depth = _drawdowns(f).min().abs()
    return _out(r, cagr(f, periods_per_year=periods_per_year) / depth.where(depth > 0))


def skew(r: Returns) -> float | pd.Series:
    """Bias-corrected sample skewness (G1)."""
    return _out(r, _frame(r, min_periods=3).skew())


def excess_kurtosis(r: Returns) -> float | pd.Series:
    """Bias-corrected sample excess kurtosis (G2; 0 for a normal distribution)."""
    return _out(r, _frame(r, min_periods=4).kurt())


def tracking_error(
    r: Returns, benchmark: pd.Series, *, periods_per_year: float
) -> float | pd.Series:
    """Annualized std (ddof=1) of the active return r - benchmark."""
    f, ppy = _frame(r), _ppy(periods_per_year)
    return _out(r, _minus(f, benchmark, "benchmark").std(ddof=1) * math.sqrt(ppy))


def information_ratio(
    r: Returns, benchmark: pd.Series, *, periods_per_year: float
) -> float | pd.Series:
    """Annualized mean active return / tracking error."""
    f, ppy = _frame(r), _ppy(periods_per_year)
    active = _minus(f, benchmark, "benchmark")
    ratio = _ratio(active.mean(), active.std(ddof=1), _scale(f, benchmark))
    return _out(r, ratio * math.sqrt(ppy))


def summary(
    r: Returns, rf: Rate, *, periods_per_year: float, benchmark: pd.Series | None = None
) -> pd.Series | pd.DataFrame:
    """All METRICS (plus tracking error and IR with a benchmark): one row per metric."""
    f = _frame(r)
    ppy = periods_per_year
    rows: dict[str, pd.Series] = {
        "cagr": cagr(f, periods_per_year=ppy),
        "annualized_vol": annualized_vol(f, periods_per_year=ppy),
        "sharpe": sharpe(f, rf, periods_per_year=ppy),
        "sortino": sortino(f, rf, periods_per_year=ppy),
        "max_drawdown": max_drawdown(f),
        "max_drawdown_duration": max_drawdown_duration(f),
        "calmar": calmar(f, periods_per_year=ppy),
        "skew": skew(f),
        "excess_kurtosis": excess_kurtosis(f),
    }
    if benchmark is not None:
        rows["tracking_error"] = tracking_error(f, benchmark, periods_per_year=ppy)
        rows["information_ratio"] = information_ratio(f, benchmark, periods_per_year=ppy)
    table = pd.DataFrame(rows).T.astype(float)
    return table.iloc[:, 0].rename(r.name) if isinstance(r, pd.Series) else table


def _frame(r: Returns, *, min_periods: int = 2) -> pd.DataFrame:
    f = r.to_frame() if isinstance(r, pd.Series) else r
    if not f.columns.is_unique:
        raise ValueError("returns need unique column names")
    if len(f) < min_periods:
        raise ValueError(f"need at least {min_periods} periods, got {len(f)}")
    try:
        values = f.to_numpy(dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("returns must be numeric") from err
    missing = np.isnan(values).any(axis=1)
    if missing.any():
        raise ValueError(f"returns are missing on {f.index[missing][0]}")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    if (values < -1).any():
        raise ValueError("a return below -100% is impossible without leverage")
    return f.astype(float)


def _drawdowns(f: pd.DataFrame) -> pd.DataFrame:
    wealth = (1 + f).cumprod()
    dd = wealth / wealth.cummax().clip(lower=1.0) - 1
    return dd.where(dd < -_RECOVERED, 0.0)


def _minus(f: pd.DataFrame, other: Rate, what: str) -> pd.DataFrame:
    if isinstance(other, pd.Series):
        if not f.index.equals(other.index):
            raise ValueError(f"{what} dates don't align with the returns")
        if other.isna().any():
            raise ValueError(f"{what} is missing on {other.index[other.isna()][0]}")
        if not np.isfinite(other.to_numpy(dtype=float)).all():
            raise ValueError(f"{what} must be finite")
        return f.sub(other, axis=0)
    if not math.isfinite(other):
        raise ValueError(f"{what} must be finite, got {other}")
    return f - other


def _scale(f: pd.DataFrame, other: Rate) -> pd.Series:
    """The size of the inputs per column, to tell real risk from float noise."""
    other_size = other.abs().max() if isinstance(other, pd.Series) else abs(other)
    return f.abs().max() + other_size


def _ratio(num: pd.Series, risk: pd.Series, scale: pd.Series) -> pd.Series:
    return num / risk.where(risk > _NOISE * scale)


def _ppy(periods_per_year: float) -> float:
    if not (math.isfinite(periods_per_year) and periods_per_year > 0):
        raise ValueError(f"periods_per_year must be positive and finite, got {periods_per_year}")
    return float(periods_per_year)


def _out(r: Returns, values: pd.Series) -> float | pd.Series:
    return float(values.iloc[0]) if isinstance(r, pd.Series) else values
