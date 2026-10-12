"""The dashboard's data (blueprint Step 8, plans/step-8-plan.md D1-D6).

The app (app/streamlit_app.py) reads a bundle of daily series that `reproduce --dashboard`
writes into app/data/ after the holdout, so the deployed app downloads nothing and never opens
the holdout itself (tests/test_holdout_scan.py covers app/ too). The bundle (`build`, `write`,
`read`):
- per view (`VIEWS`), `returns_<view>.csv.gz`: the risk-free rate, the market (Mkt-RF + RF in
  that view) and every strategy's daily gross return, from the study's backtests;
- per view, `turnover_<view>.csv.gz`: every strategy's turnover by trade date;
- `weights.csv.gz`: the targets by (view, strategy, trade date);
- `factors.csv.gz`: the six factors, RF, the USD/CAD return and the Canadian 10-year yield
  change (`FACTOR_COLUMNS`);
- `holdout/*.csv`: byte copies of the holdout's registered tables (`HOLDOUT_TABLES`);
- `design.csv`: the estimator, the end, the cost of the series and the number of strategies.

The net return at any cost is exact from the gross return and the turnover (`net_at`): after a
trade the weights are exactly the target whatever the cost, so the drift does not depend on it.
`check` refuses a bundle whose 5 bp series do not give back the holdout's own tables, and
`performance` is the study's performance table for any view, strategies, dates, cost and
benchmark. The risk tabs: `var_backtest` (rolling VaR and ES with their backtests), `betas` and
`rolling_betas` (Step 5b's partial factor betas), and `realized` (each strategy through the
registered stress windows, `scenario_days`). For the charts: `growth`, `drawdowns`, `thin` (at
most `MAX_POINTS` per series) and `slots`, which keeps each charted strategy's color while
others come and go (at most `MAX_CHARTED`, the palette's every-pair slots). Everything here is
pure except `build`, which backtests through the guarded engine.
"""

import gzip
import hashlib
import importlib.util
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_lab import study
from portfolio_lab.backtest import engine
from portfolio_lab.config import HOLDOUT_START
from portfolio_lab.construct.strategy import Strategy
from portfolio_lab.report import inference as inf
from portfolio_lab.report import metrics
from portfolio_lab.report.palette import EVERY_PAIR
from portfolio_lab.risk import backtest as bt
from portfolio_lab.risk import factors as fa
from portfolio_lab.risk import var
from portfolio_lab.stress import historical as hs

__all__ = [
    "BENCHMARKS",
    "COST_BPS",
    "FACTOR_COLUMNS",
    "HOLDOUT_TABLES",
    "MARKET",
    "RF",
    "VIEWS",
    "MAX_CHARTED",
    "MAX_POINTS",
    "Bundle",
    "betas",
    "build",
    "check",
    "drawdowns",
    "growth",
    "months",
    "net",
    "net_at",
    "performance",
    "read",
    "realized",
    "rolling_betas",
    "scenario_days",
    "slots",
    "thin",
    "var_backtest",
    "write",
]

VIEWS = ("USD", "CAD", "CAD_HEDGED")
COST_BPS = study.COST_BPS  # the cost of the study's series, which `check` compares
RF = "rf"
MARKET = "market"
BENCHMARKS = (MARKET, study.BENCHMARK)
FACTOR_COLUMNS = (*fa.FACTORS, "RF", fa.FX, fa.DY)
HOLDOUT_TABLES = ("tests", "stress", "stress_realized", "shocks")
DESIGN_KEYS = ("estimator", "end", "cost_bps", "strategies")
MAX_CHARTED = EVERY_PAIR  # lines can cross, so every pair of colors must stay apart
MAX_POINTS = 2000  # per series on a chart; the tables use every day

_SAME = 1e-12  # the bundle must give back the recorded tables to rounding
# pyarrow's CSV reader rounds every float correctly (exact, like float()) and is several times
# faster than pandas' exact parser; it comes with Streamlit, so the app has it.
_FAST_CSV = importlib.util.find_spec("pyarrow") is not None
_YEAR = math.sqrt(12)


@dataclass(frozen=True, eq=False)  # pandas fields: no value equality or hashing
class Bundle:
    """The dashboard's data (see the module doc); `tables` is empty until `read`."""

    design: pd.DataFrame
    returns: Mapping[str, pd.DataFrame]
    turnover: Mapping[str, pd.DataFrame]
    weights: pd.DataFrame
    factors: pd.DataFrame
    tables: Mapping[str, pd.DataFrame]

    @property
    def strategies(self) -> list[str]:
        """The strategies, in the registry's order."""
        return [str(c) for c in self.returns[VIEWS[0]].columns if c not in (RF, MARKET)]


# --- the bundle -------------------------------------------------------------------------------


def build(
    inputs: study.Inputs,
    *,
    end: str | pd.Timestamp,
    estimator: str,
    allow_holdout: bool = False,
    strategies: Sequence[Strategy] | None = None,
) -> Bundle:
    """Every strategy (the registry by default) backtested in every view through `end`, as the
    study runs them: one common first rebalance, 5 bp. Raises unless `net_at` gives back each
    backtest's net returns exactly."""
    group = list(study.strategies() if strategies is None else strategies)
    names = [s.name for s in group]
    if study.BENCHMARK not in names:
        raise ValueError(f"the strategies must include the benchmark {study.BENCHMARK!r}")
    returns: dict[str, pd.DataFrame] = {}
    turnover: dict[str, pd.DataFrame] = {}
    weights: dict[tuple[str, str], pd.DataFrame] = {}
    for v in VIEWS:
        runs = engine.compare(
            group, inputs.views[v], end=end, allow_holdout=allow_holdout, cost_bps=COST_BPS
        )
        first = runs[names[0]]
        days, trades = first.net.index, first.turnover.index
        for name, r in runs.items():
            if not (r.net.index.equals(days) and r.turnover.index.equals(trades)):
                raise ValueError(f"{name!r} in {v} does not share the days of {names[0]!r}")
        frame = pd.DataFrame(
            {RF: first.rf, MARKET: inputs.market[v].reindex(days)}
            | {n: runs[n].gross for n in names}
        )
        frame.index.name = "day"
        traded = pd.DataFrame({n: runs[n].turnover for n in names})
        traded.index.name = "traded"
        _check_values(frame, f"the {v} returns")
        exact = net_at(frame[names], traded, COST_BPS)
        for name in names:
            if not np.array_equal(exact[name].to_numpy(), runs[name].net.to_numpy()):
                raise ValueError(f"net_at does not give back the net returns of {name!r} in {v}")
        returns[v], turnover[v] = frame, traded
        weights |= {(v, n): runs[n].weights for n in names}

    usd = returns["USD"].index
    span = (inputs.factors.index >= usd[0]) & (inputs.factors.index <= usd[-1])
    factors = inputs.factors.loc[span, [*fa.FACTORS, "RF"]].assign(
        **{fa.FX: inputs.fx, fa.DY: inputs.dy}
    )
    factors.index.name = "day"
    design = pd.DataFrame(
        {
            "value": {
                "estimator": estimator,
                "end": str(pd.Timestamp(end).date()),
                "cost_bps": f"{COST_BPS:g}",
                "strategies": str(len(names)),
            }
        }
    )
    design.index.name = "key"
    return Bundle(
        design=design,
        returns=returns,
        turnover=turnover,
        weights=pd.concat(weights, names=["view", "strategy", "traded"]),
        factors=factors,
        tables={},
    )


def write(
    bundle: Bundle, folder: Path, *, copies: Mapping[str, bytes] | None = None
) -> dict[str, str]:
    """Write the bundle into `folder`, with `copies` (relative path -> bytes) as they are, and
    return each file's SHA-256. Gzip files carry no time stamp, so the same bundle gives the
    same bytes."""
    files = {"design.csv": _csv(bundle.design)}
    for v in VIEWS:
        files[f"returns_{v}.csv.gz"] = _gz(_csv(bundle.returns[v]))
        files[f"turnover_{v}.csv.gz"] = _gz(_csv(bundle.turnover[v]))
    files["weights.csv.gz"] = _gz(_csv(bundle.weights))
    files["factors.csv.gz"] = _gz(_csv(bundle.factors))
    files |= dict(copies or {})
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}


def read(folder: Path) -> Bundle:
    """The bundle `write` wrote into `folder`, checked against the schema: every file present,
    the registry's strategies only and the same ones everywhere, finite values."""

    def need(name: str) -> Path:
        path = folder / name
        if not path.exists():
            raise ValueError(
                f"the bundle in {folder} has no {name}: run `portfolio-lab reproduce --dashboard`"
            )
        return path

    design = pd.read_csv(need("design.csv"), index_col="key", dtype=str)
    missing = [k for k in DESIGN_KEYS if k not in design.index]
    if missing:
        raise ValueError(f"design.csv has no {missing}")
    registry = {s.name for s in study.strategies()}
    returns, turnover = {}, {}
    names: list[str] | None = None
    for v in VIEWS:
        file = f"returns_{v}.csv.gz"
        frame = _read(need(file), ["day"])
        if list(frame.columns[:2]) != [RF, MARKET]:
            raise ValueError(f"{file} must start with the columns {RF!r} and {MARKET!r}")
        columns = [str(c) for c in frame.columns[2:]]
        outside = [c for c in columns if c not in registry]
        if outside:
            raise ValueError(f"{file} has columns outside the schema: {outside}")
        names = columns if names is None else names
        if columns != names:
            raise ValueError(f"{file} holds other strategies than returns_{VIEWS[0]}.csv.gz")
        _check_values(frame, file)
        file = f"turnover_{v}.csv.gz"
        traded = _read(need(file), ["traded"])
        if [str(c) for c in traded.columns] != names:
            raise ValueError(f"{file} holds other strategies than returns_{v}.csv.gz")
        _check_values(traded, file)
        if not traded.index.isin(frame.index).all():
            raise ValueError(f"{file} has trade days that are not days of returns_{v}.csv.gz")
        returns[v], turnover[v] = frame, traded
    assert names is not None  # VIEWS is not empty
    if str(len(names)) != design.loc["strategies", "value"]:
        raise ValueError(
            f"design.csv says {design.loc['strategies', 'value']} strategies, the files hold "
            f"{len(names)}"
        )
    weights = _read(need("weights.csv.gz"), ["view", "strategy", "traded"])
    _check_values(weights, "weights.csv.gz", dated=False)  # by view and registry order
    factors = _read(need("factors.csv.gz"), ["day"])
    if list(factors.columns) != list(FACTOR_COLUMNS):
        raise ValueError(f"factors.csv.gz must have the columns {list(FACTOR_COLUMNS)}")
    _check_values(factors[[*fa.FACTORS, "RF"]], "factors.csv.gz")
    tables = {
        t: pd.read_csv(need(f"holdout/{t}.csv"), float_precision="round_trip")
        for t in HOLDOUT_TABLES
    }
    return Bundle(
        design=design,
        returns=returns,
        turnover=turnover,
        weights=weights,
        factors=factors,
        tables=tables,
    )


# --- returns at any cost ----------------------------------------------------------------------


def net_at(gross: pd.DataFrame, turnover: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Daily net returns at `cost_bps` per side: (1 + R)(1 - c) - 1 on each trade day, where
    c = cost_bps / 1e4 x 2 x turnover (the engine's cost), and R itself on every other day.

    `turnover` holds the same columns as `gross`, by trade date (days of `gross`). The result
    equals `engine.reprice` exactly: the same operations on the same numbers.
    """
    rate = engine._cost_rate(cost_bps)
    if list(turnover.columns) != list(gross.columns):
        raise ValueError(
            f"turnover has the columns {list(turnover.columns)}, the returns "
            f"{list(gross.columns)}: they must be the same"
        )
    pos = gross.index.get_indexer(turnover.index)
    if (pos < 0).any():
        day = pd.Timestamp(turnover.index[pos < 0][0])
        raise ValueError(f"trade day {day.date()} is not a day of the returns")
    g = gross.to_numpy(dtype=float)
    out = g.copy()
    cost = rate * (2 * turnover.to_numpy(dtype=float))  # 2 x turnover = the units traded
    out[pos] = g[pos] - cost * (1 + g[pos])
    return pd.DataFrame(out, index=gross.index, columns=gross.columns)


def net(bundle: Bundle, view: str, cost_bps: float) -> pd.DataFrame:
    """Every strategy's daily net returns in `view` at `cost_bps`."""
    _check_view(view)
    names = bundle.strategies
    return net_at(bundle.returns[view][names], bundle.turnover[view], cost_bps)


def months(bundle: Bundle, view: str) -> pd.PeriodIndex:
    """The calendar months of `view` after its first, partial one (as the study drops it)."""
    _check_view(view)
    days = pd.DatetimeIndex(bundle.returns[view].index)
    return pd.PeriodIndex(days.to_period("M").unique()[1:])


# --- the tables -------------------------------------------------------------------------------


def performance(
    bundle: Bundle,
    view: str,
    names: Sequence[str],
    *,
    cost_bps: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    benchmark: str = MARKET,
) -> pd.DataFrame:
    """The study's performance table (`study._performance`) for `names` in `view`, over the
    days from `start` to `end`, at `cost_bps`; tracking error and information ratio against
    `benchmark`, the market or 1/N at the same cost. One row per strategy.

    Monthly columns compound the selected days by calendar month, so `start` and `end` should
    be a month's first and last day.
    """
    _check_view(view)
    unknown = [n for n in names if n not in bundle.strategies]
    if unknown:
        raise ValueError(f"unknown strategies {unknown}: the bundle holds {bundle.strategies}")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}, got {benchmark!r}")
    rate = engine._cost_rate(cost_bps)
    returns, traded = bundle.returns[view], bundle.turnover[view]
    nets = net(bundle, view, cost_bps)
    days = np.asarray((returns.index >= start) & (returns.index <= end))
    trades = np.asarray((traded.index >= start) & (traded.index <= end))
    rf = returns.loc[days, RF]
    bench = returns.loc[days, MARKET] if benchmark == MARKET else nets.loc[days, study.BENCHMARK]
    rows = []
    for name in names:
        r = nets.loc[days, name]
        row = metrics.summary(r, rf, periods_per_year=metrics.TRADING_DAYS, benchmark=bench)
        m = inf.monthly_excess(r, rf)
        years = len(r) / metrics.TRADING_DAYS
        t = traded.loc[trades, name]
        row["sharpe_monthly"] = inf.sharpe_ratio(m) * _YEAR
        row["ceq_monthly"] = inf.ceq(m)
        row["cvar_95_daily"] = var.var_es(r, 0.95).es
        row["annual_turnover"] = t.sum() / years
        row["annual_cost"] = (rate * (2 * t)).sum() / years
        row["days"] = len(r)
        row["months"] = len(m)
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(list(names), name="strategy"))


def check(bundle: Bundle, held_performance: pd.DataFrame, held_monthly: pd.DataFrame) -> None:
    """Refuse a bundle whose 5 bp series do not give back what the holdout recorded, to 1e-12:
    - `held_performance`, the holdout's performance table indexed by (view, strategy): every
      strategy in USD and CAD, every column;
    - `held_monthly`, the robustness run's monthly net and excess returns indexed by (view,
      strategy, month): every series it holds, and at least one in each view.
    """
    start = pd.Timestamp(HOLDOUT_START)
    end = pd.Timestamp(str(bundle.design.loc["end", "value"]))
    names = bundle.strategies
    for v in study.TESTED_VIEWS:
        keys = [(v, n) for n in names]
        missing = [k for k in keys if k not in held_performance.index]
        if missing:
            raise ValueError(
                f"the holdout's performance table has no row for {missing}: the bundle must "
                "come from the holdout it shows"
            )
        mine = performance(bundle, v, names, cost_bps=COST_BPS, start=start, end=end)
        theirs = held_performance.loc[keys]
        for col in mine.columns:
            for name, here, there in zip(
                names, mine[col].to_numpy(float), theirs[col].to_numpy(float), strict=True
            ):
                _same(here, there, f"the {col} of {v} {name!r}", "the holdout's performance table")

    held = held_monthly.reset_index()
    held["month"] = pd.to_datetime(held["month"]).dt.to_period("M")
    for v in VIEWS:
        in_view = held.loc[held["view"] == v]
        present = [n for n in names if (in_view["strategy"] == n).any()]
        if not present:
            raise ValueError(f"the robustness monthly series hold nothing for {v}")
        returns = bundle.returns[v]
        days = np.asarray((returns.index >= start) & (returns.index <= end))
        nets, rf = net(bundle, v, COST_BPS).loc[days], returns.loc[days, RF]
        mine_net = inf.monthly_excess(nets[present], 0.0 * rf)
        mine_excess = inf.monthly_excess(nets[present], rf)
        for name in present:
            theirs = in_view.loc[in_view["strategy"] == name].set_index("month").sort_index()
            ours = {"net": mine_net[name], "excess": mine_excess[name]}
            labels = pd.DatetimeIndex(ours["net"].index).to_period("M")
            if not labels.equals(pd.PeriodIndex(theirs.index)):
                raise ValueError(
                    f"{v} {name!r} covers {len(labels)} holdout months here and "
                    f"{len(theirs)} in the robustness monthly series"
                )
            for col, series in ours.items():
                for month, here, there in zip(
                    labels, series.to_numpy(float), theirs[col].to_numpy(float), strict=True
                ):
                    what = f"the monthly {col} of {v} {name!r} in {month}"
                    _same(here, there, what, "the robustness monthly series")


# --- risk, factors and stress ----------------------------------------------------------------


def var_backtest(
    returns: pd.Series,
    *,
    method: str,
    level: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: int = study.VAR_WINDOW,
) -> tuple[pd.DataFrame, pd.Series]:
    """The rolling one-day VaR and ES of `returns` (Step 5a's `forecast`, from the history
    before each day) on the days from `start` to `end`, and their backtest (`backtest.report`):
    - a frame by day: the return, the VaR and ES (positive losses), and whether the loss went
      past the VaR (an exception);
    - the report's row for `method` at `level` over those days.
    Days after `end` are left out first, so nothing later is read.
    """
    r = returns.loc[returns.index <= end]
    if len(r) <= window or r.index[window] > end or not (r.index[window:] >= start).any():
        raise ValueError(
            f"no forecast day from {start.date()} to {end.date()}: the first forecast needs "
            f"{window} days of history"
        )
    fc = var.forecast(r, method=method, level=level, window=window)
    exc = bt.exceptions(r, fc["var"])
    path = pd.DataFrame(
        {"return": r.reindex(fc.index), "var": fc["var"], "es": fc["es"], "exception": exc}
    )
    path = path.loc[path.index >= start]
    report = bt.report(r, levels=(level,), methods=(method,), window=window, start=start)
    return path, report.iloc[0]


def betas(
    bundle: Bundle,
    view: str,
    names: Sequence[str],
    *,
    cost_bps: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Step 5b's weekly factor betas of each strategy's excess net returns over the days from
    `start` to `end`: FF5 + momentum in USD and CAD hedged, plus the USD/CAD return and the
    Canadian 10-year yield change in CAD (`factors.macro_betas`, all partial)."""
    excess = _excess(bundle, view, names, cost_bps, start, end)
    f = bundle.factors
    if view == "CAD":
        return fa.macro_betas(excess, f[fa.DY], f[fa.FX], f, frequency="weekly")
    return fa.factor_betas(excess, f, frequency="weekly")


def rolling_betas(
    bundle: Bundle,
    view: str,
    name: str,
    *,
    cost_bps: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: int = fa.ROLLING_WINDOW,
) -> pd.DataFrame:
    """Daily factor betas over each trailing `window` of days (with the USD/CAD return in
    CAD), from the history before `start` so the first row is `start`'s, through `end`."""
    first = bundle.returns[view].index[0]
    excess = _excess(bundle, view, [name], cost_bps, first, end)[name]
    fx = bundle.factors[fa.FX] if view == "CAD" else None
    rolled = fa.rolling_betas(excess, bundle.factors, fx=fx, window=window)
    return rolled.loc[rolled.index >= start]


def scenario_days(days: pd.Index, scenario: hs.Scenario) -> np.ndarray:
    """A mask of the days in a registered stress window, (start, end] as Step 6 defines it."""
    d = pd.DatetimeIndex(days)
    return np.asarray((d > scenario.start) & (d <= scenario.end))


def realized(bundle: Bundle, view: str, names: Sequence[str], *, cost_bps: float) -> pd.DataFrame:
    """Each strategy's own path through every registered stress window it covers, at
    `cost_bps` (Step 6's `outcome`: P&L, worst day, drawdown), by (scenario, strategy)."""
    nets = net(bundle, view, cost_bps)
    rows, keys = [], []
    for s in hs.SCENARIOS.values():
        inside = scenario_days(nets.index, s)
        if not inside.any():
            continue
        for name in names:
            rows.append(asdict(hs.outcome(nets.loc[inside, name])))
            keys.append((s.name, name))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=["scenario", "strategy"]))


# --- the charts' series -----------------------------------------------------------------------


def growth(net: pd.DataFrame) -> pd.DataFrame:
    """The value of $1 invested before the first day, at each day's close."""
    return (1 + net).cumprod()


def drawdowns(net: pd.DataFrame) -> pd.DataFrame:
    """The fall from the running peak at each close (0 at a peak, negative below it)."""
    return metrics.drawdowns(net)


def thin(frame: pd.DataFrame, how: str, max_points: int = MAX_POINTS) -> pd.DataFrame:
    """At most `max_points` rows for a chart: the days as they are if they fit, else weeks
    (to Friday), else months. Each period keeps its `how` value, the "last" close or the "min"
    (a drawdown's worst point), labelled by its last day in the data."""
    if how not in ("last", "min"):
        raise ValueError(f"how must be 'last' or 'min', got {how!r}")
    if len(frame) <= max_points:
        return frame
    days = pd.DatetimeIndex(frame.index)
    periods = days.to_period("W-FRI")
    if periods.nunique() > max_points:
        periods = days.to_period("M")
    grouped = frame.groupby(periods)
    out = grouped.last() if how == "last" else grouped.min()
    ends = pd.Series(days, index=days).groupby(periods).max()
    out.index = pd.DatetimeIndex(ends.to_numpy(), name=frame.index.name)
    return out


def slots(
    selected: Sequence[str], previous: Mapping[str, int], size: int = MAX_CHARTED
) -> dict[str, int]:
    """A palette slot for each selected strategy: the one it had in `previous` if it still
    has it, else the lowest free one, so a strategy's color never changes while others are
    added or removed (color follows the entity, not its rank)."""
    if len(selected) > size:
        raise ValueError(f"at most {size} strategies are charted at once, got {len(selected)}")
    kept = {n: s for n, s in previous.items() if n in selected and 0 <= s < size}
    free = [s for s in range(size) if s not in kept.values()]
    return {n: kept[n] if n in kept else free.pop(0) for n in selected}


# --- helpers ----------------------------------------------------------------------------------


def _excess(
    bundle: Bundle,
    view: str,
    names: Sequence[str],
    cost_bps: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """The strategies' daily net returns over the dates, less the view's risk-free rate."""
    nets = net(bundle, view, cost_bps)
    days = np.asarray((nets.index >= start) & (nets.index <= end))
    return nets.loc[days, list(names)].sub(bundle.returns[view].loc[days, RF], axis=0)


def _check_view(view: str) -> None:
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")


def _check_values(frame: pd.DataFrame, what: str, *, dated: bool = True) -> None:
    """Finite values, unique rows, and (`dated`) days in order."""
    if not frame.index.is_unique or (dated and not frame.index.is_monotonic_increasing):
        raise ValueError(f"{what}: the rows must be unique and in order")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        row, col = (int(k) for k in np.argwhere(~np.isfinite(values))[0])
        raise ValueError(f"{what}: values must be finite ({frame.columns[col]!r}, row {row})")


def _same(here: float, there: float, what: str, where: str) -> None:
    both_nan = math.isnan(here) and math.isnan(there)
    if not both_nan and not math.isclose(here, there, rel_tol=_SAME, abs_tol=_SAME):
        raise ValueError(f"{what} is {here!r} in the bundle and {there!r} in {where}")


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(lineterminator="\n").encode("utf-8")


def _gz(content: bytes) -> bytes:
    return gzip.compress(content, mtime=0)


def _read(path: Path, index: list[str]) -> pd.DataFrame:
    """A bundle file with every float exactly as written: pyarrow's reader when installed,
    else pandas' exact parser (several times slower)."""
    content = path.read_bytes()
    if path.suffix == ".gz":
        content = gzip.decompress(content)
    dates = [i for i in index if i in ("day", "traded")]
    if _FAST_CSV:
        frame = pd.read_csv(io.BytesIO(content), engine="pyarrow")
        for col in dates:
            # the unit pandas gives parsed dates (as `parse_dates` does), not seconds
            frame[col] = pd.to_datetime(frame[col], format="%Y-%m-%d").dt.as_unit("us")
        return frame.set_index(index)
    return pd.read_csv(
        io.BytesIO(content), index_col=index, parse_dates=dates, float_precision="round_trip"
    )
