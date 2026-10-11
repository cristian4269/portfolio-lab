"""The pre-registered study as code (blueprint Step 7a; research/PLAN.md is the authority).

The registry (`strategies`, `CANDIDATES`, `SEED` and the other constants) is the plan's §4-§7
written down; a test checks that every strategy name appears in PLAN.md. `run` turns the data
of one period into the result tables, `load` reads the data through the guarded loaders.

A run backtests every strategy once per tested view (USD and CAD unhedged), from the common
first rebalance of the set to `end`, at 5 bp. It then evaluates one period:
- **dev:** the months after the run's first (partial) month, through DEV_END. The primary
  estimator is chosen here by the rule of §4 (`selection`), and the minimum detectable
  difference is scaled to the holdout's length (§8).
- **holdout:** the months 2011-01 → `end` of the same continuous run, with the estimator
  frozen in PLAN.md. Only the `reproduce` command may pass `allow_holdout`.

Tables (one DataFrame each): `design`, `selection` (dev), `performance`, `tests`,
`calibration`, `block_sensitivity`, `luck`, `best_of`, `var`, `stress`, `stress_realized`,
`shocks` and `mde` (dev, when the holdout's length is given). Sharpe ratios and their
differences are monthly, annualized by √12 (§5); `performance` also has the daily metrics.
"""

import math
import zlib
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from portfolio_lab import data
from portfolio_lab.backtest import engine, luck
from portfolio_lab.backtest.benchmarks import market_views
from portfolio_lab.backtest.engine import Result
from portfolio_lab.config import DEV_END, HOLDOUT_START
from portfolio_lab.construct import strategy as st
from portfolio_lab.construct.strategy import Strategy
from portfolio_lab.data.views import View, fx_returns, make_views, yield_changes
from portfolio_lab.report import inference as inf
from portfolio_lab.report import metrics
from portfolio_lab.risk import backtest as bt
from portfolio_lab.risk import factors as fa
from portfolio_lab.risk import var
from portfolio_lab.stress import historical as hs
from portfolio_lab.stress import hypothetical as hy

__all__ = [
    "BENCHMARK",
    "CANDIDATES",
    "COST_BPS",
    "PERIODS",
    "PRIMARY_ESTIMATOR",
    "SEED",
    "TESTED_VIEWS",
    "Inputs",
    "Settings",
    "load",
    "period_months",
    "primary",
    "rng_for",
    "run",
    "selection",
    "strategies",
]

SEED = 20261009
COST_BPS = 5.0
CAP = 0.25
CANDIDATES = ("sample", "lw_identity", "lw_constant_correlation")  # §4: EWMA is not one
ESTIMATORS = (*CANDIDATES, "ewma")
TILT_STRENGTHS = (0.5, 1.0)
BENCHMARK = "1/N"
TESTED_VIEWS = ("USD", "CAD")
PERIODS = ("dev", "holdout")
ALPHA = 0.05
VAR_WINDOW = 500
VAR_SENSITIVITY_WINDOW = 1000
DEV_SHOCK_START = "2001-01-03"  # the daily Canadian 10-year starts in 2001 (Step 6)
FIRST_ASSET_DAY = "1963-07-01"  # FF5 starts in July 1963
PRIMARY_ESTIMATOR: str | None = "lw_identity"  # frozen from the dev rule (PLAN.md §13)

_YEAR = math.sqrt(12)  # monthly Sharpe ratios and their differences, annualized


def strategies() -> list[Strategy]:
    """1/N, then the 16 strategies of PLAN.md §6, in its order."""
    out = [st.equal_weight()]
    out += [st.min_variance(e) for e in ESTIMATORS]
    out += [st.min_variance(e, cap=CAP) for e in ESTIMATORS]
    out += [st.erc(e) for e in ESTIMATORS]
    out += [st.momentum_tilt(strength=s) for s in TILT_STRENGTHS]
    out += [st.low_volatility_tilt(strength=s) for s in TILT_STRENGTHS]
    return out


_REGISTRY = strategies  # `run` takes a `strategies` argument that shadows the name


def primary(estimator: str) -> tuple[str, str]:
    """The names of the primary minimum variance and ERC strategies for an estimator."""
    if estimator not in CANDIDATES:
        raise ValueError(
            f"the primary estimator must be a candidate {CANDIDATES}, got {estimator!r}"
        )
    return st.min_variance(estimator).name, st.erc(estimator).name


@dataclass(frozen=True)
class Settings:
    """The sizes of the random parts; the defaults are the registered ones (§5, §7).

    `workers` is how many processes calibrate the primary pairs at once. Each pair has its own
    seeded stream, so the results do not depend on it.
    """

    shadows: int = luck.SHADOWS
    resamples: int = inf.RESAMPLES
    pseudo: int = inf.PSEUDO
    calibration_resamples: int = inf.CALIBRATION_RESAMPLES
    blocks: tuple[int, ...] = inf.BLOCKS
    workers: int = 1


@dataclass(frozen=True, eq=False)
class Inputs:
    """The data of a run: the currency views, the market in each, the factors, and the FX
    returns and 10-year yield changes on the CAD days."""

    views: Mapping[str, View]
    market: Mapping[str, pd.Series]
    factors: pd.DataFrame
    fx: pd.Series
    dy: pd.Series


def load(*, end: str | pd.Timestamp | None, allow_holdout: bool, refresh: bool = False) -> Inputs:
    """The study's data through `end`, by the loaders' holdout rule."""
    f = data.load_factors(end=end, allow_holdout=allow_holdout, refresh=refresh)
    fx_levels = data.load_fx(end=end, allow_holdout=allow_holdout, refresh=refresh)
    rates = data.load_rates(end=end, allow_holdout=allow_holdout, refresh=refresh)
    assets = data.load_assets(12, end=end, allow_holdout=allow_holdout, refresh=refresh)
    assets = assets.loc[assets.index >= pd.Timestamp(FIRST_ASSET_DAY)]
    # RF on the asset days exactly: a missing day becomes NaN, which View refuses.
    views = make_views(assets, f["RF"].reindex(assets.index), fx_levels, rates["CA_TBILL_3M"])
    cad_days = pd.DatetimeIndex(views["CAD"].returns.index)
    return Inputs(
        views=views,
        market=market_views(f, fx_levels, rates["CA_TBILL_3M"]),
        factors=f,
        fx=fx_returns(fx_levels, cad_days),
        dy=yield_changes(rates["CA_GOVT_10Y"], cad_days),
    )


def rng_for(*labels: object) -> np.random.Generator:
    """A generator seeded from SEED and stable labels (crc32), never from global state."""
    return np.random.default_rng([SEED, *(zlib.crc32(str(x).encode()) for x in labels)])


def period_months[T: (pd.Series, pd.DataFrame)](monthly: T, period: str) -> T:
    """A run's monthly returns in one period: the first (partial) month dropped, then dev
    through DEV_END or holdout from HOLDOUT_START."""
    rest = monthly.iloc[1:]
    if period == "dev":
        return rest.loc[rest.index <= pd.Timestamp(DEV_END)]
    if period == "holdout":
        return rest.loc[rest.index >= pd.Timestamp(HOLDOUT_START)]
    raise ValueError(f"period must be one of {PERIODS}, got {period!r}")


def selection(monthly_usd: pd.DataFrame) -> pd.DataFrame:
    """The dev rule (§4): the candidate whose minimum variance has the lowest annualized
    volatility (ddof 1) of monthly net excess returns; ties go to the first candidate."""
    names = [st.min_variance(e).name for e in CANDIDATES]
    vol = np.array([monthly_usd[n].std(ddof=1) * _YEAR for n in names])
    if not np.isfinite(vol).all():
        raise ValueError(
            f"a candidate has no finite dev volatility: {dict(zip(names, vol, strict=True))}"
        )
    chosen = np.zeros(len(CANDIDATES), dtype=bool)
    chosen[int(np.argmin(vol))] = True
    index = pd.Index(CANDIDATES, name="estimator")
    return pd.DataFrame({"strategy": names, "volatility": vol, "chosen": chosen}, index=index)


def run(
    inputs: Inputs,
    *,
    period: str,
    end: str | pd.Timestamp,
    estimator: str | None = None,
    allow_holdout: bool = False,
    settings: Settings | None = None,
    strategies: Sequence[Strategy] | None = None,
    holdout_months: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Every result table of one period (see the module doc)."""
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, got {period!r}")
    if period == "dev" and estimator is not None:
        raise ValueError("on dev the estimator comes from the dev rule (§4): pass none")
    if period == "holdout" and estimator is None:
        raise ValueError("the holdout needs the estimator frozen in PLAN.md §13")
    sizes = Settings() if settings is None else settings
    group = list(_REGISTRY() if strategies is None else strategies)
    names = [s.name for s in group]
    if BENCHMARK not in names:
        raise ValueError(f"the strategies must include the benchmark {BENCHMARK!r}")

    results = {
        v: engine.compare(
            group, inputs.views[v], end=end, allow_holdout=allow_holdout, cost_bps=COST_BPS
        )
        for v in TESTED_VIEWS
    }
    monthly = {
        v: period_months(
            pd.DataFrame({n: inf.monthly_excess(r.net, r.rf) for n, r in res.items()}), period
        )
        for v, res in results.items()
    }
    if period == "holdout":
        _check_reaches(results, end)
    tables: dict[str, pd.DataFrame] = {}
    if period == "dev":
        tables["selection"] = selection(monthly["USD"])
        chosen = str(tables["selection"].index[tables["selection"]["chosen"]][0])
    else:
        assert estimator is not None  # checked above
        chosen = estimator
    pair = primary(chosen)
    missing = [n for n in pair if n not in names]
    if missing:
        raise ValueError(f"the strategies must include the primary pair: {missing}")

    usd = monthly["USD"]
    tables["design"] = pd.DataFrame(
        {
            "value": {
                "period": period,
                "estimator": chosen,
                "end": str(pd.Timestamp(end).date()),
                "first_month": str(usd.index[0].date()),
                "last_month": str(usd.index[-1].date()),
                "months_usd": str(len(usd)),
                "months_cad": str(len(monthly["CAD"])),
                "strategies": str(len(group)),
                "seed": str(SEED),
            }
        }
    )
    tables["performance"] = _performance(results, monthly, inputs)
    tests, calibration, sensitivity = _tests(monthly, pair, period, sizes)
    tables["tests"] = tests
    tables["calibration"] = calibration
    tables["block_sensitivity"] = sensitivity
    tables["luck"], tables["best_of"] = _luck(results, monthly, inputs, period, sizes)
    tables["var"] = _var(results, monthly, pair)
    tables["stress"], tables["stress_realized"] = _stress(
        results, inputs, pair, period, allow_holdout
    )
    tables["shocks"] = _shocks(results, inputs, pair, period, end)
    if period == "dev" and holdout_months is not None:
        tables["mde"] = _mde(monthly, pair, holdout_months)
    return tables


# --- the tables -------------------------------------------------------------------------------


def _performance(
    results: Mapping[str, Mapping[str, Result]],
    monthly: Mapping[str, pd.DataFrame],
    inputs: Inputs,
) -> pd.DataFrame:
    rows, keys = [], []
    for v, res in results.items():
        m = monthly[v]
        for name, r in res.items():
            days = _in_months(r.net.index, m.index)
            net, rf = r.net.loc[days], r.rf.loc[days]
            bench = inputs.market[v].reindex(net.index)
            row = metrics.summary(net, rf, periods_per_year=metrics.TRADING_DAYS, benchmark=bench)
            trades = _in_months(r.turnover.index, m.index)
            years = len(net) / metrics.TRADING_DAYS
            row["sharpe_monthly"] = inf.sharpe_ratio(m[name]) * _YEAR
            row["ceq_monthly"] = inf.ceq(m[name])
            row["cvar_95_daily"] = var.var_es(net, 0.95).es
            row["annual_turnover"] = r.turnover.loc[trades].sum() / years
            row["annual_cost"] = r.cost.loc[trades].sum() / years
            row["days"] = len(net)
            row["months"] = len(m)
            rows.append(row)
            keys.append((v, name))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=["view", "strategy"]))


def _tests(
    monthly: Mapping[str, pd.DataFrame], pair: tuple[str, str], period: str, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mv = pair[0]
    rows, cal_rows, sens_rows = [], [], []
    jobs = [
        (m[name], m[BENCHMARK], (period, v, name, "calibration"), settings)
        for v, m in monthly.items()
        for name in pair
    ]
    if settings.workers > 1:
        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            done = list(pool.map(_calibrate, jobs))
    else:
        done = [_calibrate(job) for job in jobs]
    calibrations = {(job[2][1], job[2][2]): cal for job, cal in zip(jobs, done, strict=True)}
    for v, m in monthly.items():
        bench = m[BENCHMARK]
        blocks: dict[str, int] = {}
        by_block: dict[tuple[str, int], inf.SharpeTest] = {}
        for name in pair:
            cal = calibrations[(v, name)]
            blocks[name] = cal.block
            for b, rate in zip(settings.blocks, cal.rejection.to_numpy(), strict=True):
                cal_rows.append((v, name, b, float(rate), b == cal.block))
            for b in settings.blocks:  # the sensitivity; the calibrated block is the test itself
                res = inf.lw_test(
                    m[name],
                    bench,
                    block=b,
                    resamples=settings.resamples,
                    rng=rng_for(period, v, name, "test", b),
                )
                by_block[(name, b)] = res
                sens_rows.append((v, name, b, res.diff * _YEAR, res.pvalue))
        for name in m.columns:
            if name == BENCHMARK:
                continue
            b = blocks.get(name, blocks[mv])
            if (name, b) in by_block:  # a primary pair: its test at the calibrated block
                res = by_block[(name, b)]
            else:
                res = inf.lw_test(
                    m[name],
                    bench,
                    block=b,
                    resamples=settings.resamples,
                    rng=rng_for(period, v, name, "test", b),
                )
            jk = inf.jobson_korkie_memmel(m[name], bench)
            rows.append(
                {
                    "view": v,
                    "strategy": name,
                    "family": "primary" if name in pair else "exploratory",
                    "months": len(m),
                    "diff": res.diff * _YEAR,
                    "se": res.se * _YEAR,
                    "ci_low": res.ci_low * _YEAR,
                    "ci_high": res.ci_high * _YEAR,
                    "block": b,
                    "p": res.pvalue,
                    "jk_z": jk.z,
                    "jk_p": jk.pvalue,
                }
            )
    tests = pd.DataFrame(rows).set_index(["view", "strategy"])
    tests["holm_all"] = inf.holm(tests["p"])
    is_primary = tests["family"] == "primary"
    tests["holm_primary"] = np.nan
    tests.loc[is_primary, "holm_primary"] = inf.holm(tests.loc[is_primary, "p"])
    tests["reject_primary"] = tests["holm_primary"] <= ALPHA
    calibration = pd.DataFrame(
        cal_rows, columns=["view", "strategy", "block", "rejection", "chosen"]
    ).set_index(["view", "strategy", "block"])
    sensitivity = pd.DataFrame(
        sens_rows, columns=["view", "strategy", "block", "diff", "p"]
    ).set_index(["view", "strategy", "block"])
    return tests, calibration, sensitivity


def _calibrate(
    job: tuple[pd.Series, pd.Series, tuple[object, ...], Settings],
) -> inf.Calibration:
    """One primary pair's block calibration (module level, so a worker process can run it)."""
    r1, r2, labels, settings = job
    return inf.calibrate_block(
        r1,
        r2,
        rng=rng_for(*labels),
        blocks=settings.blocks,
        pseudo=settings.pseudo,
        resamples=settings.calibration_resamples,
    )


def _luck(
    results: Mapping[str, Mapping[str, Result]],
    monthly: Mapping[str, pd.DataFrame],
    inputs: Inputs,
    period: str,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, best_rows = [], []
    for v, res in results.items():
        m = monthly[v]
        actual = inf.sharpe_ratio(m) * _YEAR
        shadow_sharpes: dict[str, np.ndarray] = {}
        returns = inputs.views[v].returns
        for name, r in res.items():
            sh = luck.shadows(
                r,
                returns.loc[returns.index <= r.net.index[-1]],
                cost_bps=COST_BPS,
                rng=rng_for(period, v, name, "luck"),
                count=settings.shadows,
            )
            sm = inf.monthly_excess(sh.net, r.rf).loc[m.index]
            ss = (sm.mean() / sm.std(ddof=0)).to_numpy() * _YEAR  # the same population Sharpe
            shadow_sharpes[name] = ss
            rows.append(
                {
                    "view": v,
                    "strategy": name,
                    "sharpe": actual[name],
                    "percentile": luck.percentile(actual[name], ss),
                    "shadow_median": float(np.median(ss)),
                    "shadow_p95": float(np.quantile(ss, 0.95)),
                    "shortfall": sh.shortfall,
                    "turnover_ratio": sh.turnover_ratio,
                }
            )
        tried = [n for n in m.columns if n != BENCHMARK]
        best = actual[tried]
        best_rows.append(
            {
                "view": v,
                "k": len(tried),
                "best": str(best.idxmax()),
                "best_sharpe": float(best.max()),
                "p_best_of": luck.best_of(
                    best, pd.DataFrame({n: shadow_sharpes[n] for n in tried})
                ),
            }
        )
    return (
        pd.DataFrame(rows).set_index(["view", "strategy"]),
        pd.DataFrame(best_rows).set_index("view"),
    )


def _var(
    results: Mapping[str, Mapping[str, Result]],
    monthly: Mapping[str, pd.DataFrame],
    pair: tuple[str, str],
) -> pd.DataFrame:
    parts = []
    for v, res in results.items():
        start = _first_day(res[BENCHMARK].net.index, monthly[v].index)
        for name in (BENCHMARK, *pair):
            net = res[name].net
            for window, methods in ((VAR_WINDOW, var.METHODS), (VAR_SENSITIVITY_WINDOW, ("fhs",))):
                table = bt.report(net, window=window, methods=methods, start=start)
                parts.append(
                    pd.concat({(v, name, window): table}, names=["view", "strategy", "window"])
                )
    return pd.concat(parts)


def _final_weights(
    results: Mapping[str, Mapping[str, Result]], pair: tuple[str, str]
) -> dict[str, dict[str, pd.Series]]:
    return {v: {n: res[n].final_weights for n in (BENCHMARK, *pair)} for v, res in results.items()}


def _stress(
    results: Mapping[str, Mapping[str, Result]],
    inputs: Inputs,
    pair: tuple[str, str],
    period: str,
    allow_holdout: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = ("gfc",) if period == "dev" else tuple(hs.SCENARIOS)
    scenarios = [hs.SCENARIOS[s] for s in names]
    views = {v: inputs.views[v] for v in TESTED_VIEWS}
    replay = hs.table(_final_weights(results, pair), views, scenarios, allow_holdout=allow_holdout)
    keys, rows = [], []
    for s in scenarios:
        for v, res in results.items():
            for name in (BENCHMARK, *pair):
                out = hs.outcome(hs.window(res[name].net, s, allow_holdout=allow_holdout))
                keys.append((s.name, v, name))
                rows.append(asdict(out))
    index = pd.MultiIndex.from_tuples(keys, names=["scenario", "view", "portfolio"])
    return replay, pd.DataFrame(rows, index=index)


def _shocks(
    results: Mapping[str, Mapping[str, Result]],
    inputs: Inputs,
    pair: tuple[str, str],
    period: str,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    # The period-end holdings, with betas over the period's span; on the holdout, also over
    # 2001 -> DEV_END, the comparison PLAN.md §10 registers.
    spans: dict[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] = {
        "dev": (DEV_SHOCK_START, DEV_END)
    }
    if period == "holdout":
        spans = {"holdout": (HOLDOUT_START, end), **spans}
    parts = {}
    for span, (lo, hi) in spans.items():
        for v, group in _final_weights(results, pair).items():
            ports = pd.DataFrame(
                {n: hy.portfolio_excess(w, inputs.views[v]) for n, w in group.items()}
            )
            keep = (ports.index >= pd.Timestamp(lo)) & (ports.index <= pd.Timestamp(hi))
            ports = ports.loc[keep]
            if v == "CAD":
                days = ports.index
                betas = fa.macro_betas(
                    ports,
                    inputs.dy.loc[days],
                    inputs.fx.loc[days],
                    inputs.factors,
                    frequency="weekly",
                )
            else:
                betas = fa.factor_betas(ports, inputs.factors, frequency="weekly")
            terms = set(betas.index.get_level_values("term"))
            parts[(span, v)] = hy.apply(betas, [s for s in hy.SHOCKS.values() if s.term in terms])
    return pd.concat(parts, names=["span", "view"])


def _mde(
    monthly: Mapping[str, pd.DataFrame], pair: tuple[str, str], holdout_months: int
) -> pd.DataFrame:
    rows = []
    family = len(monthly) * len(pair)  # Holm's first step divides alpha by the family size
    for v, m in monthly.items():
        for name in pair:
            se = inf.diff_se(m[name], m[BENCHMARK]) * math.sqrt(len(m) / holdout_months) * _YEAR
            rows.append(
                {
                    "view": v,
                    "strategy": name,
                    "dev_months": len(m),
                    "holdout_months": holdout_months,
                    "se": se,
                    "mde": inf.min_detectable_difference(se, alpha=ALPHA),
                    "mde_holm": inf.min_detectable_difference(se, alpha=ALPHA / family),
                }
            )
    return pd.DataFrame(rows).set_index(["view", "strategy"])


# --- helpers ----------------------------------------------------------------------------------


def _check_reaches(results: Mapping[str, Mapping[str, Result]], end: str | pd.Timestamp) -> None:
    """The holdout must reach HOLDOUT_END: every view's last day within its last week, in its
    month, so the last month is complete and no shorter holdout passes silently."""
    stop = pd.Timestamp(end)
    for v, res in results.items():
        last = res[BENCHMARK].net.index[-1]
        if last < stop - pd.Timedelta(days=6) or last.to_period("M") != stop.to_period("M"):
            raise ValueError(
                f"the {v} data end on {last.date()}, short of HOLDOUT_END {stop.date()}: "
                "refresh the data or check the sources"
            )


def _in_months(days: pd.Index, months: pd.Index) -> np.ndarray:
    """A mask of the days that fall in the calendar months of `months` (month-end labels)."""
    return np.asarray(
        pd.DatetimeIndex(days).to_period("M").isin(pd.DatetimeIndex(months).to_period("M"))
    )


def _first_day(days: pd.Index, months: pd.Index) -> pd.Timestamp:
    return pd.Timestamp(pd.DatetimeIndex(days)[_in_months(days, months)][0])
