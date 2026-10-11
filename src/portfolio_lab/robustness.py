"""The registered robustness checks (research/PLAN.md §6; blueprint Step 7b, plan
plans/step-7b-plan.md decisions 2-5), run after the one-shot holdout.

One change at a time from the primary configuration, for the primary pair (minimum variance and
ERC on the frozen estimator) against 1/N, in USD and CAD unhedged, over the holdout months:
- `window 126d`, `window 504d`: the strategies estimated again on their window;
- `cost 0bp`, `cost 10bp`, `cost 20bp`: the same trades repriced (`engine.reprice`), 1/N's too;
- `49 industries`: all of them from the day after the last missing cell before the holdout
  (`universe`); an industry missing a holdout day is dropped and named;
- `CAD hedged`: the three portfolios on the CAD_HEDGED view, metrics only (its excess return is
  the USD one, so it has no test).

Each test uses the block calibrated on the holdout for the same view and strategy (read from the
holdout's `tests` table) and its own seeded stream. The rows are descriptive: no Holm, no
rejection, no claim. A `primary` row recomputes the registered test with the holdout's own seed
and must equal the holdout's table, so the two cannot drift apart.

`study.py` stays as tagged (prereg-v1), so this module reuses its helpers rather than editing it.
Tables: `design`, `tests`, `performance`, `universe49` (with the 49 industries), and the series
the figures need, `monthly` (net and excess returns per month) and `weights` (the pair's targets).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_lab import data, study
from portfolio_lab.backtest import engine
from portfolio_lab.backtest.engine import Result
from portfolio_lab.config import HOLDOUT_START
from portfolio_lab.construct import strategy as st
from portfolio_lab.data.views import View, make_views
from portfolio_lab.report import inference as inf

__all__ = [
    "COSTS",
    "HEDGED",
    "IND49_SHA256",
    "PRIMARY",
    "WINDOWS",
    "Industries",
    "load_industries",
    "role",
    "run",
    "universe",
]

WINDOWS = (126, 504)
COSTS = (0.0, 10.0, 20.0)
INDUSTRIES = 49
# The cached 49-industry file, fixed in plans/step-7b-plan.md before any holdout number existed
# (it is not one of the frozen vintage's sources).
IND49_SHA256 = "8f394fe34bea54d41b9aafed410425ee8f8e252ede3c71a7c1cd20bab83040de"
PRIMARY = "primary"
HEDGED = "CAD hedged"
HEDGED_VIEW = "CAD_HEDGED"
ROLES = ("min variance", "ERC")

_YEAR = math.sqrt(12)
_SAME = 1e-12  # the primary rows must reproduce the holdout's to rounding


@dataclass(frozen=True, eq=False)
class Industries:
    """The 49-industry views and the outcome of the universe rule (`universe`)."""

    views: Mapping[str, View]
    universe: pd.DataFrame


def universe(assets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The returns the 49-industry run uses (decision 5), and a table of the rule's outcome.

    An industry with a missing day from HOLDOUT_START on is dropped. The rest start on the day
    after the last day on which any of them is missing before the holdout.
    """
    holdout = np.asarray(assets.index >= pd.Timestamp(HOLDOUT_START))
    dropped = [c for c in assets.columns if assets.loc[holdout, c].isna().any()]
    kept = assets.drop(columns=dropped)
    before = kept.loc[~holdout]
    gaps = before.index[before.isna().any(axis=1).to_numpy()]
    used = kept.loc[kept.index > gaps[-1]] if len(gaps) else kept
    if not (used.index < pd.Timestamp(HOLDOUT_START)).any():
        raise ValueError(
            "no day before the holdout has every industry: the 49-industry run needs history"
        )
    table = pd.DataFrame(
        {
            "value": {
                "first_day": str(used.index[0].date()),
                "last_gap": str(gaps[-1].date()) if len(gaps) else "none",
                "industries": str(used.shape[1]),
                "dropped": ", ".join(map(str, dropped)) or "none",
            }
        }
    )
    return used, table


def load_industries(*, end: str | pd.Timestamp | None, allow_holdout: bool) -> Industries:
    """The 49-industry views through `end`, by the loaders' holdout rule (as `study.load`)."""
    f = data.load_factors(end=end, allow_holdout=allow_holdout)
    fx_levels = data.load_fx(end=end, allow_holdout=allow_holdout)
    rates = data.load_rates(end=end, allow_holdout=allow_holdout)
    assets = data.load_assets(INDUSTRIES, end=end, allow_holdout=allow_holdout)
    used, table = universe(assets)
    views = make_views(used, f["RF"].reindex(used.index), fx_levels, rates["CA_TBILL_3M"])
    return Industries(views=views, universe=table)


def run(
    inputs: study.Inputs,
    holdout_tests: pd.DataFrame,
    *,
    industries: Industries | None,
    end: str | pd.Timestamp,
    estimator: str,
    allow_holdout: bool = False,
    resamples: int = inf.RESAMPLES,
) -> dict[str, pd.DataFrame]:
    """Every robustness table (see the module doc). `holdout_tests` is the holdout's `tests`
    table, indexed by (view, strategy); `industries=None` leaves the 49 industries out."""
    pair = study.primary(estimator)
    calibrated = holdout_tests["block"].astype(int)
    blocks = {(v, role(n)): int(calibrated[(v, n)]) for v in study.TESTED_VIEWS for n in pair}

    def compare(view: View, window: int = st.WINDOW) -> dict[str, Result]:
        group = [st.equal_weight(), st.min_variance(estimator, window=window)]
        group.append(st.erc(estimator, window=window))
        return engine.compare(
            group, view, end=end, allow_holdout=allow_holdout, cost_bps=study.COST_BPS
        )

    base = {v: compare(inputs.views[v]) for v in study.TESTED_VIEWS}
    variants: dict[str, dict[str, dict[str, Result]]] = {PRIMARY: base}
    for w in WINDOWS:
        variants[f"window {w}d"] = {v: compare(inputs.views[v], w) for v in study.TESTED_VIEWS}
    for c in COSTS:
        variants[f"cost {c:g}bp"] = {
            v: {
                n: engine.reprice(r, inputs.views[v], cost_bps=c, allow_holdout=allow_holdout)
                for n, r in res.items()
            }
            for v, res in base.items()
        }
    if industries is not None:
        variants["49 industries"] = {v: compare(industries.views[v]) for v in study.TESTED_VIEWS}
    hedged = {HEDGED_VIEW: compare(inputs.views[HEDGED_VIEW])}

    test_rows, perf_parts = [], []
    for variant, results in [*variants.items(), (HEDGED, hedged)]:
        monthly = {v: _monthly_excess(res) for v, res in results.items()}
        perf = study._performance(results, monthly, inputs)
        perf["role"] = [role(n) for n in perf.index.get_level_values("strategy")]
        perf_parts.append(pd.concat({variant: perf}, names=["variant"]))
        if variant == HEDGED:
            continue
        for v, m in monthly.items():
            for name in m.columns:
                if name == study.BENCHMARK:
                    continue
                kind = role(name)
                b = blocks[(v, kind)]
                labels = ("holdout",) if variant == PRIMARY else ("robustness", variant)
                res = inf.lw_test(
                    m[name],
                    m[study.BENCHMARK],
                    block=b,
                    resamples=resamples,
                    rng=study.rng_for(*labels, v, name, "test", b),
                )
                test_rows.append(
                    {
                        "variant": variant,
                        "view": v,
                        "strategy": name,
                        "role": kind,
                        "months": len(m),
                        "diff": res.diff * _YEAR,
                        "se": res.se * _YEAR,
                        "ci_low": res.ci_low * _YEAR,
                        "ci_high": res.ci_high * _YEAR,
                        "block": b,
                        "p": res.pvalue,
                        "sharpe": inf.sharpe_ratio(m[name]) * _YEAR,
                        "sharpe_benchmark": inf.sharpe_ratio(m[study.BENCHMARK]) * _YEAR,
                    }
                )
    tests = pd.DataFrame(test_rows).set_index(["variant", "view", "strategy"])
    _check_primary(tests, holdout_tests)

    tables = {
        "design": pd.DataFrame(
            {
                "value": {
                    "estimator": estimator,
                    "end": str(pd.Timestamp(end).date()),
                    "variants": ", ".join([*variants, HEDGED]),
                    "resamples": str(resamples),
                    "seed": str(study.SEED),
                }
            }
        ),
        "tests": tests,
        "performance": pd.concat(perf_parts),
    }
    if industries is not None:
        tables["universe49"] = industries.universe
    tables["monthly"] = _series({**base, **hedged})
    tables["weights"] = _weights(base, pair)
    return tables


def role(name: str) -> str:
    """A portfolio's role by its name: "1/N", "min variance" or "ERC"."""
    if name == study.BENCHMARK:
        return "1/N"
    return ROLES[0] if name.startswith("min variance") else ROLES[1]


def _monthly_excess(results: Mapping[str, Result]) -> pd.DataFrame:
    frame = pd.DataFrame({n: inf.monthly_excess(r.net, r.rf) for n, r in results.items()})
    return study.period_months(frame, "holdout")


def _series(results: Mapping[str, Mapping[str, Result]]) -> pd.DataFrame:
    """The monthly net and excess returns of every portfolio, holdout months, long."""
    parts = {}
    for v, res in results.items():
        for name, r in res.items():
            net = study.period_months(inf.monthly_excess(r.net, 0.0 * r.rf), "holdout")
            excess = study.period_months(inf.monthly_excess(r.net, r.rf), "holdout")
            frame = pd.DataFrame({"net": net, "excess": excess})
            frame.index.name = "month"
            parts[(v, name)] = frame
    return pd.concat(parts, names=["view", "strategy"])


def _weights(results: Mapping[str, Mapping[str, Result]], pair: tuple[str, str]) -> pd.DataFrame:
    """The pair's targets on the holdout's trade dates."""
    parts = {}
    for v, res in results.items():
        for name in pair:
            w = res[name].weights
            parts[(v, name)] = w.loc[w.index >= pd.Timestamp(HOLDOUT_START)]
    return pd.concat(parts, names=["view", "strategy"])


def _check_primary(tests: pd.DataFrame, holdout_tests: pd.DataFrame) -> None:
    """The primary rows must be the holdout's own tests (a stale or edited table fails)."""
    cols = ["diff", "se", "ci_low", "ci_high", "p"]
    is_primary = tests.index.get_level_values("variant") == PRIMARY
    mine = tests.loc[is_primary, cols].droplevel("variant")
    held = holdout_tests.reindex(mine.index)[cols]  # a missing row is NaN, which fails below
    for col in cols:
        pairs = zip(mine.index, mine[col].to_numpy(), held[col].to_numpy(), strict=True)
        for key, here, there in pairs:
            if not math.isclose(here, there, rel_tol=_SAME, abs_tol=_SAME):
                raise ValueError(
                    f"the primary {col} of {key} is {here!r} here and {there!r} in the "
                    "holdout's tests table: robustness must run on the holdout it extends"
                )
