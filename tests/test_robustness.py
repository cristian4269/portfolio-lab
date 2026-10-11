"""The registered robustness checks (Step 7b, plans/step-7b-plan.md decisions 2-6)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_study import SMALL, synthetic_inputs

from portfolio_lab import config, robustness, study
from portfolio_lab.construct import strategy as st
from portfolio_lab.data import HoldoutError
from portfolio_lab.data.views import make_views
from portfolio_lab.report import inference as inf

PLAN_7B = Path(__file__).resolve().parents[1] / "plans" / "step-7b-plan.md"
END = "2022-12-31"
RESAMPLES = SMALL.resamples


def group(estimator="sample"):
    return [st.equal_weight(), st.min_variance(estimator), st.erc(estimator)]


def assets_with(columns, *, start="1999-01-01", end=END, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    return pd.DataFrame(
        rng.normal(0.0004, 0.01, (len(days), len(columns))), index=days, columns=columns
    )


@pytest.fixture(scope="module")
def inputs():
    return synthetic_inputs(start="2007-01-01", end="2022-12-30", n=3, seed=1)


@pytest.fixture(scope="module")
def holdout_tests(inputs):
    tables = study.run(
        inputs,
        period="holdout",
        end=END,
        estimator="sample",
        allow_holdout=True,
        settings=SMALL,
        strategies=group(),
    )
    return tables["tests"]


@pytest.fixture(scope="module")
def industries(inputs):
    usd = inputs.views["USD"]
    raw = pd.concat([usd.returns, usd.returns.iloc[:, :2].add_suffix("b") * 0.9], axis=1)
    used, table = robustness.universe(raw)
    fx_levels = pd.Series(
        1.2 * np.cumprod(1 + inputs.fx.reindex(used.index).fillna(0.0)), index=used.index
    )
    views = make_views(used, usd.rf.reindex(used.index), fx_levels, pd.Series(0.03, used.index))
    return robustness.Industries(views=views, universe=table)


@pytest.fixture(scope="module")
def tables(inputs, holdout_tests, industries):
    return robustness.run(
        inputs,
        holdout_tests,
        industries=industries,
        end=END,
        estimator="sample",
        allow_holdout=True,
        resamples=RESAMPLES,
    )


# --- the 49-industry universe (decision 5) ----------------------------------------------------


def test_the_hash_of_the_49_industry_file_is_the_one_the_plan_fixed():
    assert len(robustness.IND49_SHA256) == 64
    assert robustness.IND49_SHA256 in PLAN_7B.read_text(encoding="utf-8")


def test_universe_starts_the_day_after_the_last_missing_cell_before_the_holdout():
    raw = assets_with(["A", "B", "C"])
    raw.loc[raw.index < pd.Timestamp("2001-07-02"), "B"] = np.nan  # a late start
    raw.loc[pd.Timestamp("2003-03-11"), "C"] = np.nan  # one hole, as Softw on 1971-03-11
    used, table = robustness.universe(raw)
    assert list(used.columns) == ["A", "B", "C"]
    assert used.index[0] == pd.Timestamp("2003-03-12")
    assert not used.isna().any().any()
    assert table.loc["first_day", "value"] == "2003-03-12"
    assert table.loc["last_gap", "value"] == "2003-03-11"
    assert table.loc["industries", "value"] == "3"
    assert table.loc["dropped", "value"] == "none"


def test_universe_drops_an_industry_missing_a_holdout_day_and_names_it():
    raw = assets_with(["A", "B", "C"])
    raw.loc[pd.Timestamp("2015-06-01"), "B"] = np.nan  # inside the holdout
    raw.loc[pd.Timestamp("2004-01-05"), "B"] = np.nan  # its earlier hole no longer counts
    used, table = robustness.universe(raw)
    assert list(used.columns) == ["A", "C"]
    assert used.index[0] == raw.index[0]
    assert table.loc["dropped", "value"] == "B"
    assert table.loc["last_gap", "value"] == "none"


def test_universe_needs_days_before_the_holdout():
    raw = assets_with(["A", "B"])
    raw.loc[pd.Timestamp("2010-12-31"), "A"] = np.nan
    with pytest.raises(ValueError, match="before the holdout"):
        robustness.universe(raw)


def test_load_industries_reads_49_through_the_guarded_loaders(monkeypatch, inputs):
    usd = inputs.views["USD"]
    calls = []

    def record(name, value):
        def loader(*args, **kw):
            calls.append((name, args, kw))
            return value

        return loader

    raw = pd.concat([usd.returns, usd.returns.add_suffix("b")], axis=1)
    fx = pd.Series(1.25, index=usd.returns.index)
    rates = pd.DataFrame({"CA_TBILL_3M": 0.03}, index=usd.returns.index)
    monkeypatch.setattr(robustness.data, "load_assets", record("assets", raw))
    monkeypatch.setattr(robustness.data, "load_factors", record("factors", inputs.factors))
    monkeypatch.setattr(robustness.data, "load_fx", record("fx", fx))
    monkeypatch.setattr(robustness.data, "load_rates", record("rates", rates))
    got = robustness.load_industries(end=END, allow_holdout=True)
    assert set(got.views) == {"USD", "CAD", "CAD_HEDGED"}
    assert got.views["USD"].returns.shape[1] == 6
    for name, args, kw in calls:
        assert kw["end"] == END and kw["allow_holdout"] is True, name
    assert ("assets", (49,)) in [(n, a) for n, a, _ in calls]


# --- the run ----------------------------------------------------------------------------------


def test_run_has_every_table(tables):
    assert set(tables) == {"design", "tests", "performance", "universe49", "monthly", "weights"}
    assert tables["design"].loc["estimator", "value"] == "sample"
    assert tables["design"].loc["resamples", "value"] == str(RESAMPLES)


def test_tests_cover_every_variant_view_and_primary_strategy(tables):
    tests = tables["tests"]
    variants = ["primary", "window 126d", "window 504d", "cost 0bp", "cost 10bp", "cost 20bp"]
    variants.append("49 industries")
    assert list(dict.fromkeys(tests.index.get_level_values("variant"))) == variants
    assert set(tests.index.get_level_values("view")) == set(study.TESTED_VIEWS)
    assert set(tests["role"]) == {"min variance", "ERC"}
    assert len(tests) == len(variants) * 2 * 2
    assert "holm" not in " ".join(tests.columns) and "reject" not in " ".join(tests.columns)


def test_window_variants_estimate_again_on_their_window(tables):
    tests = tables["tests"]
    for w in robustness.WINDOWS:
        names = set(tests.xs(f"window {w}d", level="variant").index.get_level_values("strategy"))
        assert names == {st.min_variance("sample", window=w).name, st.erc("sample", window=w).name}


def test_primary_rows_are_the_holdouts_own_tests(tables, holdout_tests):
    primary = tables["tests"].xs("primary", level="variant")
    for key, row in primary.iterrows():
        held = holdout_tests.loc[key]
        for col in ("diff", "se", "ci_low", "ci_high", "p", "block"):
            assert row[col] == pytest.approx(held[col], rel=1e-12, abs=1e-15), (key, col)


def test_a_holdout_table_that_differs_is_refused(inputs, holdout_tests, industries):
    edited = holdout_tests.copy()
    edited.loc[("USD", st.erc("sample").name), "diff"] += 1e-6
    with pytest.raises(ValueError, match="holdout"):
        robustness.run(
            inputs,
            edited,
            industries=industries,
            end=END,
            estimator="sample",
            allow_holdout=True,
            resamples=RESAMPLES,
        )


def test_every_variant_uses_the_holdout_block_of_its_view_and_role(tables, holdout_tests):
    tests = tables["tests"]
    mv, erc = study.primary("sample")
    by_role = {"min variance": mv, "ERC": erc}
    for (_, view, _), row in tests.iterrows():
        assert row["block"] == holdout_tests.loc[(view, by_role[row["role"]]), "block"]


def test_a_variant_test_is_lw_test_on_its_own_seed(inputs, tables):
    row = tables["tests"].loc[("cost 10bp", "CAD", st.min_variance("sample").name)]
    results = {
        r.name: r
        for r in study.engine.compare(
            group(), inputs.views["CAD"], end=END, allow_holdout=True, cost_bps=study.COST_BPS
        ).values()
    }
    priced = {
        n: study.engine.reprice(r, inputs.views["CAD"], cost_bps=10.0, allow_holdout=True)
        for n, r in results.items()
    }
    monthly = study.period_months(
        pd.DataFrame({n: inf.monthly_excess(r.net, r.rf) for n, r in priced.items()}), "holdout"
    )
    b = int(row["block"])
    name = st.min_variance("sample").name
    hand = inf.lw_test(
        monthly[name],
        monthly[study.BENCHMARK],
        block=b,
        resamples=RESAMPLES,
        rng=study.rng_for("robustness", "cost 10bp", "CAD", name, "test", b),
    )
    assert row["diff"] == pytest.approx(hand.diff * np.sqrt(12))
    assert row["p"] == pytest.approx(hand.pvalue)
    assert row["months"] == len(monthly) == 144


def test_costs_reprice_the_same_trades(tables):
    perf = tables["performance"]
    name = st.erc("sample").name
    at = {c: perf.loc[(f"cost {c:g}bp", "USD", name)] for c in robustness.COSTS}
    base = perf.loc[("primary", "USD", name)]
    assert at[0.0]["annual_cost"] == 0.0
    assert at[20.0]["annual_cost"] == pytest.approx(4 * base["annual_cost"], rel=1e-12)
    assert at[10.0]["annual_turnover"] == pytest.approx(base["annual_turnover"], rel=1e-12)


def test_the_hedged_view_has_metrics_and_no_test(tables):
    perf = tables["performance"].xs("CAD hedged", level="variant")
    assert set(perf.index.get_level_values("view")) == {"CAD_HEDGED"}
    assert set(perf["role"]) == {"1/N", "min variance", "ERC"}
    for col in ("cagr", "annualized_vol", "max_drawdown", "cvar_95_daily"):
        assert perf[col].notna().all(), col
    assert "CAD hedged" not in set(tables["tests"].index.get_level_values("variant"))


def test_performance_covers_1n_in_every_variant(tables):
    perf = tables["performance"]
    for variant in set(perf.index.get_level_values("variant")):
        assert "1/N" in set(perf.xs(variant, level="variant")["role"]), variant


def test_monthly_series_cover_the_holdout_months_of_three_views(tables, inputs):
    monthly = tables["monthly"]
    assert set(monthly.index.get_level_values("view")) == {"USD", "CAD", "CAD_HEDGED"}
    months = monthly.index.get_level_values("month")
    assert months.min() == pd.Timestamp("2011-01-31") and months.max() <= pd.Timestamp(END)
    counts = monthly.groupby(level=["view", "strategy"]).size()
    assert (counts == 144).all() and len(counts) == 9
    usd = monthly.xs(("USD", "1/N"), level=["view", "strategy"])
    rf = inputs.views["USD"].rf
    carry = (1 + rf).groupby(rf.index.to_period("M")).prod() - 1
    carry.index = carry.index.to_timestamp("M")
    np.testing.assert_allclose(usd["net"] - usd["excess"], carry.reindex(usd.index), atol=1e-12)


def test_weights_are_the_pairs_holdout_targets(tables):
    weights = tables["weights"]
    assert set(weights.index.get_level_values("view")) == set(study.TESTED_VIEWS)
    assert set(weights.index.get_level_values("strategy")) == set(study.primary("sample"))
    traded = weights.index.get_level_values("traded")
    assert traded.min() >= pd.Timestamp(config.HOLDOUT_START)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)


def test_the_49_industry_variant_is_optional(inputs, holdout_tests):
    out = robustness.run(
        inputs,
        holdout_tests,
        industries=None,
        end=END,
        estimator="sample",
        allow_holdout=True,
        resamples=RESAMPLES,
    )
    assert "universe49" not in out
    assert "49 industries" not in set(out["tests"].index.get_level_values("variant"))


def test_run_keeps_the_holdout_rule(inputs, holdout_tests):
    with pytest.raises(HoldoutError):
        robustness.run(
            inputs,
            holdout_tests,
            industries=None,
            end=END,
            estimator="sample",
            resamples=RESAMPLES,
        )
