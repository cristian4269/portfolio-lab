"""The pre-registered study as code (Step 7a): registry, dev rule, pipeline on synthetic data."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_lab import config, study
from portfolio_lab.backtest import benchmarks
from portfolio_lab.construct import strategy as st
from portfolio_lab.data import HoldoutError
from portfolio_lab.data.views import fx_returns, make_views, yield_changes
from portfolio_lab.risk.factors import FACTORS

PLAN = Path(__file__).resolve().parents[1] / "research" / "PLAN.md"
SMALL = study.Settings(shadows=20, resamples=49, pseudo=6, calibration_resamples=19, blocks=(1, 2))


def synthetic_inputs(start="2005-01-03", end="2010-12-31", n=4, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    t = len(days)
    f = pd.DataFrame(
        rng.normal(0.0002, 0.006, (t, len(FACTORS))), index=days, columns=list(FACTORS)
    )
    f["RF"] = 0.0001
    beta = rng.uniform(0.6, 1.3, n)
    assets = pd.DataFrame(
        f[["Mkt-RF"]].to_numpy() * beta + rng.normal(0.0003, 0.008, (t, n)) + 0.0001,
        index=days,
        columns=[f"Ind{i}" for i in range(n)],
    )
    fx_levels = pd.Series(1.2 * np.cumprod(1 + rng.normal(0, 0.004, t)), index=days)
    tbill = pd.Series(0.03, index=days)
    ten_year = pd.Series(0.04 + np.cumsum(rng.normal(0, 0.0004, t)), index=days)
    views = make_views(assets, f["RF"], fx_levels, tbill)
    cad_days = views["CAD"].returns.index
    return study.Inputs(
        views=views,
        market=benchmarks.market_views(f, fx_levels, tbill),
        factors=f,
        fx=fx_returns(fx_levels, cad_days),
        dy=yield_changes(ten_year, cad_days),
    )


def small_group():
    group = [st.equal_weight()]
    group += [st.min_variance(e) for e in study.CANDIDATES]
    group += [st.erc(e) for e in study.CANDIDATES]
    return group


@pytest.fixture(scope="module")
def dev_tables():
    return study.run(
        synthetic_inputs(),
        period="dev",
        end=config.DEV_END,
        settings=SMALL,
        strategies=small_group(),
        holdout_months=188,
    )


# --- the registry -----------------------------------------------------------------------------


def test_registry_is_the_planned_seventeen():
    names = [s.name for s in study.strategies()]
    assert len(names) == 17 == len(set(names))
    assert names[0] == study.BENCHMARK == "1/N"
    for e in ("sample", "lw_identity", "lw_constant_correlation", "ewma"):
        assert f"min variance ({e}, 252d)" in names
        assert f"min variance ({e}, 252d, cap 0.25)" in names
        assert f"ERC ({e}, 252d)" in names
    assert "low-vol tilt (strength 0.5, 252d) on 1/N" in names
    assert "momentum tilt (strength 1) on 1/N" in names


def test_every_strategy_is_named_in_the_plan():
    text = PLAN.read_text(encoding="utf-8")
    missing = [s.name for s in study.strategies() if f"`{s.name}`" not in text]
    assert missing == []


def test_registered_constants_match_the_plan():
    assert study.SEED == 20261009
    assert study.COST_BPS == 5.0
    assert study.CANDIDATES == ("sample", "lw_identity", "lw_constant_correlation")
    assert study.TESTED_VIEWS == ("USD", "CAD")
    assert study.VAR_WINDOW == 500 and study.VAR_SENSITIVITY_WINDOW == 1000
    text = PLAN.read_text(encoding="utf-8")
    assert "SEED = 20261009" in text
    assert str(study.SEED) in text


def test_primary_pair_names():
    assert study.primary("lw_identity") == (
        "min variance (lw_identity, 252d)",
        "ERC (lw_identity, 252d)",
    )
    with pytest.raises(ValueError, match="candidate"):
        study.primary("ewma")


# --- the dev rule, the months, the seeds ------------------------------------------------------


def test_selection_takes_the_lowest_dev_volatility_by_hand():
    months = pd.date_range("2000-01-31", periods=6, freq="ME")
    m = pd.DataFrame(
        {
            "1/N": [0.01, 0.02, -0.01, 0.03, 0.0, 0.01],
            "min variance (sample, 252d)": [0.02, -0.02, 0.02, -0.02, 0.02, -0.02],
            "min variance (lw_identity, 252d)": [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
            "min variance (lw_constant_correlation, 252d)": [0.03, -0.03, 0.0, 0.0, 0.03, -0.03],
        },
        index=months,
    )
    table = study.selection(m)
    assert table["volatility"]["lw_identity"] == pytest.approx(
        m["min variance (lw_identity, 252d)"].std(ddof=1) * math.sqrt(12)
    )
    assert table.index[table["chosen"]].tolist() == ["lw_identity"]
    assert list(table.index) == list(study.CANDIDATES)


def test_selection_breaks_ties_in_registry_order():
    months = pd.date_range("2000-01-31", periods=4, freq="ME")
    same = [0.01, -0.01, 0.01, -0.01]
    m = pd.DataFrame({st.min_variance(e).name: same for e in study.CANDIDATES}, index=months)
    assert study.selection(m).index[study.selection(m)["chosen"]].tolist() == ["sample"]


def test_period_months_drop_the_partial_first_month_and_split_at_dev_end():
    months = pd.DatetimeIndex(
        ["2010-10-29", "2010-11-30", "2010-12-31", "2011-01-31", "2011-02-28"]
    )
    m = pd.DataFrame({"x": range(5)}, index=months, dtype=float)
    assert study.period_months(m, "dev").index.tolist() == list(months[1:3])
    assert study.period_months(m, "holdout").index.tolist() == list(months[3:])
    with pytest.raises(ValueError, match="period"):
        study.period_months(m, "both")


def test_seeds_are_stable_and_distinct():
    a = study.rng_for("dev", "USD", "1/N", "luck").integers(0, 10**9, 3)
    b = study.rng_for("dev", "USD", "1/N", "luck").integers(0, 10**9, 3)
    c = study.rng_for("dev", "CAD", "1/N", "luck").integers(0, 10**9, 3)
    assert (a == b).all() and not (a == c).all()


# --- the pipeline -----------------------------------------------------------------------------


def test_dev_run_produces_every_table(dev_tables):
    assert set(dev_tables) == {
        "design",
        "selection",
        "performance",
        "tests",
        "calibration",
        "block_sensitivity",
        "luck",
        "best_of",
        "var",
        "stress",
        "stress_realized",
        "shocks",
        "mde",
    }
    for name, table in dev_tables.items():
        assert isinstance(table, pd.DataFrame) and len(table), name


def test_dev_tests_mark_the_primary_family(dev_tables):
    tests = dev_tables["tests"]
    chosen = dev_tables["selection"].index[dev_tables["selection"]["chosen"]][0]
    mv, erc = study.primary(chosen)
    primary = tests[tests["family"] == "primary"]
    assert sorted(primary.index.tolist()) == sorted(
        [(v, s) for v in ("USD", "CAD") for s in (mv, erc)]
    )
    assert primary["holm_primary"].notna().all()
    assert tests.loc[tests["family"] == "exploratory", "holm_primary"].isna().all()
    assert (tests["holm_all"] >= tests["p"] - 1e-15).all()
    assert len(tests) == 2 * 6  # 6 strategies besides 1/N in each view


def test_dev_tests_are_annualized_monthly_sharpe_differences(dev_tables):
    perf = dev_tables["performance"]
    tests = dev_tables["tests"]
    for (view, name), row in tests.iterrows():
        expected = (
            perf.loc[(view, name), "sharpe_monthly"] - perf.loc[(view, "1/N"), "sharpe_monthly"]
        )
        assert row["diff"] == pytest.approx(expected, rel=1e-12)
        assert row["ci_low"] < row["diff"] < row["ci_high"]


def test_dev_exploratory_pairs_use_the_minimum_variance_block(dev_tables):
    tests, cal = dev_tables["tests"], dev_tables["calibration"]
    chosen = dev_tables["selection"].index[dev_tables["selection"]["chosen"]][0]
    mv, _ = study.primary(chosen)
    for view in ("USD", "CAD"):
        rows = cal.reset_index()
        block = rows[(rows["view"] == view) & (rows["strategy"] == mv) & rows["chosen"]]
        block = block["block"].iloc[0]
        explo = tests.loc[view]
        explo = explo[explo["family"] == "exploratory"]
        assert (explo["block"] == block).all()


def test_dev_luck_covers_every_strategy_and_best_of_k(dev_tables):
    lk = dev_tables["luck"]
    assert len(lk) == 2 * 7
    assert lk["percentile"].between(0, 1).all()
    best = dev_tables["best_of"]
    assert list(best.index) == ["USD", "CAD"]
    assert (best["k"] == 6).all()
    assert best["p_best_of"].between(0, 1).all()


def test_dev_stress_is_the_gfc_on_dev_end_weights(dev_tables):
    stress = dev_tables["stress"]
    assert set(stress.index.get_level_values("scenario")) == {"gfc"}
    assert set(stress.index.get_level_values("view")) == {"USD", "CAD"}
    shocks = dev_tables["shocks"]
    usd_terms = set(shocks.loc["USD"].index.get_level_values("shock"))
    assert usd_terms == {"market", "momentum"}
    assert set(shocks.loc["CAD"].index.get_level_values("shock")) == {
        "market",
        "momentum",
        "rates",
        "cad_up",
        "cad_down",
    }


def test_dev_var_evaluates_dev_days_with_both_windows(dev_tables):
    var = dev_tables["var"]
    assert set(var.index.get_level_values("window")) == {500, 1000}
    assert set(var.xs(1000, level="window").index.get_level_values("method")) == {"fhs"}
    assert "kupiec_exact_p" in var.columns


def test_dev_mde_scales_to_the_holdout_length(dev_tables):
    mde = dev_tables["mde"]
    assert (mde["holdout_months"] == 188).all()
    assert (mde["mde_holm"] > mde["mde"]).all()


def test_dev_run_refuses_an_estimator_and_holdout_needs_one():
    inputs = synthetic_inputs()
    with pytest.raises(ValueError, match="dev rule"):
        study.run(
            inputs,
            period="dev",
            end=config.DEV_END,
            estimator="sample",
            settings=SMALL,
            strategies=small_group(),
        )
    with pytest.raises(ValueError, match="estimator"):
        study.run(
            inputs, period="holdout", end="2010-12-31", settings=SMALL, strategies=small_group()
        )
    with pytest.raises(ValueError, match="1/N"):
        study.run(
            inputs, period="dev", end=config.DEV_END, settings=SMALL, strategies=small_group()[1:]
        )


def test_holdout_run_needs_the_flag():
    inputs = synthetic_inputs(end="2012-06-29")
    with pytest.raises(HoldoutError):
        study.run(
            inputs,
            period="holdout",
            end="2012-06-30",
            estimator="sample",
            settings=SMALL,
            strategies=small_group(),
        )


def test_holdout_run_on_synthetic_data():
    inputs = synthetic_inputs(start="2007-01-01", end="2022-12-30", n=3, seed=1)
    group = [st.equal_weight(), st.min_variance("sample"), st.erc("sample")]
    tables = study.run(
        inputs,
        period="holdout",
        end="2022-12-31",
        estimator="sample",
        allow_holdout=True,
        settings=SMALL,
        strategies=group,
    )
    assert "selection" not in tables and "mde" not in tables
    months = tables["design"].loc["first_month", "value"]
    assert months == "2011-01-31"
    assert set(tables["stress"].index.get_level_values("scenario")) == {"gfc", "covid", "rates"}
    assert (tables["tests"]["family"] == "primary").all()
    assert tables["tests"]["months"].eq(144).all()  # 2011-01 .. 2022-12


def test_load_passes_the_holdout_rule_to_every_loader(monkeypatch):
    raw = synthetic_inputs(start="1963-01-02", end="1966-12-30")
    days = raw.factors.index
    calls = []

    def record(name, value):
        def loader(*args, **kw):
            calls.append((name, kw["end"], kw["allow_holdout"], kw["refresh"]))
            return value

        return loader

    usd = raw.views["USD"].returns
    fx_levels = pd.Series(1.2, index=days)
    rates = pd.DataFrame({"CA_TBILL_3M": 0.03, "CA_GOVT_10Y": 0.04}, index=days)
    monkeypatch.setattr(study.data, "load_factors", record("factors", raw.factors))
    monkeypatch.setattr(study.data, "load_fx", record("fx", fx_levels))
    monkeypatch.setattr(study.data, "load_rates", record("rates", rates))
    monkeypatch.setattr(study.data, "load_assets", record("assets", usd))
    inputs = study.load(end="1966-06-30", allow_holdout=False, refresh=True)
    assert sorted(calls) == sorted(
        (n, "1966-06-30", False, True) for n in ("factors", "fx", "rates", "assets")
    )
    assert inputs.views["USD"].returns.index[0] >= pd.Timestamp("1963-07-01")
    assert inputs.fx.index.equals(inputs.views["CAD"].returns.index)
    assert inputs.dy.index.equals(inputs.views["CAD"].returns.index)


def test_run_checks_the_period_and_the_primary_pair():
    inputs = synthetic_inputs()
    with pytest.raises(ValueError, match="period"):
        study.run(inputs, period="both", end=config.DEV_END, settings=SMALL)
    without_erc = [st.equal_weight(), *[st.min_variance(e) for e in study.CANDIDATES]]
    with pytest.raises(ValueError, match="primary pair"):
        study.run(inputs, period="dev", end=config.DEV_END, settings=SMALL, strategies=without_erc)
