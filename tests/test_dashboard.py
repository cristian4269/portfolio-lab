"""The dashboard's bundle and its pure functions (Step 8, plans/step-8-plan.md D1-D6)."""

import dataclasses
import hashlib
import re

import numpy as np
import pandas as pd
import pytest
from test_study import SMALL, synthetic_inputs

from portfolio_lab import dashboard as db
from portfolio_lab import robustness, study
from portfolio_lab.backtest import engine
from portfolio_lab.config import HOLDOUT_START
from portfolio_lab.construct import strategy as st
from portfolio_lab.report import inference as inf
from portfolio_lab.risk import factors as fa
from portfolio_lab.stress import historical as hs

END = "2022-12-31"


def group():
    return [st.equal_weight(), st.min_variance("sample"), st.erc("sample")]


@pytest.fixture(scope="module")
def inputs():
    return synthetic_inputs(start="2007-01-01", end="2022-12-30", n=3, seed=1)


@pytest.fixture(scope="module")
def bundle(inputs):
    return db.build(inputs, end=END, estimator="sample", allow_holdout=True, strategies=group())


def copies():
    """Stand-ins for the holdout's registered tables (the real ones are byte copies)."""
    tests = "view,strategy,diff,p\nUSD,s,0.1,0.5\n"
    return {f"holdout/{name}.csv": tests.encode() for name in db.HOLDOUT_TABLES}


# --- net_at -----------------------------------------------------------------------------------


def test_net_at_by_hand():
    days = pd.bdate_range("2020-01-01", periods=3)
    gross = pd.DataFrame({"s": [0.01, 0.02, -0.01]}, index=days)
    turnover = pd.DataFrame({"s": [0.3]}, index=days[[0]])
    net = db.net_at(gross, turnover, 10.0)
    assert net.loc[days[0], "s"] == pytest.approx(1.01 * (1 - 0.0006) - 1, abs=1e-16)
    assert net.loc[days[1], "s"] == 0.02 and net.loc[days[2], "s"] == -0.01  # no trade
    assert gross.loc[days[0], "s"] == 0.01  # the input is not changed


def test_net_at_is_the_engines_net_at_any_cost(inputs):
    view = inputs.views["USD"]
    result = engine.run(st.erc("sample"), view, end=END, allow_holdout=True, cost_bps=5.0)
    gross = result.gross.to_frame("s")
    turnover = result.turnover.to_frame("s")
    for cost in (0.0, 5.0, 20.0):
        repriced = engine.reprice(result, view, cost_bps=cost, allow_holdout=True)
        mine = db.net_at(gross, turnover, cost)["s"]
        assert np.array_equal(mine.to_numpy(), repriced.net.to_numpy())  # exactly


def test_net_at_refuses_bad_costs_and_unknown_trade_days():
    days = pd.bdate_range("2020-01-01", periods=3)
    gross = pd.DataFrame({"s": [0.01, 0.02, -0.01]}, index=days)
    turnover = pd.DataFrame({"s": [0.3]}, index=days[[0]])
    for bad in (-1.0, float("nan"), 5000.0):
        with pytest.raises(ValueError, match="cost_bps"):
            db.net_at(gross, turnover, bad)
    late = pd.DataFrame({"s": [0.3]}, index=[pd.Timestamp("2021-01-04")])
    with pytest.raises(ValueError, match="2021-01-04"):
        db.net_at(gross, late, 5.0)
    with pytest.raises(ValueError, match="columns"):
        db.net_at(gross, turnover.rename(columns={"s": "t"}), 5.0)


# --- build ------------------------------------------------------------------------------------


def test_build_holds_every_strategy_in_every_view(bundle, inputs):
    names = [s.name for s in group()]
    assert db.VIEWS == ("USD", "CAD", "CAD_HEDGED")
    assert bundle.strategies == names
    for v in db.VIEWS:
        r = bundle.returns[v]
        assert list(r.columns) == ["rf", "market", *names]
        assert r.index.name == "day" and r.index.is_monotonic_increasing
        assert np.isfinite(r.to_numpy()).all()
        assert list(bundle.turnover[v].columns) == names
        assert bundle.turnover[v].index.isin(r.index).all()
        pd.testing.assert_series_equal(
            r["market"], inputs.market[v].reindex(r.index), check_names=False
        )
    assert list(bundle.weights.index.names) == ["view", "strategy", "traded"]
    assert list(bundle.weights.columns) == list(inputs.views["USD"].returns.columns)
    assert list(bundle.factors.columns) == [*db.FACTOR_COLUMNS]
    design = bundle.design["value"]
    assert design["estimator"] == "sample" and design["end"] == END
    assert design["cost_bps"] == "5" and design["strategies"] == "3"


def test_build_series_are_the_compare_runs(bundle, inputs):
    """The bundle's 5 bp net is exactly the study's backtest, in every view."""
    for v in db.VIEWS:
        runs = engine.compare(group(), inputs.views[v], end=END, allow_holdout=True, cost_bps=5.0)
        net = db.net(bundle, v, 5.0)
        for name, r in runs.items():
            assert np.array_equal(net[name].to_numpy(), r.net.to_numpy())
            pd.testing.assert_series_equal(bundle.returns[v]["rf"], r.rf, check_names=False)


def test_months_are_the_full_months_after_the_first(bundle):
    months = db.months(bundle, "USD")
    first_day = bundle.returns["USD"].index[0]
    assert months[0] > first_day.to_period("M")  # the first, partial month is left out
    assert months[-1] == pd.Period("2022-12", "M")


# --- write and read ---------------------------------------------------------------------------


def test_write_then_read_gives_the_same_bundle(bundle, tmp_path):
    written = db.write(bundle, tmp_path, copies=copies())
    assert set(written) == {
        "design.csv",
        *(f"returns_{v}.csv.gz" for v in db.VIEWS),
        *(f"turnover_{v}.csv.gz" for v in db.VIEWS),
        "weights.csv.gz",
        "factors.csv.gz",
        *copies(),
    }
    for name, digest in written.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    back = db.read(tmp_path)
    assert back.strategies == bundle.strategies
    same = {"check_exact": True, "check_freq": False}  # synthetic days carry a frequency
    for v in db.VIEWS:
        pd.testing.assert_frame_equal(back.returns[v], bundle.returns[v], **same)
        pd.testing.assert_frame_equal(back.turnover[v], bundle.turnover[v], **same)
    pd.testing.assert_frame_equal(back.weights, bundle.weights, **same)
    pd.testing.assert_frame_equal(back.factors, bundle.factors, **same)
    pd.testing.assert_frame_equal(back.design, bundle.design)
    assert set(back.tables) == set(db.HOLDOUT_TABLES)


def test_write_is_byte_for_byte_repeatable(bundle, tmp_path):
    first = db.write(bundle, tmp_path / "a", copies=copies())
    second = db.write(bundle, tmp_path / "b", copies=copies())
    assert first == second


@pytest.fixture
def written(bundle, tmp_path):
    db.write(bundle, tmp_path, copies=copies())
    return tmp_path


def rewrite(path, edit):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    edit(frame)
    frame.to_csv(path, compression={"method": "gzip", "mtime": 0}, lineterminator="\n")


def test_read_refuses_a_column_outside_the_schema(written):
    rewrite(written / "returns_USD.csv.gz", lambda f: f.insert(2, "NoDur", 0.0))
    with pytest.raises(ValueError, match="NoDur"):
        db.read(written)


def test_read_refuses_a_missing_view(written):
    (written / "returns_CAD.csv.gz").unlink()
    with pytest.raises(ValueError, match="returns_CAD"):
        db.read(written)


def test_read_refuses_non_finite_returns(written):
    def poison(frame):
        frame.iloc[5, 3] = np.nan

    rewrite(written / "returns_USD.csv.gz", poison)
    with pytest.raises(ValueError, match="finite"):
        db.read(written)


def test_read_refuses_turnover_of_other_strategies(written):
    rewrite(written / "turnover_CAD.csv.gz", lambda f: f.pop(f.columns[-1]))
    with pytest.raises(ValueError, match="turnover_CAD"):
        db.read(written)


def test_read_refuses_a_missing_registered_table(written):
    (written / "holdout" / "tests.csv").unlink()
    with pytest.raises(ValueError, match="tests.csv"):
        db.read(written)


# --- performance and the export check ---------------------------------------------------------


@pytest.fixture(scope="module")
def references(inputs):
    """What the study and robustness record for the same data: the holdout's performance
    table (USD, CAD) and the monthly series (with CAD hedged)."""
    tables = study.run(
        inputs,
        period="holdout",
        end=END,
        estimator="sample",
        allow_holdout=True,
        settings=SMALL,
        strategies=group(),
    )
    runs = {
        v: engine.compare(group(), inputs.views[v], end=END, allow_holdout=True, cost_bps=5.0)
        for v in db.VIEWS
    }
    return tables["performance"], robustness._series(runs)


def test_performance_on_the_holdout_is_the_studys(bundle, references):
    performance, _ = references
    start, end = pd.Timestamp(HOLDOUT_START), pd.Timestamp(END)
    for v in study.TESTED_VIEWS:
        mine = db.performance(bundle, v, bundle.strategies, cost_bps=5.0, start=start, end=end)
        theirs = performance.xs(v, level="view").loc[bundle.strategies, mine.columns]
        pd.testing.assert_frame_equal(mine, theirs, rtol=1e-12, atol=1e-12, check_names=False)


def test_check_passes_on_the_recorded_tables(bundle, references):
    db.check(bundle, *references)


def test_check_fails_on_a_perturbed_gross_return(bundle, references):
    for v, name in (("USD", bundle.strategies[1]), ("CAD_HEDGED", bundle.strategies[2])):
        returns = {**bundle.returns}
        frame = returns[v].copy()
        day = frame.index[frame.index >= pd.Timestamp("2015-03-02")][0]
        frame.loc[day, name] += 1e-9
        returns[v] = frame
        changed = db.Bundle(
            design=bundle.design,
            returns=returns,
            turnover=bundle.turnover,
            weights=bundle.weights,
            factors=bundle.factors,
            tables=bundle.tables,
        )
        with pytest.raises(ValueError, match=f"{v} {re.escape(repr(name))}"):
            db.check(changed, *references)


def test_check_needs_a_reference_for_every_series(bundle, references):
    performance, monthly = references
    with pytest.raises(ValueError, match="performance"):
        db.check(bundle, performance.drop(index=("CAD", bundle.strategies[1])), monthly)
    hedged = monthly.index.get_level_values("view") == "CAD_HEDGED"
    with pytest.raises(ValueError, match="CAD_HEDGED"):
        db.check(bundle, performance, monthly.loc[~hedged])


def test_performance_against_one_over_n_and_at_other_costs(bundle):
    start, end = pd.Timestamp(HOLDOUT_START), pd.Timestamp(END)
    names = bundle.strategies
    table = db.performance(
        bundle, "USD", names, cost_bps=5.0, start=start, end=end, benchmark=study.BENCHMARK
    )
    assert table.loc[study.BENCHMARK, "tracking_error"] == 0.0
    assert np.isnan(table.loc[study.BENCHMARK, "information_ratio"])
    cheap = db.performance(bundle, "USD", names, cost_bps=0.0, start=start, end=end)
    dear = db.performance(bundle, "USD", names, cost_bps=20.0, start=start, end=end)
    assert (dear["cagr"] < cheap["cagr"]).all()  # every strategy trades after its first month
    assert (dear["annual_turnover"] == cheap["annual_turnover"]).all()  # the same trades
    assert (cheap["annual_cost"] == 0.0).all()


def test_performance_monthly_columns_use_the_selected_months(bundle):
    start, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31")
    table = db.performance(bundle, "CAD", bundle.strategies, cost_bps=5.0, start=start, end=end)
    net = db.net(bundle, "CAD", 5.0)
    rf = bundle.returns["CAD"]["rf"]
    days = (net.index >= start) & (net.index <= end)
    m = inf.monthly_excess(net.loc[days], rf.loc[days])
    assert (table["months"] == 24).all()
    name = bundle.strategies[2]
    assert table.loc[name, "sharpe_monthly"] == pytest.approx(
        inf.sharpe_ratio(m[name]) * np.sqrt(12), rel=1e-12
    )


def test_performance_refuses_unknown_strategies_views_and_benchmarks(bundle):
    start, end = pd.Timestamp(HOLDOUT_START), pd.Timestamp(END)
    with pytest.raises(ValueError, match="unknown"):
        db.performance(bundle, "USD", ["nope"], cost_bps=5.0, start=start, end=end)
    with pytest.raises(ValueError, match="view"):
        db.performance(bundle, "EUR", bundle.strategies, cost_bps=5.0, start=start, end=end)
    with pytest.raises(ValueError, match="benchmark"):
        db.performance(
            bundle, "USD", bundle.strategies, cost_bps=5.0, start=start, end=end, benchmark="x"
        )


# --- the refusals -----------------------------------------------------------------------------


def test_build_needs_one_over_n():
    with pytest.raises(ValueError, match="benchmark"):
        db.build(None, end=END, estimator="sample", strategies=[st.erc("sample")])


@pytest.fixture(scope="module")
def short_inputs():
    return synthetic_inputs(start="2007-01-01", end="2009-06-30", n=3, seed=2)


def test_build_refuses_a_net_it_cannot_give_back(short_inputs, monkeypatch):
    real = db.net_at
    monkeypatch.setattr(db, "net_at", lambda g, t, c: real(g, t, c) + 1e-15)
    with pytest.raises(ValueError, match="does not give back"):
        db.build(short_inputs, end="2009-06-30", estimator="sample", strategies=group())


def test_build_refuses_strategies_on_other_days(short_inputs, monkeypatch):
    real = engine.compare

    def shorter(*args, **kwargs):
        runs = real(*args, **kwargs)
        last = runs[st.erc("sample").name]
        runs[last.name] = engine.Result(
            gross=last.gross.iloc[:-1],
            net=last.net.iloc[:-1],
            weights=last.weights,
            turnover=last.turnover,
            cost=last.cost,
            final_weights=last.final_weights,
            name=last.name,
            rf=last.rf.iloc[:-1],
        )
        return runs

    monkeypatch.setattr(engine, "compare", shorter)
    with pytest.raises(ValueError, match="does not share the days"):
        db.build(short_inputs, end="2009-06-30", estimator="sample", strategies=group())


def drop_key(folder):
    design = pd.read_csv(folder / "design.csv", index_col="key", dtype=str)
    design.drop(index="cost_bps").to_csv(folder / "design.csv", lineterminator="\n")


def say_four(folder):
    design = pd.read_csv(folder / "design.csv", index_col="key", dtype=str)
    design.loc["strategies", "value"] = "4"
    design.to_csv(folder / "design.csv", lineterminator="\n")


def move_rf(folder):
    rewrite(folder / "returns_USD.csv.gz", lambda f: f.insert(2, "rf2", f.pop("rf")))


def drop_from_cad(folder):
    rewrite(folder / "returns_CAD.csv.gz", lambda f: f.pop(f.columns[-1]))


def reverse_days(folder):
    path = folder / "returns_USD.csv.gz"
    frame = pd.read_csv(path, index_col=0, parse_dates=True).iloc[::-1]
    frame.to_csv(path, compression={"method": "gzip", "mtime": 0}, lineterminator="\n")


def trade_on_a_sunday(folder):
    path = folder / "turnover_USD.csv.gz"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = frame.index[:-1].append(pd.DatetimeIndex(["2030-01-06"])).rename("traded")
    frame.to_csv(path, compression={"method": "gzip", "mtime": 0}, lineterminator="\n")


def drop_dy(folder):
    rewrite(folder / "factors.csv.gz", lambda f: f.pop("dy"))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (drop_key, "design.csv has no"),
        (say_four, "says 4 strategies"),
        (move_rf, "must start with the columns"),
        (drop_from_cad, "returns_CAD.csv.gz holds other strategies"),
        (reverse_days, "unique and in order"),
        (trade_on_a_sunday, "trade days that are not days"),
        (drop_dy, "factors.csv.gz must have the columns"),
    ],
)
def test_read_refuses_a_bundle_off_the_schema(written, edit, message):
    edit(written)
    with pytest.raises(ValueError, match=message):
        db.read(written)


def test_check_needs_every_holdout_month(bundle, references):
    performance, monthly = references
    hedged = monthly.xs("CAD_HEDGED", level="view", drop_level=False)
    short = monthly.drop(index=hedged.index[-1])
    with pytest.raises(ValueError, match="covers 144 holdout months here and 143"):
        db.check(bundle, performance, short)


# --- the charts' series ------------------------------------------------------------------------


def test_growth_and_drawdowns_by_hand():
    days = pd.bdate_range("2020-01-01", periods=3)
    net = pd.DataFrame({"s": [0.10, -0.50, 0.20]}, index=days)
    assert db.growth(net)["s"].tolist() == pytest.approx([1.10, 0.55, 0.66])
    assert db.drawdowns(net)["s"].tolist() == pytest.approx([0.0, -0.5, -0.4])


def test_thin_keeps_short_series_and_samples_long_ones():
    days = pd.bdate_range("1960-01-01", "2020-12-31")  # over 2000 weeks
    frame = pd.DataFrame({"s": np.arange(len(days), dtype=float)}, index=days)
    assert db.thin(frame.iloc[:500], "last").equals(frame.iloc[:500])  # daily fits
    weekly = db.thin(frame.iloc[:3000], "last")
    assert len(weekly) <= db.MAX_POINTS and weekly["s"].iloc[-1] == 2999.0  # the last close
    monthly = db.thin(frame, "min")
    assert len(monthly) == 61 * 12 and monthly["s"].iloc[0] == 0.0  # each month's worst point
    assert monthly.index[0] == pd.Timestamp("1960-01-29")  # its last day of data


def test_thin_refuses_an_unknown_rule():
    frame = pd.DataFrame({"s": [1.0]}, index=pd.bdate_range("2020-01-01", periods=1))
    with pytest.raises(ValueError, match="how"):
        db.thin(frame, "mean")


def test_slots_keep_each_strategys_color():
    first = db.slots(["a", "b", "c"], {})
    assert first == {"a": 0, "b": 1, "c": 2}
    assert db.slots(["a", "c"], first) == {"a": 0, "c": 2}  # c keeps its color
    assert db.slots(["a", "c", "d"], {"a": 0, "c": 2}) == {"a": 0, "c": 2, "d": 1}
    with pytest.raises(ValueError, match="at most 3"):
        db.slots(["a", "b", "c", "d"], {})


# --- risk, factors and stress ------------------------------------------------------------------


@pytest.fixture(scope="module")
def study_tables(inputs):
    return study.run(
        inputs,
        period="holdout",
        end=END,
        estimator="sample",
        allow_holdout=True,
        settings=SMALL,
        strategies=group(),
    )


def test_var_backtest_is_the_studys_on_the_holdout(bundle, study_tables):
    held = study_tables["var"]
    start, end = pd.Timestamp(HOLDOUT_START), pd.Timestamp(END)
    for view, name in (("USD", study.BENCHMARK), ("CAD", group()[2].name)):
        series = db.net(bundle, view, 5.0)[name]
        for method, level in (("historical", 0.99), ("fhs", 0.95)):
            path, row = db.var_backtest(series, method=method, level=level, start=start, end=end)
            theirs = held.loc[(view, name, study.VAR_WINDOW, level, method)]
            cols = ("days", "exceptions", "kupiec_p", "kupiec_exact_p", "independence_p", "cc_p")
            for col in cols:
                assert row[col] == pytest.approx(theirs[col], rel=1e-12, abs=1e-12), col
            assert list(path.columns) == ["return", "var", "es", "exception"]
            assert path.index[0] >= start and path.index[-1] == series.index[-1]
            assert int(path["exception"].sum()) == row["exceptions"]
            assert (path["exception"] == (path["return"] < -path["var"])).all()


def test_var_backtest_needs_a_forecast_in_the_dates(bundle):
    series = db.net(bundle, "USD", 5.0)[study.BENCHMARK]
    early = series.index[100]
    with pytest.raises(ValueError, match="forecast"):
        db.var_backtest(series, method="normal", level=0.95, start=series.index[0], end=early)


def test_betas_are_step_5bs_on_the_selected_days(bundle, inputs):
    start, end = pd.Timestamp("2012-01-01"), pd.Timestamp("2019-12-31")
    names = bundle.strategies[:2]
    for view in ("USD", "CAD"):
        mine = db.betas(bundle, view, names, cost_bps=5.0, start=start, end=end)
        nets = db.net(bundle, view, 5.0)
        days = (nets.index >= start) & (nets.index <= end)
        excess = nets.loc[days, names].sub(bundle.returns[view].loc[days, db.RF], axis=0)
        if view == "CAD":
            theirs = fa.macro_betas(
                excess, inputs.dy, inputs.fx, inputs.factors, frequency="weekly"
            )
        else:
            theirs = fa.factor_betas(excess, inputs.factors, frequency="weekly")
        pd.testing.assert_frame_equal(mine, theirs, check_exact=False, rtol=1e-12)
    terms = set(mine.index.get_level_values("term"))
    assert {"alpha", "Mkt-RF", "fx", "dy"} <= terms


def test_rolling_betas_use_the_history_before_the_start(bundle):
    start, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31")
    name = bundle.strategies[1]
    rolled = db.rolling_betas(bundle, "USD", name, cost_bps=5.0, start=start, end=end)
    assert rolled.index[0] == bundle.returns["USD"].index[bundle.returns["USD"].index >= start][0]
    assert rolled.index[-1] <= end and "Mkt-RF" in rolled.columns


def test_realized_stress_is_the_studys(bundle, study_tables):
    held = study_tables["stress_realized"]
    names = [study.BENCHMARK, group()[1].name]
    for view in ("USD", "CAD"):
        mine = db.realized(bundle, view, names, cost_bps=5.0)
        for (scenario, name), row in mine.iterrows():
            theirs = held.loc[(scenario, view, name)]
            assert row["pnl"] == pytest.approx(theirs["pnl"], rel=1e-12)
            assert row["max_drawdown"] == pytest.approx(theirs["max_drawdown"], rel=1e-12)
            assert row["days"] == theirs["days"]
            assert row["worst_day"] == pytest.approx(theirs["worst_day"], rel=1e-12)
            assert row["worst_date"] == theirs["worst_date"]
        assert set(mine.index.get_level_values("scenario")) == {"gfc", "covid", "rates"}


def test_scenario_days_are_the_window_after_its_start():
    days = pd.bdate_range("2008-08-25", "2008-12-05")
    mask = db.scenario_days(days, hs.SCENARIOS["gfc"])
    picked = days[mask]
    assert picked[0] == pd.Timestamp("2008-09-02") and picked[-1] == pd.Timestamp("2008-11-28")


def test_realized_skips_the_windows_outside_the_data(short_inputs):
    short = db.build(short_inputs, end="2009-06-30", estimator="sample", strategies=group())
    table = db.realized(short, "USD", [study.BENCHMARK], cost_bps=5.0)
    assert set(table.index.get_level_values("scenario")) == {"gfc"}  # covid and rates are later


def test_read_gives_the_same_bundle_with_or_without_pyarrow(written, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(db, "_FAST_CSV", True)
    fast = db.read(written)
    monkeypatch.setattr(db, "_FAST_CSV", False)
    slow = db.read(written)
    for v in db.VIEWS:
        pd.testing.assert_frame_equal(fast.returns[v], slow.returns[v], check_exact=True)
        pd.testing.assert_frame_equal(fast.turnover[v], slow.turnover[v], check_exact=True)
    pd.testing.assert_frame_equal(fast.weights, slow.weights, check_exact=True)
    pd.testing.assert_frame_equal(fast.factors, slow.factors, check_exact=True)


# --- review fixes -------------------------------------------------------------------------------


def with_late_dy(bundle, first="2015-01-02"):
    """The bundle with the 10-year yield change missing before `first`, as the real one is
    before 2001 (the daily series starts then)."""
    factors = bundle.factors.copy()
    factors.loc[factors.index < pd.Timestamp(first), fa.DY] = np.nan
    return dataclasses.replace(bundle, factors=factors)


def test_cad_betas_leave_the_yield_out_of_windows_that_start_before_it(bundle):
    late = with_late_dy(bundle)
    names = bundle.strategies[:2]
    early = db.betas(
        late, "CAD", names, cost_bps=5.0, start=pd.Timestamp("2012-01-01"), end=pd.Timestamp(END)
    )
    assert "dy" not in set(early.index.get_level_values("term"))
    assert "fx" in set(early.index.get_level_values("term"))
    first = pd.to_datetime(early["start"]).min()
    assert first < pd.Timestamp("2012-01-31")  # the whole window, not from the yield's start
    later = db.betas(
        late, "CAD", names, cost_bps=5.0, start=pd.Timestamp("2015-02-01"), end=pd.Timestamp(END)
    )
    assert "dy" in set(later.index.get_level_values("term"))


def test_beta_table_annualizes_the_weekly_alpha_by_hand():
    index = pd.MultiIndex.from_product([["s"], ["alpha", "Mkt-RF"]], names=["asset", "term"])
    table = pd.DataFrame({"coef": [0.001, 0.9], "t": [2.0, 30.0]}, index=index)
    coef, tstat = db.beta_table(table)
    assert coef.loc["s", "alpha"] == pytest.approx(0.052)  # 0.1% a week, 52 weeks
    assert coef.loc["s", "Mkt-RF"] == 0.9 and tstat.loc["s", "alpha"] == 2.0
    assert list(coef.columns) == ["alpha", "Mkt-RF"]


def test_holdings_keep_the_largest_industries_and_fold_the_rest():
    traded = pd.DatetimeIndex(["2020-01-02", "2020-02-03"], name="traded")
    w = pd.DataFrame(
        {"A": [0.5, 0.3], "B": [0.3, 0.5], "C": [0.15, 0.1], "D": [0.05, 0.1]}, index=traded
    )
    weights = pd.concat({("USD", "s"): w}, names=["view", "strategy", "traded"])
    bundle = db.Bundle(
        design=pd.DataFrame(),
        returns={},
        turnover={},
        weights=weights,
        factors=pd.DataFrame(),
        tables={},
    )
    held = db.holdings(bundle, "USD", "s", top=2)
    assert list(held.columns) == ["A", "B", db.OTHER]  # by average weight, largest first
    assert held[db.OTHER].tolist() == pytest.approx([0.2, 0.2])
    assert np.allclose(held.sum(axis=1), 1.0)
    assert list(db.holdings(bundle, "USD", "s", top=6).columns) == ["A", "B", "C", "D"]


def test_check_refuses_a_performance_table_with_other_columns(bundle, references):
    performance, monthly = references
    with pytest.raises(ValueError, match="columns"):
        db.check(bundle, performance.drop(columns="calmar"), monthly)
    with pytest.raises(ValueError, match="columns"):
        db.check(bundle, performance.assign(extra=1.0), monthly)


def test_write_removes_files_of_an_older_bundle(bundle, tmp_path):
    (tmp_path / "holdout").mkdir()
    (tmp_path / "holdout" / "stress_realized.csv").write_text("old")
    (tmp_path / "returns_EUR.csv.gz").write_bytes(b"old")
    (tmp_path / "README.md").write_text("kept: not a bundle file")
    db.write(bundle, tmp_path, copies=copies())
    assert not (tmp_path / "holdout" / "stress_realized.csv").exists()
    assert not (tmp_path / "returns_EUR.csv.gz").exists()
    assert (tmp_path / "README.md").exists()


def test_holdings_keep_an_industry_named_other_apart_from_the_fold():
    traded = pd.DatetimeIndex(["2020-01-02"], name="traded")
    w = pd.DataFrame({"A": [0.2], "Other": [0.7], "B": [0.1]}, index=traded)
    weights = pd.concat({("USD", "s"): w}, names=["view", "strategy", "traded"])
    bundle = db.Bundle(
        design=pd.DataFrame(),
        returns={},
        turnover={},
        weights=weights,
        factors=pd.DataFrame(),
        tables={},
    )
    held = db.holdings(bundle, "USD", "s", top=1)
    assert list(held.columns) == ["Other", db.OTHER]  # French's "Other" industry, then the rest
    assert held.iloc[0].tolist() == pytest.approx([0.7, 0.3])


# --- /code-review fixes ---------------------------------------------------------------------------


def test_read_needs_one_over_n(bundle, tmp_path):
    names = [n for n in bundle.strategies if n != study.BENCHMARK]
    without = dataclasses.replace(
        bundle,
        returns={v: f.drop(columns=study.BENCHMARK) for v, f in bundle.returns.items()},
        turnover={v: f[names] for v, f in bundle.turnover.items()},
        design=bundle.design.assign(value=bundle.design["value"].replace({"3": "2"})),
    )
    db.write(without, tmp_path, copies=copies())
    with pytest.raises(ValueError, match="1/N"):
        db.read(tmp_path)


def test_read_detects_a_bundle_written_only_in_part(bundle, tmp_path):
    db.write(bundle, tmp_path, copies=copies())
    old = (tmp_path / "turnover_USD.csv.gz").read_bytes()
    later = dataclasses.replace(bundle, turnover={v: f * 2 for v, f in bundle.turnover.items()})
    db.write(later, tmp_path, copies=copies())
    (tmp_path / "turnover_USD.csv.gz").write_bytes(old)  # as if this file was never replaced
    with pytest.raises(ValueError, match="turnover_USD.csv.gz"):
        db.read(tmp_path)


def test_net_for_some_strategies_is_the_same_numbers(bundle):
    some = bundle.strategies[1:]
    pd.testing.assert_frame_equal(
        db.net(bundle, "CAD", 7.0, some), db.net(bundle, "CAD", 7.0)[some]
    )


def test_performance_needs_a_year(bundle):
    with pytest.raises(ValueError, match="12 months"):
        db.performance(
            bundle,
            "USD",
            bundle.strategies,
            cost_bps=5.0,
            start=pd.Timestamp("2015-01-01"),
            end=pd.Timestamp("2015-06-30"),
        )


def test_holdings_in_a_window_start_with_the_weights_in_force():
    traded = pd.DatetimeIndex(["2020-01-02", "2020-02-03", "2020-03-02"], name="traded")
    w = pd.DataFrame({"A": [0.6, 0.5, 0.4], "B": [0.4, 0.5, 0.6]}, index=traded)
    weights = pd.concat({("USD", "s"): w}, names=["view", "strategy", "traded"])
    bundle = db.Bundle(
        design=pd.DataFrame(),
        returns={},
        turnover={},
        weights=weights,
        factors=pd.DataFrame(),
        tables={},
    )
    held = db.holdings(
        bundle, "USD", "s", start=pd.Timestamp("2020-02-15"), end=pd.Timestamp("2020-03-31")
    )
    assert list(held.index) == list(traded[1:])  # February's targets held on 15 February
