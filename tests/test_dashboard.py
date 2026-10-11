"""The dashboard's bundle and its pure functions (Step 8, plans/step-8-plan.md D1-D6)."""

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
