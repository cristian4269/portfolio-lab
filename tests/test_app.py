"""The dashboard app (app/streamlit_app.py) end to end, on a synthetic bundle: no network, no
real data (blueprint Step 8: an `AppTest` smoke test)."""

from pathlib import Path

import pandas as pd
import pytest
from test_study import SMALL, synthetic_inputs

from portfolio_lab import dashboard as db
from portfolio_lab import study
from portfolio_lab.construct import strategy as st

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
END = "2022-12-31"


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    """A bundle of the primary three on synthetic data, with the registered tables that the
    study itself writes for the same data."""
    inputs = synthetic_inputs(start="2007-01-01", end="2022-12-30", n=3, seed=1)
    group = [st.equal_weight(), st.min_variance("lw_identity"), st.erc("lw_identity")]
    bundle = db.build(
        inputs, end=END, estimator="lw_identity", allow_holdout=True, strategies=group
    )
    tables = study.run(
        inputs,
        period="holdout",
        end=END,
        estimator="lw_identity",
        allow_holdout=True,
        settings=SMALL,
        strategies=group,
    )
    copies = {
        f"holdout/{t}.csv": tables[t].to_csv(lineterminator="\n").encode()
        for t in db.HOLDOUT_TABLES
    }
    path = tmp_path_factory.mktemp("bundle")
    db.write(bundle, path, copies=copies)
    return path


@pytest.fixture
def app(folder, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_LAB_APP_DATA", str(folder))
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def cagr(at) -> pd.Series:
    return at.dataframe[0].value["CAGR"]


def test_every_tab_renders(app):
    assert [t.label for t in app.tabs] == ["Performance", "Risk", "Factors", "Stress", "About"]
    assert any("registered tests" in h.value for h in app.subheader)
    assert any("VaR" in h.value for h in app.subheader)
    assert any("Factor betas" in h.value for h in app.subheader)
    assert any("stress" in h.value.lower() for h in app.subheader)
    assert len(app.metric) == 4  # the VaR backtest


def test_the_cost_slider_changes_the_metrics(app):
    at_5bp = cagr(app)
    app.slider(key="cost").set_value(30).run()
    assert not app.exception
    assert (cagr(app) < at_5bp).all()


def test_at_most_three_strategies_are_charted(app):
    assert app.multiselect(key="charted").proto.max_selections == db.MAX_CHARTED
    assert list(app.multiselect(key="charted").value) == [
        s.name for s in study.strategies()[:1]
    ] + list(study.primary("lw_identity"))


def test_every_view_renders(app):
    for label in ("Canadian investor, unhedged", "Canadian investor, hedged"):
        app.radio(key="view").set_value(label).run()  # AppTest selects by the shown label
        assert not app.exception, label
    assert any("no test" in c.value for c in app.caption)  # the hedged view's registered tests


def test_months_without_a_forecast_say_so(app, folder):
    months = db.months(db.read(folder), "USD")  # the slider's values, before its format_func
    app.select_slider(key="months_USD").set_range(months[0], months[13]).run()
    assert not app.exception
    assert any("Choose later months" in i.value for i in app.info)  # VaR needs 500 days


def test_less_than_a_year_asks_for_more(app, folder):
    months = db.months(db.read(folder), "USD")
    app.select_slider(key="months_USD").set_range(months[40], months[45]).run()
    assert not app.exception
    assert any("at least 12 months" in i.value for i in app.info)


def test_no_strategy_asks_for_one(app):
    app.multiselect(key="charted").set_value([]).run()
    assert not app.exception
    assert any("at least one strategy" in i.value for i in app.info)
