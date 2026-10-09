"""Historical stress replays (blueprint Step 6, plans/step-6-plan.md).

A scenario covers the returns of the days in (start, end]: the weights are bought at the close
of the last day on or before `start` and held, drifting with no trades, to the close of the last
day on or before `end`. Holdout windows need `allow_holdout=True`, which tests may pass (the
text scan exempts tests/); every window here runs on synthetic data. Hand examples are worked in
exact fractions.
"""

import dataclasses
from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.data import HoldoutError
from portfolio_lab.data.views import View, make_views
from portfolio_lab.report import metrics
from portfolio_lab.stress import historical as hs

ASSETS = ["A", "B"]


def ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day)


def scenario(start: str, end: str, name: str = "test") -> hs.Scenario:
    return hs.Scenario(name, ts(start), ts(end), "a test window")


def calendar(first: str, last: str, *, holidays: tuple[str, ...] = ()) -> pd.DatetimeIndex:
    days = pd.bdate_range(first, last, name="date")
    return days.drop(pd.DatetimeIndex(holidays)) if holidays else days


def frame(days: pd.DatetimeIndex, seed: int = 0, columns=ASSETS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-0.05, 0.05, size=(len(days), len(columns)))
    return pd.DataFrame(values, index=days, columns=columns)


def view(returns: pd.DataFrame) -> View:
    return View(returns, pd.Series(0.0, index=returns.index, name="rf"))


def weights(*values: float, assets=ASSETS) -> pd.Series:
    return pd.Series(values, index=assets, dtype=float, name="w")


def approx(values, fractions) -> None:
    expected = [float(f) for f in fractions]
    np.testing.assert_allclose(np.asarray(values, dtype=float), expected, rtol=0, atol=1e-15)


GFC = hs.SCENARIOS["gfc"]
# Labor Day (2008-09-01) and Thanksgiving (2008-11-27) closed the US market.
GFC_DAYS = calendar("2008-08-25", "2008-12-05", holidays=("2008-09-01", "2008-11-27"))

# The hand example: bought at the close of Tue 2008-09-09, held Wed to Fri. The rows up to and
# including the anchor are NaN and the day after the window is +500%: none may be read.
HAND_DAYS = pd.DatetimeIndex(
    ["2008-09-08", "2008-09-09", "2008-09-10", "2008-09-11", "2008-09-12", "2008-09-15"],
    name="date",
)
HAND = pd.DataFrame(
    {
        "A": [np.nan, np.nan, 0.10, -0.05, 0.02, 5.0],
        "B": [np.nan, np.nan, -0.02, 0.03, 0.00, 5.0],
    },
    index=HAND_DAYS,
)
HAND_W = {"A": F(6, 10), "B": F(4, 10)}
HAND_R = {"A": [F(10, 100), F(-5, 100), F(2, 100)], "B": [F(-2, 100), F(3, 100), F(0)]}


def hand_navs() -> list[F]:
    """Buy-and-hold NAV at each close of the window, from 1 at the anchor."""
    growth = {k: F(1) for k in HAND_W}
    navs = []
    for t in range(3):
        for k in growth:
            growth[k] *= 1 + HAND_R[k][t]
        navs.append(sum(HAND_W[k] * growth[k] for k in HAND_W))
    return navs


# --- the registry ---------------------------------------------------------------------------


def test_scenarios_are_the_blueprints_windows():
    got = {key: (s.name, s.start, s.end) for key, s in hs.SCENARIOS.items()}
    assert got == {
        "gfc": ("gfc", ts("2008-09-01"), ts("2008-11-30")),
        "covid": ("covid", ts("2020-02-19"), ts("2020-03-23")),
        "rates": ("rates", ts("2022-01-01"), ts("2022-12-31")),
    }
    assert all(s.note for s in hs.SCENARIOS.values())


def test_scenarios_are_read_only():
    with pytest.raises(TypeError):
        hs.SCENARIOS["gfc"] = scenario("2001-01-01", "2001-02-01")
    with pytest.raises(dataclasses.FrozenInstanceError):
        hs.SCENARIOS["gfc"].end = ts("2008-12-31")


@pytest.mark.parametrize("start, end", [("2008-11-30", "2008-09-01"), ("2008-09-01", "2008-09-01")])
def test_a_scenario_ends_after_it_starts(start, end):
    with pytest.raises(ValueError, match="after"):
        scenario(start, end)


@pytest.mark.parametrize("bad", ["", "NaT", 20080901, None])
def test_a_scenario_needs_real_dates(bad):
    with pytest.raises(ValueError, match="not a date"):
        hs.Scenario("x", bad, ts("2008-11-30"), "")
    with pytest.raises(ValueError, match="not a date"):
        hs.Scenario("x", ts("2008-09-01"), bad, "")


def test_a_scenario_takes_date_strings():
    s = hs.Scenario("x", "2008-09-01", "2008-11-30", "")
    assert (s.start, s.end) == (ts("2008-09-01"), ts("2008-11-30"))


# --- window ---------------------------------------------------------------------------------


def test_window_is_start_exclusive_and_end_inclusive_around_a_holiday_and_a_weekend():
    r = frame(GFC_DAYS)
    got = hs.window(r, GFC)
    inside = GFC_DAYS[(GFC_DAYS > "2008-09-01") & (GFC_DAYS <= "2008-11-30")]
    pd.testing.assert_frame_equal(got, r.loc[inside])
    # Labor Day start, Sunday end: Tue 2008-09-02 to Fri 2008-11-28.
    assert (got.index[0], got.index[-1]) == (ts("2008-09-02"), ts("2008-11-28"))


def test_window_leaves_out_the_start_days_own_return():
    days = calendar("2020-02-10", "2020-03-31")
    got = hs.window(frame(days), hs.SCENARIOS["covid"], allow_holdout=True)
    # Peak to trough: held from the close of 2020-02-19, so its return is not in the window.
    assert (got.index[0], got.index[-1]) == (ts("2020-02-20"), ts("2020-03-23"))


def test_window_keeps_a_series_a_series():
    r = frame(GFC_DAYS)
    got = hs.window(r["A"], GFC)
    pd.testing.assert_series_equal(got, hs.window(r, GFC)["A"])


def test_window_needs_a_close_on_or_before_the_start():
    assert hs.window(frame(GFC_DAYS[GFC_DAYS >= "2008-08-29"]), GFC).index[0] == ts("2008-09-02")
    covid_days = calendar("2020-02-19", "2020-03-31")  # starts on the scenario's start day
    covid = hs.window(frame(covid_days), hs.SCENARIOS["covid"], allow_holdout=True)
    assert covid.index[0] == ts("2020-02-20")
    with pytest.raises(ValueError, match="2008-09-02"):
        hs.window(frame(GFC_DAYS[GFC_DAYS >= "2008-09-02"]), GFC)


def test_window_needs_data_through_the_last_weekday_of_the_window():
    # A weekend end is reached on the Friday before it. A holiday is not known, so data ending
    # before a weekday end (even a holiday) is refused.
    assert hs.window(frame(GFC_DAYS[GFC_DAYS <= "2008-11-28"]), GFC).index[-1] == ts("2008-11-28")
    with pytest.raises(ValueError, match="2008-11-26"):
        hs.window(frame(GFC_DAYS[GFC_DAYS <= "2008-11-26"]), GFC)
    with pytest.raises(ValueError, match="2008-10-31"):
        hs.window(frame(GFC_DAYS[GFC_DAYS <= "2008-10-31"]), GFC)
    covid = frame(calendar("2020-02-10", "2020-03-20"))  # ends the Friday before Monday's end
    with pytest.raises(ValueError, match="2020-03-20"):
        hs.window(covid, hs.SCENARIOS["covid"], allow_holdout=True)


def test_window_compares_calendar_dates_not_times_of_day():
    days = calendar("2020-02-19", "2020-03-31") + pd.Timedelta(hours=16)  # closes at 16:00
    r = frame(days)
    got = hs.window(r, hs.SCENARIOS["covid"], allow_holdout=True)
    assert (got.index[0], got.index[-1]) == (ts("2020-02-20 16:00"), ts("2020-03-23 16:00"))
    replayed = hs.replay(weights(0.5, 0.5), view(r), hs.SCENARIOS["covid"], allow_holdout=True)
    assert replayed.index.equals(got.index)


def test_window_with_no_day_inside_raises():
    days = calendar("2008-09-01", "2008-09-30")
    with pytest.raises(ValueError, match="no day"):
        hs.window(frame(days), scenario("2008-09-05", "2008-09-07"))  # Friday to Sunday


@pytest.mark.parametrize("key", ["covid", "rates"])
def test_holdout_scenarios_need_the_flag_even_on_synthetic_data(key):
    r = frame(calendar("2019-12-02", "2023-01-31"))
    with pytest.raises(HoldoutError):
        hs.window(r, hs.SCENARIOS[key])
    assert len(hs.window(r, hs.SCENARIOS[key], allow_holdout=True)) > 0


def test_the_holdout_rule_comes_before_the_coverage_check():
    with pytest.raises(HoldoutError):
        hs.window(frame(GFC_DAYS), hs.SCENARIOS["covid"])


def test_window_uses_the_loaders_holdout_rule():
    r = frame(calendar("2010-11-01", "2011-01-31"))
    assert hs.window(r, scenario("2010-12-01", "2010-12-31")).index[-1] == ts("2010-12-31")
    with pytest.raises(HoldoutError):
        hs.window(r, scenario("2010-12-01", "2011-01-01"))


def test_window_rejects_unsorted_or_undated_returns():
    r = frame(GFC_DAYS)
    with pytest.raises(ValueError, match="sorted"):
        hs.window(r.iloc[::-1], GFC)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        hs.window(r.reset_index(drop=True), GFC)


def test_window_needs_a_scenario_and_returns():
    with pytest.raises(TypeError, match="Scenario"):
        hs.window(frame(GFC_DAYS), "gfc")
    with pytest.raises(TypeError, match="Series or DataFrame"):
        hs.window(frame(GFC_DAYS).to_numpy(), GFC)


# --- replay ---------------------------------------------------------------------------------


def test_replay_is_buy_and_hold_by_hand():
    got = hs.replay(weights(0.6, 0.4), view(HAND), scenario("2008-09-09", "2008-09-12"))
    n1, n2, n3 = hand_navs()
    assert got.index.equals(HAND_DAYS[2:5])
    # Drift, not constant weights: day 2 is not 0.6 x -5% + 0.4 x 3% = -1.8%.
    approx(got, [n1 - 1, n2 / n1 - 1, n3 / n2 - 1])
    assert float(np.prod(1 + got.to_numpy()) - 1) == pytest.approx(float(n3 - 1), abs=1e-15)


@pytest.mark.parametrize("seed", range(5))
def test_replay_pnl_is_the_weighted_sum_of_each_assets_compounded_return(seed):
    rng = np.random.default_rng(seed)
    r = pd.DataFrame(
        rng.uniform(-0.1, 0.1, size=(len(GFC_DAYS), 4)), index=GFC_DAYS, columns=list("WXYZ")
    )
    w = pd.Series(rng.dirichlet(np.ones(4)), index=r.columns)
    got = hs.replay(w, view(r), GFC)
    inside = r.loc[(GFC_DAYS > "2008-09-01") & (GFC_DAYS <= "2008-11-30")]
    expected = float(w @ ((1 + inside).prod() - 1))
    assert float(np.prod(1 + got.to_numpy()) - 1) == pytest.approx(expected, abs=1e-12)
    assert got.index.equals(inside.index)


@pytest.mark.parametrize(
    "w, error, match",
    [
        (pd.Series([0.4, 0.6], index=["B", "A"]), ValueError, "same assets"),
        (pd.Series([1.0], index=["A"]), ValueError, "same assets"),
        (pd.Series([1.2, -0.2], index=ASSETS), ValueError, ">= 0"),
        (pd.Series([0.5, 0.4], index=ASSETS), ValueError, "sum to 1"),
        (pd.Series([np.nan, 1.0], index=ASSETS), ValueError, "finite"),
        ({"A": 0.5, "B": 0.5}, TypeError, "Series"),
    ],
)
def test_replay_checks_the_weights_like_the_engine(w, error, match):
    with pytest.raises(error, match=match):
        hs.replay(w, view(HAND), scenario("2008-09-09", "2008-09-12"))


def test_replay_needs_a_view():
    with pytest.raises(TypeError, match="View"):
        hs.replay(weights(0.6, 0.4), HAND, scenario("2008-09-09", "2008-09-12"))


def test_replay_rejects_a_gap_inside_the_window_even_at_zero_weight():
    r = HAND.copy()
    r.loc["2008-09-11", "B"] = np.nan
    with pytest.raises(ValueError, match="missing for B"):
        hs.replay(weights(1.0, 0.0), view(r), scenario("2008-09-09", "2008-09-12"))


def test_replay_applies_the_holdout_rule():
    v = view(frame(calendar("2020-02-03", "2020-03-31")))
    with pytest.raises(HoldoutError):
        hs.replay(weights(0.5, 0.5), v, hs.SCENARIOS["covid"])
    got = hs.replay(weights(0.5, 0.5), v, hs.SCENARIOS["covid"], allow_holdout=True)
    assert (got.index[0], got.index[-1]) == (ts("2020-02-20"), ts("2020-03-23"))


def test_a_usd_rally_is_a_gain_in_the_cad_replay_only():
    days = calendar("2008-09-01", "2008-09-15")
    zeros = pd.Series(0.0, index=days)
    usdcad = pd.Series(1.0, index=days)  # CAD per USD
    usdcad.loc["2008-09-10":] = 1.05  # the USD gains 5% on 2008-09-10
    views = make_views(pd.DataFrame(0.0, index=days, columns=ASSETS), zeros, usdcad, zeros)
    s = scenario("2008-09-02", "2008-09-12")
    cad = hs.replay(weights(0.5, 0.5), views["CAD"], s)
    assert cad.loc["2008-09-10"] == pytest.approx(0.05, abs=1e-15)
    assert hs.outcome(cad).pnl == pytest.approx(0.05, abs=1e-15)
    assert hs.outcome(hs.replay(weights(0.5, 0.5), views["USD"], s)).pnl == 0
    assert hs.outcome(hs.replay(weights(0.5, 0.5), views["CAD_HEDGED"], s)).pnl == 0


def test_the_cad_replay_of_one_asset_compounds_asset_and_currency():
    rng = np.random.default_rng(7)
    assets = frame(GFC_DAYS, seed=3)
    usdcad = pd.Series(np.cumprod(1 + rng.uniform(-0.01, 0.01, len(GFC_DAYS))), index=GFC_DAYS)
    zeros = pd.Series(0.0, index=GFC_DAYS)
    got = hs.replay(weights(1.0, 0.0), make_views(assets, zeros, usdcad, zeros)["CAD"], GFC)
    fx = usdcad / usdcad.shift(1) - 1
    expected = ((1 + assets["A"]) * (1 + fx) - 1).loc[got.index]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-14)


# --- outcome --------------------------------------------------------------------------------


def test_outcome_by_hand():
    days = calendar("2008-09-02", "2008-09-05")
    got = hs.outcome(pd.Series([-0.05, 0.10, -0.20, 0.05], index=days))
    # NAV 0.95, 1.045, 0.836, 0.8778: the deepest fall is 0.836 / 1.045 - 1 = -20%.
    assert got.pnl == pytest.approx(-0.1222, abs=1e-15)
    assert (got.worst_day, got.worst_date) == (-0.20, ts("2008-09-04"))
    assert got.max_drawdown == pytest.approx(-0.20, abs=1e-15)
    assert (got.days, got.first, got.last) == (4, ts("2008-09-02"), ts("2008-09-05"))


def test_a_loss_on_day_one_is_a_drawdown_from_the_start():
    got = hs.outcome(pd.Series([-0.10, 0.05], index=calendar("2008-09-02", "2008-09-03")))
    assert got.max_drawdown == pytest.approx(-0.10, abs=1e-15)
    assert got.pnl == pytest.approx(-0.055, abs=1e-15)


def test_outcome_without_a_loss():
    got = hs.outcome(pd.Series([0.02, 0.01], index=calendar("2008-09-02", "2008-09-03")))
    assert (got.max_drawdown, got.worst_day, got.worst_date) == (0.0, 0.01, ts("2008-09-03"))


def test_outcome_of_the_hand_replay():
    replay = hs.replay(weights(0.6, 0.4), view(HAND), scenario("2008-09-09", "2008-09-12"))
    got = hs.outcome(replay)
    n1, n2, n3 = hand_navs()
    assert got.pnl == pytest.approx(float(n3 - 1), abs=1e-15)
    assert got.max_drawdown == pytest.approx(float(n2 / n1 - 1), abs=1e-15)  # peak day 1
    assert got.worst_date == ts("2008-09-11")


@pytest.mark.parametrize("seed", range(3))
def test_outcome_agrees_with_the_metrics(seed):
    r = pd.Series(np.random.default_rng(seed).normal(0, 0.03, 60), index=GFC_DAYS[:60])
    got = hs.outcome(r)
    assert got.max_drawdown == metrics.max_drawdown(r)
    assert got.pnl == pytest.approx(float(np.prod(1 + r.to_numpy()) - 1), abs=1e-14)
    assert (got.worst_day, got.worst_date) == (r.min(), r.idxmin())


@pytest.mark.parametrize(
    "bad, error",
    [
        (frame(GFC_DAYS[:3]), TypeError),
        (pd.Series([0.01, np.nan], index=GFC_DAYS[:2]), ValueError),
        (pd.Series([], index=GFC_DAYS[:0], dtype=float), ValueError),
        (pd.Series([0.01, -1.5], index=GFC_DAYS[:2]), ValueError),
        (pd.Series([0.01, 0.02], index=GFC_DAYS[:2][::-1]), ValueError),
    ],
)
def test_outcome_rejects_what_is_not_one_portfolios_daily_returns(bad, error):
    with pytest.raises(error):
        hs.outcome(bad)


# --- table ----------------------------------------------------------------------------------


def test_table_has_one_outcome_per_scenario_view_and_portfolio():
    usd, cad = view(frame(GFC_DAYS, seed=1)), view(frame(GFC_DAYS, seed=2))
    october = scenario("2008-09-30", "2008-10-31", name="october")
    w = {
        "USD": {"even": weights(0.5, 0.5), "tilted": weights(0.8, 0.2)},
        "CAD": {"even": weights(0.5, 0.5)},
    }
    got = hs.table(w, {"USD": usd, "CAD": cad}, [GFC, october])
    assert got.index.names == ["scenario", "view", "portfolio"]
    assert list(got.index) == [
        ("gfc", "USD", "even"),
        ("gfc", "USD", "tilted"),
        ("gfc", "CAD", "even"),
        ("october", "USD", "even"),
        ("october", "USD", "tilted"),
        ("october", "CAD", "even"),
    ]
    assert list(got.columns) == [f.name for f in dataclasses.fields(hs.Outcome)]
    expected = hs.outcome(hs.replay(weights(0.8, 0.2), usd, october))
    assert got.loc[("october", "USD", "tilted")].to_dict() == dataclasses.asdict(expected)


def test_table_checks_every_scenario_against_the_holdout_rule_before_replaying(monkeypatch):
    calls = []
    monkeypatch.setattr(hs, "replay", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(HoldoutError):
        hs.table(
            {"USD": {"even": weights(0.5, 0.5)}},
            {"USD": view(frame(GFC_DAYS))},
            [GFC, hs.SCENARIOS["covid"]],
        )
    assert calls == []


def test_table_needs_a_view_for_every_set_of_weights():
    with pytest.raises(ValueError, match="CAD"):
        hs.table({"CAD": {"even": weights(0.5, 0.5)}}, {"USD": view(frame(GFC_DAYS))}, [GFC])


def test_table_needs_unique_scenario_names():
    october = hs.Scenario("gfc", ts("2008-09-30"), ts("2008-10-31"), "same name, other dates")
    with pytest.raises(ValueError, match="unique"):
        hs.table(
            {"USD": {"even": weights(0.5, 0.5)}}, {"USD": view(frame(GFC_DAYS))}, [GFC, october]
        )


def test_table_needs_scenarios_and_portfolios():
    v = {"USD": view(frame(GFC_DAYS))}
    with pytest.raises(ValueError, match="scenario"):
        hs.table({"USD": {"even": weights(0.5, 0.5)}}, v, [])
    with pytest.raises(ValueError, match="portfolio"):
        hs.table({"USD": {}}, v, [GFC])
