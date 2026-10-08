"""Strategies in the engine's loop: targets, run, compare and Result (plans/step-4-plan.md).

At each month's last trading day t the engine gives the strategy the returns history ending at
the close of t, and the portfolio trades at the close of the next day. No look-ahead is checked
three ways: a spy sees every history the engine passes, and changing the returns after any day
k changes nothing estimated or earned through k, both for a rule that reads its whole window and
for a real minimum-variance strategy. `run` stops at DEV_END unless the holdout is allowed.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from portfolio_lab.backtest import engine as bt
from portfolio_lab.config import DEV_END
from portfolio_lab.construct import strategy as sg
from portfolio_lab.data import HoldoutError
from portfolio_lab.data.views import View
from portfolio_lab.report import metrics


def returns_frame(seed: int, t: int, n: int = 3, start: str = "2001-01-01") -> pd.DataFrame:
    """t business days of independent daily returns (vol 1%), oldest first."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=t, name="date")
    values = rng.normal(0.0003, 0.01, size=(t, n))
    return pd.DataFrame(values, index=index, columns=[f"x{i}" for i in range(n)])


def view_of(returns: pd.DataFrame, rf: float = 0.0001) -> View:
    return View(returns, pd.Series(rf, index=returns.index, name="RF"))


class Spy:
    """A duck-typed strategy that records every history the engine passes and holds 1/N."""

    def __init__(self, lookback: int, name: str = "spy"):
        self.name = name
        self.lookback = lookback
        self.histories: list[pd.DataFrame] = []

    def weights(self, history: pd.DataFrame) -> pd.Series:
        self.histories.append(history)
        return pd.Series(1 / history.shape[1], index=history.columns)


def whole_window(lookback: int = 10) -> sg.Strategy:
    """A rule that reads every value of its window: weights proportional to exp(50 x means)."""

    def rule(window: pd.DataFrame) -> pd.Series:
        z = 50 * window.mean()
        w = np.exp(z - z.max())
        return w / w.sum()

    return sg.Strategy("whole window", lookback, rule)


# 2001 has 23 business days in January, 20 in February, 22 in March, 21 in April, 23 in May
# and 21 in June: 130 days from 2001-01-01 end on Friday 2001-06-29, a month end.
HALF_YEAR = returns_frame(3, 130)
# 600 business days from 2009-01-01 run into the holdout (to April 2011).
LONG = returns_frame(4, 600, start="2009-01-01")


# --- targets: what the strategy sees, and when it trades ---


def test_each_history_ends_at_its_estimation_date_and_the_next_day_trades():
    spy = Spy(lookback=30)
    tg = bt.targets(spy, HALF_YEAR)
    schedule = bt.rebalance_dates(HALF_YEAR.index, start="2001-02-28")  # first with 30 rows
    assert len(spy.histories) == len(schedule) == 4
    for history, t in zip(spy.histories, schedule.index, strict=True):
        pos = HALF_YEAR.index.get_loc(t)
        pd.testing.assert_frame_equal(history, HALF_YEAR.iloc[: pos + 1])  # through t, no later
    assert list(tg.index) == list(schedule)
    assert tg.index.name == "traded"


def test_targets_are_the_strategy_weights_on_each_history():
    returns = returns_frame(2, 200, n=4)
    s = sg.min_variance("sample", window=40)
    tg = bt.targets(s, returns)
    assert len(tg) > 3
    for traded, row in tg.iterrows():
        t = returns.index[returns.index.get_loc(traded) - 1]
        pd.testing.assert_series_equal(
            row, s.weights(returns.loc[:t]), check_exact=True, check_names=False
        )


@pytest.mark.parametrize(
    ("lookback", "first"),
    [
        (1, "2001-01-31"),
        (23, "2001-01-31"),  # exactly the 23 rows of January
        (24, "2001-02-28"),
        (43, "2001-02-28"),
        (44, "2001-03-30"),
    ],
)
def test_without_start_the_first_rebalance_waits_for_lookback_rows(lookback, first):
    spy = Spy(lookback)
    bt.targets(spy, HALF_YEAR)
    assert spy.histories[0].index[-1] == pd.Timestamp(first)
    assert len(spy.histories[0]) >= lookback


def test_a_start_too_early_for_the_lookback_raises_and_names_the_earliest_date():
    spy = Spy(30)
    with pytest.raises(ValueError, match="needs 30 days.*2001-02-28"):
        bt.targets(spy, HALF_YEAR, start="2001-01-15")
    assert spy.histories == []  # failed before estimating anything


def test_targets_align_weights_to_the_return_columns_by_label():
    class Reversed(Spy):
        """Weights 1, 2, 3 (over 6) for x0, x1, x2, returned in reverse label order."""

        def weights(self, history: pd.DataFrame) -> pd.Series:
            w = pd.Series([1.0, 2.0, 3.0], index=history.columns) / 6
            return w.iloc[::-1]

    tg = bt.targets(Reversed(5), HALF_YEAR)
    assert tg.columns.equals(HALF_YEAR.columns)
    assert tg.iloc[0].tolist() == pytest.approx([1 / 6, 2 / 6, 3 / 6], rel=1e-15)
    bt.simulate(tg, HALF_YEAR, cost_bps=5)  # accepted as is


def test_a_later_start_is_kept():
    spy = Spy(30)
    bt.targets(spy, HALF_YEAR, start="2001-03-15")
    assert spy.histories[0].index[-1] == pd.Timestamp("2001-03-30")


def test_no_month_end_with_enough_history_raises():
    with pytest.raises(ValueError, match="no month end"):
        bt.targets(Spy(200), HALF_YEAR)


def test_a_start_after_the_last_rebalance_raises():
    with pytest.raises(ValueError, match="start"):
        bt.targets(Spy(5), HALF_YEAR, start="2001-06-29")


# --- no look-ahead: later returns change nothing estimated or earned before them ---


def perturbed(returns: pd.DataFrame, cut: int, seed: int) -> pd.DataFrame:
    """The same returns through row `cut`, different (and wilder) ones after it."""
    rng = np.random.default_rng(seed)
    moved = returns.copy()
    later = moved.iloc[cut + 1 :]
    moved.iloc[cut + 1 :] = rng.normal(0.0, 0.05, size=later.shape)
    return moved


def assert_nothing_through_cut_changes(s, returns: pd.DataFrame, cut: int, seed: int) -> None:
    base = bt.run(s, view_of(returns), cost_bps=5)
    alt = bt.run(s, view_of(perturbed(returns, cut, seed)), cost_bps=5)
    # A target traded at position p was estimated at p - 1: known by `cut` if p - 1 <= cut.
    traded_at = returns.index.get_indexer(base.weights.index)
    known = base.weights.index[traded_at - 1 <= cut]
    pd.testing.assert_frame_equal(alt.weights.loc[known], base.weights.loc[known], check_exact=True)
    day = returns.index[cut]
    pd.testing.assert_series_equal(alt.net.loc[:day], base.net.loc[:day], check_exact=True)
    pd.testing.assert_series_equal(alt.gross.loc[:day], base.gross.loc[:day], check_exact=True)
    pd.testing.assert_series_equal(
        alt.turnover.loc[:day], base.turnover.loc[:day], check_exact=True
    )


@given(seed=st.integers(0, 2**32 - 1), cut=st.integers(0, 179))
def test_later_returns_change_nothing_through_any_day(seed, cut):
    assert_nothing_through_cut_changes(whole_window(10), returns_frame(seed, 180), cut, seed + 1)


@settings(max_examples=25)
@given(seed=st.integers(0, 2**32 - 1), cut=st.integers(0, 199))
def test_later_returns_change_nothing_through_any_day_for_min_variance(seed, cut):
    s = sg.min_variance("sample", window=30)
    assert_nothing_through_cut_changes(s, returns_frame(seed, 200, n=4), cut, seed + 1)


def test_the_estimate_ignores_the_trade_day_return():
    """A huge return on a trade day t + 1 can't change the target traded at its close.

    The 10-day window is shorter than a month, so no later window holds an earlier shocked
    trade day either: every target must be unchanged. (With a lookback of a month or more,
    later targets would rightly change: those rows are history by then.)"""
    returns = returns_frame(9, 130)
    tg = bt.targets(whole_window(10), returns)
    shocked = returns.copy()
    shocked.loc[tg.index] = 0.9
    pd.testing.assert_frame_equal(bt.targets(whole_window(10), shocked), tg, check_exact=True)


# --- run: one strategy on one view, guarded like the loaders ---


def test_run_is_targets_then_simulate_with_the_strategy_name_and_the_view_rf():
    returns = returns_frame(5, 300, n=4)
    rf = pd.Series(np.linspace(1e-4, 2e-4, 300), index=returns.index, name="RF")
    s = sg.erc("sample", window=60)
    result = bt.run(s, View(returns, rf), cost_bps=7)
    ledger = bt.simulate(bt.targets(s, returns), returns, cost_bps=7)
    assert result.name == s.name
    for field in ("gross", "net", "turnover", "cost", "final_weights"):
        pd.testing.assert_series_equal(getattr(result, field), getattr(ledger, field))
    pd.testing.assert_frame_equal(result.weights, ledger.weights)
    pd.testing.assert_series_equal(result.rf, rf.loc[result.net.index])


def test_run_defaults_to_5bp():
    returns = returns_frame(5, 300, n=4)
    result = bt.run(sg.equal_weight(), view_of(returns))
    ledger = bt.simulate(bt.targets(sg.equal_weight(), returns), returns, cost_bps=5)
    pd.testing.assert_series_equal(result.net, ledger.net)


def test_run_needs_a_view():
    with pytest.raises(TypeError, match="View"):
        bt.run(sg.equal_weight(), HALF_YEAR)


def test_run_checks_cost_bps_before_estimating():
    spy = Spy(5)
    with pytest.raises(ValueError, match="cost_bps"):
        bt.run(spy, view_of(HALF_YEAR), cost_bps=-1)
    assert spy.histories == []


def test_run_stops_at_dev_end_and_no_later_row_reaches_the_strategy():
    spy = Spy(5)
    result = bt.run(spy, view_of(LONG))
    assert result.net.index[-1] == pd.Timestamp(DEV_END)  # a Friday
    # The last estimate is 2010-11-30: DEV_END itself has no day after it to trade on.
    assert max(h.index[-1] for h in spy.histories) == pd.Timestamp("2010-11-30")


def test_run_past_dev_end_needs_allow_holdout():
    spy = Spy(5)
    with pytest.raises(HoldoutError):
        bt.run(spy, view_of(LONG), end="2011-03-31")
    with pytest.raises(HoldoutError):
        bt.run(spy, view_of(LONG), end=None)
    assert spy.histories == []


def test_allow_holdout_opens_the_later_rows():
    result = bt.run(Spy(5), view_of(LONG), end=None, allow_holdout=True)
    assert result.net.index[-1] == LONG.index[-1]


def test_an_earlier_end_truncates():
    result = bt.run(Spy(5), view_of(LONG), end="2010-06-30")
    assert result.net.index[-1] == pd.Timestamp("2010-06-30")
    assert result.rf.index[-1] == pd.Timestamp("2010-06-30")


@pytest.mark.parametrize("end", ["", "NaT", pd.NaT, 20101231, pd.Timestamp("2010-06-30", tz="UTC")])
def test_an_end_that_is_not_a_date_raises(end):
    with pytest.raises(ValueError, match="not a date"):
        bt.run(Spy(5), view_of(LONG), end=end)


# --- compare: one common first rebalance ---


def test_compare_starts_every_strategy_on_the_date_the_longest_lookback_allows():
    view = view_of(returns_frame(6, 200, n=4))
    group = [sg.equal_weight(), sg.min_variance("lw_identity", window=60), Spy(30)]
    results = bt.compare(group, view)
    assert list(results) == [s.name for s in group]
    firsts = {result.weights.index[0] for result in results.values()}
    assert firsts == {pd.Timestamp("2001-04-02")}  # 60 rows first by 2001-03-30, traded next day
    alone = bt.run(sg.equal_weight(), view)
    assert alone.weights.index[0] == pd.Timestamp("2001-02-01")  # 1/N alone would start sooner
    # Both hold 1/N from the close of 2001-04-02, so every later day is the same.
    pd.testing.assert_series_equal(results["1/N"].net, alone.net.loc["2001-04-03":])


def test_compare_with_a_start_too_early_for_one_strategy_raises():
    view = view_of(returns_frame(6, 200, n=4))
    group = [sg.equal_weight(), sg.min_variance("sample", window=60)]
    with pytest.raises(ValueError, match="needs 60"):
        bt.compare(group, view, start="2001-01-31")


def test_compare_checks_a_start_for_every_strategy_before_running_any():
    spy = Spy(5, name="short lookback, listed first")
    group = [spy, sg.min_variance("sample", window=60)]
    with pytest.raises(ValueError, match="needs 60"):
        bt.compare(group, view_of(returns_frame(6, 200, n=4)), start="2001-01-31")
    assert spy.histories == []  # nothing estimated before the check


def test_compare_needs_at_least_one_strategy_and_unique_names():
    view = view_of(HALF_YEAR)
    with pytest.raises(ValueError, match="at least one"):
        bt.compare([], view)
    with pytest.raises(ValueError, match="unique"):
        bt.compare([sg.equal_weight(), sg.equal_weight()], view)


def test_compare_passes_end_and_cost_through():
    group = [sg.equal_weight(), whole_window(10)]
    results = bt.compare(group, view_of(LONG), end="2010-06-30", cost_bps=0)
    for result in results.values():
        assert result.net.index[-1] == pd.Timestamp("2010-06-30")
        pd.testing.assert_series_equal(result.net, result.gross, check_names=False)
    with pytest.raises(HoldoutError):
        bt.compare(group, view_of(LONG), end="2011-03-31")


# --- reprice: the same targets at another cost, on the same days ---


def test_reprice_is_run_at_another_cost_without_estimating_again():
    view = view_of(returns_frame(5, 300, n=4))
    s = sg.min_variance("sample", window=60)
    base = bt.run(s, view, cost_bps=5)
    for bps in (0.0, 5.0, 20.0):
        again, ref = bt.reprice(base, view, cost_bps=bps), bt.run(s, view, cost_bps=bps)
        for field in ("gross", "net", "turnover", "cost", "final_weights", "rf"):
            pd.testing.assert_series_equal(getattr(again, field), getattr(ref, field))
        pd.testing.assert_frame_equal(again.weights, ref.weights)
        assert again.name == base.name


def test_reprice_stops_where_the_result_stops():
    view = view_of(LONG)  # runs on past DEV_END
    base = bt.run(sg.equal_weight(), view, end="2010-06-30")
    again = bt.reprice(base, view, cost_bps=10)
    assert again.net.index.equals(base.net.index)


def test_reprice_into_the_holdout_needs_the_flag():
    view = view_of(LONG)
    base = bt.run(sg.equal_weight(), view, end=None, allow_holdout=True)
    with pytest.raises(HoldoutError):
        bt.reprice(base, view, cost_bps=10)
    again = bt.reprice(base, view, cost_bps=10, allow_holdout=True)
    assert again.net.index.equals(base.net.index)


def test_reprice_needs_every_day_the_result_was_run_on():
    base = bt.run(sg.equal_weight(), view_of(LONG), end="2010-06-30")
    gappy = view_of(LONG.drop(pd.Timestamp("2010-03-10")))  # a mid-month day
    with pytest.raises(ValueError, match="days"):
        bt.reprice(base, gappy, cost_bps=10)


def test_reprice_checks_cost_bps_and_its_inputs():
    view = view_of(HALF_YEAR)
    base = bt.run(sg.equal_weight(), view)
    with pytest.raises(ValueError, match="cost_bps"):
        bt.reprice(base, view, cost_bps=-1)
    with pytest.raises(TypeError, match="Result"):
        bt.reprice(base.net, view, cost_bps=5)
    with pytest.raises(TypeError, match="View"):
        bt.reprice(base, HALF_YEAR, cost_bps=5)


# --- Result: summary on the step-2 metrics, plus the trades ---

HAND_INDEX = pd.bdate_range("2001-01-01", periods=126, name="date")  # half a year of days


def hand_result(turnover=(0.1, 0.3), cost=(1e-4, 3e-4)) -> bt.Result:
    rng = np.random.default_rng(0)
    gross = pd.Series(rng.normal(0.0004, 0.01, 126), index=HAND_INDEX, name="gross")
    trades = HAND_INDEX[[40, 80]][: len(turnover)]
    net = gross.copy().rename("net")
    net.loc[trades] -= np.asarray(cost) * (1 + gross.loc[trades])
    return bt.Result(
        gross=gross,
        net=net,
        weights=pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3}, index=HAND_INDEX[[0, 40, 80]]),
        turnover=pd.Series(turnover, index=trades, dtype=float, name="turnover"),
        cost=pd.Series(cost, index=trades, dtype=float, name="cost"),
        final_weights=pd.Series([0.5, 0.5], index=["A", "B"]),
        name="hand",
        rf=pd.Series(1e-4, index=HAND_INDEX, name="RF"),
    )


def test_summary_is_the_step_2_metrics_of_gross_and_net_plus_the_trades():
    """126 days are half a year (252 a year): two trades with turnover 0.1 and 0.3 and costs
    1e-4 and 3e-4 give 0.2 per rebalance, 0.8 a year, and 8e-4 a year of cost. Gross pays no
    cost by definition: its annual cost is 0, its turnover the same."""
    r = hand_result()
    table = r.summary()
    frame = pd.DataFrame({"gross": r.gross, "net": r.net})
    expected = metrics.summary(frame, r.rf, periods_per_year=252)
    pd.testing.assert_frame_equal(table.loc[expected.index], expected)
    assert list(table.columns) == ["gross", "net"]
    assert table.loc["mean_turnover"].tolist() == pytest.approx([0.2, 0.2], rel=1e-15)
    assert table.loc["annual_turnover"].tolist() == pytest.approx([0.8, 0.8], rel=1e-15)
    assert table.loc["annual_cost"].tolist() == pytest.approx([0.0, 8e-4], rel=1e-15)


def test_summary_cuts_a_longer_benchmark_to_the_result_days():
    r = hand_result()
    rng = np.random.default_rng(1)
    longer_days = pd.bdate_range("2000-12-01", periods=200, name="date")
    longer = pd.Series(rng.normal(0.0003, 0.01, 200), index=longer_days, name="Market")
    table = r.summary(benchmark=longer)
    frame = pd.DataFrame({"gross": r.gross, "net": r.net})
    bench = longer.loc[HAND_INDEX]
    expected = metrics.summary(frame, r.rf, periods_per_year=252, benchmark=bench)
    pd.testing.assert_frame_equal(table.loc[expected.index], expected)
    assert {"tracking_error", "information_ratio"} <= set(table.index)


def test_a_benchmark_missing_a_day_raises():
    r = hand_result()
    with pytest.raises(ValueError, match="no return on 2001-01-08"):
        r.summary(benchmark=pd.Series(0.0, index=HAND_INDEX.delete(5)))


def test_a_benchmark_with_a_missing_value_raises():
    with pytest.raises(ValueError, match="benchmark is missing"):
        hand_result().summary(benchmark=pd.Series(np.nan, index=HAND_INDEX))


def test_a_benchmark_must_be_a_series():
    frame = pd.DataFrame({"Market": 0.0}, index=HAND_INDEX)
    with pytest.raises(TypeError, match="Series"):
        hand_result().summary(benchmark=frame)


def test_without_a_second_trade_turnover_per_rebalance_is_nan_and_the_annual_rows_zero():
    r = hand_result(turnover=(), cost=())
    table = r.summary()
    assert table.loc["mean_turnover"].isna().all()
    assert (table.loc[["annual_turnover", "annual_cost"]] == 0).all().all()


def test_result_rf_must_align_with_its_returns():
    r = hand_result()
    with pytest.raises(ValueError, match="rf"):
        bt.Result(**{**vars(r), "rf": r.rf.iloc[1:]})


def test_result_gross_must_align_with_net():
    r = hand_result()
    with pytest.raises(ValueError, match="gross"):
        bt.Result(**{**vars(r), "gross": r.gross.iloc[1:]})


def test_result_turnover_and_cost_must_share_trade_dates():
    r = hand_result()
    with pytest.raises(ValueError, match="cost"):
        bt.Result(**{**vars(r), "cost": r.cost.iloc[:1]})
