"""The backtest engine's calendar and accounting (blueprint Step 4, plans/step-4-plan.md).

Calendar: estimate at the last trading day t of each month, trade at the close of the next
trading day. Accounting: between trades the weights drift with returns; a trade day's return is
earned by the drifted weights, then the trade costs cost_bps / 1e4 x sum|target - drifted| of
the NAV. Hand examples are worked in exact fractions; a plain day-by-day loop, written here
independently of the engine, is the reference on random cases.
"""

from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from portfolio_lab.backtest import engine as bt

ASSETS = ["A", "B"]


def days(*dates: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates, name="date")


def assert_schedule(got: pd.Series, pairs: dict[str, str]) -> None:
    """`got` maps each estimation date to its trade date as `pairs` does."""
    assert list(got.index) == [pd.Timestamp(d) for d in pairs]
    assert list(got) == [pd.Timestamp(d) for d in pairs.values()]


def targets(rows: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows.values()), index=days(*rows), columns=ASSETS)


def approx(values, fractions) -> None:
    expected = [float(f) for f in fractions]
    np.testing.assert_allclose(np.asarray(values, dtype=float), expected, rtol=0, atol=1e-15)


def random_case(seed: int, n: int, t: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random daily returns and long-only targets (some weights 0) on random trade days.

    Returns up to and including the first trade day are NaN: the portfolio is bought at that
    day's close, so the engine must never read them.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2001-01-01", periods=t, name="date")
    names = [f"x{i}" for i in range(n)]
    returns = pd.DataFrame(rng.uniform(-0.3, 0.3, size=(t, n)), index=index, columns=names)
    first = int(rng.integers(0, t - 1))
    later = np.arange(first + 1, t)
    k = int(rng.integers(0, min(len(later), 8) + 1))
    pos = np.r_[first, np.sort(rng.choice(later, size=k, replace=False))]
    returns.iloc[: first + 1] = np.nan
    w = rng.dirichlet(np.ones(n), size=len(pos))
    w[rng.random(w.shape) < 0.3] = 0
    w[np.arange(len(pos)), rng.integers(0, n, size=len(pos))] += 0.1
    w /= w.sum(axis=1, keepdims=True)
    return returns, pd.DataFrame(w, index=index[pos], columns=names)


def daily_loop(tg: pd.DataFrame, returns: pd.DataFrame, cost_bps: float):
    """The reference: one day at a time, in the blueprint's words."""
    rate = cost_bps / 1e4
    trades = {day: row.to_numpy(dtype=float) for day, row in tg.iterrows()}
    first = returns.index.get_loc(tg.index[0])
    w = trades[tg.index[0]].copy()
    gross, net, turnover, cost = [], [], [], []
    for day in returns.index[first + 1 :]:
        r = returns.loc[day].to_numpy(dtype=float)
        g = float(w @ r)  # earned by the weights held at the previous close
        w = w * (1 + r) / (1 + g)  # drift
        n = g
        if day in trades:  # trade at this close, after this day's return
            delta = float(np.abs(trades[day] - w).sum())
            turnover.append(delta / 2)
            cost.append(rate * delta)
            n = (1 + g) * (1 - rate * delta) - 1  # cost deducted from NAV after the return
            w = trades[day].copy()
        gross.append(g)
        net.append(n)
    return np.array(gross), np.array(net), np.array(turnover), np.array(cost), w


# --- rebalance_dates: estimate at the last trading day of a month, trade the next day ---


def test_the_last_day_of_each_month_estimates_and_the_next_day_trades():
    d = pd.bdate_range("2001-01-29", "2001-03-02", name="date")
    got = bt.rebalance_dates(d)
    assert_schedule(got, {"2001-01-31": "2001-02-01", "2001-02-28": "2001-03-01"})
    assert got.index.name == "estimated"
    assert got.name == "traded"


def test_a_holiday_on_the_last_weekday_moves_the_estimate_earlier_and_the_trade_later():
    d = days("2001-08-29", "2001-08-30", "2001-09-04", "2001-09-05")  # no Aug 31, no Sep 3
    assert_schedule(bt.rebalance_dates(d), {"2001-08-30": "2001-09-04"})


def test_a_month_with_one_trading_day_both_trades_and_estimates_on_it():
    d = days("2001-01-30", "2001-01-31", "2001-02-15", "2001-03-01", "2001-03-02")
    assert_schedule(bt.rebalance_dates(d), {"2001-01-31": "2001-02-15", "2001-02-15": "2001-03-01"})


def test_the_last_day_never_estimates_since_nothing_trades_after_it():
    d = pd.bdate_range("2001-01-29", "2001-02-28", name="date")  # ends on a month end
    assert_schedule(bt.rebalance_dates(d), {"2001-01-31": "2001-02-01"})


@pytest.mark.parametrize(
    ("start", "first"),
    [
        (None, "2001-01-31"),
        ("2001-01-01", "2001-01-31"),
        ("2001-01-31", "2001-01-31"),  # a month end itself is on or after
        ("2001-02-01", "2001-02-28"),
        (pd.Timestamp("2001-02-28"), "2001-02-28"),
    ],
)
def test_start_keeps_the_estimation_dates_on_or_after_it(start, first):
    d = pd.bdate_range("2001-01-29", "2001-03-02", name="date")
    assert bt.rebalance_dates(d, start=start).index[0] == pd.Timestamp(first)


def test_a_start_after_the_last_estimation_date_leaves_nothing():
    d = pd.bdate_range("2001-01-29", "2001-03-02", name="date")
    assert bt.rebalance_dates(d, start="2001-03-01").empty


@pytest.mark.parametrize("n", [0, 1])
def test_too_few_days_have_no_rebalance(n):
    assert bt.rebalance_dates(pd.bdate_range("2001-01-31", periods=n)).empty


@pytest.mark.parametrize("start", ["", "NaT", pd.NaT, float("nan")])
def test_a_start_that_is_not_a_date_raises(start):
    d = pd.bdate_range("2001-01-29", "2001-03-02")
    with pytest.raises(ValueError, match="start"):
        bt.rebalance_dates(d, start=start)


@pytest.mark.parametrize(
    "bad",
    [
        days("2001-02-01", "2001-01-31"),
        days("2001-01-31", "2001-01-31", "2001-02-01"),
        pd.Index([1, 2, 3]),
        pd.date_range("2001-01-29", periods=5, tz="UTC"),
    ],
    ids=["unsorted", "repeated", "not dates", "tz-aware"],
)
def test_days_must_be_sorted_unique_naive_dates(bad):
    with pytest.raises(ValueError, match="days"):
        bt.rebalance_dates(bad)


@given(seed=st.integers(0, 2**32 - 1))
def test_every_month_with_a_later_day_estimates_once_on_its_last_day(seed):
    rng = np.random.default_rng(seed)
    every = pd.bdate_range("2000-12-01", "2002-02-28")
    d = every[rng.random(len(every)) < rng.uniform(0.02, 1)]
    got = bt.rebalance_dates(d)
    months = d.to_period("M")
    for t, traded in got.items():
        pos = d.get_loc(t)
        assert traded == d[pos + 1]  # the next day trades
        assert months[pos + 1] != months[pos]  # t is the last day of its month
    with_a_later_day = sorted(set(months[:-1]) - {months[-1]}) if len(d) else []
    assert list(got.index.to_period("M")) == with_a_later_day


# --- simulate: drift, then trade at the close after the day's return, then costs ---

HAND_DAYS = days("2001-01-31", "2001-02-01", "2001-02-02", "2001-02-05")
# The first trade day's own return is never used (the portfolio is bought at its close): NaN.
HAND_RETURNS = pd.DataFrame(
    [[np.nan, np.nan], [0.10, -0.10], [0.00, 0.10], [-0.10, 0.00]],
    index=HAND_DAYS,
    columns=ASSETS,
)


def test_hand_example_drift_then_trade_at_the_close():
    """Bought at (1/2, 1/2) at the close of day 0, 5 bp per unit traded.

    Day 1: (+10%, -10%) -> R = 0, weights (0.55, 0.45).
    Day 2: (0, +10%) -> R = 0.045 = 9/200, weights (0.55, 0.495) / 1.045 = (10/19, 9/19).
    Day 3: (-10%, 0) -> R = -(10/19)(1/10) = -1/19, weights (9/19, 9/19) / (18/19) = (1/2, 1/2).
    Trade at day 3's close to (1, 0): sum |delta| = 1, turnover 1/2, cost 5e-4 of NAV, so
    net = (18/19)(1 - 5e-4) - 1.
    """
    tg = targets({"2001-01-31": [0.5, 0.5], "2001-02-05": [1.0, 0.0]})
    got = bt.simulate(tg, HAND_RETURNS, cost_bps=5)
    assert got.gross.index.equals(HAND_DAYS[1:])
    assert got.net.index.equals(HAND_DAYS[1:])
    approx(got.gross, [F(0), F(9, 200), F(-1, 19)])
    approx(got.net, [F(0), F(9, 200), F(18, 19) * (1 - F(5, 10_000)) - 1])
    assert got.turnover.index.equals(HAND_DAYS[[3]])
    assert got.cost.index.equals(HAND_DAYS[[3]])
    approx(got.turnover, [F(1, 2)])
    approx(got.cost, [F(5, 10_000)])
    assert list(got.final_weights.index) == ASSETS
    approx(got.final_weights, [F(1), F(0)])  # a trade on the last day leaves the target
    pd.testing.assert_frame_equal(got.weights, tg)


def test_turnover_is_measured_from_the_drifted_weights():
    """Back to (1/2, 1/2) at day 2's close, from (10/19, 9/19): sum |delta| = 1/19, turnover
    1/38, cost 5e-4 / 19. Day 3 drifts (1/2, 1/2) by (-10%, 0) to (9/19, 10/19), R = -1/20."""
    tg = targets({"2001-01-31": [0.5, 0.5], "2001-02-02": [0.5, 0.5]})
    got = bt.simulate(tg, HAND_RETURNS, cost_bps=5)
    approx(got.turnover, [F(1, 38)])
    approx(got.cost, [F(5, 10_000) / 19])
    approx(got.gross, [F(0), F(9, 200), F(-1, 20)])
    approx(got.net, [F(0), (1 + F(9, 200)) * (1 - F(5, 10_000) / 19) - 1, F(-1, 20)])
    approx(got.final_weights, [F(9, 19), F(10, 19)])


def test_the_trade_day_return_is_earned_by_the_old_weights():
    """Estimated at t, traded at the close of t + 1. A's +50% on the trade day goes to the old
    portfolio (all B, flat); only A's +10% the day after reaches the new one (all A). Trading
    at t would have earned the +50%. Selling all of B to buy all of A: turnover 1, cost 1e-3."""
    r = pd.DataFrame(
        [[np.nan, np.nan], [0.0, 0.0], [0.5, 0.0], [0.1, 0.0]], index=HAND_DAYS, columns=ASSETS
    )
    tg = targets({"2001-01-31": [0.0, 1.0], "2001-02-02": [1.0, 0.0]})
    got = bt.simulate(tg, r, cost_bps=5)
    approx(got.gross, [F(0), F(0), F(1, 10)])
    approx(got.turnover, [F(1)])
    approx(got.net, [F(0), -F(10, 10_000), F(1, 10)])


def test_zero_cost_net_is_exactly_gross_and_costs_never_change_gross_or_turnover():
    returns, tg = random_case(7, n=4, t=80)
    free = bt.simulate(tg, returns, cost_bps=0)
    paid = bt.simulate(tg, returns, cost_bps=5)
    assert len(free.turnover) > 0
    pd.testing.assert_series_equal(free.net, free.gross, check_exact=True, check_names=False)
    assert (free.cost == 0).all()
    pd.testing.assert_series_equal(paid.gross, free.gross, check_exact=True)
    pd.testing.assert_series_equal(paid.turnover, free.turnover, check_exact=True)


def test_the_cost_is_cost_bps_per_unit_traded():
    returns, tg = random_case(11, n=5, t=120)
    five = bt.simulate(tg, returns, cost_bps=5)
    twenty = bt.simulate(tg, returns, cost_bps=20)
    np.testing.assert_allclose(five.cost, 5e-4 * 2 * five.turnover, rtol=1e-15)
    np.testing.assert_allclose(twenty.cost, 4 * five.cost, rtol=1e-15)


@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(1, 5),
    t=st.integers(2, 60),
    cost_bps=st.sampled_from([0.0, 5.0, 20.0]),
)
def test_matches_a_plain_daily_loop(seed, n, t, cost_bps):
    returns, tg = random_case(seed, n, t)
    got = bt.simulate(tg, returns, cost_bps=cost_bps)
    gross, net, turnover, cost, final = daily_loop(tg, returns, cost_bps)
    first = returns.index.get_loc(tg.index[0])
    assert got.gross.index.equals(returns.index[first + 1 :])
    assert got.turnover.index.equals(tg.index[1:])
    np.testing.assert_allclose(got.gross, gross, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.net, net, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.turnover, turnover, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.cost, cost, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.final_weights, final, rtol=0, atol=1e-12)
    # Costs only ever lower the net return, and only on trade days.
    traded = got.net.index.isin(tg.index[1:])
    assert (got.net <= got.gross).all()
    assert (got.net[~traded] == got.gross[~traded]).all()


@given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 6))
def test_one_target_is_buy_and_hold(seed, n):
    """Wealth from a single trade is sum_i w_i prod(1 + r_i), with no turnover or cost."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2001-01-01", periods=40, name="date")
    returns = pd.DataFrame(rng.uniform(-0.3, 0.3, size=(40, n)), index=index)
    w = rng.dirichlet(np.ones(n))
    tg = pd.DataFrame([w], index=index[[0]], columns=returns.columns)
    got = bt.simulate(tg, returns, cost_bps=5)
    held = float(w @ (1 + returns.iloc[1:]).prod().to_numpy())
    assert float((1 + got.gross).prod()) == pytest.approx(held, rel=1e-12)
    assert got.turnover.empty
    assert got.cost.empty
    assert isinstance(got.turnover.index, pd.DatetimeIndex)
    pd.testing.assert_series_equal(got.net, got.gross, check_exact=True, check_names=False)


def test_an_asset_can_lose_everything():
    r = pd.DataFrame(
        [[np.nan, np.nan], [-1.0, 0.0], [0.0, 0.1], [0.0, 0.0]], index=HAND_DAYS, columns=ASSETS
    )
    got = bt.simulate(targets({"2001-01-31": [0.5, 0.5]}), r, cost_bps=5)
    approx(got.gross, [F(-1, 2), F(1, 10), F(0)])
    approx(got.final_weights, [F(0), F(1)])


def test_a_portfolio_that_loses_everything_raises():
    r = pd.DataFrame(
        [[np.nan, np.nan], [-1.0, 0.0], [0.0, 0.1], [0.0, 0.0]], index=HAND_DAYS, columns=ASSETS
    )
    with pytest.raises(ValueError, match="lost everything on 2001-02-01"):
        bt.simulate(targets({"2001-01-31": [1.0, 0.0]}), r, cost_bps=5)


# --- simulate: inputs ---

GOOD = targets({"2001-01-31": [0.5, 0.5], "2001-02-02": [1.0, 0.0]})


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (GOOD.iloc[:0], "at least one"),
        (targets({"2001-02-03": [0.5, 0.5]}), "not a day of the returns"),  # a Saturday
        (GOOD.iloc[::-1], "sorted"),
        (pd.concat([GOOD.iloc[:1], GOOD.iloc[:1]]), "unique"),
        (GOOD[["B", "A"]], "same assets"),
        (GOOD[["A"]], "same assets"),
        (targets({"2001-01-31": [0.5, 0.4]}), "sum to 1"),
        (targets({"2001-01-31": [1.5, -0.5]}), ">= 0"),
        (targets({"2001-01-31": [np.nan, 1.0]}), "finite"),
        (targets({"2001-02-05": [0.5, 0.5]}), "no day after the first trade"),
    ],
    ids=[
        "empty",
        "not a return day",
        "unsorted",
        "repeated",
        "assets reordered",
        "asset missing",
        "budget",
        "short",
        "NaN",
        "last day first",
    ],
)
def test_targets_must_be_long_only_portfolios_on_days_of_the_returns(bad, match):
    with pytest.raises(ValueError, match=match):
        bt.simulate(bad, HAND_RETURNS, cost_bps=5)


def test_a_bad_target_names_its_trade_date():
    bad = targets({"2001-01-31": [0.5, 0.5], "2001-02-02": [0.5, 0.4]})
    with pytest.raises(ValueError, match="2001-02-02"):
        bt.simulate(bad, HAND_RETURNS, cost_bps=5)


def test_inputs_must_be_dataframes():
    with pytest.raises(TypeError, match="targets"):
        bt.simulate(GOOD.iloc[0], HAND_RETURNS, cost_bps=5)
    with pytest.raises(TypeError, match="returns"):
        bt.simulate(GOOD, HAND_RETURNS["A"], cost_bps=5)


@pytest.mark.parametrize(
    ("cell", "value", "match"),
    [
        ((2, "B"), np.nan, "missing for B"),
        ((3, "A"), np.nan, "missing for A"),  # a later trade day
        ((1, "A"), -1.5, "-100%"),
        ((2, "B"), np.inf, "finite"),
    ],
)
def test_returns_on_simulated_days_must_be_complete_finite_and_at_least_minus_100pct(
    cell, value, match
):
    r = HAND_RETURNS.copy()
    r.loc[HAND_DAYS[cell[0]], cell[1]] = value
    with pytest.raises(ValueError, match=match):
        bt.simulate(GOOD, r, cost_bps=5)


def test_returns_must_be_in_date_order():
    with pytest.raises(ValueError, match="sorted"):
        bt.simulate(GOOD, HAND_RETURNS.iloc[[0, 2, 1, 3]], cost_bps=5)


@pytest.mark.parametrize("cost_bps", [-1.0, np.nan, np.inf])
def test_cost_bps_must_be_finite_and_not_negative(cost_bps):
    with pytest.raises(ValueError, match="cost_bps"):
        bt.simulate(GOOD, HAND_RETURNS, cost_bps=cost_bps)
