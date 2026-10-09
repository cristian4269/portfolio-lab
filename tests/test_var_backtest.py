"""VaR backtests: exceptions, Kupiec, Christoffersen, conditional coverage, Basel zones, report."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from portfolio_lab.risk import backtest as bt
from portfolio_lab.risk import var as v


def days(n):
    return pd.bdate_range("2000-01-03", periods=n)


def hits(flags):
    return pd.Series([bool(f) for f in flags], index=days(len(flags)), dtype=bool)


def spread(x, t):
    """t days with x exceptions spread out (never two in a row while x <= t / 2)."""
    flags = np.zeros(t, dtype=bool)
    flags[np.linspace(0, t - 1, x, dtype=int)] = True
    return hits(flags)


bools = st.lists(st.booleans(), min_size=2, max_size=300)


# --- exceptions -------------------------------------------------------------------------------


def test_exceptions_by_hand_and_strict():
    r = pd.Series([-0.03, -0.01, 0.02, -0.02, 0.5], index=days(5), name="P")
    var = pd.Series([0.02, 0.01, 0.02, 0.02], index=days(5)[:4])
    exc = bt.exceptions(r, var)
    # -1% against a 1% VaR is not an exception: the loss must be strictly larger.
    assert exc.tolist() == [True, False, False, False]
    assert exc.index.equals(var.index)
    assert exc.dtype == bool


def test_exceptions_on_the_forecasts_days_only():
    r = pd.Series(np.linspace(-0.05, 0.05, 30), index=days(30))
    fc = v.forecast(r, method="historical", level=0.95, window=10)
    exc = bt.exceptions(r, fc["var"])
    assert exc.index.equals(r.index[10:])
    assert exc.tolist() == (r.iloc[10:] < -fc["var"]).tolist()


def test_exceptions_reject_forecast_days_without_a_return():
    r = pd.Series([0.01, 0.02], index=days(2))
    var = pd.Series([0.02, 0.02], index=days(3)[1:])
    with pytest.raises(ValueError, match="no return"):
        bt.exceptions(r, var)


def test_exceptions_reject_a_missing_forecast():
    r = pd.Series([0.01, 0.02], index=days(2))
    with pytest.raises(ValueError, match="finite"):
        bt.exceptions(r, pd.Series([0.02, float("nan")], index=days(2)))
    with pytest.raises(TypeError, match="Series"):
        bt.exceptions(r, [0.02, 0.02])


def test_exceptions_must_be_numeric_flags():
    with pytest.raises(ValueError, match="True/False"):
        bt.kupiec(pd.Series(["yes", "no"], index=days(2)), 0.99)


# --- Kupiec POF -------------------------------------------------------------------------------


def test_kupiec_blueprint_case():
    # T = 250, x = 8, p = 0.01, computed here from the definition with math.log.
    t, x, p = 250, 8, 0.01
    expected = -2 * ((t - x) * math.log(1 - p) + x * math.log(p)) + 2 * (
        (t - x) * math.log(1 - x / t) + x * math.log(x / t)
    )
    res = bt.kupiec(spread(8, 250), 0.99)
    assert res.stat == pytest.approx(expected, rel=1e-12)
    assert res.stat == pytest.approx(7.7335507244945205, rel=1e-12)
    assert res.pvalue == pytest.approx(0.0054204051941277994, rel=1e-9)
    assert (res.df, res.exceptions, res.days) == (1, 8, 250)


def test_kupiec_with_no_exceptions():
    # 0 of 250 at 99%: LR = -2 * 250 * ln(0.99) = 5.025, rejected at 5% (too few exceptions).
    res = bt.kupiec(spread(0, 250), 0.99)
    assert res.stat == pytest.approx(-2 * 250 * math.log(0.99), rel=1e-12)
    assert res.pvalue < 0.05


def test_kupiec_with_only_exceptions():
    res = bt.kupiec(hits([True] * 20), 0.95)
    assert res.stat == pytest.approx(-2 * 20 * math.log(0.05), rel=1e-12)


def test_kupiec_is_zero_at_the_expected_rate():
    # 5 of 100 at 95%; 1 - 0.95 is not exactly 0.05 in floats, hence the tolerance.
    res = bt.kupiec(spread(5, 100), 0.95)
    assert res.stat == pytest.approx(0.0, abs=1e-12)
    assert res.pvalue == pytest.approx(1.0, abs=1e-6)


@given(bools, st.sampled_from(v.LEVELS))
def test_kupiec_pvalue_is_chi2_with_one_degree(flags, level):
    res = bt.kupiec(hits(flags), level)
    assert res.stat >= 0
    assert res.pvalue == pytest.approx(stats.chi2.sf(res.stat, 1), rel=1e-12, abs=1e-300)


def test_kupiec_size_on_simulated_correct_forecasts():
    # iid normal returns and the true normal VaR: exceptions are Bernoulli(p). At 95% over
    # 500 days the exact size of a 5% Kupiec test is 5.39% (discreteness, from the binomial).
    rng = np.random.default_rng(20261009)
    reps, t = 4000, 500
    var = -0.01 * stats.norm.ppf(0.05)
    r = rng.normal(0.0, 0.01, (reps, t))
    index = days(t)  # built once: 4000 date ranges would dominate the test's time
    rejected = sum(bt.kupiec(pd.Series(row < -var, index=index), 0.95).pvalue < 0.05 for row in r)
    rate = rejected / reps
    size = 0.05393307391942694
    assert abs(rate - size) < 4 * math.sqrt(size * (1 - size) / reps)


# --- Christoffersen independence --------------------------------------------------------------


def test_transitions_by_hand():
    #        0  0  1  1  0  0  0  1  1  1  0  0
    flags = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    # pairs: 00 01 11 10 00 00 01 11 11 10 00 -> n00 4, n01 2, n10 2, n11 3
    assert bt.transitions(hits(flags)) == (4, 2, 2, 3)


def test_christoffersen_clustered_by_hand():
    flags = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    n00, n01, n10, n11 = 4, 2, 2, 3
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 11
    log_l0 = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    log_l1 = (
        n00 * math.log(1 - pi01)
        + n01 * math.log(pi01)
        + n10 * math.log(1 - pi11)
        + n11 * math.log(pi11)
    )
    res = bt.christoffersen(hits(flags))
    assert res.stat == pytest.approx(-2 * (log_l0 - log_l1), rel=1e-12)
    assert res.pvalue == pytest.approx(stats.chi2.sf(res.stat, 1), rel=1e-12)
    assert (res.df, res.exceptions, res.days) == (1, 5, 12)


def test_christoffersen_without_consecutive_exceptions():
    # n11 = 0: pi11 = 0 and its 0 * ln(0) terms vanish.
    flags = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    assert bt.transitions(hits(flags)) == (5, 2, 2, 0)
    log_l0 = 7 * math.log(7 / 9) + 2 * math.log(2 / 9)
    log_l1 = 5 * math.log(5 / 7) + 2 * math.log(2 / 7)
    res = bt.christoffersen(hits(flags))
    assert res.stat == pytest.approx(-2 * (log_l0 - log_l1), rel=1e-12)
    assert res.stat == pytest.approx(1.1589373428441796, rel=1e-12)


def test_christoffersen_with_the_only_exception_on_the_last_day():
    # No day follows an exception: n10 + n11 = 0, so pi11 is undefined and drops out.
    res = bt.christoffersen(hits([0, 0, 0, 1]))
    assert res.stat == 0.0
    assert res.pvalue == 1.0


@pytest.mark.parametrize("flags", [[0] * 30, [1] * 30])
def test_christoffersen_with_no_variation(flags):
    res = bt.christoffersen(hits(flags))
    assert res.stat == 0.0


def test_christoffersen_needs_two_days():
    with pytest.raises(ValueError, match="at least 2"):
        bt.christoffersen(hits([1]))


@given(bools)
def test_christoffersen_is_non_negative_and_chi2(flags):
    res = bt.christoffersen(hits(flags))
    assert res.stat >= 0
    assert res.pvalue == pytest.approx(stats.chi2.sf(res.stat, 1), rel=1e-12, abs=1e-300)


# --- conditional coverage ---------------------------------------------------------------------


@given(bools, st.sampled_from(v.LEVELS))
def test_conditional_coverage_is_kupiec_on_the_transitions_plus_independence(flags, level):
    # LR_cc = LR_uc + LR_ind exactly when LR_uc is computed on the T - 1 days that have a
    # previous day (Christoffersen 1998).
    exc = hits(flags)
    cc = bt.conditional_coverage(exc, level)
    uc = bt.kupiec(exc.iloc[1:], level).stat
    ind = bt.christoffersen(exc).stat
    assert cc.stat == pytest.approx(uc + ind, rel=1e-10, abs=1e-10)
    assert cc.df == 2
    assert cc.pvalue == pytest.approx(stats.chi2.sf(cc.stat, 2), rel=1e-12, abs=1e-300)
    assert (cc.exceptions, cc.days) == (sum(flags), len(flags))


def test_conditional_coverage_by_hand():
    flags = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  # transitions 5, 2, 2, 0
    p = 0.05
    log_lp = 7 * math.log(1 - p) + 2 * math.log(p)
    log_l1 = 5 * math.log(5 / 7) + 2 * math.log(2 / 7)
    assert bt.conditional_coverage(hits(flags), 0.95).stat == pytest.approx(
        -2 * (log_lp - log_l1), rel=1e-12
    )


# --- Basel traffic light ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "zone"),
    [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red"), (40, "red")],
)
def test_basel_zone_boundaries(x, zone):
    assert bt.basel_zone(spread(x, 250), 0.99) == zone


def test_basel_zone_only_for_99_over_250_days():
    with pytest.raises(ValueError, match="0.99"):
        bt.basel_zone(spread(3, 250), 0.95)
    with pytest.raises(ValueError, match="250"):
        bt.basel_zone(spread(3, 249), 0.99)
    with pytest.raises(ValueError, match="250"):
        bt.basel_zone(spread(3, 500), 0.99)


def test_basel_zones_over_trailing_years():
    flags = np.zeros(300, dtype=bool)
    flags[[10, 20, 30, 40, 50, 270]] = True  # 5 in the first 250 days
    exc = hits(flags)
    zones = bt.basel_zones(exc, 0.99)
    assert zones.index.equals(exc.index[249:])
    counts = exc.astype(int).rolling(250).sum().iloc[249:]
    expected = ["green" if c <= 4 else "yellow" if c <= 9 else "red" for c in counts]
    assert zones.tolist() == expected
    assert zones.iloc[0] == "yellow"  # days 0-249 hold 5
    assert zones.iloc[11] == "green"  # day 10 has left the window, day 270 is not in yet
    assert zones.iloc[21] == "green"  # days 21-270 hold 30, 40, 50 and 270


def test_basel_zones_need_a_year_and_99():
    with pytest.raises(ValueError, match="250"):
        bt.basel_zones(spread(1, 249), 0.99)
    with pytest.raises(ValueError, match="0.99"):
        bt.basel_zones(spread(1, 300), 0.95)


# --- inputs -----------------------------------------------------------------------------------


def test_exception_series_checks():
    with pytest.raises(TypeError, match="Series"):
        bt.kupiec([True, False], 0.99)
    with pytest.raises(ValueError, match="at least 1"):
        bt.kupiec(hits([]), 0.99)
    with pytest.raises(ValueError, match="True/False"):
        bt.kupiec(pd.Series([0.5, 1.0], index=days(2)), 0.99)
    with pytest.raises(ValueError, match="True/False"):
        bt.kupiec(pd.Series([1.0, float("nan")], index=days(2)), 0.99)
    with pytest.raises(ValueError, match="sorted"):
        bt.christoffersen(hits([0, 1, 0]).iloc[::-1])
    with pytest.raises(ValueError, match="level"):
        bt.kupiec(hits([0, 1]), 0.4)


def test_zero_one_integers_count_as_exceptions():
    ints = pd.Series([0, 1, 0, 1, 0, 0, 0, 0, 0, 0], index=days(10))
    assert bt.christoffersen(ints) == bt.christoffersen(ints.astype(bool))


# --- report -----------------------------------------------------------------------------------


def test_report_collects_every_test_for_every_method_and_level():
    r = pd.Series(np.random.default_rng(8).standard_t(4, 700) * 0.01, index=days(700))
    table = bt.report(r, window=100)
    assert list(table.index.names) == ["level", "method"]
    assert list(table.index) == [(lv, m) for lv in v.LEVELS for m in v.METHODS]
    assert list(table.columns) == [
        "days",
        "exceptions",
        "expected",
        "rate",
        "kupiec_stat",
        "kupiec_p",
        "independence_stat",
        "independence_p",
        "cc_stat",
        "cc_p",
        "basel",
    ]
    for level in v.LEVELS:
        for method in v.METHODS:
            fc = v.forecast(r, method=method, level=level, window=100)
            exc = bt.exceptions(r, fc["var"])
            row = table.loc[(level, method)]
            uc, ind = bt.kupiec(exc, level), bt.christoffersen(exc)
            cc = bt.conditional_coverage(exc, level)
            assert row["days"] == 600
            assert row["exceptions"] == exc.sum()
            assert row["expected"] == pytest.approx(600 * (1 - level))
            assert row["rate"] == pytest.approx(exc.mean())
            assert (row["kupiec_stat"], row["kupiec_p"]) == (uc.stat, uc.pvalue)
            assert (row["independence_stat"], row["independence_p"]) == (ind.stat, ind.pvalue)
            assert (row["cc_stat"], row["cc_p"]) == (cc.stat, cc.pvalue)
            if level == 0.99:
                assert row["basel"] == bt.basel_zone(exc.iloc[-250:], 0.99)
            else:
                assert row["basel"] is None


def test_report_has_no_basel_zone_with_less_than_a_year():
    r = pd.Series(np.random.default_rng(8).normal(0, 0.01, 300), index=days(300))
    table = bt.report(r, window=100, methods=("historical",), levels=(0.99,))
    assert table.loc[(0.99, "historical"), "basel"] is None
    assert table.loc[(0.99, "historical"), "days"] == 200
