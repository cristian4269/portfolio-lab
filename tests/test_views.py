"""Currency views: calendar alignment, FX and rate conversion, CAD unhedged/hedged, excess."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from portfolio_lab.data import views

# Thu 31 Dec 2009 .. Wed 6 Jan 2010; Fri 1 Jan is a US holiday (no trading day).
DAYS = pd.DatetimeIndex(["2009-12-30", "2009-12-31", "2010-01-04", "2010-01-05", "2010-01-06"])

small_returns = st.floats(-0.5, 0.5, allow_nan=False)


def series(values, index=DAYS, name=None):
    return pd.Series(values, index=pd.DatetimeIndex(index), name=name, dtype=float)


def _days(n):
    return pd.bdate_range("2000-01-03", periods=n)


# --- align_levels ---------------------------------------------------------------------------


def test_align_levels_carries_the_last_observation():
    levels = series([1.0, 2.0], ["2009-12-30", "2010-01-01"])  # 1 Jan: observed, not traded
    out = views.align_levels(levels, DAYS)
    assert out.index.equals(DAYS)
    assert out.tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_align_levels_skips_missing_values():
    levels = series([1.0, np.nan, np.nan, 4.0, 5.0])
    assert views.align_levels(levels, DAYS).tolist() == [1.0, 1.0, 1.0, 4.0, 5.0]


def test_align_levels_is_nan_before_the_first_observation():
    levels = series([3.0], ["2010-01-04"])
    out = views.align_levels(levels, DAYS)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [3.0, 3.0, 3.0]


def test_align_levels_rejects_a_stale_value():
    levels = series([1.0], ["2009-12-20"])
    with pytest.raises(ValueError, match="stale"):
        views.align_levels(levels, DAYS, max_age_days=7)
    assert views.align_levels(levels, DAYS, max_age_days=20).notna().all()


def test_align_levels_rejects_unsorted_inputs():
    with pytest.raises(ValueError, match="sorted"):
        views.align_levels(series([1.0, 2.0], ["2010-01-04", "2009-12-30"]), DAYS)
    with pytest.raises(ValueError, match="sorted"):
        views.align_levels(series([1.0], ["2009-12-30"]), DAYS[::-1])


def test_align_levels_rejects_duplicate_days():
    with pytest.raises(ValueError, match="unique"):
        views.align_levels(series([1.0], ["2009-12-30"]), DAYS.append(DAYS[-1:]))
    with pytest.raises(ValueError, match="unique"):
        views.align_levels(series([1.0, 2.0], ["2009-12-30", "2009-12-30"]), DAYS)


def test_align_levels_rejects_timezones_and_non_dates():
    with pytest.raises(ValueError, match="tz-naive"):
        views.align_levels(series([1.0], ["2009-12-30"]), DAYS.tz_localize("UTC"))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        views.align_levels(pd.Series([1.0], index=[0]), DAYS)


def test_align_levels_handles_empty_inputs():
    empty = pd.DatetimeIndex([])
    assert views.align_levels(series([1.0], ["2009-12-30"]), empty).empty
    assert views.align_levels(series([], []), DAYS).isna().all()


# --- fx_returns -----------------------------------------------------------------------------


def test_fx_returns_hand_values():
    # CAD per USD: 1.00 -> 1.10 (USD +10%) -> 0.99 (USD -10%)
    usdcad = series([1.00, 1.10, 0.99], DAYS[:3])
    fx = views.fx_returns(usdcad, DAYS[:3])
    assert np.isnan(fx.iloc[0])
    assert fx.iloc[1:].to_numpy() == pytest.approx([0.10, -0.10])


def test_fx_returns_missing_fix_on_a_trading_day_carries_the_move():
    # No fix on 4 Jan: that day is flat, the 5 Jan return covers both days' move.
    usdcad = series([1.00, 1.00, np.nan, 1.21, 1.21])
    fx = views.fx_returns(usdcad, DAYS)
    assert fx.iloc[1:].to_numpy() == pytest.approx([0.0, 0.0, 0.21, 0.0])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
def test_fx_returns_rejects_impossible_levels(bad):
    with pytest.raises(ValueError, match="positive and finite"):
        views.fx_returns(series([1.0, bad, 1.0, 1.0, 1.0]), DAYS)


# --- yield_changes --------------------------------------------------------------------------


def test_yield_changes_hand_values_and_a_canadian_holiday():
    # 3.50% -> 3.60% -> 3.55%, no Canadian close on 5 Jan (carried: change 0), then 3.40%.
    y10 = series([0.0350, 0.0360, 0.0355, np.nan, 0.0340], name="CA_GOVT_10Y")
    dy = views.yield_changes(y10, DAYS)
    assert np.isnan(dy.iloc[0])
    assert dy.iloc[1:].to_numpy() == pytest.approx([0.0010, -0.0005, 0.0, -0.0015], abs=1e-15)
    assert dy.name == "CA_GOVT_10Y"
    assert dy.index.equals(DAYS)


def test_yield_changes_are_nan_before_the_first_level():
    # The daily 10-year starts in 2001: earlier US days get NaN, not a change from 0.
    y10 = series([0.05, 0.051], DAYS[2:4])
    dy = views.yield_changes(y10, DAYS)
    assert dy.iloc[:3].isna().all()
    assert dy.iloc[3:].to_numpy() == pytest.approx([0.001, 0.0], abs=1e-15)


def test_yield_changes_allow_negative_yields():
    dy = views.yield_changes(series([-0.002, -0.001], DAYS[:2]), DAYS[:2])
    assert dy.iloc[1] == pytest.approx(0.001, abs=1e-15)


def test_yield_changes_reject_stale_and_non_finite_levels():
    with pytest.raises(ValueError, match="stale"):
        views.yield_changes(
            series([0.05], DAYS[:1]), pd.DatetimeIndex(["2009-12-30", "2010-01-15"])
        )
    with pytest.raises(ValueError, match="finite"):
        views.yield_changes(series([0.05, np.inf, 0.05, 0.05, 0.05]), DAYS)
    # max_age_days reaches the carry: 4 Jan is 4 days after the 31 Dec close.
    y10 = series([0.05, 0.05], DAYS[:2])
    assert views.yield_changes(y10, DAYS[:3], max_age_days=4).iloc[2] == 0.0
    with pytest.raises(ValueError, match="stale"):
        views.yield_changes(y10, DAYS[:3], max_age_days=3)


# --- to_cad_unhedged ------------------------------------------------------------------------


def test_usd_up_is_a_gain_for_a_cad_investor():
    r = series([0.0, 0.0], DAYS[:2])
    fx = series([0.10, -0.10], DAYS[:2])  # USD +10%, then USD -10%
    assert views.to_cad_unhedged(r, fx).to_numpy() == pytest.approx([0.10, -0.10])


def test_to_cad_unhedged_compounds_asset_and_currency():
    r = series([0.02], DAYS[:1])
    fx = series([0.05], DAYS[:1])
    assert views.to_cad_unhedged(r, fx).iloc[0] == pytest.approx(1.02 * 1.05 - 1)  # 0.071


def test_to_cad_unhedged_applies_fx_to_every_column():
    r = pd.DataFrame({"A": [0.01, -0.02], "B": [0.00, 0.03]}, index=DAYS[:2])
    fx = series([0.10, -0.05], DAYS[:2])
    out = views.to_cad_unhedged(r, fx)
    expected = (1 + r).mul(1 + fx, axis=0) - 1
    pd.testing.assert_frame_equal(out, expected)


@given(st.lists(st.tuples(small_returns, small_returns), min_size=1, max_size=20))
def test_to_cad_unhedged_round_trips(pairs):
    r = series([p[0] for p in pairs], _days(len(pairs)))
    fx = series([p[1] for p in pairs], _days(len(pairs)))
    back = (1 + views.to_cad_unhedged(r, fx)) / (1 + fx) - 1
    assert back.to_numpy() == pytest.approx(r.to_numpy(), abs=1e-12)


# --- daily_rate -----------------------------------------------------------------------------


def test_daily_rate_rolls_three_month_bills():
    y = series([0.08] * 5)
    rate = views.daily_rate(y, DAYS)
    one_day = (1 + 0.08 * 91 / 365) ** (1 / 91) - 1
    assert np.isnan(rate.iloc[0])  # no previous trading day
    assert rate.iloc[1] == pytest.approx(one_day)  # Wed -> Thu: 1 day


def test_daily_rate_accrues_weekends_and_holidays():
    y = series([0.08] * 5)
    rate = views.daily_rate(y, DAYS)
    one_day = rate.iloc[1]
    # Thu 31 Dec -> Mon 4 Jan is 4 calendar days (holiday + weekend).
    assert 1 + rate.iloc[2] == pytest.approx((1 + one_day) ** 4)
    assert rate.iloc[3] == pytest.approx(one_day)


def test_daily_rate_compounds_to_the_quoted_bill_return_over_its_term():
    days = pd.date_range("2000-01-03", periods=92, freq="D")  # 91 one-day periods
    rate = views.daily_rate(series([0.20] * 92, days), days)
    assert (1 + rate.iloc[1:]).prod() == pytest.approx(1 + 0.20 * 91 / 365)


def test_daily_rate_uses_the_yield_known_at_the_start_of_the_period():
    # The yield jumps on 4 Jan; the 4 Jan return still earns the old 1%.
    y = series([0.01, 0.01, 0.50, 0.50, 0.50])
    rate = views.daily_rate(y, DAYS)
    assert 1 + rate.iloc[2] == pytest.approx((1 + 0.01 * 91 / 365) ** (4 / 91))
    assert 1 + rate.iloc[3] == pytest.approx((1 + 0.50 * 91 / 365) ** (1 / 91))


def test_daily_rate_term_is_configurable():
    y = series([0.05] * 5)
    rate = views.daily_rate(y, DAYS, term_days=30)
    assert rate.iloc[1] == pytest.approx((1 + 0.05 * 30 / 365) ** (1 / 30) - 1)


def test_daily_rate_rejects_a_bad_term():
    with pytest.raises(ValueError, match="term_days"):
        views.daily_rate(series([0.05] * 5), DAYS, term_days=0)


# --- to_cad_hedged and excess ---------------------------------------------------------------


def test_cip_hedge_toy_example():
    # Asset +1% in USD; CAD bills pay 3 bp, USD bills 1 bp: the hedge adds the 2 bp differential.
    r = series([0.01], DAYS[:1])
    out = views.to_cad_hedged(r, series([0.0003], DAYS[:1]), series([0.0001], DAYS[:1]))
    assert out.iloc[0] == pytest.approx(0.0102)


def test_cip_hedge_ignores_the_currency_move():
    r = pd.DataFrame({"A": [0.01, 0.01]}, index=DAYS[:2])
    rf = series([0.0002, 0.0002], DAYS[:2])
    out = views.to_cad_hedged(r, rf, rf)
    pd.testing.assert_frame_equal(out, r)


@given(st.lists(st.tuples(small_returns, small_returns, small_returns), min_size=1, max_size=20))
def test_hedged_excess_return_equals_usd_excess_return(triples):
    days = _days(len(triples))
    r = series([t[0] for t in triples], days)
    rf_cad = series([t[1] / 100 for t in triples], days)
    rf_usd = series([t[2] / 100 for t in triples], days)
    hedged_excess = views.excess(views.to_cad_hedged(r, rf_cad, rf_usd), rf_cad)
    assert hedged_excess.to_numpy() == pytest.approx(views.excess(r, rf_usd).to_numpy(), abs=1e-12)


def test_excess_subtracts_per_row():
    r = pd.DataFrame({"A": [0.010, 0.020], "B": [0.0, -0.01]}, index=DAYS[:2])
    rf = series([0.001, 0.002], DAYS[:2])
    expected = pd.DataFrame({"A": [0.009, 0.018], "B": [-0.001, -0.012]}, index=DAYS[:2])
    pd.testing.assert_frame_equal(views.excess(r, rf), expected)


def test_excess_of_a_series_keeps_its_name():
    out = views.excess(
        series([0.010, 0.020], DAYS[:2], name="1/N"), series([0.001, 0.002], DAYS[:2])
    )
    assert out.name == "1/N"
    assert out.to_numpy() == pytest.approx([0.009, 0.018])


@pytest.mark.parametrize(
    "func",
    [
        lambda r, s: views.excess(r, s),
        lambda r, s: views.to_cad_unhedged(r, s),
        lambda r, s: views.to_cad_hedged(r, s, r * 0),  # rf_cad misaligned
        lambda r, s: views.to_cad_hedged(r, r * 0, s),  # rf_usd misaligned
    ],
)
def test_misaligned_dates_raise(func):
    r = series([0.01, 0.02], DAYS[:2])
    with pytest.raises(ValueError, match="align"):
        func(r, series([0.001, 0.001], DAYS[1:3]))


# --- make_views -----------------------------------------------------------------------------


@pytest.fixture
def inputs():
    days = pd.bdate_range("1970-12-28", "1971-01-15")
    assets = pd.DataFrame(
        {"A": np.linspace(-0.01, 0.01, len(days)), "B": 0.002}, index=days, dtype=float
    )
    rf_usd = series([0.0002] * len(days), days, name="RF")
    usdcad = series(np.linspace(1.00, 1.02, 10), pd.bdate_range("1971-01-04", periods=10))
    # Daily, as load_rates delivers it: 5% until a new value enters on 8 Jan.
    rate_days = pd.bdate_range("1970-12-01", "1971-01-15")
    ca_tbill = series(np.where(rate_days < "1971-01-08", 0.05, 0.06), rate_days)
    return assets, rf_usd, usdcad, ca_tbill


def test_make_views_returns_three_views(inputs):
    assets, rf_usd, usdcad, ca_tbill = inputs
    out = views.make_views(assets, rf_usd, usdcad, ca_tbill)
    assert set(out) == {"USD", "CAD", "CAD_HEDGED"}
    pd.testing.assert_frame_equal(out["USD"].returns, assets)
    pd.testing.assert_series_equal(out["USD"].rf, rf_usd)


def test_cad_views_start_on_the_second_day_with_fx(inputs):
    assets, rf_usd, usdcad, ca_tbill = inputs
    out = views.make_views(assets, rf_usd, usdcad, ca_tbill)
    for name in ("CAD", "CAD_HEDGED"):
        view = out[name]
        assert view.returns.index[0] == pd.Timestamp("1971-01-05")
        assert view.returns.index.equals(view.rf.index)
        assert view.returns.notna().all().all()
        assert view.rf.notna().all()


def test_make_views_hand_values(inputs):
    out = views.make_views(*inputs)
    cad, hedged = out["CAD"], out["CAD_HEDGED"]
    fx_step = 0.02 / 9  # USD/CAD rises 1.00 -> 1.02 in 9 equal steps; first step on 5 Jan

    def bill(y, d):
        return (1 + y * 91 / 365) ** (d / 91) - 1

    # Tue 5 Jan: asset B earns 0.2% in USD; USD +0.222%; CAD rate known on 4 Jan is 5%.
    assert cad.returns.loc["1971-01-05", "B"] == pytest.approx(1.002 * (1 + fx_step) - 1)
    assert cad.rf.loc["1971-01-05"] == pytest.approx(bill(0.05, 1))
    # Fri 8 Jan: the 6% rate enters today, so today still earns 5%.
    assert cad.rf.loc["1971-01-08"] == pytest.approx(bill(0.05, 1))
    # Mon 11 Jan: 6% known on Friday, accrued over 3 calendar days.
    assert cad.rf.loc["1971-01-11"] == pytest.approx(bill(0.06, 3))
    assert hedged.returns.loc["1971-01-11", "B"] == pytest.approx(0.002 + bill(0.06, 3) - 0.0002)
    assert hedged.rf.equals(cad.rf)
    pd.testing.assert_frame_equal(
        hedged.excess, out["USD"].excess.loc[hedged.returns.index], atol=1e-15
    )


@pytest.mark.parametrize("cut", ["1971-01-06", "1971-01-08", "1971-01-12"])
def test_make_views_has_no_look_ahead(inputs, cut):
    """Changing every input after day `cut` leaves all rows up to `cut` unchanged."""
    assets, rf_usd, usdcad, ca_tbill = inputs

    def scramble(obj):
        obj = obj.copy()
        obj.loc[obj.index > cut] *= 3.7
        return obj

    base = views.make_views(assets, rf_usd, usdcad, ca_tbill)
    moved = views.make_views(
        scramble(assets), scramble(rf_usd), scramble(usdcad), scramble(ca_tbill)
    )
    for name in base:
        pd.testing.assert_frame_equal(base[name].returns.loc[:cut], moved[name].returns.loc[:cut])
        pd.testing.assert_series_equal(base[name].rf.loc[:cut], moved[name].rf.loc[:cut])
        assert (
            not base[name].returns.loc[cut:].iloc[1:].equals(moved[name].returns.loc[cut:].iloc[1:])
        )


def test_view_allows_missing_returns_but_not_missing_rf():
    returns = pd.DataFrame({"A": [np.nan, 0.01]}, index=DAYS[:2])
    views.View(returns, series([0.0, 0.0], DAYS[:2]))
    with pytest.raises(ValueError, match="rf is missing"):
        views.View(returns, series([0.0, np.nan], DAYS[:2]))


def test_view_rejects_an_unsorted_calendar():
    returns = pd.DataFrame({"A": [0.0, 0.01]}, index=DAYS[:2][::-1])
    with pytest.raises(ValueError, match="sorted"):
        views.View(returns, series([0.0, 0.0], DAYS[:2][::-1]))


def test_view_excess_uses_its_own_rf(inputs):
    out = views.make_views(*inputs)
    cad = out["CAD"]
    pd.testing.assert_frame_equal(cad.excess, views.excess(cad.returns, cad.rf))


def test_make_views_rejects_misaligned_rf(inputs):
    assets, rf_usd, usdcad, ca_tbill = inputs
    with pytest.raises(ValueError, match="align"):
        views.make_views(assets, rf_usd.iloc[1:], usdcad, ca_tbill)


def test_cad_views_end_when_fx_would_go_stale(inputs):
    # FRED posts USD/CAD with a lag: the CAD views stop once the last fix is over
    # 7 days old (6 Jan + 7 = 13 Jan); the USD view does not.
    assets, rf_usd, usdcad, ca_tbill = inputs
    out = views.make_views(assets, rf_usd, usdcad.loc[:"1971-01-06"], ca_tbill)
    assert out["USD"].returns.index[-1] == pd.Timestamp("1971-01-15")
    for name in ("CAD", "CAD_HEDGED"):
        assert out[name].returns.index[-1] == pd.Timestamp("1971-01-13")


def test_a_holiday_on_the_last_day_is_carried(inputs):
    # No fix on the last trading day (like the Fed holiday on 2010-12-31): carry it.
    assets, rf_usd, usdcad, ca_tbill = inputs
    out = views.make_views(assets, rf_usd, usdcad.loc[:"1971-01-14"], ca_tbill)
    assert out["CAD"].returns.index[-1] == pd.Timestamp("1971-01-15")


def test_make_views_rejects_a_gap_inside_the_cad_period(inputs):
    assets, rf_usd, usdcad, ca_tbill = inputs
    gappy = usdcad.drop(pd.bdate_range("1971-01-05", "1971-01-13"))  # 4 Jan, then 14 Jan
    with pytest.raises(ValueError, match="stale"):
        views.make_views(assets, rf_usd, gappy, ca_tbill)


def test_make_views_needs_both_sources(inputs):
    assets, rf_usd, usdcad, ca_tbill = inputs
    with pytest.raises(ValueError, match="no day"):
        views.make_views(assets, rf_usd, usdcad * np.nan, ca_tbill)


def test_views_are_compared_by_identity(inputs):
    out = views.make_views(*inputs)
    assert out["CAD"] != out["CAD_HEDGED"]
    assert len({out["CAD"], out["USD"]}) == 2  # hashable


def test_fx_returns_rejects_text():
    with pytest.raises(ValueError, match="numeric"):
        views.fx_returns(pd.Series(["n/a"] * 5, index=DAYS, dtype=object), DAYS)


def test_make_views_needs_some_cad_history(inputs):
    assets, rf_usd, usdcad, ca_tbill = inputs
    late_fx = series([1.0], ["1971-02-01"])
    with pytest.raises(ValueError, match="no day"):
        views.make_views(assets, rf_usd, late_fx, ca_tbill)
