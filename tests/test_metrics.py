"""Performance metrics: hand-computed values, textbook formulas, properties, input checks."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from portfolio_lab.report import metrics as m

# The docstring example: 4 periods = 1 year (periods_per_year=4).
# Wealth 1.10, 0.88, 0.924, 1.0164; peak 1.10 before the 20% loss.
DAYS = pd.bdate_range("2000-01-03", periods=4)
R = pd.Series([0.10, -0.20, 0.05, 0.10], index=DAYS, name="P")
PPY = 4

returns_lists = st.lists(st.floats(-0.2, 0.2, allow_nan=False), min_size=5, max_size=60)


def daily(values, name=None):
    return pd.Series(values, index=pd.bdate_range("2000-01-03", periods=len(values)), name=name)


# --- the 4-period hand example --------------------------------------------------------------


def test_cagr_hand_example():
    assert m.cagr(R, periods_per_year=PPY) == pytest.approx(0.0164)


def test_cagr_annualizes_by_the_number_of_years():
    # The same 4 periods spread over 2 years: 1.0164 ** (1/2) - 1.
    assert m.cagr(R, periods_per_year=2) == pytest.approx(math.sqrt(1.0164) - 1)


def test_annualized_vol_hand_example():
    # deviations from the mean 0.0125: .0875, -.2125, .0375, .0875; sum of squares .061875
    assert m.annualized_vol(R, periods_per_year=PPY) == pytest.approx(math.sqrt(0.061875 / 3) * 2)


def test_sharpe_hand_example():
    expected = 0.0125 / math.sqrt(0.061875 / 3) * 2  # 0.1741
    assert m.sharpe(R, 0.0, periods_per_year=PPY) == pytest.approx(expected)


def test_sharpe_uses_the_risk_free_rate_per_period():
    rf = pd.Series([0.01] * 4, index=DAYS)
    expected = (0.0125 - 0.01) / math.sqrt(0.061875 / 3) * 2
    assert m.sharpe(R, rf, periods_per_year=PPY) == pytest.approx(expected)
    assert m.sharpe(R, 0.01, periods_per_year=PPY) == pytest.approx(expected)


def test_sortino_hand_example():
    # Downside deviation: sqrt((0 + 0.2^2 + 0 + 0) / 4) = 0.1 (all periods in the mean).
    assert m.sortino(R, 0.0, periods_per_year=PPY) == pytest.approx(0.0125 / 0.1 * 2)


def test_drawdowns_hand_example():
    expected = [0.0, 0.88 / 1.10 - 1, 0.924 / 1.10 - 1, 1.0164 / 1.10 - 1]
    assert m.drawdowns(R).to_numpy() == pytest.approx(expected)


def test_max_drawdown_hand_example():
    assert m.max_drawdown(R) == pytest.approx(-0.20)


def test_max_drawdown_duration_counts_an_unrecovered_drawdown():
    # Under the 1.10 peak for periods 2, 3 and 4, still not recovered at the end.
    assert m.max_drawdown_duration(R) == 3


def test_calmar_hand_example():
    assert m.calmar(R, periods_per_year=PPY) == pytest.approx(0.0164 / 0.20)


# --- drawdown details -----------------------------------------------------------------------


def test_a_first_period_loss_is_a_drawdown_from_the_starting_wealth():
    r = daily([-0.10, 0.05])
    assert m.drawdowns(r).to_numpy() == pytest.approx([-0.10, 0.9 * 1.05 - 1])
    assert m.max_drawdown_duration(r) == 2


def test_duration_ends_when_the_old_peak_is_regained():
    # Peak 1.1 after day 1; under it on days 2-3; back to exactly 1.1 on day 4.
    r = daily([0.10, -0.50, 0.50, 1.1 / 0.825 - 1, -0.01])
    assert m.max_drawdown_duration(r) == 2
    assert m.drawdowns(r).iloc[3] == pytest.approx(0.0)


def test_recovery_with_float_noise_counts_as_recovered():
    # 1.3 * 0.7 * (1 + 0.3/0.7) lands 2.2e-16 below 1.3 in floating point.
    r = daily([0.3, -0.3, 0.3 / 0.7])
    assert float(np.prod(1 + r.to_numpy())) < 1.3
    assert m.drawdowns(r).iloc[2] == 0.0
    assert m.max_drawdown_duration(r) == 1
    assert m.max_drawdown(r) == pytest.approx(-0.3)


def test_a_dip_within_float_noise_is_no_drawdown():
    r = daily([0.1, -1e-14, 0.01, 0.02])
    assert m.max_drawdown(r) == 0.0
    assert m.max_drawdown_duration(r) == 0
    assert math.isnan(m.calmar(r, periods_per_year=252))


def test_a_total_loss():
    r = daily([-1.0, 0.5])
    assert m.cagr(r, periods_per_year=2) == -1.0
    assert m.max_drawdown(r) == -1.0
    assert m.max_drawdown_duration(r) == 2


def test_duration_takes_the_longest_spell():
    r = daily([-0.1, 0.2, -0.01, -0.01, -0.01, 0.5, -0.1])
    assert m.max_drawdown_duration(r) == 3


def test_no_loss_means_no_drawdown():
    r = daily([0.01] * 10)
    assert m.max_drawdown(r) == 0.0
    assert m.max_drawdown_duration(r) == 0
    assert math.isnan(m.calmar(r, periods_per_year=252))  # no risk: undefined, never ranks first


# --- higher moments -------------------------------------------------------------------------

X = daily([0.01, 0.02, 0.03, 0.10, -0.04])


def test_skew_is_the_bias_corrected_sample_skewness():
    x = X.to_numpy()
    n, d = len(x), x - x.mean()
    g1 = (d**3).mean() / (d**2).mean() ** 1.5
    assert m.skew(X) == pytest.approx(math.sqrt(n * (n - 1)) / (n - 2) * g1)


def test_excess_kurtosis_is_the_bias_corrected_sample_value():
    x = X.to_numpy()
    n, d = len(x), x - x.mean()
    g2 = (d**4).mean() / (d**2).mean() ** 2 - 3
    expected = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
    assert m.excess_kurtosis(X) == pytest.approx(expected)


def test_higher_moments_need_enough_periods():
    with pytest.raises(ValueError, match="at least 3"):
        m.skew(daily([0.01, 0.02]))
    with pytest.raises(ValueError, match="at least 4"):
        m.excess_kurtosis(daily([0.01, 0.02, 0.03]))


# --- active risk ----------------------------------------------------------------------------


def test_tracking_error_and_information_ratio_hand_example():
    bench = pd.Series([0.08, -0.18, 0.05, 0.09], index=DAYS)
    active = np.array([0.02, -0.02, 0.0, 0.01])  # mean 0.0025; squared deviations sum 0.000875
    te = math.sqrt(0.000875 / 3) * 2
    assert m.tracking_error(R, bench, periods_per_year=PPY) == pytest.approx(te)
    assert m.information_ratio(R, bench, periods_per_year=PPY) == pytest.approx(
        active.mean() * PPY / te
    )


def test_a_portfolio_has_no_tracking_error_against_itself():
    assert m.tracking_error(R, R, periods_per_year=PPY) == 0.0


# --- DataFrames, summary --------------------------------------------------------------------


def test_metrics_apply_per_column():
    frame = pd.DataFrame({"P": R, "Q": -R})
    out = m.max_drawdown(frame)
    assert isinstance(out, pd.Series)
    assert out["P"] == pytest.approx(m.max_drawdown(R))
    assert out["Q"] == pytest.approx(m.max_drawdown(-R))
    durations = m.max_drawdown_duration(frame)
    assert durations.tolist() == [3, m.max_drawdown_duration(-R)]


def test_summary_of_a_series():
    out = m.summary(R, 0.0, periods_per_year=PPY)
    assert isinstance(out, pd.Series)
    assert out["cagr"] == pytest.approx(0.0164)
    assert out["max_drawdown"] == pytest.approx(-0.20)
    assert out["max_drawdown_duration"] == 3
    assert "tracking_error" not in out


def test_summary_of_a_frame_with_a_benchmark():
    frame = pd.DataFrame({"P": R, "Q": R * 0.5})
    out = m.summary(frame, 0.0, periods_per_year=PPY, benchmark=R)
    assert list(out.columns) == ["P", "Q"]
    assert out.loc["tracking_error", "P"] == 0.0
    assert out.loc["sharpe", "Q"] == pytest.approx(m.sharpe(R * 0.5, 0.0, periods_per_year=PPY))
    assert set(m.METRICS) | {"tracking_error", "information_ratio"} == set(out.index)


# --- properties -----------------------------------------------------------------------------


@given(returns_lists, st.floats(0.1, 4))  # 4 x -20% stays above -100%
def test_vol_scales_and_sharpe_does_not(values, k):
    r = daily(values)
    assume(r.std() > 1e-6)
    assert m.annualized_vol(r * k, periods_per_year=252) == pytest.approx(
        k * m.annualized_vol(r, periods_per_year=252)
    )
    assert m.sharpe(r * k, 0.0, periods_per_year=252) == pytest.approx(
        m.sharpe(r, 0.0, periods_per_year=252)
    )


@given(returns_lists)
def test_cagr_compounds_back_to_final_wealth(values):
    r = daily(values)
    years = len(r) / 12  # monthly: a moderate exponent keeps 1 + cagr precise
    growth = (1 + m.cagr(r, periods_per_year=12)) ** years
    assert growth == pytest.approx(float((1 + r).prod()))


@given(returns_lists)
def test_drawdowns_are_between_minus_one_and_zero(values):
    dd = m.drawdowns(daily(values))
    assert ((dd <= 0) & (dd >= -1)).all()
    assert m.max_drawdown_duration(daily(values)) <= len(values)


# --- input checks ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda r: m.cagr(r, periods_per_year=252),
        lambda r: m.sharpe(r, 0.0, periods_per_year=252),
        lambda r: m.max_drawdown(r),
        lambda r: m.skew(r),
    ],
)
def test_nan_raises(call):
    with pytest.raises(ValueError, match="missing"):
        call(daily([0.01, np.nan, 0.02, 0.03]))


def test_periods_per_year_is_required_and_positive():
    with pytest.raises(TypeError):
        m.cagr(R)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="periods_per_year"):
        m.annualized_vol(R, periods_per_year=0)


def test_too_short_or_impossible_returns_raise():
    with pytest.raises(ValueError, match="at least 2"):
        m.annualized_vol(daily([0.01]), periods_per_year=252)
    with pytest.raises(ValueError, match="-100%"):
        m.cagr(daily([0.1, -1.5]), periods_per_year=252)


def test_misaligned_rf_or_benchmark_raises():
    other = pd.Series([0.0] * 4, index=pd.bdate_range("2001-01-01", periods=4))
    with pytest.raises(ValueError, match="align"):
        m.sharpe(R, other, periods_per_year=PPY)
    with pytest.raises(ValueError, match="align"):
        m.tracking_error(R, other, periods_per_year=PPY)


def test_missing_rf_raises():
    rf = pd.Series([0.0, np.nan, 0.0, 0.0], index=DAYS)
    with pytest.raises(ValueError, match="missing"):
        m.sortino(R, rf, periods_per_year=PPY)


def test_a_constant_rf_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        m.sharpe(R, float("nan"), periods_per_year=PPY)


# --- zero risk is NaN, never a huge number --------------------------------------------------


@pytest.mark.parametrize("level", [0.0, 0.01, 0.1])
def test_constant_returns_have_no_sharpe(level):
    r = daily([level] * 10)
    assert math.isnan(m.sharpe(r, 0.0, periods_per_year=252))
    assert math.isnan(m.sortino(r, 0.0, periods_per_year=252))


def test_cash_against_its_own_rate_has_no_sharpe():
    rf = daily([0.0001, 0.0002, 0.00015, 0.0001])
    assert math.isnan(m.sharpe(rf, rf, periods_per_year=252))


def test_no_period_below_rf_has_no_sortino():
    assert math.isnan(m.sortino(daily([0.01, 0.03, 0.02]), 0.0, periods_per_year=252))


def test_constant_active_return_has_no_information_ratio():
    bench = daily([0.01, -0.02, 0.03, 0.0])
    assert math.isnan(m.information_ratio(bench + 0.001, bench, periods_per_year=252))
    assert math.isnan(m.information_ratio(bench, bench, periods_per_year=252))


# --- more inputs ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "match"),
    [([0.1, np.inf, 0.1], "finite"), ([0.1, -np.inf, 0.1], "finite"), (["a", "b"], "numeric")],
)
def test_bad_values_raise(values, match):
    with pytest.raises(ValueError, match=match):
        m.max_drawdown(daily(values))


def test_infinite_periods_per_year_raises():
    with pytest.raises(ValueError, match="finite"):
        m.cagr(R, periods_per_year=math.inf)


def test_duplicate_columns_raise():
    frame = pd.concat([R, R], axis=1)
    with pytest.raises(ValueError, match="unique column"):
        m.sharpe(frame, 0.0, periods_per_year=PPY)


def test_infinite_rf_raises():
    with pytest.raises(ValueError, match="finite"):
        m.sharpe(R, pd.Series([0.0, np.inf, 0.0, 0.0], index=DAYS), periods_per_year=PPY)


def test_integer_returns_and_a_plain_index_work():
    assert m.max_drawdown(pd.Series([0, 0, 1])) == 0.0


def test_frame_metrics_with_a_series_rf():
    frame = pd.DataFrame({"P": R, "Q": R * 0.5})
    rf = pd.Series([0.001, 0.0, 0.002, 0.001], index=DAYS)
    for func in (m.sharpe, m.sortino):
        out = func(frame, rf, periods_per_year=PPY)
        assert out["Q"] == pytest.approx(func(R * 0.5, rf, periods_per_year=PPY))
    assert m.skew(frame)["P"] == pytest.approx(m.skew(R))
    ir = m.information_ratio(frame, R * 0.8, periods_per_year=PPY)
    assert ir["P"] == pytest.approx(m.information_ratio(R, R * 0.8, periods_per_year=PPY))


def test_names_pass_through():
    assert m.drawdowns(R).name == "P"
    assert m.summary(R, 0.0, periods_per_year=PPY).name == "P"


@given(returns_lists, returns_lists)
def test_sharpe_with_a_series_rf_is_sharpe_of_excess(values, rf_values):
    n = min(len(values), len(rf_values))
    r, rf = daily(values[:n]), daily([v / 10 for v in rf_values[:n]])
    assume((r - rf).std() > 1e-6)
    assert m.sharpe(r, rf, periods_per_year=252) == pytest.approx(
        m.sharpe(r - rf, 0.0, periods_per_year=252)
    )


@given(returns_lists)
def test_information_ratio_sign_follows_mean_active_return(values):
    bench = daily(values)
    noise = daily([(-1) ** i * 0.001 for i in range(len(values))])
    for shift, sign in ((0.0005, 1), (-0.0005, -1)):
        ir = m.information_ratio(bench + noise + shift, bench, periods_per_year=252)
        assert np.sign(ir) == sign


def test_returns_out_of_date_order_raise():
    # Reversed, the same returns would give a 2-period drawdown instead of 3.
    with pytest.raises(ValueError, match="sorted"):
        m.max_drawdown_duration(R.iloc[::-1])
    with pytest.raises(ValueError, match="unique"):
        m.cagr(pd.concat([R, R.iloc[-1:]]), periods_per_year=PPY)


def test_text_rf_raises():
    with pytest.raises(ValueError, match="numeric"):
        m.sharpe(R, pd.Series(["n/a"] * 4, index=DAYS, dtype=object), periods_per_year=PPY)
