"""Rolling one-day-ahead VaR/ES forecasts: a plain loop as reference, timing, no look-ahead."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from portfolio_lab.construct.covariance import EWMA_DECAY
from portfolio_lab.report import metrics
from portfolio_lab.risk import var as v


def days(n):
    return pd.bdate_range("2000-01-03", periods=n)


def fat_tailed(seed, n, scale=0.01):
    return pd.Series(np.random.default_rng(seed).standard_t(4, n) * scale, index=days(n), name="P")


def loop_forecast(returns, method, level, window):
    """One window at a time with the point functions: the reference for `forecast`."""
    r = returns.to_numpy()
    # FHS's filter by a plain recursion, independent of the code under test.
    var, sigma = float(np.mean(r[:window] ** 2)), []
    for x in r:
        sigma.append(var**0.5)  # sigma[s]: the forecast for day s, from returns before it
        var = EWMA_DECAY * var + (1 - EWMA_DECAY) * x * x
    sigma = np.array(sigma)
    rows = []
    for t in range(window, len(r)):
        past = returns.iloc[t - window : t]
        if method == "historical":
            risk = v.var_es(past, level)
        elif method == "normal":
            risk = v.normal_var_es(float(past.mean()), float(past.std(ddof=1)), level)
        elif method == "cornish_fisher":
            risk = v.cornish_fisher_var_es(
                float(past.mean()),
                float(past.std(ddof=1)),
                metrics.skew(past),
                metrics.excess_kurtosis(past),
                level,
            )
        else:  # fhs: residuals of the window, rescaled by the forecast vol for day t
            residuals = past.to_numpy() / sigma[t - window : t]
            risk = v.var_es(residuals * sigma[t], level)
        rows.append(risk)
    return pd.DataFrame(rows, columns=["var", "es"], index=returns.index[window:])


# --- shape and timing -------------------------------------------------------------------------


def test_methods_and_default_window():
    assert v.METHODS == ("historical", "normal", "cornish_fisher", "fhs")
    assert v.WINDOW == 500


@pytest.mark.parametrize("method", v.METHODS)
def test_first_forecast_is_for_the_day_after_the_first_window(method):
    r = fat_tailed(0, 30)
    fc = v.forecast(r, method=method, level=0.95, window=10)
    assert list(fc.columns) == ["var", "es"]
    assert fc.index.equals(r.index[10:])
    assert len(fc) == 20


def test_historical_forecast_by_hand():
    # window 4; the forecast for day 5 uses days 1-4 only, never day 5 itself.
    r = pd.Series([-0.04, 0.01, -0.02, 0.03, -0.50, 0.02], index=days(6))
    fc = v.forecast(r, method="historical", level=0.75, window=4)
    # Days 1-4 sorted: -0.04, -0.02, 0.01, 0.03; h = 3 * 0.25 = 0.75 -> -0.04 + 0.75 * 0.02.
    assert fc["var"].iloc[0] == pytest.approx(0.025, abs=1e-15)
    assert fc["es"].iloc[0] == pytest.approx(0.04, abs=1e-15)
    # Day 6 uses days 2-5: -0.50 is in its window now.
    assert fc["es"].iloc[1] == pytest.approx(0.50, abs=1e-15)


def test_fhs_by_hand():
    # decay 0.5, window 4, returns 2%, -2%, 2%, -2%, then 4%, then anything.
    # Seed = mean of r^2 over days 1-4 = 4e-4; every r^2 equals it, so sigma = 0.02 on days
    # 1-5, and the residuals of days 1-4 are (1, -1, 1, -1).
    # Day 5: sample 0.02 x (1, -1, 1, -1); type 7 at 0.25 sits between the two -0.02s.
    # Day 6: residuals of days 2-5 are (-1, 1, -1, 0.04 / 0.02 = 2), and
    # sigma(6)^2 = 0.5 * 4e-4 + 0.5 * 0.04^2 = 1e-3: the 4% day lifts the next VaR to 3.16%.
    r = pd.Series([0.02, -0.02, 0.02, -0.02, 0.04, 0.0], index=days(6))
    fc = v.forecast(r, method="fhs", level=0.75, window=4, decay=0.5)
    assert fc["var"].iloc[0] == pytest.approx(0.02, abs=1e-15)
    assert fc["es"].iloc[0] == pytest.approx(0.02, abs=1e-15)
    assert fc["var"].iloc[1] == pytest.approx(math.sqrt(1e-3), rel=1e-14)
    assert fc["es"].iloc[1] == pytest.approx(math.sqrt(1e-3), rel=1e-14)


@pytest.mark.parametrize("method", v.METHODS)
@pytest.mark.parametrize("level", v.LEVELS)
def test_forecast_matches_a_plain_loop(method, level):
    r = fat_tailed(7, 400)
    fc = v.forecast(r, method=method, level=level, window=60)
    expected = loop_forecast(r, method, level, 60)
    assert fc.index.equals(expected.index)
    np.testing.assert_allclose(fc.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-15)


@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(6, 120),
    window=st.integers(4, 40),
    method=st.sampled_from(v.METHODS),
    level=st.sampled_from(v.LEVELS),
)
def test_forecast_matches_a_plain_loop_on_random_cases(seed, n, window, method, level):
    if n <= window:
        n = window + 1
    r = fat_tailed(seed, n)
    fc = v.forecast(r, method=method, level=level, window=window)
    expected = loop_forecast(r, method, level, window)
    np.testing.assert_allclose(fc.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("method", v.METHODS)
def test_forecast_across_many_blocks_matches_the_loop(method, monkeypatch):
    # Blocks of 3 windows (90 cells / window 30): dozens of seams, every row compared.
    monkeypatch.setattr(v, "_BLOCK_CELLS", 90)
    r = fat_tailed(11, 400)
    fc = v.forecast(r, method=method, level=0.99, window=30)
    expected = loop_forecast(r, method, 0.99, 30)
    np.testing.assert_allclose(fc.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-15)


def test_blocks_hold_at_least_one_window(monkeypatch):
    monkeypatch.setattr(v, "_BLOCK_CELLS", 1)
    r = fat_tailed(12, 60)
    expected = loop_forecast(r, "historical", 0.95, 20)
    fc = v.forecast(r, method="historical", level=0.95, window=20)
    np.testing.assert_allclose(fc.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-15)


# --- no look-ahead ------------------------------------------------------------------------------


@given(
    seed=st.integers(0, 2**32 - 1),
    k=st.integers(0, 59),
    method=st.sampled_from(v.METHODS),
    shock=st.floats(-0.3, 0.3).filter(lambda x: abs(x) > 1e-6),
)
def test_no_look_ahead(seed, k, method, shock):
    # Changing returns from day k on can't change any forecast for a day <= k: the forecast
    # for day t uses returns through t - 1 only (and FHS's seed only the first window).
    r = fat_tailed(seed, 60)
    base = v.forecast(r, method=method, level=0.99, window=12)
    bumped = r.copy()
    bumped.iloc[k:] = (bumped.iloc[k:] + shock).clip(lower=-0.9)
    moved = v.forecast(bumped, method=method, level=0.99, window=12)
    before = base.index[base.index <= r.index[k]]
    assert moved.loc[before].equals(base.loc[before])


@pytest.mark.parametrize("method", v.METHODS)
def test_a_crash_enters_the_next_forecast_not_its_own(method):
    r = fat_tailed(3, 80)
    crash = r.copy()
    crash.iloc[50] = -0.25
    base = v.forecast(r, method=method, level=0.99, window=20)
    hit = v.forecast(crash, method=method, level=0.99, window=20)
    assert hit.iloc[: 50 - 20 + 1].equals(base.iloc[: 50 - 20 + 1])  # through day 50 itself
    assert hit["var"].iloc[51 - 20] > base["var"].iloc[51 - 20]  # day 51 sees it


# --- next_day: "today's VaR" ----------------------------------------------------------------------


@pytest.mark.parametrize("method", v.METHODS)
@pytest.mark.parametrize("level", v.LEVELS)
def test_next_day_is_the_forecast_one_row_later(method, level):
    # The forecast for the day after the last close uses the same rule: appending any return
    # makes it an ordinary forecast row.
    r = fat_tailed(5, 90)
    extended = pd.concat([r, pd.Series([0.123], index=pd.bdate_range(r.index[-1], periods=2)[1:])])
    nxt = v.next_day(r, method=method, level=level, window=30)
    row = v.forecast(extended, method=method, level=level, window=30).iloc[-1]
    assert isinstance(nxt, v.VarEs)
    assert nxt.var == row["var"]
    assert nxt.es == row["es"]


def test_next_day_needs_only_one_window():
    r = fat_tailed(2, 10)
    assert v.next_day(r, method="historical", level=0.95, window=10) == v.var_es(r, 0.95)


# --- the window's moments -------------------------------------------------------------------------


def test_cornish_fisher_uses_steps_2_bias_corrected_moments():
    r = fat_tailed(9, 41)
    fc = v.forecast(r, method="cornish_fisher", level=0.99, window=40)
    past = r.iloc[:40]
    expected = v.cornish_fisher_var_es(
        float(past.mean()),
        float(past.std(ddof=1)),
        metrics.skew(past),
        metrics.excess_kurtosis(past),
        0.99,
    )
    assert fc.iloc[0].to_numpy() == pytest.approx(list(expected), rel=1e-12)


@pytest.mark.parametrize("method", ["normal", "cornish_fisher"])
def test_a_window_without_variance_raises_naming_the_day(method):
    r = pd.Series([0.01] * 12 + [0.02, -0.01], index=days(14))
    with pytest.raises(ValueError, match=f"no variance.*{r.index[10].date()}"):
        v.forecast(r, method=method, level=0.99, window=10)


def test_historical_and_fhs_accept_a_flat_window():
    r = pd.Series([0.01] * 12 + [0.02, -0.01], index=days(14))
    assert v.forecast(r, method="historical", level=0.99, window=10)["var"].iloc[0] == -0.01
    fhs = v.forecast(r, method="fhs", level=0.99, window=10)
    assert np.isfinite(fhs.to_numpy()).all()


def test_fhs_needs_a_first_window_that_moves():
    r = pd.Series([0.0] * 10 + [0.01, -0.02], index=days(12))
    with pytest.raises(ValueError, match="first window"):
        v.forecast(r, method="fhs", level=0.99, window=10)


def test_fhs_uses_the_given_decay():
    r = fat_tailed(4, 60)
    assert not v.forecast(r, method="fhs", level=0.99, window=20, decay=0.97).equals(
        v.forecast(r, method="fhs", level=0.99, window=20)
    )
    assert v.forecast(r, method="fhs", level=0.99, window=20, decay=EWMA_DECAY).equals(
        v.forecast(r, method="fhs", level=0.99, window=20)
    )


# --- input checks ---------------------------------------------------------------------------------


def test_forecast_rejects_bad_arguments():
    r = fat_tailed(1, 30)
    with pytest.raises(ValueError, match="method"):
        v.forecast(r, method="monte_carlo", level=0.99, window=10)
    with pytest.raises(ValueError, match="level"):
        v.forecast(r, method="historical", level=99, window=10)
    for window in (3, 0, 10.0, True):
        with pytest.raises(ValueError, match="window"):
            v.forecast(r, method="historical", level=0.99, window=window)
    with pytest.raises(ValueError, match="more than 30"):
        v.forecast(r, method="historical", level=0.99, window=30)
    with pytest.raises(ValueError, match="at least 31"):
        v.next_day(r, method="historical", level=0.99, window=31)
    with pytest.raises(ValueError, match="decay"):
        v.forecast(r, method="fhs", level=0.99, window=10, decay=1.5)


def test_forecast_rejects_bad_returns():
    r = fat_tailed(1, 30)
    with pytest.raises(ValueError, match="finite"):
        v.forecast(r.where(r.index != r.index[20]), method="historical", level=0.99, window=10)
    with pytest.raises(ValueError, match="-100%"):
        v.forecast(r.where(r.index != r.index[20], -1.2), method="normal", level=0.99, window=10)
    with pytest.raises(ValueError, match="sorted"):
        v.forecast(r.iloc[::-1], method="historical", level=0.99, window=10)
    with pytest.raises(TypeError, match="Series"):
        v.forecast(r.to_frame(), method="historical", level=0.99, window=10)


def test_fhs_raises_if_the_volatility_underflows_to_zero():
    # With a tiny decay the filter forgets in a day; ~30 zero returns take sigma below the
    # smallest float, and residuals r / 0 would be inf or NaN.
    r = pd.Series([0.01, -0.01, 0.01, -0.01] + [0.0] * 40, index=days(44))
    with pytest.raises(ValueError, match="reached 0"):
        v.forecast(r, method="fhs", level=0.99, window=4, decay=1e-10)


def test_next_day_names_the_day_after_the_data_for_a_flat_last_window():
    r = pd.Series([0.02, -0.01] + [0.01] * 10, index=days(12))
    with pytest.raises(ValueError, match=f"no variance.*the day after {r.index[-1].date()}"):
        v.next_day(r, method="normal", level=0.99, window=10)


# --- review fixes ---------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["normal", "cornish_fisher"])
def test_forecast_ignores_a_problem_only_in_the_next_day_row(method):
    # Days 0-19 move, days 20-29 are 0. The last forecast row (day 29) still sees day 19;
    # only the day after the data would have a flat window, and forecast doesn't return it.
    r = pd.concat([fat_tailed(13, 20), pd.Series(0.0, index=days(30)[20:])])
    fc = v.forecast(r, method=method, level=0.99, window=10)
    assert fc.index[-1] == r.index[-1]
    assert np.isfinite(fc.to_numpy()).all()
    with pytest.raises(ValueError, match="no variance.*the day after"):
        v.next_day(r, method=method, level=0.99, window=10)


def test_forecast_accepts_numpy_integers():
    r = fat_tailed(14, 40)
    assert v.forecast(r, method="historical", level=np.float64(0.99), window=np.int64(10)).equals(
        v.forecast(r, method="historical", level=0.99, window=10)
    )
    assert v.next_day(r, method="normal", level=0.99, window=np.int32(10)) == v.next_day(
        r, method="normal", level=0.99, window=10
    )


@pytest.mark.parametrize("method", ["normal", "cornish_fisher", "fhs"])
def test_forecast_raises_on_results_that_overflow(method):
    # A finite but absurd return (+1e200): its square overflows, and the forecasts that see it
    # would be inf or NaN. NaN VaR would silently count as "no exception".
    r = fat_tailed(15, 40)
    r.iloc[25] = 1e200
    with pytest.raises(ValueError, match=f"not finite.*{r.index[26].date()}"):
        v.forecast(r, method=method, level=0.99, window=10)


def test_fhs_rejects_a_first_window_that_overflows():
    r = fat_tailed(16, 30)
    r.iloc[3] = 1e200
    with pytest.raises(ValueError, match="first window"):
        v.forecast(r, method="fhs", level=0.99, window=10)
