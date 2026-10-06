"""The strategy contract that Step 4's engine calls: a history in, a long-only portfolio out.

Every strategy declares its lookback, uses only the last `lookback` rows of the history it is
given, and returns weights >= 0 summing to 1, labeled by the history's columns. Each
constructor is checked against the rule it wraps, called directly on the same window, with the
estimators named here independently of the module's registry.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.construct import covariance as cv
from portfolio_lab.construct import strategy as sg
from portfolio_lab.construct import weights as wt

ESTIMATES = {
    "sample": cv.sample_cov,
    "lw_identity": lambda r: cv.ledoit_wolf_identity(r).cov,
    "lw_constant_correlation": lambda r: cv.ledoit_wolf_constant_correlation(r).cov,
    "ewma": cv.ewma_cov,
}

STRATEGIES = {
    "1/N": sg.equal_weight,
    "min_variance": lambda: sg.min_variance("lw_identity"),
    "min_variance_capped": lambda: sg.min_variance("sample", window=60, cap=0.3),
    "erc": lambda: sg.erc("lw_constant_correlation", window=126),
    "erc_ewma": lambda: sg.erc("ewma", window=60),
    "momentum": lambda: sg.momentum_tilt(strength=0.5),
    "momentum_on_min_variance": lambda: sg.momentum_tilt(
        strength=1.0, base=sg.min_variance("lw_identity", window=300)
    ),
    "low_volatility": lambda: sg.low_volatility_tilt(strength=0.5, window=126),
    "low_volatility_on_erc": lambda: sg.low_volatility_tilt(
        strength=-0.5, window=60, base=sg.erc("sample", window=126)
    ),
}
each_strategy = pytest.mark.parametrize("make", STRATEGIES.values(), ids=list(STRATEGIES))
USES_DATA = {k: m for k, m in STRATEGIES.items() if k != "1/N"}
uses_data = pytest.mark.parametrize("make", USES_DATA.values(), ids=list(USES_DATA))


def history(seed: int, t: int, n: int = 6, names=None) -> pd.DataFrame:
    """t days of correlated daily returns (vols near 1%), oldest first."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(t, n)) @ rng.normal(size=(n, n)) * 0.01 / np.sqrt(n)
    days = pd.bdate_range("2001-01-01", periods=t)
    return pd.DataFrame(values, index=days, columns=names or [f"x{i}" for i in range(n)])


def assert_portfolio(w: pd.Series, columns: pd.Index) -> None:
    assert isinstance(w, pd.Series)
    assert w.index.equals(columns)
    x = w.to_numpy()
    assert np.isfinite(x).all()
    assert (x >= 0).all()
    assert x.sum() == pytest.approx(1, abs=1e-12)


def same(a: pd.Series, b: pd.Series) -> None:
    pd.testing.assert_series_equal(a, b, check_exact=True)


def fixed(weights: pd.Series | np.ndarray, lookback: int = 5) -> sg.Strategy:
    """A strategy whose rule ignores the window and returns `weights`."""
    return sg.Strategy("fixed", lookback, lambda window: weights)


# --- The contract, for every constructor ---


@each_strategy
def test_weights_are_a_long_only_portfolio_labeled_by_the_history(make):
    s = make()
    h = history(1, s.lookback + 20, names=[5, 3, 9, 1, 7, 2])
    assert_portfolio(s.weights(h), h.columns)


@each_strategy
@pytest.mark.parametrize("older", ["nan", "noise"])
def test_weights_depend_only_on_the_last_lookback_rows(make, older):
    s = make()
    h = history(2, s.lookback + 40)
    changed = h.copy()
    rng = np.random.default_rng(3)
    changed.iloc[:40] = np.nan if older == "nan" else rng.normal(scale=0.05, size=(40, 6))
    same(s.weights(changed), s.weights(h))


@uses_data
def test_the_oldest_row_of_the_lookback_matters(make):
    s = make()
    h = history(4, s.lookback + 10)
    changed = h.copy()
    # A big day for every asset at the start of the window (one asset alone might sit at zero
    # weight, where its own variance cannot move a sample-matrix minimum-variance portfolio).
    changed.iloc[-s.lookback] += np.linspace(-0.1, 0.2, 6)
    assert not np.allclose(s.weights(changed), s.weights(h), rtol=0, atol=1e-6)


@each_strategy
def test_a_history_shorter_than_the_lookback_raises(make):
    s = make()
    assert_portfolio(s.weights(history(5, s.lookback)), history(5, 1).columns)
    with pytest.raises(ValueError, match=f"at least {s.lookback} periods"):
        s.weights(history(5, s.lookback - 1))


@each_strategy
def test_a_missing_return_in_the_window_raises(make):
    s = make()
    h = history(6, s.lookback + 5)
    h.iloc[-1, 2] = np.nan
    with pytest.raises(ValueError, match="missing for x2"):
        s.weights(h)


def test_strategy_names_are_unique():
    names = [make().name for make in STRATEGIES.values()]
    assert len(set(names)) == len(names)


# --- Each constructor wraps its rule ---


def test_equal_weight_strategy():
    s = sg.equal_weight()
    assert (s.name, s.lookback) == ("1/N", 1)
    h = history(7, 30)
    same(s.weights(h), wt.equal_weight(h.columns))


def test_the_registered_estimators():
    assert set(sg.ESTIMATORS) == set(ESTIMATES)


@pytest.mark.parametrize("estimator", list(ESTIMATES))
def test_min_variance_strategy_runs_the_rule_on_the_estimate_of_its_window(estimator):
    s = sg.min_variance(estimator, window=120, cap=0.4)
    assert s.lookback == 120
    h = history(8, 200)
    same(s.weights(h), wt.min_variance(ESTIMATES[estimator](h.iloc[-120:]), cap=0.4))


@pytest.mark.parametrize("estimator", list(ESTIMATES))
def test_erc_strategy_runs_the_rule_on_the_estimate_of_its_window(estimator):
    s = sg.erc(estimator, window=90)
    assert s.lookback == 90
    h = history(9, 150)
    same(s.weights(h), wt.equal_risk_contribution(ESTIMATES[estimator](h.iloc[-90:])))


def test_the_default_covariance_window_is_252_days():
    assert sg.WINDOW == 252
    assert sg.min_variance("sample").lookback == sg.erc("sample").lookback == 252
    assert sg.low_volatility_tilt(strength=1.0).lookback == 252


def test_the_minimum_variance_cap_defaults_to_none():
    h = history(10, 260)
    same(sg.min_variance("sample").weights(h), wt.min_variance(cv.sample_cov(h.iloc[-252:])))


def test_momentum_tilt_of_equal_weights():
    s = sg.momentum_tilt(strength=0.7)
    assert s.lookback == 252
    h = history(11, 300)
    same(s.weights(h), wt.tilt(wt.momentum_12_1(h.iloc[-252:]), strength=0.7))


@pytest.mark.parametrize(("window", "lookback"), [(126, 252), (252, 252), (300, 300)])
def test_momentum_tilt_of_a_base_strategy_looks_back_as_far_as_either_needs(window, lookback):
    base = sg.min_variance("lw_identity", window=window)
    s = sg.momentum_tilt(strength=0.7, base=base)
    assert s.lookback == lookback
    h = history(12, 320)
    expected = wt.tilt(wt.momentum_12_1(h.iloc[-252:]), strength=0.7, base=base.weights(h))
    same(s.weights(h), expected)


def test_low_volatility_tilt_of_equal_weights():
    s = sg.low_volatility_tilt(strength=0.5, window=60)
    assert s.lookback == 60
    h = history(13, 100)
    same(s.weights(h), wt.tilt(wt.low_volatility(h.iloc[-60:]), strength=0.5))


@pytest.mark.parametrize(("window", "lookback"), [(60, 126), (200, 200)])
def test_low_volatility_tilt_of_a_base_strategy_looks_back_as_far_as_either_needs(window, lookback):
    base = sg.erc("sample", window=126)
    s = sg.low_volatility_tilt(strength=0.5, window=window, base=base)
    assert s.lookback == lookback
    h = history(14, 260)
    expected = wt.tilt(wt.low_volatility(h.iloc[-window:]), strength=0.5, base=base.weights(h))
    same(s.weights(h), expected)


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (lambda: sg.min_variance("lw_identity"), "min variance (lw_identity, 252d)"),
        (
            lambda: sg.min_variance("sample", window=60, cap=0.3),
            "min variance (sample, 60d, cap 0.3)",
        ),
        (lambda: sg.erc("ewma"), "ERC (ewma, 252d)"),
        (lambda: sg.momentum_tilt(strength=0.5), "momentum tilt (strength 0.5) on 1/N"),
        (
            lambda: sg.low_volatility_tilt(strength=1, window=126, base=sg.erc("sample")),
            "low-vol tilt (strength 1, 126d) on ERC (sample, 252d)",
        ),
    ],
)
def test_names_say_what_the_strategy_does(make, name):
    assert make().name == name


# --- Constructor inputs ---


@pytest.mark.parametrize("make", [sg.min_variance, sg.erc], ids=["min_variance", "erc"])
def test_an_unknown_estimator_raises(make):
    with pytest.raises(ValueError, match="unknown estimator 'ledoit'.*lw_identity"):
        make("ledoit")


@pytest.mark.parametrize("window", [0, -5, 2.5, True])
@pytest.mark.parametrize(
    "make",
    [
        lambda window: sg.min_variance("sample", window=window),
        lambda window: sg.erc("sample", window=window),
        lambda window: sg.low_volatility_tilt(strength=1.0, window=window),
    ],
    ids=["min_variance", "erc", "low_volatility_tilt"],
)
def test_the_window_must_be_a_positive_integer(make, window):
    # Checked up front: a tilt's lookback also covers its base, so it can hide a bad window.
    with pytest.raises(ValueError, match="window must be a positive integer"):
        make(window)


@pytest.mark.parametrize(
    "make",
    [sg.min_variance, sg.erc, sg.momentum_tilt, sg.low_volatility_tilt],
    ids=["min_variance", "erc", "momentum_tilt", "low_volatility_tilt"],
)
def test_estimator_and_strength_have_no_defaults(make):
    with pytest.raises(TypeError):
        make()


# --- The Strategy class ---


def test_the_rule_gets_exactly_the_last_lookback_rows():
    seen = []

    def rule(window: pd.DataFrame) -> pd.Series:
        seen.append(window)
        return wt.equal_weight(window.columns)

    h = history(15, 30)
    sg.Strategy("recorder", 5, rule).weights(h)
    pd.testing.assert_frame_equal(seen[0], h.iloc[-5:], check_exact=True)


@pytest.mark.parametrize("lookback", [0, -3, 2.5, True, "252", None])
def test_the_lookback_must_be_a_positive_integer(lookback):
    with pytest.raises(ValueError, match="lookback"):
        sg.Strategy("x", lookback, wt.equal_weight)


def test_a_numpy_integer_lookback_is_fine():
    assert sg.Strategy("x", np.int64(5), lambda w: wt.equal_weight(w.columns)).lookback == 5


@pytest.mark.parametrize("name", ["", "   ", 5, None])
def test_the_name_must_be_text(name):
    with pytest.raises(ValueError, match="name"):
        sg.Strategy(name, 5, wt.equal_weight)


def test_the_rule_must_be_callable():
    with pytest.raises(TypeError, match="rule"):
        sg.Strategy("x", 5, "min variance")


def test_the_history_must_be_a_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        sg.equal_weight().weights(np.zeros((10, 3)))


@pytest.mark.parametrize(
    "dates",
    [
        pd.bdate_range("2001-01-01", periods=30)[::-1],
        pd.DatetimeIndex(["2000-12-29"] * 2).append(pd.bdate_range("2001-01-01", periods=28)),
    ],
    ids=["reversed", "repeated-before-the-window"],
)
def test_the_whole_history_must_be_in_date_order(dates):
    h = history(16, 30).set_axis(dates)
    with pytest.raises(ValueError, match="sorted|unique"):
        fixed(wt.equal_weight(h.columns)).weights(h)


H = history(17, 10)
COLUMNS = H.columns


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (pd.Series(1 / 5, index=COLUMNS[:5]), "same assets"),
        (pd.Series(1 / 6, index=COLUMNS[::-1]), "same assets"),
        (pd.Series([0.5, 0.3, 0.3, 0, 0, -0.1], index=COLUMNS), "negative weight for x5"),
        (pd.Series([0.5, 0.5, np.nan, 0, 0, 0], index=COLUMNS), "finite"),
        (pd.Series(0.15, index=COLUMNS), "sum to 1"),
    ],
    ids=["fewer-assets", "other-order", "negative", "nan", "under-invested"],
)
def test_a_rule_that_returns_an_invalid_portfolio_raises(weights, match):
    with pytest.raises(ValueError, match=f"fixed.*{match}"):
        fixed(weights).weights(H)


def test_a_rule_that_returns_no_series_raises():
    with pytest.raises(TypeError, match="fixed.*Series"):
        fixed(np.full(6, 1 / 6)).weights(H)
