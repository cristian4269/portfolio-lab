"""Signals and tilts: momentum 12-1, low volatility, and tilting a base portfolio toward a signal.

Momentum is pinned down at both ends of its window by planted moves (including big moves in the
last 21 days, which it must ignore) and by its length (231 returns). The tilt is checked against
a hand example, its defining identity (log(w_i / b_i) - strength * z_i is the same for every
asset) and its limits: strength 0, equal signals, very large strengths.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from portfolio_lab.construct import covariance as cv
from portfolio_lab.construct import weights as wt

seeds = st.integers(0, 2**32 - 1)


def flat(t: int, names: str = "abc") -> pd.DataFrame:
    """t days of zero returns."""
    days = pd.bdate_range("2001-01-01", periods=t)
    return pd.DataFrame(0.0, index=days, columns=list(names))


def random_returns(seed: int, t: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(t, n)) @ rng.normal(size=(n, n)) * 0.01 / math.sqrt(n)
    days = pd.bdate_range("2001-01-01", periods=t)
    return pd.DataFrame(values, index=days, columns=[f"x{i}" for i in range(n)])


def series(values, names: str = "abcdefghij") -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=list(names[: len(values)]))


def random_case(seed: int, n: int) -> tuple[pd.Series, pd.Series]:
    """A random signal and a random fully invested base portfolio."""
    rng = np.random.default_rng(seed)
    names = [f"x{i}" for i in range(n)]
    base = rng.dirichlet(np.ones(n))
    return pd.Series(rng.normal(size=n), index=names), pd.Series(base / base.sum(), index=names)


def zscore(signal: pd.Series) -> np.ndarray:
    """Cross-sectional z-score with the population standard deviation (divides by N)."""
    s = signal.to_numpy()
    return (s - s.mean()) / s.std()


def close(actual, expected, atol: float = 1e-15) -> None:
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=0, atol=atol)


def assert_portfolio(w: pd.Series, index: pd.Index) -> None:
    assert isinstance(w, pd.Series)
    assert w.index.equals(index)
    x = w.to_numpy()
    assert np.isfinite(x).all()
    assert (x >= 0).all()
    assert x.sum() == pytest.approx(1, abs=1e-12)


# --- Momentum 12-1 ---


def test_momentum_hand_example():
    r = flat(260)
    r.iloc[-252, 0] = 0.10  # a: the first day of the window counts...
    r.iloc[-1, 0] = -0.50  # ...the last day of the history does not
    r.iloc[-253, 1] = -0.50  # b: the day before the window does not count...
    r.iloc[-22, 1] = 1.00  # ...the last day of the formation period does
    r.iloc[-100, 2] = 0.20  # c: 1.2 * 0.75 - 1 = -0.1...
    r.iloc[-50, 2] = -0.25
    r.iloc[-21, 2] = 3.00  # ...and a +300% day in the skipped month is ignored
    m = wt.momentum_12_1(r)
    assert list(m.index) == ["a", "b", "c"]
    close(m, [0.10, 1.00, -0.10])


def test_momentum_compounds_exactly_231_returns():
    r = flat(252) + 0.001
    close(wt.momentum_12_1(r), [1.001**231 - 1] * 3, atol=1e-13)


def test_momentum_uses_the_constants_it_documents():
    assert (wt.MOMENTUM_LOOKBACK, wt.MOMENTUM_SKIP) == (252, 21)


def test_momentum_needs_252_days():
    wt.momentum_12_1(flat(252))
    with pytest.raises(ValueError, match="at least 252"):
        wt.momentum_12_1(flat(251))


def test_momentum_ignores_rows_before_its_window():
    r = flat(260)
    r.iloc[:8] = np.nan
    close(wt.momentum_12_1(r), [0.0] * 3, atol=0)


def test_momentum_rejects_missing_returns_in_its_window():
    r = flat(252)
    r.iloc[-1, 1] = np.nan  # even in the skipped month: windows are complete
    with pytest.raises(ValueError, match="missing for b"):
        wt.momentum_12_1(r)


def test_momentum_accepts_a_total_loss_and_rejects_anything_worse():
    r = flat(252)
    r.iloc[-30, 0] = -1.0
    close(wt.momentum_12_1(r), [-1.0, 0.0, 0.0], atol=0)
    r.iloc[-30, 0] = -1.5
    with pytest.raises(ValueError, match="below -100%"):
        wt.momentum_12_1(r)


def test_momentum_needs_a_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        wt.momentum_12_1(np.zeros((252, 3)))


# --- Low volatility ---


def test_low_volatility_hand_example():
    # The covariance module's hand example: sd 3%, 1% and 1% (demeaned, divided by T).
    days = pd.bdate_range("2001-01-01", periods=4)
    pct = [[6, 1, 3], [0, 1, 1], [0, -1, 3], [-2, -1, 1]]
    r = pd.DataFrame(np.array(pct) / 100, index=days, columns=list("abc"))
    signal = wt.low_volatility(r)
    assert list(signal.index) == ["a", "b", "c"]
    close(signal, [-0.03, -0.01, -0.01], atol=1e-16)


@given(seeds, st.integers(2, 60), st.integers(1, 8))
def test_low_volatility_is_minus_the_sample_volatility(seed, t, n):
    r = random_returns(seed, t, n)
    expected = -np.sqrt(np.diag(cv.sample_cov(r)))
    np.testing.assert_allclose(wt.low_volatility(r), expected, rtol=1e-12, atol=0)


def test_low_volatility_needs_two_periods():
    with pytest.raises(ValueError, match="at least 2"):
        wt.low_volatility(flat(1))


# --- Tilt: hand examples and limits ---

# For the signal (1, 2, 3), z = (-sqrt(3/2), 0, sqrt(3/2)), so exp(strength * z) = (1/2, 1, 2).
LN2 = math.log(2) * math.sqrt(2 / 3)
SIGNAL = series([1, 2, 3])


def test_tilt_hand_example():
    close(wt.tilt(SIGNAL, strength=LN2), [1 / 7, 2 / 7, 4 / 7])


def test_tilt_of_a_given_base_hand_example():
    base = series([1 / 2, 1 / 4, 1 / 4])
    close(wt.tilt(SIGNAL, strength=LN2, base=base), [1 / 4, 1 / 4, 1 / 2])


def test_the_default_base_is_equal_weight():
    w = wt.tilt(SIGNAL, strength=0.3)
    expected = wt.tilt(SIGNAL, strength=0.3, base=wt.equal_weight(SIGNAL.index))
    pd.testing.assert_series_equal(w, expected, check_exact=True)


@pytest.mark.parametrize("base", [None, series([0.5, 0.2, 0.3])], ids=["equal", "given"])
def test_strength_zero_returns_the_base(base):
    expected = [1 / 3] * 3 if base is None else base
    close(wt.tilt(SIGNAL, strength=0.0, base=base), expected, atol=1e-16)


@pytest.mark.parametrize(
    "signal",
    [series([0.2, 0.2, 0.2]), series([0.0, 0.0, 0.0]), series([0.3, 0.1 + 0.2, 0.3])],
    ids=["equal", "zero", "equal-up-to-rounding"],
)
def test_equal_signals_return_the_base(signal):
    base = series([0.5, 0.2, 0.3])
    close(wt.tilt(signal, strength=5.0, base=base), base, atol=1e-16)


def test_tiny_signals_that_differ_still_tilt():
    tiny = series([1e-20, 2e-20, 3e-20])  # equal only in absolute terms
    close(wt.tilt(tiny, strength=LN2), [1 / 7, 2 / 7, 4 / 7])


def test_a_huge_strength_puts_everything_on_the_best_signal_without_overflow():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        close(wt.tilt(SIGNAL, strength=1e6), [0, 0, 1], atol=0)
        close(wt.tilt(series([1, 3, 3]), strength=1e6), [0, 0.5, 0.5], atol=1e-16)
        close(wt.tilt(SIGNAL, strength=-1e6), [1, 0, 0], atol=0)


def test_an_asset_outside_the_base_stays_out():
    base = series([0.5, 0.5, 0.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        close(wt.tilt(SIGNAL, strength=LN2, base=base), [1 / 3, 2 / 3, 0], atol=1e-16)
        close(wt.tilt(SIGNAL, strength=1e6, base=base), [0, 1, 0], atol=0)


def test_a_single_asset_gets_everything():
    close(wt.tilt(series([0.7]), strength=2.0), [1.0], atol=0)


def test_labels_follow_the_signal():
    signal = pd.Series([3.0, 1.0, 2.0], index=[30, 10, 20])
    w = wt.tilt(signal, strength=1.0)
    assert list(w.index) == [30, 10, 20]
    assert w[30] > w[20] > w[10]


# --- Tilt: properties ---


@given(seeds, st.integers(2, 12), st.floats(-3, 3))
def test_tilt_is_exponential_in_the_zscore(seed, n, strength):
    signal, base = random_case(seed, n)
    w = wt.tilt(signal, strength=strength, base=base)
    assert_portfolio(w, signal.index)
    excess = np.log(w.to_numpy() / base.to_numpy()) - strength * zscore(signal)
    assert np.ptp(excess) < 1e-12


@given(seeds, st.integers(2, 12), st.floats(0.01, 3))
def test_with_equal_base_weights_a_higher_signal_never_gets_less(seed, n, strength):
    signal, _ = random_case(seed, n)
    w = wt.tilt(signal, strength=strength)
    order = np.argsort(signal.to_numpy())
    assert (np.diff(w.to_numpy()[order]) >= 0).all()


@given(seeds, st.integers(2, 12), st.floats(1e-3, 1e3), st.floats(-10, 10))
def test_tilt_ignores_the_signals_location_and_scale(seed, n, scale, shift):
    signal, base = random_case(seed, n)
    w = wt.tilt(signal, strength=1.5, base=base)
    close(wt.tilt(scale * signal + shift, strength=1.5, base=base), w, atol=1e-12)


@given(seeds, st.integers(2, 12), st.floats(0, 3))
def test_a_negative_strength_tilts_toward_the_opposite_signal(seed, n, strength):
    signal, base = random_case(seed, n)
    w = wt.tilt(signal, strength=-strength, base=base)
    close(w, wt.tilt(-signal, strength=strength, base=base), atol=1e-15)


# --- Tilt: inputs ---


def test_strength_is_required_and_keyword_only():
    with pytest.raises(TypeError):
        wt.tilt(SIGNAL)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        wt.tilt(SIGNAL, 1.0)  # type: ignore[misc]


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -float("inf")])
def test_a_strength_that_is_not_finite_raises(strength):
    with pytest.raises(ValueError, match="strength"):
        wt.tilt(SIGNAL, strength=strength)


@pytest.mark.parametrize(
    ("signal", "match"),
    [
        (pd.Series([], dtype=float), "no assets"),
        (pd.Series([1.0, 2.0], index=["a", "a"]), "unique"),
        (series([1, np.nan, 3]), "finite"),
        (series([1, np.inf, 3]), "finite"),
        (pd.Series(["x", "y"], index=["a", "b"]), "numeric"),
    ],
    ids=["empty", "repeated", "nan", "inf", "text"],
)
def test_bad_signals_raise(signal, match):
    with pytest.raises(ValueError, match=match):
        wt.tilt(signal, strength=1.0)


def test_a_signal_must_be_a_series():
    with pytest.raises(TypeError, match="Series"):
        wt.tilt(np.array([1.0, 2.0, 3.0]), strength=1.0)


@pytest.mark.parametrize(
    ("base", "match"),
    [
        (series([0.5, 0.5]), "same assets"),
        (pd.Series([0.2, 0.3, 0.5], index=list("acb")), "same assets"),
        (pd.Series([0.2, 0.3, 0.5], index=list("abd")), "same assets"),
        (series([0.5, 0.6, -0.1]), "negative weight for c"),
        (series([0.5, np.nan, 0.5]), "finite"),
        (series([0.3, 0.3, 0.3]), "sum to 1"),
        (series([0.0, 0.0, 0.0]), "sum to 1"),
    ],
    ids=["shorter", "other-order", "other-assets", "negative", "nan", "under-invested", "zero"],
)
def test_bad_bases_raise(base, match):
    with pytest.raises(ValueError, match=match):
        wt.tilt(SIGNAL, strength=1.0, base=base)


def test_a_base_must_be_a_series():
    with pytest.raises(TypeError, match="Series"):
        wt.tilt(SIGNAL, strength=1.0, base=np.array([0.2, 0.3, 0.5]))
