"""VaR and ES point estimators: hand values, textbook formulas, properties, input checks."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from scipy.integrate import quad

from portfolio_lab.construct.covariance import EWMA_DECAY
from portfolio_lab.risk import var as v

# The 20-value hand list: worst -10%, then -6%, then 18 values from -5% to +12%.
TWENTY = np.array([-0.10, -0.06, *np.linspace(-0.05, 0.12, 18)])
Z99 = -2.3263478740408408  # Phi^-1(0.01)
Z95 = -1.6448536269514722  # Phi^-1(0.05)

samples = st.lists(st.floats(-0.5, 0.5, allow_nan=False), min_size=1, max_size=80)
levels = st.floats(0.51, 0.999)
# Far from subnormals, where halving a number drops bits (returns are never 1e-300).
normal_samples = st.lists(
    st.floats(-0.5, 0.5).filter(lambda f: f == 0 or abs(f) > 1e-300), min_size=1, max_size=80
)


def phi(z):
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def cf_quantile(z, s, k):
    """Cornish-Fisher quantile of a standardized variable (Zangari 1996, 4th moment)."""
    return z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36


def days(n):
    return pd.bdate_range("2000-01-03", periods=n)


# --- historical: the 20-value list ------------------------------------------------------------


def test_historical_hand_list_at_95():
    # Type 7: h = (20 - 1) * 0.05 = 0.95, between the worst two: -0.10 + 0.95 * 0.04 = -0.062.
    # Only -10% is at or below -6.2%, so ES = 10%.
    risk = v.var_es(TWENTY, 0.95)
    assert risk.var == pytest.approx(0.062, abs=1e-15)
    assert risk.es == pytest.approx(0.10, abs=1e-15)


def test_historical_hand_list_at_99():
    # h = 19 * 0.01 = 0.19: -0.10 + 0.19 * 0.04 = -0.0924.
    risk = v.var_es(TWENTY, 0.99)
    assert risk.var == pytest.approx(0.0924, abs=1e-15)
    assert risk.es == pytest.approx(0.10, abs=1e-15)


def test_historical_tail_mean_over_several_values():
    # 10 values -9%..0%: at 75%, h = 9 * 0.25 = 2.25 -> -0.07 + 0.25 * 0.01 = -0.0675.
    # At or below: -9%, -8%, -7% -> ES = 8%.
    x = np.arange(-9, 1) / 100
    risk = v.var_es(x, 0.75)
    assert risk.var == pytest.approx(0.0675, abs=1e-15)
    assert risk.es == pytest.approx(0.08, abs=1e-15)


def test_historical_var_is_numpys_linear_quantile():
    x = np.random.default_rng(1).normal(0, 0.01, 500)
    for level in v.LEVELS:
        assert v.var_es(x, level).var == -np.quantile(x, 1 - level, method="linear")


def test_historical_ignores_order_and_takes_series_arrays_and_lists():
    x = np.random.default_rng(2).normal(0, 0.01, 50)
    expected = v.var_es(x, 0.95)
    shuffled = np.random.default_rng(3).permutation(x)
    for sample in (shuffled, list(x), pd.Series(x, index=days(50))):
        assert v.var_es(sample, 0.95) == expected


def test_historical_returns_plain_floats():
    risk = v.var_es(TWENTY, 0.95)
    assert type(risk.var) is float
    assert type(risk.es) is float
    var, es = risk  # unpacks like a pair
    assert (var, es) == (risk.var, risk.es)


def test_historical_constant_sample_has_a_tail():
    risk = v.var_es([0.01] * 7, 0.99)
    assert risk.var == -0.01
    assert risk.es == -0.01


def test_historical_single_value():
    assert v.var_es([-0.03], 0.95) == (0.03, 0.03)


def test_historical_tail_survives_values_one_ulp_apart():
    a = 0.1
    b = np.nextafter(a, 1.0)
    for x in ([a, b], [a, b, b], [a, a, b]):
        risk = v.var_es(x, 0.95)
        assert math.isfinite(risk.es)
        assert risk.es >= -b


def test_historical_es_never_rounds_below_var():
    # The sum of 7 copies divided by 7 lands 1 ulp above the value itself.
    c = 0.045899312196799685
    assert sum([c] * 7) / 7 > c
    risk = v.var_es([c] * 7, 0.95)
    assert risk.es == risk.var == -c


@given(samples, levels)
def test_historical_es_at_least_var_and_at_most_the_worst_loss(x, level):
    risk = v.var_es(x, level)
    worst = -min(x)
    assert risk.es >= risk.var  # exactly
    assert risk.var <= worst  # exactly: interpolation stays between neighbours
    assert risk.es <= worst + 1e-15 * max(1.0, abs(worst))  # a mean can round past its min


@given(samples)
def test_historical_higher_level_never_lowers_var_or_es(x):
    low, high = v.var_es(x, 0.95), v.var_es(x, 0.99)
    assert high.var >= low.var
    assert high.es >= low.es - 1e-15


@given(normal_samples, levels, st.integers(-8, 8))
def test_historical_power_of_two_scale_is_exact(x, level, k):
    # FHS rests on VaR(c x) = c VaR(x) for c > 0; a power of 2 makes it exact in floats.
    c = 2.0**k
    base = v.var_es(x, level)
    assert v.var_es(np.array(x) * c, level) == (base.var * c, base.es * c)


@given(normal_samples, levels, st.floats(1e-3, 1e3))
def test_historical_positive_scale_commutes(x, level, c):
    base = v.var_es(x, level)
    scaled = v.var_es(np.array(x) * c, level)
    assert scaled.var == pytest.approx(base.var * c, rel=1e-12, abs=1e-15 * c)
    assert scaled.es == pytest.approx(base.es * c, rel=1e-12, abs=1e-15 * c)


def test_historical_takes_standardized_residuals_below_minus_one():
    # FHS feeds residuals r / sigma, which can be far below -1: not a return check.
    assert v.var_es([-4.0, -1.0, 0.5, 2.0], 0.95).es == 4.0


@pytest.mark.parametrize(
    ("sample", "match"),
    [
        ([], "at least 1"),
        ([0.01, float("nan")], "finite"),
        ([0.01, float("inf")], "finite"),
        (np.zeros((3, 2)), "one-dimensional"),
        (["a", "b"], "numeric"),
    ],
)
def test_historical_rejects_bad_samples(sample, match):
    with pytest.raises(ValueError, match=match):
        v.var_es(sample, 0.95)


@pytest.mark.parametrize("level", [0.5, 1.0, 0.3, 1.5, float("nan"), 95, True])
def test_level_must_be_a_confidence_above_one_half(level):
    with pytest.raises(ValueError, match="level"):
        v.var_es(TWENTY, level)
    with pytest.raises(ValueError, match="level"):
        v.normal_var_es(0.0, 0.01, level)
    with pytest.raises(ValueError, match="level"):
        v.cornish_fisher_var_es(0.0, 0.01, 0.0, 0.0, level)


def test_levels_are_the_blueprints():
    assert v.LEVELS == (0.95, 0.99)


# --- parametric normal ----------------------------------------------------------------------


def test_normal_hand_values_at_99():
    # mu = 0.1%, sigma = 1%: VaR = -(0.001 + Z99 * 0.01), ES = -0.001 + 0.01 * phi(Z99) / 0.01.
    risk = v.normal_var_es(0.001, 0.01, 0.99)
    assert risk.var == pytest.approx(-(0.001 + Z99 * 0.01), rel=1e-14)
    assert risk.var == pytest.approx(0.022263478740408408, rel=1e-14)
    assert risk.es == pytest.approx(-0.001 + 0.01 * phi(Z99) / 0.01, rel=1e-14)
    assert risk.es == pytest.approx(0.02565214220345808, rel=1e-14)


def test_normal_hand_values_at_95():
    risk = v.normal_var_es(0.0, 0.02, 0.95)
    assert risk.var == pytest.approx(-Z95 * 0.02, rel=1e-14)
    assert risk.es == pytest.approx(0.02 * phi(Z95) / 0.05, rel=1e-14)


@pytest.mark.parametrize(("mu", "sigma"), [(0.0, 0.01), (0.002, 0.015), (-0.001, 0.03)])
@pytest.mark.parametrize("level", v.LEVELS)
def test_normal_matches_scipy(mu, sigma, level):
    p = 1 - level
    q = stats.norm.ppf(p, loc=mu, scale=sigma)
    tail_mean = stats.norm.expect(lambda x: x, loc=mu, scale=sigma, ub=q, conditional=True)
    risk = v.normal_var_es(mu, sigma, level)
    assert risk.var == pytest.approx(-q, rel=1e-12)
    assert risk.es == pytest.approx(-tail_mean, rel=1e-8)


def test_normal_sign_a_higher_mean_lowers_var_one_for_one():
    low, high = v.normal_var_es(0.0, 0.01, 0.99), v.normal_var_es(0.002, 0.01, 0.99)
    assert high.var == pytest.approx(low.var - 0.002, abs=1e-15)
    assert high.es == pytest.approx(low.es - 0.002, abs=1e-15)
    assert low.var > 0  # a loss, not a return


def test_normal_more_volatility_more_risk_and_es_above_var():
    a, b = v.normal_var_es(0.0, 0.01, 0.99), v.normal_var_es(0.0, 0.02, 0.99)
    assert b.var == pytest.approx(2 * a.var)
    assert b.es > b.var > 0


@pytest.mark.parametrize(
    ("mu", "sigma", "match"),
    [
        (0.0, 0.0, "sigma"),
        (0.0, -0.01, "sigma"),
        (0.0, float("nan"), "sigma"),
        (0.0, float("inf"), "sigma"),
        (float("nan"), 0.01, "mu"),
        (float("inf"), 0.01, "mu"),
    ],
)
def test_normal_rejects_bad_moments(mu, sigma, match):
    with pytest.raises(ValueError, match=match):
        v.normal_var_es(mu, sigma, 0.99)


# --- Cornish-Fisher -------------------------------------------------------------------------


@pytest.mark.parametrize("level", v.LEVELS)
@pytest.mark.parametrize(("mu", "sigma"), [(0.0, 0.01), (0.001, 0.02)])
def test_cornish_fisher_is_normal_without_skew_or_excess_kurtosis(mu, sigma, level):
    assert v.cornish_fisher_var_es(mu, sigma, 0.0, 0.0, level) == v.normal_var_es(mu, sigma, level)


@pytest.mark.parametrize(
    ("s", "k", "level"),
    [(-0.8, 6.0, 0.99), (-0.5, 3.0, 0.99), (0.0, 6.0, 0.95), (-0.5, 0.0, 0.99), (0.3, 2.0, 0.95)],
)
def test_cornish_fisher_var_is_the_textbook_quantile(s, k, level):
    mu, sigma = 0.0005, 0.012
    z = stats.norm.ppf(1 - level)
    risk = v.cornish_fisher_var_es(mu, sigma, s, k, level)
    assert risk.var == pytest.approx(-(mu + cf_quantile(z, s, k) * sigma), rel=1e-13)


def test_cornish_fisher_hand_value():
    # S = -0.5, K = 3 at 99%, standardized: z_cf = -3.3013 (vs -2.3263 for the normal).
    risk = v.cornish_fisher_var_es(0.0, 1.0, -0.5, 3.0, 0.99)
    assert risk.var == pytest.approx(3.301284492180553, rel=1e-13)
    assert risk.es == pytest.approx(4.469906383029829, rel=1e-9)


@pytest.mark.parametrize(
    ("s", "k", "level"),
    [(-0.8, 6.0, 0.99), (-0.5, 3.0, 0.99), (0.0, 6.0, 0.95), (1.0, 0.0, 0.99), (0.3, 2.0, 0.95)],
)
def test_cornish_fisher_es_is_the_tail_mean_of_the_cf_quantile(s, k, level):
    # ES = -(mu + sigma * (1/p) * integral_0^p z_cf(Phi^-1(u)) du), by numerical integration.
    mu, sigma, p = 0.0005, 0.012, 1 - level
    integral = quad(lambda u: cf_quantile(stats.norm.ppf(u), s, k), 0, p, limit=200)[0]
    risk = v.cornish_fisher_var_es(mu, sigma, s, k, level)
    assert risk.es == pytest.approx(-(mu + sigma * integral / p), rel=1e-9)


def test_cornish_fisher_negative_skew_and_fat_tails_raise_the_99_var():
    normal = v.normal_var_es(0.0, 0.01, 0.99).var
    assert v.cornish_fisher_var_es(0.0, 0.01, -0.5, 0.0, 0.99).var > normal
    assert v.cornish_fisher_var_es(0.0, 0.01, 0.0, 6.0, 0.99).var > normal


def test_cornish_fisher_excess_kurtosis_lowers_the_95_var():
    # A known feature of the expansion: (z^3 - 3z) > 0 at z = -1.645, so K pulls the 95%
    # quantile toward the mean (fat tails, thin shoulders), while it pushes the 99% one out.
    assert (
        v.cornish_fisher_var_es(0.0, 0.01, 0.0, 6.0, 0.95).var
        < v.normal_var_es(0.0, 0.01, 0.95).var
    )


def test_cornish_fisher_documented_failure_es_below_var():
    # Positive skew with no excess kurtosis: the CF quantile turns back up deep in the tail
    # (it is not monotone), so its tail mean is smaller than its VaR. Reported, not corrected.
    risk = v.cornish_fisher_var_es(0.0, 1.0, 1.0, 0.0, 0.99)
    assert risk.var == pytest.approx(1.2146943900254161, rel=1e-13)
    assert risk.es == pytest.approx(0.9045532651730611, rel=1e-9)
    assert risk.es < risk.var


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((0.0, 0.0, 0.0, 0.0), "sigma"),
        ((0.0, 0.01, float("nan"), 0.0), "skew"),
        ((0.0, 0.01, 0.0, float("inf")), "exkurt"),
        ((float("nan"), 0.01, 0.0, 0.0), "mu"),
    ],
)
def test_cornish_fisher_rejects_bad_moments(args, match):
    with pytest.raises(ValueError, match=match):
        v.cornish_fisher_var_es(*args, 0.99)


# --- EWMA volatility (the FHS filter) ---------------------------------------------------------


def test_ewma_vol_hand_recursion():
    # decay 0.5, seed variance 4e-4 for day 1; returns 2%, -1%, 3%:
    # day 2: 0.5 * 4e-4 + 0.5 * 0.02^2 = 4e-4; day 3: 0.5 * 4e-4 + 0.5 * 0.01^2 = 2.5e-4.
    r = pd.Series([0.02, -0.01, 0.03], index=days(3), name="P")
    vol = v.ewma_vol(r, seed_var=4e-4, decay=0.5)
    assert vol.to_numpy() == pytest.approx([0.02, 0.02, math.sqrt(2.5e-4)], rel=1e-15)
    assert vol.index.equals(r.index)
    assert vol.name == "P"


def test_ewma_vol_default_decay_is_riskmetrics():
    r = pd.Series([0.01, -0.02], index=days(2))
    vol = v.ewma_vol(r, seed_var=1e-4)
    assert vol.iloc[1] == pytest.approx(math.sqrt(EWMA_DECAY * 1e-4 + (1 - EWMA_DECAY) * 1e-4))
    r = pd.Series([0.03, 0.0], index=days(2))
    vol = v.ewma_vol(r, seed_var=1e-4)
    assert vol.iloc[1] == pytest.approx(math.sqrt(0.94 * 1e-4 + 0.06 * 9e-4), rel=1e-15)


@given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 300))
def test_ewma_vol_matches_a_plain_loop(seed, n):
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.standard_t(4, n) * 0.01, index=days(n))
    var = 2e-4
    expected = []
    for x in r:
        expected.append(math.sqrt(var))
        var = EWMA_DECAY * var + (1 - EWMA_DECAY) * x * x
    assert v.ewma_vol(r, seed_var=2e-4).to_numpy() == pytest.approx(expected, rel=1e-12)


@given(seed=st.integers(0, 2**32 - 1), k=st.integers(0, 39), shock=st.floats(-0.5, 0.5))
def test_ewma_vol_is_causal(seed, k, shock):
    # sigma(s) is the forecast for day s made at the close of s - 1: changing r(k) can't move
    # sigma(k) or anything before it, but does move sigma(k + 1).
    r = pd.Series(np.random.default_rng(seed).normal(0, 0.01, 40), index=days(40))
    base = v.ewma_vol(r, seed_var=1e-4)
    bumped = r.copy()
    bumped.iloc[k] += shock
    moved = v.ewma_vol(bumped, seed_var=1e-4)
    assert moved.iloc[: k + 1].equals(base.iloc[: k + 1])
    if k + 1 < 40 and abs(shock) > 1e-6:  # a 1e-300 shock leaves r(k) unchanged in floats
        assert moved.iloc[k + 1] != base.iloc[k + 1]


def test_ewma_vol_stays_positive_through_a_run_of_zero_returns():
    r = pd.Series(np.zeros(200), index=days(200))
    vol = v.ewma_vol(r, seed_var=1e-4)
    assert (vol > 0).all()
    assert vol.iloc[-1] == pytest.approx(0.01 * EWMA_DECAY ** (199 / 2), rel=1e-12)


def test_ewma_vol_decay_one_keeps_the_seed():
    r = pd.Series([0.05, -0.05, 0.1], index=days(3))
    assert (v.ewma_vol(r, seed_var=4e-4, decay=1.0) == 0.02).all()


@pytest.mark.parametrize("seed_var", [0.0, -1e-4, float("nan"), float("inf")])
def test_ewma_vol_rejects_a_bad_seed(seed_var):
    r = pd.Series([0.01, 0.02], index=days(2))
    with pytest.raises(ValueError, match="seed_var"):
        v.ewma_vol(r, seed_var=seed_var)


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.01, float("nan")])
def test_ewma_vol_rejects_a_bad_decay(decay):
    r = pd.Series([0.01, 0.02], index=days(2))
    with pytest.raises(ValueError, match="decay"):
        v.ewma_vol(r, seed_var=1e-4, decay=decay)


def test_ewma_vol_rejects_bad_returns():
    good = pd.Series([0.01, 0.02, 0.03], index=days(3))
    with pytest.raises(ValueError, match="finite"):
        v.ewma_vol(good.where(good != 0.02), seed_var=1e-4)
    with pytest.raises(ValueError, match="sorted"):
        v.ewma_vol(good.iloc[::-1], seed_var=1e-4)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        v.ewma_vol(good.reset_index(drop=True), seed_var=1e-4)
    with pytest.raises(TypeError, match="Series"):
        v.ewma_vol(good.to_frame(), seed_var=1e-4)
    with pytest.raises(ValueError, match="at least 1"):
        v.ewma_vol(good.iloc[:0], seed_var=1e-4)
    with pytest.raises(ValueError, match="-100%"):
        v.ewma_vol(good.where(good != 0.02, -1.5), seed_var=1e-4)


# --- horizon scaling --------------------------------------------------------------------------


def test_scale_horizon_is_square_root_of_time():
    assert v.scale_horizon(0.02) == pytest.approx(0.02 * math.sqrt(10), rel=1e-15)
    assert v.scale_horizon(0.02, days=1) == 0.02
    assert v.scale_horizon(0.02, days=4) == pytest.approx(0.04, rel=1e-15)


def test_scale_horizon_keeps_series_and_frames():
    s = pd.Series([0.01, 0.02], index=days(2), name="var")
    f = pd.DataFrame({"var": [0.01, 0.02], "es": [0.015, 0.03]}, index=days(2))
    out_s, out_f = v.scale_horizon(s, days=9), v.scale_horizon(f, days=9)
    assert isinstance(out_s, pd.Series)
    assert out_s.name == "var"
    assert out_s.to_numpy() == pytest.approx([0.03, 0.06])
    assert isinstance(out_f, pd.DataFrame)
    assert out_f.to_numpy() == pytest.approx(3 * f.to_numpy())


@pytest.mark.parametrize("bad", [0, -10, 2.5, True, "10"])
def test_scale_horizon_needs_a_positive_whole_number_of_days(bad):
    with pytest.raises(ValueError, match="days"):
        v.scale_horizon(0.02, days=bad)
