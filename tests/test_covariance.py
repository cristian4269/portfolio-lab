"""Covariance estimators: hand examples, scikit-learn, an independent implementation, properties.

The constant-correlation estimator has no library oracle, so it is checked five ways: the
docstring's hand example redone in exact fractions; PyPortfolioOpt's implementation (vendored in
`_reference_pypfopt.py`) given the sample matrix it uses; a loop-by-loop transcription of the
paper's appendix; properties; and a mutation check run by hand (see NOTES.md).
"""

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from _reference_pypfopt import pypfopt_ledoit_wolf_constant_correlation
from hypothesis import given
from hypothesis import strategies as st
from sklearn.covariance import LedoitWolf

from portfolio_lab.construct import covariance as cv

PCT2 = 1e-4  # (1%)^2: the docstring's hand example is in percent units
DAYS = pd.bdate_range("2000-01-03", periods=4)
# The docstring's hand example in decimals; a has a mean of 1% and c of 2%, for demeaning to remove.
HAND = pd.DataFrame(
    {
        "a": [0.06, 0.00, 0.00, -0.02],
        "b": [0.01, 0.01, -0.01, -0.01],
        "c": [0.03, 0.01, 0.03, 0.01],
    },
    index=DAYS,
)
S_HAND = np.array([[9, 2, 2], [2, 1, 0], [2, 0, 1]])  # PCT2 units; sd 3, 1, 1
F_HAND = np.array([[9, 4 / 3, 4 / 3], [4 / 3, 1, 4 / 9], [4 / 3, 4 / 9, 1]])  # rbar = 4/9

ESTIMATORS = {
    "sample": cv.sample_cov,
    "identity": lambda r: cv.ledoit_wolf_identity(r).cov,
    "constant_correlation": lambda r: cv.ledoit_wolf_constant_correlation(r).cov,
    "ewma": cv.ewma_cov,
}
each_estimator = pytest.mark.parametrize("estimate", ESTIMATORS.values(), ids=list(ESTIMATORS))
seeds = st.integers(0, 2**32 - 1)


def twice(returns: pd.DataFrame) -> pd.DataFrame:
    """The same periods twice, on new dates: T doubles and every average stays the same."""
    doubled = pd.concat([returns, returns])
    doubled.index = pd.bdate_range("2000-01-03", periods=len(doubled))
    return doubled


def random_returns(seed: int, t: int, n: int) -> pd.DataFrame:
    """Correlated returns of about 1% a period, with different means."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(t, n)) @ rng.normal(size=(n, n)) * 0.01 + rng.normal(0, 0.001, n)
    days = pd.bdate_range("2001-01-01", periods=t)
    return pd.DataFrame(values, index=days, columns=[f"x{i}" for i in range(n)])


def with_sample_cov(cov: np.ndarray, t: int, seed: int = 0) -> pd.DataFrame:
    """Returns whose sample matrix (demeaned, divided by T) is `cov`, up to rounding."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(t, len(cov)))
    q, _ = np.linalg.qr(z - z.mean(axis=0))  # orthonormal columns, orthogonal to a constant
    x = math.sqrt(t) * q @ np.linalg.cholesky(cov).T
    return pd.DataFrame(x + 0.0005, index=pd.bdate_range("2001-01-01", periods=t))


def close(actual, expected, rtol: float = 1e-12) -> None:
    """Matrices equal to `rtol`, relative to the largest expected entry."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    atol = rtol * np.abs(expected).max()
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def exact_sqrt(q: Fraction) -> Fraction:
    root = Fraction(math.isqrt(q.numerator), math.isqrt(q.denominator))
    assert root**2 == q, f"{q} is not the square of a fraction"
    return root


def paper_lw_cc(rows, sqrt):
    """Ledoit and Wolf (2004, JPM) appendix, loop by loop; rows[t][i] = asset i in period t.

    Works on floats (sqrt=math.sqrt) or exact fractions (sqrt=exact_sqrt).
    """
    T, N = len(rows), len(rows[0])
    mean = [sum(row[i] for row in rows) / T for i in range(N)]
    x = [[row[i] - mean[i] for i in range(N)] for row in rows]
    s = [[sum(xt[i] * xt[j] for xt in x) / T for j in range(N)] for i in range(N)]
    pairs = [(i, j) for i in range(N - 1) for j in range(i + 1, N)]
    rbar = 2 * sum(s[i][j] / sqrt(s[i][i] * s[j][j]) for i, j in pairs) / ((N - 1) * N)
    f = [
        [s[i][i] if i == j else rbar * sqrt(s[i][i] * s[j][j]) for j in range(N)] for i in range(N)
    ]
    pi = [[sum((xt[i] * xt[j] - s[i][j]) ** 2 for xt in x) / T for j in range(N)] for i in range(N)]

    def theta(k, i, j):  # theta_kk,ij
        return sum((xt[k] ** 2 - s[k][k]) * (xt[i] * xt[j] - s[i][j]) for xt in x) / T

    rho = sum(pi[i][i] for i in range(N)) + sum(
        rbar
        / 2
        * (sqrt(s[j][j] / s[i][i]) * theta(i, i, j) + sqrt(s[i][i] / s[j][j]) * theta(j, i, j))
        for i in range(N)
        for j in range(N)
        if j != i
    )
    gamma = sum((f[i][j] - s[i][j]) ** 2 for i in range(N) for j in range(N))
    pi_total = sum(sum(row) for row in pi)
    kappa = (pi_total - rho) / gamma
    return {
        "s": s,
        "rbar": rbar,
        "target": f,
        "pi": pi_total,
        "rho": rho,
        "gamma": gamma,
        "kappa": kappa,
        "delta": max(0, min(1, kappa / T)),
    }


# --- Hand examples (module docstring) ---


def test_sample_cov_hand_example():
    close(cv.sample_cov(HAND), S_HAND * PCT2)


def test_sample_cov_divides_by_t():
    returns = random_returns(1, 30, 4)
    close(cv.sample_cov(returns), np.cov(returns.to_numpy(), rowvar=False, ddof=0))


def test_identity_hand_example():
    shrunk = cv.ledoit_wolf_identity(HAND)
    delta, mu = 177 / 352, 11 / 3
    assert shrunk.intensity == pytest.approx(delta, rel=1e-12)
    assert LedoitWolf().fit(HAND.to_numpy()).shrinkage_ == pytest.approx(delta, rel=1e-12)
    close(shrunk.target, mu * np.eye(3) * PCT2)
    close(shrunk.cov, (delta * mu * np.eye(3) + (1 - delta) * S_HAND) * PCT2)


def test_identity_hand_example_with_every_period_twice():
    assert cv.ledoit_wolf_identity(twice(HAND)).intensity == pytest.approx(177 / 704, rel=1e-12)


def test_constant_correlation_hand_example_is_clipped_at_full_shrinkage():
    shrunk = cv.ledoit_wolf_constant_correlation(HAND)  # kappa / T = 651/352 > 1
    assert shrunk.intensity == 1
    close(shrunk.target, F_HAND * PCT2)
    close(shrunk.cov, F_HAND * PCT2)


def test_constant_correlation_hand_example_with_every_period_twice():
    shrunk = cv.ledoit_wolf_constant_correlation(twice(HAND))
    delta = 651 / 704
    assert shrunk.intensity == pytest.approx(delta, rel=1e-12)
    expected = np.array(
        [[9, 487 / 352, 487 / 352], [487 / 352, 1, 217 / 528], [487 / 352, 217 / 528, 1]]
    )
    close(shrunk.cov, expected * PCT2)
    close(shrunk.cov, (delta * F_HAND + (1 - delta) * S_HAND) * PCT2)


def test_hand_example_in_exact_fractions():
    pct = [[Fraction(round(v * 100)) for v in row] for row in HAND.to_numpy()]  # whole percents
    paper = paper_lw_cc(pct, exact_sqrt)
    assert paper["s"] == S_HAND.tolist()
    assert paper["rbar"] == Fraction(4, 9)
    assert paper["gamma"] == Fraction(176, 81)
    assert paper["pi"] == 118
    assert paper["rho"] == Fraction(2752, 27)
    assert paper["kappa"] == Fraction(651, 88)
    assert paper["delta"] == 1
    doubled = paper_lw_cc(pct + pct, exact_sqrt)
    assert doubled["kappa"] == Fraction(651, 88)
    assert doubled["delta"] == Fraction(651, 704)


# --- Oracles ---


@pytest.mark.parametrize(
    ("t", "n", "seed"),
    [(50, 5, 1), (10, 20, 2), (252, 12, 3), (2, 3, 4)],
    ids=["T>N", "T<N", "daily-year", "T=2"],
)
def test_identity_matches_scikit_learn(t, n, seed):
    returns = random_returns(seed, t, n)
    ours = cv.ledoit_wolf_identity(returns)
    theirs = LedoitWolf().fit(returns.to_numpy())
    assert ours.intensity == pytest.approx(theirs.shrinkage_, abs=1e-10)
    close(ours.cov, theirs.covariance_, rtol=1e-10)


@pytest.mark.parametrize(("t", "n", "seed"), [(30, 3, 30), (60, 5, 31), (12, 4, 24)])
def test_constant_correlation_matches_the_paper_loop_by_loop(t, n, seed):
    returns = random_returns(seed, t, n)
    paper = paper_lw_cc(returns.to_numpy().tolist(), math.sqrt)
    shrunk = cv.ledoit_wolf_constant_correlation(returns)
    assert 0 < paper["delta"] < 1  # an interior intensity: nothing hidden by clipping
    assert shrunk.intensity == pytest.approx(paper["delta"], rel=1e-10)
    close(shrunk.target, paper["target"])
    close(cv.sample_cov(returns), paper["s"])


@pytest.mark.parametrize(("t", "n", "seed"), [(40, 4, 20), (100, 6, 21), (8, 5, 22), (252, 12, 23)])
def test_constant_correlation_matches_pyportfolioopt_given_its_sample_matrix(t, n, seed):
    """PyPortfolioOpt divides its sample matrix by T - 1 but averages pi and theta over T.

    The internal `_lw_cc` takes the sample matrix as an argument, so it can be given the same one.
    """
    values = random_returns(seed, t, n).to_numpy()
    s = np.cov(values, rowvar=False, ddof=1)
    theirs, their_delta = pypfopt_ledoit_wolf_constant_correlation(values, s)
    target, delta = cv._lw_cc(values - values.mean(axis=0), s)
    assert 0 < their_delta < 1
    assert delta == pytest.approx(their_delta, rel=1e-12)
    close(delta * target + (1 - delta) * s, theirs)


# --- Properties ---


@given(seeds, st.integers(2, 40), st.integers(1, 6))
def test_shrinkage_blends_target_and_sample_into_a_valid_matrix(seed, t, n):
    returns = random_returns(seed, t, n)
    sample = cv.sample_cov(returns)
    estimates = [cv.ledoit_wolf_identity(returns)]
    if n >= 2:
        estimates.append(cv.ledoit_wolf_constant_correlation(returns))
    for shrunk in estimates:
        d = shrunk.intensity
        assert 0 <= d <= 1
        close(shrunk.cov, d * shrunk.target + (1 - d) * sample)
        cov = shrunk.cov.to_numpy()
        np.testing.assert_array_equal(cov, cov.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert eigenvalues.min() >= -1e-12 * eigenvalues.max()


@given(seeds, st.integers(2, 40), st.integers(2, 6))
def test_the_constant_correlation_target_keeps_variances_and_averages_correlations(seed, t, n):
    returns = random_returns(seed, t, n)
    s = cv.sample_cov(returns).to_numpy()
    f = cv.ledoit_wolf_constant_correlation(returns).target.to_numpy()
    sd = np.sqrt(np.diag(s))
    off = ~np.eye(n, dtype=bool)
    np.testing.assert_array_equal(np.diag(f), np.diag(s))
    rbar = (s / np.outer(sd, sd))[off].mean()
    np.testing.assert_allclose((f / np.outer(sd, sd))[off], rbar, rtol=1e-12, atol=1e-12)


@given(seeds, st.integers(2, 40), st.integers(2, 6))
def test_reordering_the_assets_reorders_the_estimates(seed, t, n):
    returns = random_returns(seed, t, n)
    order = list(returns.columns[np.random.default_rng(seed).permutation(n)])
    for estimate in (cv.ledoit_wolf_identity, cv.ledoit_wolf_constant_correlation):
        base, moved = estimate(returns), estimate(returns[order])
        assert moved.intensity == pytest.approx(base.intensity, rel=1e-9, abs=1e-12)
        close(moved.cov, base.cov.loc[order, order], rtol=1e-9)
        assert list(moved.cov.index) == order
        assert list(moved.cov.columns) == order


@given(seeds, st.integers(2, 40), st.integers(2, 6), st.floats(0.01, 100))
def test_rescaling_the_returns_leaves_the_intensities_unchanged(seed, t, n, c):
    returns = random_returns(seed, t, n)
    for estimate in (cv.ledoit_wolf_identity, cv.ledoit_wolf_constant_correlation):
        base, scaled = estimate(returns), estimate(returns * c)
        assert scaled.intensity == pytest.approx(base.intensity, rel=1e-9, abs=1e-12)
        close(scaled.cov, base.cov * c**2, rtol=1e-9)


@given(seeds, st.integers(2, 6), st.integers(0, 30))
def test_identity_shrinkage_never_worsens_the_condition_number(seed, n, extra):
    returns = random_returns(seed, n + 2 + extra, n)
    shrunk = cv.ledoit_wolf_identity(returns)
    assert np.linalg.cond(shrunk.cov) <= np.linalg.cond(cv.sample_cov(returns)) * (1 + 1e-9)


def test_identity_shrinkage_makes_a_singular_sample_invertible():
    returns = random_returns(5, 10, 20)  # T < N: the sample matrix has rank 9
    assert np.linalg.matrix_rank(cv.sample_cov(returns).to_numpy()) < 20
    shrunk = cv.ledoit_wolf_identity(returns)
    assert shrunk.intensity > 0
    assert np.linalg.eigvalsh(shrunk.cov.to_numpy()).min() > 0


@given(seeds, st.integers(2, 40))
def test_two_assets_are_their_own_constant_correlation_target(seed, t):
    returns = random_returns(seed, t, 2)
    shrunk = cv.ledoit_wolf_constant_correlation(returns)
    assert shrunk.intensity == 0
    pd.testing.assert_frame_equal(shrunk.cov, cv.sample_cov(returns))


def test_equal_sample_correlations_mean_no_constant_correlation_shrinkage():
    sd = np.array([0.01, 0.02, 0.015, 0.03])
    corr = np.full((4, 4), 0.3)
    np.fill_diagonal(corr, 1.0)
    returns = with_sample_cov(corr * np.outer(sd, sd), t=50)
    shrunk = cv.ledoit_wolf_constant_correlation(returns)
    assert shrunk.intensity == 0
    pd.testing.assert_frame_equal(shrunk.cov, cv.sample_cov(returns))


def test_a_sample_matrix_proportional_to_the_identity_is_not_shrunk():
    returns = with_sample_cov(np.eye(4) * 4e-4, t=30)
    shrunk = cv.ledoit_wolf_identity(returns)
    assert shrunk.intensity == 0
    pd.testing.assert_frame_equal(shrunk.cov, cv.sample_cov(returns))


def test_a_single_asset_is_not_shrunk_toward_the_identity():
    shrunk = cv.ledoit_wolf_identity(HAND[["a"]])
    assert shrunk.intensity == 0
    close(shrunk.cov, [[9 * PCT2]])


# --- EWMA ---


def test_ewma_hand_example():
    # Weights 1/7, 2/7, 4/7 (decay 0.5, oldest first), zero mean.
    returns = pd.DataFrame({"a": [0.01, -0.01, 0.02], "b": [0.02, 0.00, -0.01]})
    expected = np.array([[19, -6], [-6, 8]]) / 7 * PCT2
    close(cv.ewma_cov(returns, decay=0.5), expected)


def test_ewma_without_decay_is_the_zero_mean_second_moment():
    values = random_returns(6, 20, 3).to_numpy()
    returns = pd.DataFrame(values)
    close(cv.ewma_cov(returns, decay=1.0), values.T @ values / len(values))


def test_ewma_weighs_the_latest_periods_most():
    returns = random_returns(7, 20, 3)
    backwards = returns.iloc[::-1].set_axis(returns.index)
    assert not np.allclose(cv.ewma_cov(returns), cv.ewma_cov(backwards))
    close(cv.ewma_cov(returns, decay=1.0), cv.ewma_cov(backwards, decay=1.0))


def test_ewma_default_decay_is_riskmetrics():
    assert cv.EWMA_DECAY == 0.94
    returns = random_returns(8, 30, 3)
    close(cv.ewma_cov(returns), cv.ewma_cov(returns, decay=0.94))


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.5, float("nan")])
def test_ewma_rejects_a_decay_outside_zero_to_one(decay):
    with pytest.raises(ValueError, match="decay"):
        cv.ewma_cov(HAND, decay=decay)


# --- Inputs ---


@each_estimator
def test_labels_follow_the_assets(estimate):
    returns = HAND.set_axis([3, 1, 2], axis=1)
    cov = estimate(returns)
    assert list(cov.index) == [3, 1, 2]
    assert list(cov.columns) == [3, 1, 2]


@each_estimator
def test_integer_returns_and_a_plain_index_work(estimate):
    returns = pd.DataFrame({"a": [1, 0, 0, -1], "b": [0, 1, -1, 1], "c": [1, 1, 0, 0]})
    assert estimate(returns).shape == (3, 3)


@each_estimator
def test_missing_values_raise_with_the_asset_and_date(estimate):
    gappy = HAND.assign(b=[0.01, np.nan, -0.01, -0.01])
    with pytest.raises(ValueError, match="missing for b on 2000-01-04"):
        estimate(gappy)


@each_estimator
@pytest.mark.parametrize(
    ("returns", "match"),
    [
        (HAND.assign(b=[0.01, np.inf, -0.01, -0.01]), "finite"),
        (HAND.assign(b=["0.01", "n/a", "-0.01", "-0.01"]), "numeric"),
        (HAND.iloc[:1], "at least 2 periods"),
        (HAND.iloc[::-1], "sorted"),
        (HAND.set_axis([DAYS[0], DAYS[0], DAYS[2], DAYS[3]]), "unique dates"),
        (HAND.set_axis(["a", "a", "c"], axis=1), "unique asset names"),
        (HAND.iloc[:, :0], "no assets"),
    ],
    ids=["inf", "text", "one-period", "unsorted", "repeated-date", "repeated-asset", "no-assets"],
)
def test_bad_windows_raise(estimate, returns, match):
    with pytest.raises(ValueError, match=match):
        estimate(returns)


@each_estimator
def test_a_series_is_not_a_window(estimate):
    with pytest.raises(TypeError, match="DataFrame"):
        estimate(HAND["a"])


def test_constant_correlation_needs_two_assets():
    with pytest.raises(ValueError, match="at least 2 assets"):
        cv.ledoit_wolf_constant_correlation(HAND[["a"]])


@pytest.mark.parametrize("level", [0.0, 0.1, -0.003])
def test_constant_correlation_rejects_an_asset_without_variance(level):
    # 0.1 repeated has a float-noise mean: its demeaned values are not exactly 0.
    with pytest.raises(ValueError, match="c has no variance"):
        cv.ledoit_wolf_constant_correlation(HAND.assign(c=level))
