"""Weight rules: hand examples, an exhaustive oracle, optimality conditions, properties.

Minimum variance is checked against an exact solver for small N (every assignment of each asset
to zero, free or capped, solved as a linear system), against the KKT conditions on random
matrices (sufficient for a convex problem), and against the 2-asset closed form. Equal risk
contribution is pinned down by its definition (equal risk contributions; the long-only solution
is unique) and by closed forms: 2 assets, uncorrelated assets, equal correlations.
"""

import itertools
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.optimize import OptimizeResult, check_grad

from portfolio_lab.construct import covariance as cv
from portfolio_lab.construct import weights as wt

PCT2 = 1e-4  # (1%)^2: hand examples are in percent units
RULES = {"min_variance": wt.min_variance, "erc": wt.equal_risk_contribution}
each_rule = pytest.mark.parametrize("rule", RULES.values(), ids=list(RULES))
seeds = st.integers(0, 2**32 - 1)


def frame(m, names=None) -> pd.DataFrame:
    """A labeled covariance matrix in PCT2 units."""
    m = np.asarray(m, dtype=float) * PCT2
    names = names or list("abcdefghij")[: len(m)]
    return pd.DataFrame(m, index=names, columns=names)


def from_vols(sd, corr) -> pd.DataFrame:
    sd = np.asarray(sd, dtype=float)
    return frame(np.asarray(corr, dtype=float) * np.outer(sd, sd))


def equal_corr(n: int, rho: float) -> np.ndarray:
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return corr


def random_cov(seed: int, n: int) -> pd.DataFrame:
    """A random positive definite matrix with daily-sized variances (vols of 0.5% to 3%)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    s = a @ a.T / n + 0.05 * np.eye(n)
    d = np.sqrt(np.diag(s))
    sd = rng.uniform(0.005, 0.03, n)
    names = [f"x{i}" for i in range(n)]
    return pd.DataFrame(s / np.outer(d, d) * np.outer(sd, sd), index=names, columns=names)


def random_returns(seed: int, t: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(t, n)) @ rng.normal(size=(n, n)) * 0.01
    days = pd.bdate_range("2001-01-01", periods=t)
    return pd.DataFrame(values, index=days, columns=[f"x{i}" for i in range(n)])


def cap_for(n: int, u: float) -> float:
    """A feasible cap: from 1/N (only equal weights fit) to 1 (no cap)."""
    return 1 / n + u * (1 - 1 / n)


# Caps from 1/N (u = 0) or clearly above it. Just above 1/N every feasible portfolio is within
# N * cap - 1 of equal weights; that band has its own test.
caps = st.one_of(st.just(0.0), st.floats(0.01, 1))


def exact_min_variance(cov: pd.DataFrame, cap: float = 1.0) -> np.ndarray:
    """Exhaustive: each asset at zero, free or at the cap; the free ones solve the
    equality-constrained problem; the lowest-variance feasible candidate wins (3^N solves)."""
    s = cov.to_numpy()
    n = len(s)
    best, best_var = None, np.inf
    for states in itertools.product((0, 1, 2), repeat=n):
        free = [i for i in range(n) if states[i] == 1]
        capped = [i for i in range(n) if states[i] == 2]
        w = np.zeros(n)
        w[capped] = cap
        rest = 1 - cap * len(capped)
        if free:
            k = len(free)
            kkt = np.block([[s[np.ix_(free, free)], np.ones((k, 1))], [np.ones((1, k)), 0]])
            rhs = np.append(-s[np.ix_(free, capped)] @ w[capped], rest)
            w[free] = np.linalg.solve(kkt, rhs)[:k]
        elif abs(rest) > 1e-12:
            continue
        if (w < -1e-12).any() or (w > cap + 1e-12).any():
            continue
        if w @ s @ w < best_var:
            best, best_var = w, w @ s @ w
    assert best is not None
    return best


def kkt_gap(cov: pd.DataFrame, w: pd.Series, cap: float = 1.0) -> float:
    """Largest marginal variance (Sigma w)_i among assets that could be sold, minus the smallest
    among assets that could be bought, relative to the mean variance. At the long-only optimum
    it is <= 0: no switch between two assets lowers the variance."""
    s, x = cov.to_numpy(), w.to_numpy()
    marginal = s @ x / np.mean(np.diag(s))
    can_buy = x < cap - 1e-12
    if not can_buy.any():
        return -np.inf
    return float(marginal[x > 1e-12].max() - marginal[can_buy].min())


def risk_contributions(cov: pd.DataFrame, w: pd.Series) -> np.ndarray:
    x = w.to_numpy()
    return x * (cov.to_numpy() @ x)


def rc_spread(cov: pd.DataFrame, w: pd.Series) -> float:
    rc = risk_contributions(cov, w)
    return float((rc.max() - rc.min()) / rc.mean())


def assert_valid(w: pd.Series, cov: pd.DataFrame, cap: float = 1.0) -> None:
    assert isinstance(w, pd.Series)
    assert w.index.equals(cov.index)
    x = w.to_numpy()
    assert np.isfinite(x).all()
    assert (x >= 0).all()
    assert (x <= cap + 1e-12).all()
    assert x.sum() == pytest.approx(1, abs=1e-12)


def close(actual, expected, atol: float = 1e-12) -> None:
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=0, atol=atol)


# --- Equal weight ---


def test_equal_weight_splits_evenly():
    w = wt.equal_weight(["a", "b", "c", "d"])
    assert list(w.index) == ["a", "b", "c", "d"]
    close(w, [0.25] * 4, atol=0)


def test_equal_weight_keeps_the_labels_of_an_index():
    columns = pd.Index([3, 1, 2])
    w = wt.equal_weight(columns)
    assert w.index.equals(columns)
    close(w, [1 / 3] * 3, atol=1e-16)


@pytest.mark.parametrize(
    ("assets", "match"), [([], "no assets"), (["a", "b", "a"], "unique")], ids=["empty", "repeated"]
)
def test_equal_weight_rejects_bad_assets(assets, match):
    with pytest.raises(ValueError, match=match):
        wt.equal_weight(assets)


# --- Minimum variance: hand examples ---


def two_asset_closed_form(s1: float, s2: float, s12: float) -> float:
    return (s2**2 - s12) / (s1**2 + s2**2 - 2 * s12)


def test_min_variance_two_asset_hand_example():
    # sd 2% and 3%, covariance 1 (%^2): w1 = (9 - 1) / (4 + 9 - 2) = 8/11.
    w = wt.min_variance(frame([[4, 1], [1, 9]]))
    close(w, [8 / 11, 3 / 11])


@pytest.mark.parametrize(
    ("s1", "s2", "rho"),
    [(1, 2, 0.2), (2, 3, -0.5), (1.5, 1, 0.6), (1, 1, 0.0), (2, 1, -0.9)],
)
def test_min_variance_matches_the_two_asset_closed_form(s1, s2, rho):
    s12 = rho * s1 * s2
    w1 = two_asset_closed_form(s1, s2, s12)
    assert 0 < w1 < 1  # the long-only constraint does not bind
    w = wt.min_variance(frame([[s1**2, s12], [s12, s2**2]]))
    close(w, [w1, 1 - w1])


def test_min_variance_two_asset_corner():
    # Covariance 2 exceeds the first variance 1: the closed form says 7/6 and -1/6, so the
    # long-only portfolio holds only the first asset.
    assert two_asset_closed_form(1, 3, 2) == pytest.approx(7 / 6)
    w = wt.min_variance(frame([[1, 2], [2, 9]]))
    assert list(w) == [1.0, 0.0]


def test_min_variance_drops_an_asset_with_a_negative_unconstrained_weight():
    # Without the long-only constraint: (3/5, 3/5, -1/5). Long-only: (1/2, 1/2, 0), where c's
    # marginal variance (1) exceeds a's and b's (1/2).
    cov = frame([[1, 0, 1], [0, 1, 1], [1, 1, 4]])
    unconstrained = np.linalg.solve(cov.to_numpy(), np.ones(3))
    close(unconstrained / unconstrained.sum(), [3 / 5, 3 / 5, -1 / 5])
    w = wt.min_variance(cov)
    close(w, [1 / 2, 1 / 2, 0])
    assert w["c"] == 0  # exactly: no dust left for the engine to trade


def test_min_variance_of_uncorrelated_assets_is_inverse_variance():
    # Variances 1, 4, 9: weights proportional to 1, 1/4, 1/9 = (36, 9, 4) / 49.
    close(wt.min_variance(frame(np.diag([1, 4, 9]))), [36 / 49, 9 / 49, 4 / 49])


def test_min_variance_with_a_binding_cap():
    # The cap holds a at 1/2 (it wanted 36/49); b and c split the other half 1/4 : 1/9, i.e.
    # 9/26 and 4/26. Marginal variances: a 1/2 <= b = c = 18/13.
    cov = frame(np.diag([1, 4, 9]))
    w = wt.min_variance(cov, cap=0.5)
    close(w, [1 / 2, 9 / 26, 4 / 26])
    assert w["a"] == 0.5
    close(cov.to_numpy() @ w.to_numpy() / PCT2, [1 / 2, 18 / 13, 18 / 13])


def test_min_variance_with_a_cap_that_does_not_bind():
    cov = frame(np.diag([1, 4, 9]))
    close(wt.min_variance(cov, cap=0.8), [36 / 49, 9 / 49, 4 / 49])


@pytest.mark.parametrize(("n", "cap"), [(3, 1 / 3), (4, 0.25), (10, 0.1), (1, 1.0)])
def test_a_cap_of_one_over_n_leaves_only_equal_weights(n, cap):
    w = wt.min_variance(random_cov(n, n), cap=cap)
    close(w, [1 / n] * n, atol=1e-15)


@pytest.mark.parametrize("excess", [1e-13, 1e-9, 1e-6, 1e-4])
def test_a_cap_just_above_one_over_n_leaves_nearly_equal_weights(excess):
    # Each weight lies within [1/N - excess, 1/N + excess / N]: nearly no room to optimize.
    n = 5
    cov, cap = random_cov(16, n), (1 + excess) / n
    w = wt.min_variance(cov, cap=cap)
    assert_valid(w, cov, cap)
    close(w, [1 / n] * n, atol=excess)


def test_a_cap_too_small_to_hold_the_assets_raises():
    with pytest.raises(ValueError, match=r"cap of 0\.2 .* 3 assets"):
        wt.min_variance(frame(np.diag([1, 4, 9])), cap=0.2)


@pytest.mark.parametrize("cap", [0.0, -0.1, 1.5, float("nan")])
def test_a_cap_outside_zero_to_one_raises(cap):
    with pytest.raises(ValueError, match="cap"):
        wt.min_variance(frame(np.diag([1, 4, 9])), cap=cap)


# --- Minimum variance: oracle, optimality, invariances ---


@given(seeds, st.integers(2, 5), caps)
def test_min_variance_matches_an_exhaustive_search(seed, n, u):
    cov, cap = random_cov(seed, n), cap_for(n, u)
    close(wt.min_variance(cov, cap=cap), exact_min_variance(cov, cap), atol=1e-10)


@given(seeds, st.integers(2, 25), caps)
def test_min_variance_satisfies_the_kkt_conditions(seed, n, u):
    cov, cap = random_cov(seed, n), cap_for(n, u)
    w = wt.min_variance(cov, cap=cap)
    assert_valid(w, cov, cap)
    assert kkt_gap(cov, w, cap) <= 1e-9


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_gradient_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    s = random_cov(seed, 8).to_numpy()
    a = s / np.mean(np.diag(s))
    w = rng.dirichlet(np.ones(8))
    gradient = wt._gradient(w, a)
    assert check_grad(wt._objective, wt._gradient, w, a) <= 1e-6 * np.linalg.norm(gradient)
    assert wt._objective(w, a) == pytest.approx(0.5 * w @ a @ w, rel=1e-14)


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1.0, 1e4])
@pytest.mark.parametrize("cap", [1.0, 0.3])
def test_rescaling_the_matrix_leaves_the_weights_unchanged(scale, cap):
    # Daily variances are ~1e-4: unscaled, SLSQP's absolute tolerance stops it too early.
    cov = random_cov(11, 12)
    close(wt.min_variance(cov * scale, cap=cap), wt.min_variance(cov, cap=cap), atol=1e-10)


@each_rule
def test_dividing_the_sample_matrix_by_t_or_t_minus_1_gives_the_same_weights(rule):
    returns = random_returns(12, 40, 6)
    by_t = rule(cv.sample_cov(returns))  # divides by T
    by_t_minus_1 = rule(returns.cov())  # pandas divides by T - 1
    close(by_t, by_t_minus_1, atol=1e-12)


@each_rule
@given(seed=seeds, n=st.integers(2, 10))
def test_reordering_the_assets_reorders_the_weights(rule, seed, n):
    cov = random_cov(seed, n)
    order = list(cov.index[np.random.default_rng(seed).permutation(n)])
    base, moved = rule(cov), rule(cov.loc[order, order])
    assert list(moved.index) == order
    close(moved, base[order], atol=1e-10)


def test_polishing_recovers_the_exact_optimum_from_a_rough_solution():
    s = np.diag([1.0, 4.0, 9.0])
    a = s / np.mean(np.diag(s))
    exact = np.array([1 / 2, 9 / 26, 4 / 26])
    rough = exact + np.array([3e-8, -2e-8, -1e-8])
    close(wt._polish(a, rough, 0.5), exact, atol=1e-15)


@pytest.mark.parametrize(
    ("s", "guess"),
    [
        (np.diag([1.0, 4.0, 9.0]), [0.5, 0.5, 0.0]),  # c belongs in the portfolio: KKT fails
        (np.array([[1.0, 0, 1], [0, 1, 1], [1, 1, 4]]), [1 / 3] * 3),  # free solve: c = -1/5
    ],
    ids=["missing-asset", "negative-weight"],
)
def test_polishing_keeps_the_solver_weights_when_its_active_set_is_wrong(s, guess):
    a = s / np.mean(np.diag(s))
    np.testing.assert_array_equal(wt._polish(a, np.array(guess), 1.0), guess)


def test_a_solver_failure_raises(monkeypatch):
    def failing(*args, **kwargs):
        return OptimizeResult(success=False, message="Iteration limit reached", x=np.ones(3) / 3)

    monkeypatch.setattr(wt, "minimize", failing)
    with pytest.raises(RuntimeError, match="Iteration limit reached"):
        wt.min_variance(frame(np.diag([1, 4, 9])))


# --- Equal risk contribution ---


def test_erc_hand_example_in_exact_fractions():
    # sd 1%, 2%, 4%; correlations a-b = a-c = 1/2, b-c = -1/16. Weights (1/2, 1/3, 1/6):
    # Sigma w = (7/6, 7/4, 7/2), so every risk contribution is 7/12.
    sigma = [
        [Fraction(1), Fraction(1), Fraction(2)],
        [Fraction(1), Fraction(4), Fraction(-1, 2)],
        [Fraction(2), Fraction(-1, 2), Fraction(16)],
    ]
    w = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)]
    marginal = [sum(row[j] * w[j] for j in range(3)) for row in sigma]
    assert marginal == [Fraction(7, 6), Fraction(7, 4), Fraction(7, 2)]
    assert [w[i] * marginal[i] for i in range(3)] == [Fraction(7, 12)] * 3
    corr = [[1, 1 / 2, 1 / 2], [1 / 2, 1, -1 / 16], [1 / 2, -1 / 16, 1]]
    close(wt.equal_risk_contribution(from_vols([1, 2, 4], corr)), [1 / 2, 1 / 3, 1 / 6])


def test_erc_of_uncorrelated_assets_is_inverse_volatility():
    # sd 1, 2, 3: weights proportional to 1, 1/2, 1/3 = (6, 3, 2) / 11.
    w = wt.equal_risk_contribution(frame(np.diag([1, 4, 9])))
    close(w, [6 / 11, 3 / 11, 2 / 11])


@pytest.mark.parametrize("rho", [0.5, -0.2, 0.9])
def test_erc_with_equal_correlations_is_inverse_volatility(rho):
    # Maillard, Roncalli and Teiletche (2010): one common correlation -> w_i proportional to
    # 1/sigma_i. sd 1, 2, 3, 4: (12, 6, 4, 3) / 25.
    w = wt.equal_risk_contribution(from_vols([1, 2, 3, 4], equal_corr(4, rho)))
    close(w, np.array([12, 6, 4, 3]) / 25)


@pytest.mark.parametrize("rho", [-0.95, -0.5, 0.0, 0.5, 0.95])
def test_erc_of_two_assets_ignores_the_correlation(rho):
    # w1 = sigma2 / (sigma1 + sigma2) = 3/4 for sd 1% and 3%, whatever the correlation.
    w = wt.equal_risk_contribution(from_vols([1, 3], equal_corr(2, rho)))
    close(w, [3 / 4, 1 / 4])


@given(seeds, st.integers(1, 30))
def test_erc_risk_contributions_are_equal(seed, n):
    cov = random_cov(seed, n)
    w = wt.equal_risk_contribution(cov)
    assert_valid(w, cov)
    assert (w > 0).all()
    assert rc_spread(cov, w) <= 1e-10


@pytest.mark.parametrize("condition", [1e4, 1e6, 1e8])
def test_erc_converges_on_ill_conditioned_matrices(condition):
    rng = np.random.default_rng(13)
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    s = q @ np.diag(np.geomspace(1, 1 / condition, 10)) @ q.T
    cov = frame((s + s.T) / 2)
    w = wt.equal_risk_contribution(cov)
    assert_valid(w, cov)
    assert rc_spread(cov, w) <= 1e-8


@given(seeds, st.integers(2, 10), st.floats(1e-6, 1e6))
def test_erc_ignores_a_common_rescaling(seed, n, c):
    cov = random_cov(seed, n)
    close(wt.equal_risk_contribution(cov * c), wt.equal_risk_contribution(cov), atol=1e-12)


def test_erc_that_does_not_converge_raises(monkeypatch):
    monkeypatch.setattr(wt, "_NEWTON_STEPS", 1)
    corr = [[1, 1 / 2, 1 / 2], [1 / 2, 1, -1 / 16], [1 / 2, -1 / 16, 1]]
    with pytest.raises(RuntimeError, match="converge"):
        wt.equal_risk_contribution(from_vols([1, 2, 4], corr))


# --- Both rules: inputs ---


@each_rule
def test_a_single_asset_gets_everything(rule):
    close(rule(frame([[4]])), [1.0], atol=0)


@each_rule
def test_labels_follow_the_matrix(rule):
    cov = frame(np.diag([1, 4, 9]), names=[3, 1, 2])
    w = rule(cov)
    assert list(w.index) == [3, 1, 2]


@each_rule
@pytest.mark.parametrize(
    "estimate",
    [
        cv.sample_cov,
        lambda r: cv.ledoit_wolf_identity(r).cov,
        lambda r: cv.ledoit_wolf_constant_correlation(r).cov,
        cv.ewma_cov,
    ],
    ids=["sample", "identity", "constant_correlation", "ewma"],
)
def test_every_estimator_feeds_every_rule(rule, estimate):
    cov = estimate(random_returns(14, 60, 5))
    assert_valid(rule(cov), cov)


@each_rule
def test_shrinkage_makes_a_window_shorter_than_the_universe_usable(rule):
    returns = random_returns(15, 8, 12)  # T < N: the sample matrix is singular
    with pytest.raises(ValueError, match="positive definite"):
        rule(cv.sample_cov(returns))
    cov = cv.ledoit_wolf_identity(returns).cov
    assert_valid(rule(cov), cov)


def _bad(m, names=None) -> pd.DataFrame:
    return pd.DataFrame(m, index=names, columns=names)


GOOD = frame(np.diag([1, 4, 9]))
INDEFINITE = from_vols([1, 1, 1], [[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
COLLINEAR = from_vols([1, 1, 1], [[1, 0, 1 / 2**0.5], [0, 1, 1 / 2**0.5], [1 / 2**0.5] * 2 + [1]])


@each_rule
@pytest.mark.parametrize(
    ("cov", "match"),
    [
        (GOOD.iloc[:2], "square"),
        (GOOD.set_axis(["a", "b", "d"], axis=1), "same assets"),
        (GOOD.loc[:, ["b", "a", "c"]], "same assets"),
        (_bad(np.eye(2), ["a", "a"]), "unique"),
        (GOOD.iloc[:0, :0], "no assets"),
        (GOOD.assign(b=[0, np.nan, 0]), "finite"),
        (GOOD.astype(object).where(GOOD > 0, "n/a"), "numeric"),
        (GOOD.assign(a=GOOD["a"] + np.array([0, 1e-6, 0])), "symmetric"),
        (frame(np.diag([1, 0, 9])), "b has no positive variance"),
        (frame(np.diag([1, -4, 9])), "b has no positive variance"),
        (INDEFINITE, "positive definite"),
        (COLLINEAR, "positive definite"),
    ],
    ids=[
        "not-square",
        "other-assets",
        "other-order",
        "repeated",
        "empty",
        "nan",
        "text",
        "asymmetric",
        "zero-variance",
        "negative-variance",
        "indefinite",
        "singular",
    ],
)
def test_bad_matrices_raise(rule, cov, match):
    with pytest.raises(ValueError, match=match):
        rule(cov)


@each_rule
def test_a_numpy_array_is_not_a_covariance_frame(rule):
    with pytest.raises(TypeError, match="DataFrame"):
        rule(np.eye(3))
