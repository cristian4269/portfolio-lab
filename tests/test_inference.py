"""Sharpe-ratio inference (Step 7a): Ledoit-Wolf (2008), Jobson-Korkie-Memmel, Holm, MDE, CEQ."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portfolio_lab.report import inference as inf


def months(n):
    return pd.date_range("2000-01-31", periods=n, freq="ME")


def pair(a, b):
    ix = months(len(a))
    return pd.Series(a, index=ix, dtype=float), pd.Series(b, index=ix, dtype=float)


# r1: mean 0.01, gamma 16e-4/6, variance 1/6000 -> SR sqrt(0.6).
# r2: mean 0.01, gamma 14e-4/6, variance 1/7500 -> SR sqrt(0.75).
R1 = [0.02, -0.01, 0.03, 0.00, 0.01, 0.01]
R2 = [0.01, 0.01, -0.01, 0.03, 0.01, 0.01]


def normal_pair(rng, t, rho=0.5, mu=(0.01, 0.01), sigma=(0.04, 0.04)):
    cov = np.array(
        [[sigma[0] ** 2, rho * sigma[0] * sigma[1]], [rho * sigma[0] * sigma[1], sigma[1] ** 2]]
    )
    x = rng.multivariate_normal(mu, cov, size=t)
    return pair(x[:, 0], x[:, 1])


def ar_pair(rng, t, phi=0.7):
    """Two correlated AR(1) series with strong serial dependence."""
    e = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=t + 100) * 0.03
    x = np.zeros_like(e)
    for s in range(1, len(e)):
        x[s] = phi * x[s - 1] + e[s]
    x = x[100:] + 0.01
    return pair(x[:, 0], x[:, 1])


# --- Sharpe ratio and its difference ----------------------------------------------------------


def test_sharpe_diff_by_hand():
    r1, r2 = pair(R1, R2)
    assert inf.sharpe_ratio(r1) == pytest.approx(math.sqrt(0.6), rel=1e-12)
    assert inf.sharpe_ratio(r2) == pytest.approx(math.sqrt(0.75), rel=1e-12)
    assert inf.sharpe_diff(r1, r2) == pytest.approx(math.sqrt(0.6) - math.sqrt(0.75), rel=1e-12)


def test_sharpe_uses_population_moments_not_ddof_1():
    r1, _ = pair(R1, R2)
    ddof1 = r1.mean() / r1.std(ddof=1)
    assert inf.sharpe_ratio(r1) == pytest.approx(ddof1 * math.sqrt(6 / 5), rel=1e-12)


def test_gradient_matches_finite_differences():
    v = np.array([0.008, 0.011, 0.0022, 0.0031])

    def f(x):
        return x[0] / math.sqrt(x[2] - x[0] ** 2) - x[1] / math.sqrt(x[3] - x[1] ** 2)

    g = inf._gradient(*v)
    h = 1e-7
    for k in range(4):
        up, down = v.copy(), v.copy()
        up[k] += h
        down[k] -= h
        assert g[k] == pytest.approx((f(up) - f(down)) / (2 * h), rel=1e-6)


def test_gradient_by_hand():
    # mu1 = 0.1, gamma1 = 0.02 -> v1 = 0.01, v1^1.5 = 0.001; same for asset 2 with mu2 = 0.
    g = inf._gradient(0.1, 0.0, 0.02, 0.01)
    assert g == pytest.approx([20.0, -10.0, -50.0, 0.0])


def test_sharpe_needs_aligned_finite_variable_series():
    r1, r2 = pair(R1, R2)
    with pytest.raises(ValueError, match="dates"):
        inf.sharpe_diff(r1, r2.iloc[1:])
    with pytest.raises(ValueError, match="finite"):
        inf.sharpe_diff(r1.where(r1 > 0), r2)
    with pytest.raises(ValueError, match="variance"):
        inf.sharpe_diff(pd.Series(0.01, index=r1.index), r2)
    with pytest.raises(TypeError, match="Series"):
        inf.sharpe_diff(np.array(R1), r2)


# --- HAC: Parzen kernel, Andrews bandwidth, prewhitening -------------------------------------


@pytest.mark.parametrize(
    ("x", "k"),
    [(0, 1), (0.25, 0.71875), (0.5, 0.25), (-0.5, 0.25), (0.75, 0.03125), (1, 0), (1.5, 0)],
)
def test_parzen_kernel_by_hand(x, k):
    assert inf._parzen(np.array([x]))[0] == pytest.approx(k, abs=1e-15)


def test_andrews_alpha_one_column_by_hand():
    # One column: alpha = 4 rho^2 / (1 - rho)^4 whatever sigma is.
    rng = np.random.default_rng(1)
    e = rng.standard_normal(400)
    x = np.zeros(400)
    for t in range(1, 400):
        x[t] = 0.5 * x[t - 1] + e[t]
    xc = x - x.mean()
    rho = np.polyfit(xc[:-1], xc[1:], 1)[0]  # OLS slope with an intercept
    assert inf._andrews_alpha(x[:, None]) == pytest.approx(4 * rho**2 / (1 - rho) ** 4, rel=1e-10)


def test_andrews_alpha_pools_columns_and_ignores_a_common_scale():
    rng = np.random.default_rng(2)
    e = rng.standard_normal((300, 2))
    alpha = inf._andrews_alpha(e)
    assert inf._andrews_alpha(e * 7.0) == pytest.approx(alpha, rel=1e-12)
    parts = []
    for col in e.T:
        c = col - col.mean()
        slope, icpt = np.polyfit(c[:-1], c[1:], 1)
        s2 = np.mean((c[1:] - icpt - slope * c[:-1]) ** 2)
        parts.append((slope, s2))
    num = sum(4 * r**2 * s2**2 / (1 - r) ** 8 for r, s2 in parts)
    den = sum(s2**2 / (1 - r) ** 4 for r, s2 in parts)
    assert alpha == pytest.approx(num / den, rel=1e-10)


def test_kernel_sum_by_hand():
    # n = 6 rows, 1 column, bandwidth 2: lags with j < 2 only, k(1/2) = 1/4.
    e = np.array([[1.0], [-1.0], [2.0], [0.0], [-2.0], [0.0]])
    g0 = (1 + 1 + 4 + 0 + 4 + 0) / 6
    g1 = (-1 - 2 + 0 + 0 + 0) / 6
    expected = 6 / (6 - 4) * (g0 + 0.25 * 2 * g1)
    assert inf._kernel_psi(e, 2.0)[0, 0] == pytest.approx(expected, rel=1e-12)


def test_kernel_sum_stops_below_the_bandwidth():
    # Bandwidth exactly 1: no lag j >= 1 is used (j < S), so only Gamma(0).
    e = np.array([[1.0], [-1.0], [2.0], [0.0], [-2.0], [0.0]])
    assert inf._kernel_psi(e, 1.0)[0, 0] == pytest.approx(6 / 2 * 10 / 6, rel=1e-12)


def test_singular_values_capped_at_097():
    a = np.diag([1.5, 0.5, 0.2, 0.1])
    assert np.diag(inf._cap_singular(a)) == pytest.approx([0.97, 0.5, 0.2, 0.1])
    small = np.array([[0.3, 0.1], [0.0, 0.2]])
    assert inf._cap_singular(small) == pytest.approx(small, abs=1e-15)


def _reference_hac(y):
    """A plain-loop prewhitened Parzen HAC, written from the formulas (Andrews and Monahan 1992;
    Andrews 1991), for the vectorized `hac_psi` to match."""
    t, p = y.shape
    a = np.zeros((p, p))
    for j in range(p):
        a[j] = np.linalg.lstsq(y[:-1], y[1:, j], rcond=None)[0]
    u, d, vt = np.linalg.svd(a)
    a = u @ np.diag(np.minimum(d, 0.97)) @ vt
    e = np.array([y[s] - a @ y[s - 1] for s in range(1, t)])
    n = len(e)
    num = den = 0.0
    for col in e.T:
        c = col - col.mean()
        x = np.column_stack([np.ones(n - 1), c[:-1]])
        b = np.linalg.lstsq(x, c[1:], rcond=None)[0]
        s2 = np.mean((c[1:] - x @ b) ** 2)
        num += 4 * b[1] ** 2 * s2**2 / (1 - b[1]) ** 8
        den += s2**2 / (1 - b[1]) ** 4
    bw = 2.6614 * (num / den * n) ** 0.2
    psi = sum(np.outer(e[s], e[s]) for s in range(n)) / n
    j = 1
    while j < bw:
        x = j / bw
        k = 1 - 6 * x**2 + 6 * x**3 if x <= 0.5 else 2 * (1 - x) ** 3
        g = sum(np.outer(e[s], e[s - j]) for s in range(j, n)) / n
        psi = psi + k * (g + g.T)
        j += 1
    psi = n / (n - 4) * psi
    dmat = np.linalg.inv(np.eye(p) - a)
    return dmat @ psi @ dmat.T


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_hac_matches_the_loop_reference(seed):
    rng = np.random.default_rng(seed)
    r1, r2 = ar_pair(rng, 150, phi=0.3)
    y = inf._moment_matrix(r1.to_numpy(), r2.to_numpy())
    assert inf.hac_psi(y) == pytest.approx(_reference_hac(y), rel=1e-10, abs=1e-18)


def test_hac_is_symmetric_and_close_to_iid_on_iid_data():
    rng = np.random.default_rng(6)
    r1, r2 = normal_pair(rng, 6000)
    y = inf._moment_matrix(r1.to_numpy(), r2.to_numpy())
    psi = inf.hac_psi(y)
    assert psi == pytest.approx(psi.T, rel=1e-12)
    assert np.linalg.eigvalsh(psi).min() > 0
    iid = y.T @ y / len(y)
    assert np.diag(psi) == pytest.approx(np.diag(iid), rel=0.1)


def test_moment_matrix_by_hand():
    y = inf._moment_matrix(np.array(R1), np.array(R2))
    assert y[0] == pytest.approx([0.01, 0.0, 4e-4 - 16e-4 / 6, 1e-4 - 14e-4 / 6])
    assert y.mean(axis=0) == pytest.approx([0, 0, 0, 0], abs=1e-18)


def test_diff_se_is_the_delta_method_on_the_hac():
    rng = np.random.default_rng(7)
    r1, r2 = normal_pair(rng, 200)
    x1, x2 = r1.to_numpy(), r2.to_numpy()
    g = inf._gradient(x1.mean(), x2.mean(), (x1**2).mean(), (x2**2).mean())
    psi = inf.hac_psi(inf._moment_matrix(x1, x2))
    assert inf.diff_se(r1, r2) == pytest.approx(math.sqrt(g @ psi @ g / 200), rel=1e-12)


# --- the studentized circular block bootstrap -------------------------------------------------


def test_circular_blocks_wrap_around():
    rows = inf._cbb_rows(np.array([[3, 0]]), block=3, n=5)
    assert rows.tolist() == [[3, 4, 0, 0, 1, 2]]


def test_block_psi_by_hand():
    y = np.array([1.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0])[None, :, None]
    # Block sums 0, 2, -1, -1 over sqrt(2), squared, averaged over 4 blocks: 6 / 8.
    assert inf._block_psi(y, 2)[0, 0, 0] == pytest.approx(0.75, rel=1e-12)
    # b = 1 is the sample (1/T) covariance (Ledoit-Wolf footnote 9).
    assert inf._block_psi(y, 1)[0, 0, 0] == pytest.approx(12 / 8, rel=1e-12)


def test_block_psi_at_1_is_the_sample_covariance_in_4_dimensions():
    rng = np.random.default_rng(8)
    y = rng.standard_normal((1, 40, 4))
    y = y - y.mean(axis=1, keepdims=True)
    assert inf._block_psi(y, 1)[0] == pytest.approx(y[0].T @ y[0] / 40, rel=1e-12)


def test_p_value_counts_ties_and_adds_one():
    d_star = np.array([0.5, 1.0, 2.0])
    assert inf._p_value(1.0, d_star) == pytest.approx(3 / 4)
    assert inf._p_value(5.0, d_star) == pytest.approx(1 / 4)


def test_lw_test_reports_the_statistic_and_its_se():
    rng = np.random.default_rng(9)
    r1, r2 = normal_pair(rng, 120)
    res = inf.lw_test(r1, r2, block=4, resamples=199, rng=np.random.default_rng(1))
    assert res.diff == pytest.approx(inf.sharpe_diff(r1, r2), rel=1e-12)
    assert res.se == pytest.approx(inf.diff_se(r1, r2), rel=1e-12)
    assert res.block == 4
    assert res.resamples == 199
    assert 1 / 200 <= res.pvalue <= 1
    assert res.ci_low < res.diff < res.ci_high
    assert res.diff - res.ci_low == pytest.approx(res.ci_high - res.diff, rel=1e-12)


def test_lw_test_is_repeatable_with_a_seed():
    rng = np.random.default_rng(10)
    r1, r2 = normal_pair(rng, 100)
    a = inf.lw_test(r1, r2, block=2, resamples=99, rng=np.random.default_rng(5))
    b = inf.lw_test(r1, r2, block=2, resamples=99, rng=np.random.default_rng(5))
    assert a == b


def test_lw_test_ci_inverts_the_test():
    """0 is outside the (1 - alpha) interval exactly when p <= alpha (up to the quantile's
    interpolation between resamples)."""
    rng = np.random.default_rng(11)
    r1, r2 = normal_pair(rng, 150, mu=(0.03, 0.0), rho=0.8)
    res = inf.lw_test(r1, r2, block=1, resamples=999, rng=np.random.default_rng(2))
    assert res.pvalue < 0.05
    assert res.ci_low > 0


def test_lw_test_null_value_shifts_the_statistic():
    rng = np.random.default_rng(12)
    r1, r2 = normal_pair(rng, 120)
    at_estimate = inf.lw_test(
        r1, r2, block=1, resamples=99, null=inf.sharpe_diff(r1, r2), rng=np.random.default_rng(3)
    )
    assert at_estimate.pvalue == 1.0  # d = 0: every resample is at least as extreme


def test_lw_test_refuses_identical_series_and_bad_blocks():
    r1, r2 = pair(R1, R2)
    with pytest.raises(ValueError, match="identical"):
        inf.lw_test(r1, r1, block=1, resamples=9, rng=np.random.default_rng(0))
    for bad in (0, 7, 2.0, True):
        with pytest.raises(ValueError, match="block"):
            inf.lw_test(r1, r2, block=bad, resamples=9, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="resamples"):
        inf.lw_test(r1, r2, block=1, resamples=0, rng=np.random.default_rng(0))


def test_lw_test_has_the_right_size_on_iid_data():
    """Ledoit-Wolf's Table 1 setting: iid normal pairs with equal Sharpe ratios, T = 120.

    300 tests at 5% should reject about 15 times; the 99% binomial band is [5, 27]."""
    rng = np.random.default_rng(20261009)
    rejects = 0
    for _ in range(300):
        r1, r2 = normal_pair(rng, 120)
        rejects += inf.lw_test(r1, r2, block=1, resamples=199, rng=rng).pvalue <= 0.05
    lo, hi = stats.binom.ppf([0.005, 0.995], 300, 0.05)
    assert lo <= rejects <= hi


# --- block-size calibration (Ledoit-Wolf Algorithm 3.1) ---------------------------------------


def test_var1_fit_recovers_a_known_system():
    rng = np.random.default_rng(13)
    c, b = np.array([0.01, -0.02]), np.array([[0.5, 0.1], [-0.2, 0.3]])
    x = np.zeros((20000, 2))
    for t in range(1, 20000):
        x[t] = c + b @ x[t - 1] + rng.standard_normal(2) * 0.01
    c_hat, b_hat, resid = inf._var1_fit(x)
    assert c_hat == pytest.approx(c, abs=2e-3)
    assert b_hat == pytest.approx(b, abs=0.02)
    assert resid.shape == (19999, 2)
    assert resid.mean(axis=0) == pytest.approx([0, 0], abs=1e-15)


def test_var1_paths_by_hand():
    c = np.array([1.0, 0.0])
    b = np.array([[0.5, 0.0], [0.0, -1.0]])
    eps = np.array([[[0.0, 1.0], [2.0, 0.0]]])  # one path, two steps
    paths = inf._var1_paths(c, b, np.array([2.0, 3.0]), eps)
    assert paths.tolist() == [[[2.0, 3.0], [2.0, -2.0], [4.0, 2.0]]]


def test_stationary_rows_have_geometric_blocks_with_the_right_mean():
    rng = np.random.default_rng(14)
    rows = inf._stationary_rows(rng, n=1000, mean_block=5, length=200000, paths=1)[0]
    assert rows.min() >= 0 and rows.max() < 1000
    breaks = np.flatnonzero(np.diff(rows) % 1000 != 1)
    assert np.mean(np.diff(breaks)) == pytest.approx(5, rel=0.05)


def test_stationary_rows_wrap_around():
    class Fixed:
        def integers(self, low, high, size):
            return np.full(size, 998)

        def geometric(self, p, size):
            return np.full(size, 4)

    rows = inf._stationary_rows(Fixed(), n=1000, mean_block=5, length=6, paths=1)
    assert rows.tolist() == [[998, 999, 0, 1, 998, 999]]


def test_calibration_picks_the_rate_closest_to_alpha_smaller_block_on_ties():
    rates = pd.Series([0.09, 0.06, 0.04, 0.02], index=[1, 2, 4, 6])
    assert inf._pick_block(rates, 0.05) == 2  # 0.06 and 0.04 tie: the smaller block
    assert inf._pick_block(pd.Series([0.2, 0.051], index=[1, 4]), 0.05) == 4


def test_calibration_wants_longer_blocks_for_dependent_data():
    rng = np.random.default_rng(15)
    iid = normal_pair(rng, 120)
    dep = ar_pair(rng, 120, phi=0.8)
    kw = dict(blocks=(1, 8), pseudo=150, resamples=99)
    cal_iid = inf.calibrate_block(*iid, rng=np.random.default_rng(1), **kw)
    cal_dep = inf.calibrate_block(*dep, rng=np.random.default_rng(1), **kw)
    assert cal_dep.rejection[1] > cal_dep.rejection[8]
    assert cal_dep.rejection[1] > cal_iid.rejection[1]
    assert cal_dep.block == 8
    assert list(cal_iid.rejection.index) == [1, 8]


def test_calibration_is_repeatable_and_checks_its_grid():
    rng = np.random.default_rng(16)
    r1, r2 = normal_pair(rng, 60)
    kw = dict(blocks=(1, 2), pseudo=20, resamples=19)
    a = inf.calibrate_block(r1, r2, rng=np.random.default_rng(4), **kw)
    b = inf.calibrate_block(r1, r2, rng=np.random.default_rng(4), **kw)
    assert a.block == b.block and a.rejection.equals(b.rejection)
    for bad in ((), (2, 1), (1, 1), (0, 2)):
        with pytest.raises(ValueError, match="blocks"):
            inf.calibrate_block(r1, r2, rng=np.random.default_rng(4), **{**kw, "blocks": bad})


def test_block_grid_is_the_registered_one():
    assert inf.BLOCKS == (1, 2, 4, 6, 8, 10, 12)


# --- Jobson-Korkie with Memmel's correction ---------------------------------------------------


def test_jobson_korkie_memmel_matches_the_moment_form():
    """Memmel (2003): z = (s2 m1 - s1 m2) / sqrt(theta), with theta from the moments; this is
    the same z as the Sharpe-ratio form used in the code."""
    rng = np.random.default_rng(17)
    r1, r2 = normal_pair(rng, 90, mu=(0.012, 0.008), rho=0.6)
    x1, x2 = r1.to_numpy(), r2.to_numpy()
    t = len(x1)
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(), x2.std()
    s12 = np.mean((x1 - m1) * (x2 - m2))
    theta = (
        2 * s1**2 * s2**2
        - 2 * s1 * s2 * s12
        + 0.5 * m1**2 * s2**2
        + 0.5 * m2**2 * s1**2
        - m1 * m2 / (s1 * s2) * s12**2
    ) / t
    z = (s2 * m1 - s1 * m2) / math.sqrt(theta)
    res = inf.jobson_korkie_memmel(r1, r2)
    assert res.z == pytest.approx(z, rel=1e-12)
    assert res.pvalue == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-12)


def test_jobson_korkie_memmel_by_hand():
    # The same six returns in another order: equal Sharpe ratios, so z = 0 and p = 1.
    r1, r2 = pair([0.01, 0.03, -0.01, 0.02, 0.0, 0.04], [0.04, 0.0, 0.02, -0.01, 0.03, 0.01])
    res = inf.jobson_korkie_memmel(r1, r2)
    assert res.z == pytest.approx(0, abs=1e-12)
    assert res.pvalue == pytest.approx(1)


# --- Holm, MDE, CEQ, monthly excess returns ---------------------------------------------------


def test_holm_textbook_example():
    adjusted = inf.holm([0.03, 0.04, 0.01, 0.005])
    # Sorted 0.005 x 4, 0.01 x 3, 0.03 x 2, 0.04 x 1 -> 0.02, 0.03, 0.06, max(0.04, 0.06).
    assert adjusted == pytest.approx([0.06, 0.06, 0.03, 0.02])


def test_holm_caps_at_one_keeps_labels_and_handles_one_test():
    adj = inf.holm(pd.Series([0.5, 0.9], index=["a", "b"]))
    assert adj.tolist() == [1.0, 1.0]
    assert list(adj.index) == ["a", "b"]
    assert inf.holm([0.2]) == pytest.approx([0.2])
    with pytest.raises(ValueError, match="p-values"):
        inf.holm([0.2, 1.5])
    with pytest.raises(ValueError, match="p-values"):
        inf.holm([])


def test_holm_rejects_like_the_step_down_rule():
    p = [0.011, 0.015, 0.04, 0.3]
    adj = inf.holm(p)
    # 0.011 <= 0.05/4, 0.015 <= 0.05/3, 0.04 > 0.05/2: the first two reject.
    assert [a <= 0.05 for a in adj] == [True, True, False, False]


def test_min_detectable_difference_closed_form():
    assert inf.min_detectable_difference(0.1) == pytest.approx(
        0.1 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)), rel=1e-12
    )
    assert inf.min_detectable_difference(0.1, alpha=0.0125, power=0.9) == pytest.approx(
        0.1 * (stats.norm.ppf(1 - 0.00625) + stats.norm.ppf(0.9)), rel=1e-12
    )
    with pytest.raises(ValueError):
        inf.min_detectable_difference(-0.1)
    with pytest.raises(ValueError):
        inf.min_detectable_difference(0.1, alpha=1.2)


def test_ceq_by_hand():
    r, _ = pair(R1, R2)
    # mean 0.01, sample variance (ddof 1) = (1/6000) * 6/5 = 0.0002; gamma = 1 -> 0.01 - 0.0001.
    assert inf.ceq(r) == pytest.approx(0.0099, rel=1e-12)
    assert inf.ceq(r, gamma=3) == pytest.approx(0.0097, rel=1e-12)


def test_monthly_excess_compounds_within_calendar_months():
    days = pd.DatetimeIndex(["2001-01-30", "2001-01-31", "2001-02-01", "2001-02-28"])
    r = pd.Series([0.10, -0.05, 0.02, 0.03], index=days)
    rf = pd.Series([0.001, 0.001, 0.002, 0.0], index=days)
    out = inf.monthly_excess(r, rf)
    jan = 1.10 * 0.95 - 1 - (1.001**2 - 1)
    feb = 1.02 * 1.03 - 1 - (1.002 - 1)
    assert out.tolist() == pytest.approx([jan, feb], rel=1e-12)
    assert list(out.index) == [pd.Timestamp("2001-01-31"), pd.Timestamp("2001-02-28")]


def test_monthly_excess_needs_matching_dates():
    days = pd.bdate_range("2001-01-01", periods=5)
    r = pd.Series(0.01, index=days)
    with pytest.raises(ValueError, match="dates"):
        inf.monthly_excess(r, pd.Series(0.0, index=days[1:]))


# --- input checks -----------------------------------------------------------------------------


def test_single_series_checks():
    with pytest.raises(TypeError, match="Series"):
        inf.sharpe_ratio(np.array(R1))
    with pytest.raises(ValueError, match="at least 6 periods"):
        inf.sharpe_ratio(pd.Series(R1[:5], index=months(5)))
    with pytest.raises(ValueError, match="gamma"):
        inf.ceq(pair(R1, R2)[0], gamma=True)
    with pytest.raises(ValueError, match="gamma"):
        inf.ceq(pair(R1, R2)[0], gamma=float("nan"))
    with pytest.raises(TypeError, match="Series"):
        inf.monthly_excess([0.01, 0.02], pd.Series([0.0, 0.0]))


def test_level_and_alpha_checks():
    r1, r2 = pair(R1, R2)
    with pytest.raises(ValueError, match="level"):
        inf.lw_test(r1, r2, block=1, resamples=9, level=1.5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="alpha"):
        inf.calibrate_block(
            r1, r2, blocks=(1,), pseudo=2, resamples=9, alpha=0, rng=np.random.default_rng(0)
        )


def test_jobson_korkie_memmel_refuses_perfect_correlation():
    r1, _ = pair(R1, R2)
    with pytest.raises(ValueError, match="perfectly correlated"):
        inf.jobson_korkie_memmel(r1, 2 * r1)


def test_hac_refuses_a_constant_moment_column():
    with pytest.raises(ValueError, match="bandwidth"):
        inf.hac_psi(np.zeros((10, 4)))


def test_se_refuses_a_zero_variance_estimate(monkeypatch):
    monkeypatch.setattr(inf, "hac_psi", lambda y: np.zeros((4, 4)))
    with pytest.raises(ValueError, match="no estimated variance"):
        inf.diff_se(*pair(R1, R2))


def test_bootstrap_refuses_resamples_without_variance():
    # Five of six returns equal: some resamples of single rows draw only those five.
    r1, r2 = pair([0.01] * 5 + [0.05], R2)
    with pytest.raises(ValueError, match="no variance"):
        inf.lw_test(r1, r2, block=1, resamples=99, rng=np.random.default_rng(0))
