"""Factor regressions: the lag rule, HAC errors by hand, alignment, synthetic betas, weekly data."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from portfolio_lab.risk import factors as fa

TERMS = [fa.ALPHA, *fa.FACTORS]


def days(n, start="2000-01-03"):
    return pd.bdate_range(start, periods=n, name="date")


def factor_frame(n, seed=0, start="2000-01-03"):
    """Synthetic daily factors (FF5 + momentum) and a constant RF, as `load_factors` names them."""
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(rng.normal(0, 0.01, (n, 6)), index=days(n, start), columns=list(fa.FACTORS))
    f["RF"] = 0.0001
    return f


def hac_by_hand(X, e, lags):
    """Newey-West covariance with plain loops: (X'X)^-1 Omega (X'X)^-1, Bartlett weights
    1 - j/(L+1), no 1/T factor and no small-sample correction."""
    T, k = len(X), len(X[0])
    omega = [[0.0] * k for _ in range(k)]
    for t in range(T):
        for a in range(k):
            for b in range(k):
                omega[a][b] += e[t] * e[t] * X[t][a] * X[t][b]
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        for t in range(j, T):
            for a in range(k):
                for b in range(k):
                    omega[a][b] += (
                        w * e[t] * e[t - j] * (X[t][a] * X[t - j][b] + X[t - j][a] * X[t][b])
                    )
    bread = np.linalg.inv(np.array(X).T @ np.array(X))
    return bread @ np.array(omega) @ bread


def ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta, y - X @ beta


# --- the lag rule -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nobs", "lags"),
    [(1, 1), (100, 4), (250, 4), (500, 5), (2000, 7), (11959, 11), (51199, 15), (51200, 16)],
)
def test_nw_lags(nobs, lags):
    # floor(4 (T/100)^(2/9)). At T = 51,200 the value is exactly 16 (100 * 4^4.5 = 51,200), which
    # floats compute as 15.999...; the rule must give the whole number.
    assert fa.nw_lags(nobs) == lags


def test_nw_lags_matches_the_formula_away_from_whole_numbers():
    for nobs in range(2, 30000, 37):
        x = 4 * (nobs / 100) ** (2 / 9)
        if abs(x - round(x)) > 1e-9:
            assert fa.nw_lags(nobs) == math.floor(x)


@pytest.mark.parametrize("nobs", [0, -5, 2.5, True, "100"])
def test_nw_lags_rejects_non_positive_or_non_integers(nobs):
    with pytest.raises(ValueError, match="number of observations"):
        fa.nw_lags(nobs)


# --- regress: hand values ---------------------------------------------------------------------


X6 = [0.01, -0.02, 0.015, 0.03, -0.01, 0.005]
Y6 = [0.012, -0.018, 0.02, 0.025, -0.004, 0.001]


def six_rows():
    idx = days(6)
    return pd.Series(Y6, index=idx, name="y"), pd.DataFrame({"x": X6}, index=idx)


def test_hac_lag_1_by_hand_on_six_rows():
    y, X = six_rows()
    design = [[1.0, x] for x in X6]
    beta, e = ols(np.array(design), np.array(Y6))
    se = np.sqrt(np.diag(hac_by_hand(design, list(e), 1)))

    fit = fa.regress(y, X, lags=1)
    assert fit.lags == 1
    assert fit.coef.to_numpy() == pytest.approx(beta, rel=1e-12, abs=1e-15)
    assert fit.se.to_numpy() == pytest.approx(se, rel=1e-12)


def test_hac_general_lag_matches_the_loops_on_random_data():
    rng = np.random.default_rng(7)
    T = 300
    x = rng.normal(size=(T, 2))
    e_true = np.convolve(rng.normal(size=T + 2), [1, 0.5, 0.25])[2 : T + 2]  # autocorrelated
    yv = 0.1 + x @ [0.5, -0.2] + e_true
    idx = days(T)
    fit = fa.regress(
        pd.Series(yv, index=idx), pd.DataFrame(x, index=idx, columns=["a", "b"]), lags=3
    )

    design = np.column_stack([np.ones(T), x])
    beta, e = ols(design, yv)
    se = np.sqrt(np.diag(hac_by_hand(design.tolist(), list(e), 3)))
    assert fit.coef.to_numpy() == pytest.approx(beta, rel=1e-10)
    assert fit.se.to_numpy() == pytest.approx(se, rel=1e-10)


def test_lag_zero_is_white_errors():
    y, X = six_rows()
    design = [[1.0, x] for x in X6]
    _, e = ols(np.array(design), np.array(Y6))
    se = np.sqrt(np.diag(hac_by_hand(design, list(e), 0)))
    assert fa.regress(y, X, lags=0).se.to_numpy() == pytest.approx(se, rel=1e-12)


def test_default_lag_is_the_rule():
    f = factor_frame(2000)
    fit = fa.regress(f["Mkt-RF"] * 0.3 + f["SMB"], f[["Mkt-RF"]])
    assert fit.lags == fa.nw_lags(2000) == 7


def test_terms_stats_and_table():
    y, X = six_rows()
    fit = fa.regress(y, X, lags=1)
    assert list(fit.coef.index) == [fa.ALPHA, "x"]
    for s in (fit.se, fit.tstat, fit.pvalue):
        assert s.index.equals(fit.coef.index)
    assert fit.tstat.to_numpy() == pytest.approx((fit.coef / fit.se).to_numpy(), rel=1e-12)
    # HAC inference is asymptotic: normal p-values, not Student t.
    p = 2 * stats.norm.sf(np.abs(fit.tstat.to_numpy()))
    assert fit.pvalue.to_numpy() == pytest.approx(p, rel=1e-10)

    design = np.column_stack([np.ones(6), X6])
    _, e = ols(design, np.array(Y6))
    r2 = 1 - (e @ e) / np.sum((np.array(Y6) - np.mean(Y6)) ** 2)
    assert fit.r2 == pytest.approx(r2, rel=1e-12)
    assert fit.adj_r2 == pytest.approx(1 - (1 - r2) * 5 / 4, rel=1e-12)
    assert (fit.nobs, fit.start, fit.end) == (6, y.index[0], y.index[-1])

    table = fit.table()
    assert list(table.columns) == ["coef", "se", "t", "p"]
    assert table.index.equals(fit.coef.index)
    assert table["se"].equals(fit.se)


# --- regress: properties ----------------------------------------------------------------------


def random_problem(seed, T=60):
    rng = np.random.default_rng(seed)
    idx = days(T)
    X = pd.DataFrame(rng.normal(0, 0.01, (T, 2)), index=idx, columns=["a", "b"])
    y = pd.Series(0.001 + 0.7 * X["a"] - 0.3 * X["b"] + rng.normal(0, 0.005, T), index=idx)
    return y, X


@given(st.integers(0, 2**32 - 1), st.floats(-100, 100).filter(lambda c: abs(c) > 1e-3))
def test_scaling_y_scales_coefficients_and_errors(seed, c):
    y, X = random_problem(seed)
    base, scaled = fa.regress(y, X), fa.regress(c * y, X)
    assert scaled.coef.to_numpy() == pytest.approx(c * base.coef.to_numpy(), rel=1e-9, abs=1e-14)
    assert scaled.se.to_numpy() == pytest.approx(abs(c) * base.se.to_numpy(), rel=1e-9)


@given(st.integers(0, 2**32 - 1), st.floats(-10, 10))
def test_adding_a_regressor_to_y_shifts_only_its_beta(seed, c):
    y, X = random_problem(seed)
    base, shifted = fa.regress(y, X), fa.regress(y + c * X["a"], X)
    expected = base.coef.to_numpy() + np.array([0, c, 0])
    assert shifted.coef.to_numpy() == pytest.approx(expected, rel=1e-9, abs=1e-11)
    assert shifted.se.to_numpy() == pytest.approx(base.se.to_numpy(), rel=1e-6)


# --- regress: input checks --------------------------------------------------------------------


def test_collinear_regressors_raise():
    y, X = six_rows()
    with pytest.raises(ValueError, match="collinear"):
        fa.regress(y, X.assign(x2=2 * X["x"]))
    with pytest.raises(ValueError, match="collinear"):
        fa.regress(y, X.assign(c=1.0))  # a constant duplicates the intercept


def test_too_few_rows_raise():
    y, X = six_rows()
    X3 = X.assign(z=[1.0, 0.0, 2.0, 1.0, 5.0, 3.0], w=[0.0, 1.0, 0.0, 3.0, 1.0, 1.0])
    fa.regress(y.iloc[:5], X3.iloc[:5], lags=0)  # 5 rows, 4 parameters: fine
    with pytest.raises(ValueError, match="more than 4 observations"):
        fa.regress(y.iloc[:4], X3.iloc[:4], lags=0)


@pytest.mark.parametrize("lags", [-1, 6, 1.5, True])
def test_bad_lags_raise(lags):
    y, X = six_rows()
    with pytest.raises(ValueError, match="lags"):
        fa.regress(y, X, lags=lags)


def test_input_types_and_names():
    y, X = six_rows()
    with pytest.raises(TypeError, match="Series"):
        fa.regress(y.to_frame(), X)
    with pytest.raises(TypeError, match="DataFrame"):
        fa.regress(y, X["x"])
    with pytest.raises(ValueError, match="alpha"):
        fa.regress(y, X.rename(columns={"x": fa.ALPHA}))
    with pytest.raises(ValueError, match="unique"):
        fa.regress(y, pd.concat([X, X * 2], axis=1))
    with pytest.raises(ValueError, match="at least one regressor"):
        fa.regress(y, X[[]])


def test_non_finite_values_raise():
    y, X = six_rows()
    with pytest.raises(ValueError, match="finite"):
        fa.regress(y.where(y.index != y.index[2], np.inf), X)
    # In a regressor too, naming the column and the day.
    f = factor_frame(50)
    f.iloc[7, f.columns.get_loc("HML")] = -np.inf
    with pytest.raises(ValueError, match=rf"HML.*finite.*{f.index[7].date()}"):
        fa.factor_betas(f["SMB"].rename("p"), f)


def test_a_constant_y_raises():
    # Nothing to explain: R2 and the t-stats would be 0/0.
    y, X = six_rows()
    with pytest.raises(ValueError, match="constant"):
        fa.regress(y * 0, X)


def test_the_largest_lag_is_one_below_the_sample():
    y, X = six_rows()
    assert fa.regress(y, X, lags=5).lags == 5


# --- alignment (decision 7) -------------------------------------------------------------------


def test_inputs_are_trimmed_to_their_shared_span():
    f = factor_frame(400)
    y = (0.5 * f["Mkt-RF"]).iloc[50:300]
    fit = fa.regress(y, f[["Mkt-RF", "SMB"]])
    assert (fit.nobs, fit.start, fit.end) == (250, f.index[50], f.index[299])


def test_dates_must_match_within_the_span():
    f = factor_frame(300)
    monthly = f["Mkt-RF"].groupby(f.index.to_period("M")).tail(1)  # month-end days only
    with pytest.raises(ValueError, match="don't align"):
        fa.regress(monthly, f[["SMB"]])


def test_leading_and_trailing_nan_are_dropped_and_reported():
    f = factor_frame(200)
    y = 0.5 * f["Mkt-RF"] + 0.001 * f["SMB"]
    y.iloc[:10] = np.nan  # a portfolio that starts later
    X = f[["Mkt-RF", "SMB"]].copy()
    X.iloc[-5:, 1] = np.nan  # a regressor that ends earlier
    fit = fa.regress(y, X)
    assert (fit.nobs, fit.start, fit.end) == (185, f.index[10], f.index[194])


def test_interior_nan_raises_naming_the_day_and_input():
    f = factor_frame(200)
    X = f[["Mkt-RF", "SMB"]].copy()
    X.iloc[100, 1] = np.nan
    with pytest.raises(ValueError, match=rf"SMB.*{f.index[100].date()}"):
        fa.regress(f["HML"], X)
    y = f["HML"].copy()
    y.iloc[50] = np.nan
    with pytest.raises(ValueError, match=rf"{f.index[50].date()}"):
        fa.regress(y, f[["Mkt-RF"]])


def test_no_shared_dates_or_no_complete_rows_raise():
    f = factor_frame(100)
    g = factor_frame(100, start="2010-01-04")
    with pytest.raises(ValueError, match="no dates in common"):
        fa.regress(f["SMB"], g[["Mkt-RF"]])
    with pytest.raises(ValueError, match="no complete rows"):
        fa.regress(f["SMB"] * np.nan, f[["Mkt-RF"]])


def test_unsorted_dates_raise():
    f = factor_frame(50)
    with pytest.raises(ValueError, match="sorted"):
        fa.regress(f["SMB"].iloc[::-1], f[["Mkt-RF"]])


# --- factor_betas -----------------------------------------------------------------------------


def within(fit_table, asset, truth, n_se=4):
    """Every coefficient within n_se HAC standard errors of its true value (0 if not given)."""
    block = fit_table.loc[asset]
    for term, row in block.iterrows():
        assert abs(row["coef"] - truth.get(term, 0.0)) < n_se * row["se"], term


def test_usd_synthetic_betas_are_recovered():
    f = factor_frame(5000, seed=1)
    noise = np.random.default_rng(2).normal(0, 0.005, 5000)
    y = (0.5 * f["Mkt-RF"] + 0.2 * f["SMB"] + noise).rename("p")
    table = fa.factor_betas(y, f)
    assert list(table.loc["p"].index) == TERMS
    within(table, "p", {"Mkt-RF": 0.5, "SMB": 0.2})
    assert table.loc[("p", "Mkt-RF"), "se"] < 0.02  # informative, not a vacuous check


def test_cad_synthetic_fx_beta_is_one():
    f = factor_frame(5000, seed=3)
    rng = np.random.default_rng(4)
    fx = pd.Series(rng.normal(0, 0.006, 5000), index=f.index, name="usdcad")
    r_usd = 0.5 * f["Mkt-RF"] + rng.normal(0, 0.005, 5000)
    r_cad = ((1 + r_usd) * (1 + fx) - 1).rename("p")  # Step 2's unhedged conversion
    table = fa.factor_betas(r_cad, f, fx=fx)
    assert list(table.loc["p"].index) == [*TERMS, "fx"]  # named fx whatever the input name
    within(table, "p", {"Mkt-RF": 0.5, "fx": 1.0})


def test_one_block_per_column_equal_to_regress():
    f = factor_frame(600, seed=5)
    rng = np.random.default_rng(6)
    excess = pd.DataFrame(
        {"A": 0.9 * f["Mkt-RF"] + rng.normal(0, 0.004, 600), "B": -0.3 * f["HML"] + 0.001},
        index=f.index,
    )
    excess.iloc[:20, 1] = np.nan  # B starts later
    table = fa.factor_betas(excess, f)
    assert table.index.names == ["asset", "term"]
    assert list(table.columns) == [
        "coef",
        "se",
        "t",
        "p",
        "r2",
        "adj_r2",
        "nobs",
        "lags",
        "start",
        "end",
    ]
    for asset in ("A", "B"):
        fit = fa.regress(excess[asset], f[list(fa.FACTORS)])
        block = table.loc[asset]
        expected = fit.table()
        for col in ("coef", "se", "t", "p"):
            assert block[col].to_numpy() == pytest.approx(expected[col].to_numpy(), rel=1e-12)
        assert (block["adj_r2"] == fit.adj_r2).all()
        assert (block["lags"] == fit.lags).all()
        assert (block["end"] == fit.end).all()
        assert (block["r2"] == fit.r2).all()
        assert (block["nobs"] == fit.nobs).all()
        assert (block["start"] == fit.start).all()
    assert table.loc[("B", fa.ALPHA), "nobs"] == 580


def test_factors_may_run_longer_than_fx():
    # The CAD view starts in 1971, the factors in 1963: the shared span is used.
    f = factor_frame(800, seed=8)
    fx = pd.Series(np.random.default_rng(9).normal(0, 0.006, 500), index=f.index[300:])
    y = (0.4 * f["Mkt-RF"] + fx).iloc[300:].rename("p")
    table = fa.factor_betas(y, f, fx=fx)
    assert table.loc[("p", "fx"), "nobs"] == 500
    assert table.loc[("p", "fx"), "start"] == f.index[300]


def test_missing_factor_columns_raise_and_extra_columns_are_ignored():
    f = factor_frame(300)
    y = f["Mkt-RF"].rename("p")
    with pytest.raises(ValueError, match="RMW, Mom"):
        fa.factor_betas(y, f.drop(columns=["RMW", "Mom"]))
    table = fa.factor_betas(y, f.assign(extra=1.0))  # RF and "extra" are not regressors
    assert list(table.loc["p"].index) == TERMS


def test_weekly_equals_regress_on_compounded_weeks():
    f = factor_frame(1500, seed=10)
    rng = np.random.default_rng(11)
    fx = pd.Series(rng.normal(0, 0.006, 1500), index=f.index)
    y = (0.6 * f["Mkt-RF"] + 0.8 * fx + rng.normal(0, 0.004, 1500)).rename("p")
    table = fa.factor_betas(y, f, fx=fx, frequency="weekly")

    X = f[list(fa.FACTORS)].assign(fx=fx)
    fit = fa.regress(fa.to_weekly(y), fa.to_weekly(X))
    assert table.loc["p", "coef"].to_numpy() == pytest.approx(fit.coef.to_numpy(), rel=1e-12)
    assert table.loc["p", "se"].to_numpy() == pytest.approx(fit.se.to_numpy(), rel=1e-12)
    assert table.loc[("p", "fx"), "lags"] == fa.nw_lags(fit.nobs)


def test_weekly_fit_reports_the_daily_days_it_covers():
    # Dev data ends on Friday 2010-12-31, whose week is labelled Wednesday 2011-01-05: the fit
    # must report the days it used, not a label inside the holdout.
    idx = pd.bdate_range("2010-01-07", "2010-12-31", name="date")  # Thursday .. Friday
    f = factor_frame(len(idx), seed=13, start="2010-01-07")
    assert f.index.equals(idx)
    y = (0.5 * f["Mkt-RF"] + np.random.default_rng(14).normal(0, 0.003, len(idx))).rename("p")
    table = fa.factor_betas(y, f, frequency="weekly")
    assert table.loc[("p", fa.ALPHA), "start"] == pd.Timestamp("2010-01-07")
    assert table.loc[("p", fa.ALPHA), "end"] == pd.Timestamp("2010-12-31")
    assert table.loc[("p", fa.ALPHA), "nobs"] == len(fa.to_weekly(y))  # nobs still counts weeks


def test_weekly_drops_the_nan_edges_before_making_weeks():
    # A portfolio starting mid-week: its first week compounds only its own days, the same days
    # as the factors' first week.
    f = factor_frame(400, seed=12)
    y = (0.5 * f["Mkt-RF"]).rename("p")
    y.iloc[:3] = np.nan  # 2000-01-03 is a Monday: starts on Thursday the 6th
    table = fa.factor_betas(y, f, frequency="weekly")
    X = f[list(fa.FACTORS)].iloc[3:]
    fit = fa.regress(fa.to_weekly(y.iloc[3:]), fa.to_weekly(X))
    assert table.loc["p", "coef"].to_numpy() == pytest.approx(fit.coef.to_numpy(), rel=1e-12)


def test_bad_frequency_and_unnamed_series():
    f = factor_frame(300)
    with pytest.raises(ValueError, match="frequency"):
        fa.factor_betas(f["SMB"], f, frequency="monthly")
    table = fa.factor_betas(f["SMB"].rename(None) * 0.5 + f["HML"], f)
    assert table.index.get_level_values("asset").unique().tolist() == ["returns"]


def test_asset_labels_become_text():
    f = factor_frame(300)
    y = 0.5 * f["Mkt-RF"] + 0.1 * f["HML"]
    table = fa.factor_betas(pd.DataFrame({0: y, 1: -y}), f)
    assert table.index.get_level_values("asset").unique().tolist() == ["0", "1"]
    assert fa.factor_betas(y.rename(7), f).index.get_level_values("asset")[0] == "7"


def test_factor_betas_input_types():
    f = factor_frame(300)
    y = f["SMB"].rename("p")
    with pytest.raises(TypeError, match="Series or a DataFrame"):
        fa.factor_betas(y.to_numpy(), f)
    with pytest.raises(ValueError, match="unique column names"):
        fa.factor_betas(pd.concat([y, y], axis=1), f)
    with pytest.raises(ValueError, match="unique column names"):
        # 1 and "1" differ as labels but name the same asset in the table
        fa.factor_betas(pd.DataFrame({1: y, "1": 2 * y}), f)
    with pytest.raises(ValueError, match="no columns"):
        fa.factor_betas(f[[]], f)
    with pytest.raises(TypeError, match="factors must be a DataFrame"):
        fa.factor_betas(y, f.to_numpy())
    with pytest.raises(TypeError, match="fx must be a Series"):
        fa.factor_betas(y, f, fx=f[["SMB"]])
    with pytest.raises(ValueError, match="no rows"):
        fa.factor_betas(y.iloc[:0], f)


# --- macro_betas (decision 6: partial betas) --------------------------------------------------


def macro_problem(n=5000, seed=20):
    """y = -7 dy + 1 fx + 0.8 MKT + noise, where the yield change co-moves with the market (as
    on dev data, where yields rose on good equity days)."""
    f = factor_frame(n, seed)
    rng = np.random.default_rng(seed + 1)
    fx = pd.Series(rng.normal(0, 0.006, n), index=f.index, name="usdcad")
    dy = pd.Series(0.03 * f["Mkt-RF"] + rng.normal(0, 0.0004, n), index=f.index, name="y10")
    y = (-7 * dy + 1.0 * fx + 0.8 * f["Mkt-RF"] + rng.normal(0, 0.004, n)).rename("p")
    return f, fx, dy, y


def test_macro_betas_are_partial_with_the_factors_held_fixed():
    f, fx, dy, y = macro_problem()
    table = fa.macro_betas(y, dy, fx, f)
    assert list(table.loc["p"].index) == [*TERMS, fa.FX, fa.DY]
    within(table, "p", {fa.DY: -7.0, fa.FX: 1.0, "Mkt-RF": 0.8})
    assert table.loc[("p", fa.DY), "se"] < 0.5


def test_without_the_factors_the_yield_beta_takes_the_market_s_sign():
    # The omitted market adds 0.8 cov(MKT, dy) / var(dy) = 9.6 to the yield beta: a +200 bp
    # shock would look like a gain. This is why macro_betas holds the factors fixed.
    f, fx, dy, y = macro_problem()
    total = fa.regress(y, pd.DataFrame({"dy": dy, "fx": fx}))
    assert total.coef["dy"] > 0
    assert total.coef["dy"] == pytest.approx(-7 + 9.6, abs=0.6)


def test_macro_weekly_sums_yield_changes_and_compounds_returns():
    f, fx, dy, y = macro_problem(n=3000)
    table = fa.macro_betas(y, dy, fx, f, frequency="weekly")
    X = pd.concat(
        [
            fa.to_weekly(f[list(fa.FACTORS)].assign(fx=fx)),
            fa.to_weekly(dy.rename(fa.DY), how="sum"),
        ],
        axis=1,
    )
    fit = fa.regress(fa.to_weekly(y), X)
    assert table.loc["p", "coef"].to_numpy() == pytest.approx(fit.coef.to_numpy(), rel=1e-12)
    assert table.loc["p", "se"].to_numpy() == pytest.approx(fit.se.to_numpy(), rel=1e-12)


def test_macro_sample_starts_with_the_first_yield_change():
    # The daily 10-year starts in 2001: earlier days are NaN and drop out, reported.
    f, fx, dy, y = macro_problem(n=1500)
    dy.iloc[:300] = np.nan
    table = fa.macro_betas(y, dy, fx, f)
    assert table.loc[("p", fa.DY), "start"] == f.index[300]
    assert table.loc[("p", fa.DY), "nobs"] == 1200


def test_macro_betas_one_block_per_column():
    f, fx, dy, y = macro_problem(n=1000)
    excess = pd.DataFrame({"A": y, "B": 0.5 * y + 0.001})
    table = fa.macro_betas(excess, dy, fx, f)
    assert table.index.get_level_values("asset").unique().tolist() == ["A", "B"]
    assert table.loc[("B", fa.DY), "coef"] == pytest.approx(
        0.5 * table.loc[("A", fa.DY), "coef"], rel=1e-9
    )


def test_macro_betas_input_types():
    f, fx, dy, y = macro_problem(n=300)
    with pytest.raises(TypeError, match="fx must be a Series"):
        fa.macro_betas(y, dy, None, f)
    with pytest.raises(TypeError, match="d_cad10y must be a Series"):
        fa.macro_betas(y, dy.to_frame(), fx, f)
    with pytest.raises(ValueError, match="frequency"):
        fa.macro_betas(y, dy, fx, f, frequency="monthly")


# --- rolling_betas ----------------------------------------------------------------------------


def rolling_loop(y, X, window):
    """Each window's OLS by lstsq, one at a time: the reference for the vectorized version."""
    design = np.column_stack([np.ones(len(y)), X.to_numpy()])
    yv = y.to_numpy()
    return np.array(
        [
            np.linalg.lstsq(design[i - window + 1 : i + 1], yv[i - window + 1 : i + 1], rcond=None)[
                0
            ]
            for i in range(window - 1, len(y))
        ]
    )


def rolling_problem(n=700, seed=30):
    f = factor_frame(n, seed)
    rng = np.random.default_rng(seed + 1)
    fx = pd.Series(rng.normal(0, 0.006, n), index=f.index)
    y = (0.7 * f["Mkt-RF"] + 0.9 * fx + rng.normal(0, 0.004, n)).rename("p")
    return f, fx, y


@pytest.mark.parametrize("window", [fa.ROLLING_WINDOW, 30])
def test_rolling_betas_equal_a_per_window_loop(window):
    f, fx, y = rolling_problem()
    betas = fa.rolling_betas(y, f, fx=fx, window=window)
    X = f[list(fa.FACTORS)].assign(fx=fx)
    assert betas.to_numpy() == pytest.approx(rolling_loop(y, X, window), rel=1e-10, abs=1e-12)
    assert betas.index.equals(f.index[window - 1 :])  # indexed by each window's last day
    assert list(betas.columns) == [*TERMS, fa.FX]
    assert betas.columns.name == "term"


def test_rolling_betas_default_window_is_one_year():
    f, fx, y = rolling_problem()
    assert fa.ROLLING_WINDOW == 252
    assert fa.rolling_betas(y, f).index[0] == f.index[251]


def test_rolling_betas_have_no_look_ahead():
    f, fx, y = rolling_problem()
    k = 400
    bumped = y.copy()
    bumped.iloc[k] += 0.05
    before, after = fa.rolling_betas(y, f, fx=fx), fa.rolling_betas(bumped, f, fx=fx)
    earlier = before.index < f.index[k]
    assert np.array_equal(before[earlier].to_numpy(), after[earlier].to_numpy())
    assert not np.allclose(before.loc[f.index[k]], after.loc[f.index[k]])


def test_rolling_betas_drop_the_nan_edges():
    f, fx, y = rolling_problem()
    y.iloc[:10] = np.nan
    assert fa.rolling_betas(y, f).index[0] == f.index[10 + 251]


@pytest.mark.parametrize("window", [7, 701, 25.0, True])
def test_rolling_window_checks(window):
    # 7 parameters (alpha + 6 factors): a window must have more rows than that, and fit the data.
    f, fx, y = rolling_problem()
    with pytest.raises(ValueError, match="window must be a whole number of days"):
        fa.rolling_betas(y, f, window=window)
    assert len(fa.rolling_betas(y, f, window=8)) == 700 - 7


def test_a_window_as_long_as_the_data_is_the_full_sample_fit():
    f, fx, y = rolling_problem()
    one = fa.rolling_betas(y, f, fx=fx, window=700)
    assert one.index.tolist() == [f.index[-1]]
    full = fa.regress(y, f[list(fa.FACTORS)].assign(fx=fx))
    assert one.iloc[0].to_numpy() == pytest.approx(full.coef.to_numpy(), rel=1e-10, abs=1e-12)


def test_a_singular_window_raises_naming_its_last_day():
    f, fx, y = rolling_problem()
    f.iloc[:300, f.columns.get_loc("SMB")] = 0.0  # SMB flat for the first 300 days
    with pytest.raises(ValueError, match=rf"collinear.*{f.index[251].date()}"):
        fa.rolling_betas(y, f)


def test_rolling_betas_take_one_series():
    f, fx, y = rolling_problem()
    with pytest.raises(TypeError, match="Series"):
        fa.rolling_betas(y.to_frame(), f)


# --- to_weekly --------------------------------------------------------------------------------


def hand_week_returns():
    """Wed 3 Jan 2024 alone; then Thu 4 - Wed 10; then Thu 11 - Wed 17 with Monday the 15th a
    holiday; nothing in the week to Wed 24; then Wed 31 alone."""
    data = {
        "2024-01-03": 0.01,
        "2024-01-04": 0.02,
        "2024-01-05": -0.01,
        "2024-01-08": 0.03,
        "2024-01-09": 0.00,
        "2024-01-10": 0.01,
        "2024-01-11": 0.01,
        "2024-01-12": 0.02,
        "2024-01-16": -0.02,
        "2024-01-17": 0.005,
        "2024-01-31": -0.03,
    }
    return pd.Series(data, name="r").rename_axis("date").set_axis(pd.to_datetime(list(data)))


def test_weekly_compounding_by_hand():
    w = fa.to_weekly(hand_week_returns())
    assert list(w.index) == list(
        pd.to_datetime(["2024-01-03", "2024-01-10", "2024-01-17", "2024-01-31"])
    )
    expected = [
        0.01,
        1.02 * 0.99 * 1.03 * 1.00 * 1.01 - 1,
        1.01 * 1.02 * 0.98 * 1.005 - 1,
        -0.03,
    ]
    assert w.to_numpy() == pytest.approx(expected, rel=1e-14)
    assert w.name == "r"


def test_weekly_sum_for_yield_changes():
    w = fa.to_weekly(hand_week_returns(), how="sum")
    expected = [0.01, 0.02 - 0.01 + 0.03 + 0.00 + 0.01, 0.01 + 0.02 - 0.02 + 0.005, -0.03]
    assert w.to_numpy() == pytest.approx(expected, rel=1e-14)


def test_weeks_ignore_the_time_of_day():
    # Stamps at the close (16:00), or one at 09:30, are still the days they fall on.
    r = hand_week_returns()
    stamped = r.set_axis(r.index + pd.Timedelta(hours=16))
    stamped.index = stamped.index.where(stamped.index != "2024-01-11 16:00", "2024-01-11 09:30")
    w, ref = fa.to_weekly(stamped), fa.to_weekly(r)
    assert w.index.equals(ref.index)  # labelled by the Wednesday's date
    assert w.to_numpy() == pytest.approx(ref.to_numpy(), rel=1e-15)


def test_week_label_is_the_wednesday_even_when_it_is_a_holiday():
    # 4 July 2018 was a Wednesday holiday.
    idx = pd.to_datetime(["2018-07-02", "2018-07-03", "2018-07-05"])
    w = fa.to_weekly(pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [0.0, 0.0, 0.0]}, index=idx))
    assert list(w.index) == list(pd.to_datetime(["2018-07-04", "2018-07-11"]))
    assert w["a"].to_numpy() == pytest.approx([1.01 * 1.02 - 1, 0.03], rel=1e-14)
    assert list(w.columns) == ["a", "b"]


def test_to_weekly_input_checks():
    r = hand_week_returns()
    with pytest.raises(ValueError, match="how"):
        fa.to_weekly(r, how="mean")
    with pytest.raises(ValueError, match=r"missing.*2024-01-08"):
        fa.to_weekly(r.where(r.index != "2024-01-08"))
    with pytest.raises(ValueError, match="finite"):
        fa.to_weekly(r.where(r.index != "2024-01-08", np.inf))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        fa.to_weekly(r.reset_index(drop=True))
    with pytest.raises(TypeError, match="Series or a DataFrame"):
        fa.to_weekly(r.to_numpy())


@given(st.integers(0, 2**32 - 1), st.integers(1, 120), st.sampled_from(["compound", "sum"]))
def test_to_weekly_equals_pandas_resample(seed, n, how):
    # Random trading days (gaps of 1-12 calendar days, so some weeks are empty) against pandas'
    # own Wednesday resampling.
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(
        pd.Timestamp("2001-01-01") + pd.to_timedelta(np.cumsum(rng.integers(1, 13, n)), "D")
    )
    r = pd.DataFrame(rng.normal(0, 0.02, (n, 3)), index=idx, columns=["a", "b", "c"])
    weeks = r.resample("W-WED")
    ref = (1 + r).resample("W-WED").prod() - 1 if how == "compound" else weeks.sum()
    ref = ref[weeks.size().to_numpy() > 0]
    out = fa.to_weekly(r, how=how)
    assert out.index.equals(ref.index)
    assert out.to_numpy() == pytest.approx(ref.to_numpy(), rel=1e-12, abs=1e-15)
    assert fa.to_weekly(r["b"], how=how).to_numpy() == pytest.approx(
        ref["b"].to_numpy(), rel=1e-12, abs=1e-15
    )


def test_to_weekly_of_nothing_is_empty():
    r = hand_week_returns().iloc[:0]
    w = fa.to_weekly(r)
    assert isinstance(w, pd.Series) and len(w) == 0 and w.name == "r"
    frame = fa.to_weekly(pd.DataFrame({"a": r, "b": r}), how="sum")
    assert list(frame.columns) == ["a", "b"] and len(frame) == 0
