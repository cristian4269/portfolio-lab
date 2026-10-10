"""Luck benchmark (Step 7a): turnover-matched random shadow strategies."""

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.backtest import engine, luck
from portfolio_lab.construct import strategy as st
from portfolio_lab.data.views import View


def asset_returns(n_days=130, n_assets=4, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2001-01-01", periods=n_days)
    r = rng.normal(0.0004, 0.01, size=(n_days, n_assets))
    return pd.DataFrame(r, index=days, columns=[f"A{i}" for i in range(n_assets)])


def a_result(returns, strategy=None, cost_bps=5.0):
    view = View(returns, pd.Series(0.0001, index=returns.index))
    return engine.run(strategy or st.min_variance("sample", window=20), view, cost_bps=cost_bps)


# --- the shadows ------------------------------------------------------------------------------


def test_shadows_equal_the_engine_on_their_own_targets():
    returns = asset_returns()
    result = a_result(returns, st.equal_weight())  # turnover about 1%: never capped
    paths = luck._shadow_paths(
        result, returns, cost_bps=5.0, count=5, rng=np.random.default_rng(1), keep_targets=True
    )
    assert paths.targets is not None
    for s in range(5):
        targets = pd.DataFrame(
            paths.targets[s], index=result.weights.index, columns=returns.columns
        )
        ledger = engine.simulate(targets, returns.loc[: result.net.index[-1]], cost_bps=5.0)
        assert paths.net[:, s] == pytest.approx(ledger.net.to_numpy(), rel=0, abs=1e-12)
        # Each later trade turns over exactly what the strategy did (no cap was needed here).
        assert ledger.turnover.to_numpy() == pytest.approx(result.turnover.to_numpy(), abs=1e-12)


def test_capped_shadows_trade_no_more_than_the_strategy():
    returns = asset_returns()
    result = a_result(returns)  # minimum variance on 20 days: up to 36% a month
    paths = luck._shadow_paths(
        result, returns, cost_bps=5.0, count=40, rng=np.random.default_rng(1), keep_targets=True
    )
    assert paths.targets is not None
    tau = result.turnover.to_numpy()
    matched = 0
    for s in range(40):
        targets = pd.DataFrame(
            paths.targets[s], index=result.weights.index, columns=returns.columns
        )
        ledger = engine.simulate(targets, returns.loc[: result.net.index[-1]], cost_bps=5.0)
        got = ledger.turnover.to_numpy()
        assert (got <= tau + 1e-12).all()
        matched += int(np.isclose(got, tau, rtol=0, atol=1e-12).sum())
    assert paths.shortfall == pytest.approx(1 - matched / (40 * len(tau)))


def test_shadows_are_long_only_portfolios():
    returns = asset_returns()
    paths = luck._shadow_paths(
        a_result(returns),
        returns,
        cost_bps=5.0,
        count=50,
        rng=np.random.default_rng(2),
        keep_targets=True,
    )
    assert paths.targets is not None
    assert (paths.targets >= 0).all()
    assert paths.targets.sum(axis=2) == pytest.approx(np.ones(paths.targets.shape[:2]))


def test_shadows_report_the_turnover_they_could_not_reach():
    """A strategy that switches assets completely trades 100% of NAV; one Dirichlet step from
    a random holding can rarely move that far, so the shadow falls short and says so."""
    returns = asset_returns(n_days=200, n_assets=3, seed=3)

    def rotate(window):
        k = window.index[-1].month % 3  # a different asset every month
        w = np.zeros(3)
        w[k] = 1.0
        return pd.Series(w, index=window.columns)

    result = a_result(returns, st.Strategy("rotation", 1, rotate))
    assert (result.turnover > 0.9).all()
    out = luck.shadows(result, returns, cost_bps=5.0, count=200, rng=np.random.default_rng(4))
    assert 0.5 < out.shortfall <= 1.0
    calm = luck.shadows(
        a_result(returns, st.equal_weight()),
        returns,
        cost_bps=5.0,
        count=200,
        rng=np.random.default_rng(4),
    )
    assert calm.shortfall < 0.01  # 1/N barely trades: a random step almost always reaches it


def test_shadows_are_repeatable_and_labelled_by_the_results_days():
    returns = asset_returns()
    result = a_result(returns)
    a = luck.shadows(result, returns, cost_bps=5.0, count=7, rng=np.random.default_rng(5))
    b = luck.shadows(result, returns, cost_bps=5.0, count=7, rng=np.random.default_rng(5))
    assert a.net.equals(b.net)
    assert a.net.index.equals(result.net.index)
    assert list(a.net.columns) == list(range(7))


def test_shadows_pay_the_cost_on_trade_days_only():
    returns = asset_returns()
    result = a_result(returns)
    free = luck._shadow_paths(
        result, returns, cost_bps=0.0, count=3, rng=np.random.default_rng(6), keep_targets=False
    )
    paid = luck._shadow_paths(
        result, returns, cost_bps=50.0, count=3, rng=np.random.default_rng(6), keep_targets=False
    )
    differs = ~np.isclose(free.net, paid.net, rtol=0, atol=1e-15).all(axis=1)
    trade_rows = result.net.index.get_indexer(result.turnover.index)
    assert set(np.flatnonzero(differs)) == set(trade_rows)
    assert (paid.net <= free.net + 1e-15).all()


def test_shadows_check_their_inputs():
    returns = asset_returns()
    result = a_result(returns)
    with pytest.raises(TypeError, match="Result"):
        luck.shadows("x", returns, cost_bps=5.0, count=2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="count"):
        luck.shadows(result, returns, cost_bps=5.0, count=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="days"):
        luck.shadows(
            result,
            returns.drop(index=returns.index[60]),
            cost_bps=5.0,
            count=2,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match="assets"):
        luck.shadows(
            result, returns.iloc[:, :3], cost_bps=5.0, count=2, rng=np.random.default_rng(0)
        )


# --- percentiles and best of K ---------------------------------------------------------------


def test_percentile_counts_the_shadows_below():
    assert luck.percentile(0.5, [0.1, 0.5, 0.7, 0.2]) == pytest.approx(0.5)
    assert luck.percentile(1.0, [0.1, 0.5]) == 1.0
    assert luck.percentile(-1.0, [0.1, 0.5]) == 0.0


def test_best_of_k_by_hand():
    actual = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    shadow = pd.DataFrame(
        [[0.1, 0.6, 0.2], [0.4, 0.3, 0.1], [0.5, 0.2, 0.0], [0.0, 0.1, 0.2]],
        columns=["a", "b", "c"],
    )
    # Replicate maxima 0.6, 0.4, 0.5, 0.2 against the best actual 0.5: two reach it.
    assert luck.best_of(actual, shadow) == pytest.approx(0.5)


def test_best_of_k_needs_matching_strategies():
    actual = pd.Series({"a": 0.5, "b": 0.3})
    with pytest.raises(ValueError, match="strategies"):
        luck.best_of(actual, pd.DataFrame([[0.1, 0.2]], columns=["a", "c"]))
