"""Benchmarks: the market index, Mkt-RF + RF, in each currency view (plans/step-4-plan.md).

1/N needs nothing here: it is `equal_weight()` run through the engine like any strategy.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.backtest import benchmarks as bm
from portfolio_lab.data import views

DAYS = pd.bdate_range("1970-12-28", "1971-01-15", name="date")


def factors() -> pd.DataFrame:
    """Shaped like `load_factors`: daily decimal factor returns and RF."""
    n = len(DAYS)
    return pd.DataFrame(
        {"Mkt-RF": np.linspace(-0.01, 0.012, n), "SMB": 0.001, "RF": np.linspace(2e-4, 3e-4, n)},
        index=DAYS,
    )


def test_the_market_is_mkt_rf_plus_rf():
    f = pd.DataFrame(
        {"Mkt-RF": [0.01, -0.02, 0.005], "SMB": [0.3, 0.3, 0.3], "RF": [1e-4, 1e-4, 2e-4]},
        index=DAYS[:3],
    )
    out = bm.market(f)
    np.testing.assert_allclose(out, [0.0101, -0.0199, 0.0052], rtol=0, atol=1e-15)
    assert out.name == "Market"
    assert out.index.equals(DAYS[:3])


@pytest.mark.parametrize("missing", ["Mkt-RF", "RF"])
def test_the_market_needs_both_columns(missing):
    with pytest.raises(ValueError, match=missing):
        bm.market(factors().drop(columns=missing))


def test_the_market_is_converted_to_each_view_like_the_assets():
    f = factors()
    usdcad = pd.Series(np.linspace(1.00, 1.02, 10), index=pd.bdate_range("1971-01-04", periods=10))
    rate_days = pd.bdate_range("1970-12-01", "1971-01-15")
    ca_tbill = pd.Series(np.where(rate_days < "1971-01-08", 0.05, 0.06), index=rate_days)
    out = bm.market_views(f, usdcad, ca_tbill)
    assert set(out) == {"USD", "CAD", "CAD_HEDGED"}
    mkt = bm.market(f)
    pd.testing.assert_series_equal(out["USD"], mkt)

    cad_days = out["CAD"].index
    assert cad_days[0] == pd.Timestamp("1971-01-05")  # the second day with a USD/CAD fix
    fx = views.fx_returns(usdcad, DAYS).loc[cad_days]
    unhedged = views.to_cad_unhedged(mkt.loc[cad_days], fx)
    pd.testing.assert_series_equal(out["CAD"], unhedged, check_freq=False)
    rf_cad = views.daily_rate(ca_tbill, DAYS).loc[cad_days]
    hedged = views.to_cad_hedged(mkt.loc[cad_days], rf_cad, f["RF"].loc[cad_days])
    pd.testing.assert_series_equal(out["CAD_HEDGED"], hedged, check_freq=False)
