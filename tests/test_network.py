"""Smoke tests against the real sources. Excluded by default; run with `uv run pytest -m network`.

They check that each live file still parses into the shape the loaders expect,
which catches upstream format changes. Apart from start dates and the timing
cross-check below, they do not assert specific values.
"""

import pandas as pd
import pytest

from portfolio_lab.data import boc, fred, french, statcan

pytestmark = pytest.mark.network


def test_fred_usdcad_live():
    s = fred.usdcad()
    assert s.index[0] == pd.Timestamp("1971-01-04")
    assert s.dropna().between(0.9, 2.0).all()  # CAD per USD, never outside this range


@pytest.mark.parametrize("series", sorted(boc.APPROVED))
def test_valet_live(series):
    s = boc.fetch_valet(series)
    assert s.index[0] == pd.Timestamp("2001-01-02")
    assert s.dropna().between(-0.01, 0.25).all()  # decimal yields


@pytest.mark.parametrize("vector", sorted(statcan.APPROVED))
def test_statcan_live(vector):
    s = statcan.fetch_vector(vector)
    assert s.index[0] <= pd.Timestamp("1971-01-27")  # covers the whole CAD view
    assert (s.index.dayofweek == 2).all()
    assert s.dropna().between(-0.01, 0.25).all()


def test_statcan_monthly_matches_valet_daily_on_last_wednesday():
    """Checks the timing rule: the monthly T-bill is the last-Wednesday value, not the 1st's."""
    monthly = statcan.fetch_vector("v122531").loc["2001-02":].dropna()
    daily = boc.fetch_valet("TB.CDN.90D.MID").dropna()
    on_wednesday = daily.reindex(monthly.index).dropna()
    gap_wednesday = (monthly.loc[on_wednesday.index] - on_wednesday).abs()
    first_days = daily.groupby(daily.index.to_period("M")).first()
    on_first = first_days.reindex(monthly.index.to_period("M")).to_numpy()
    gap_first = (monthly - on_first).abs().dropna()
    assert len(gap_wednesday) > 0.9 * len(monthly)  # the Wednesdays exist in the daily data
    assert gap_wednesday.median() < 0.0002  # within 2 bp on the observation day
    assert gap_wednesday.median() < gap_first.median()


def test_french_industries_12_live():
    df = french.industry_returns(12)
    assert df.shape[1] == 12
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("1926-07-01")
    # Daily industry returns are decimals: well inside +/-50% on every day.
    assert df.abs().max().max() < 0.5


def test_french_industries_49_live():
    assert french.industry_returns(49).shape[1] == 49


def test_french_factors_live():
    df = french.factors()
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    assert df.index[0] == pd.Timestamp("1963-07-01")
    assert df["Mom"].notna().mean() > 0.99
    assert french.fetch_french("ff5_daily").vintage is not None
