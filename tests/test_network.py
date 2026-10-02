"""Smoke tests against the real sources. Excluded by default; run with `uv run pytest -m network`.

They check that each live file still parses into the shape the loaders expect,
which catches upstream format changes. They do not assert specific values.
"""

import pandas as pd
import pytest

from portfolio_lab.data import french

pytestmark = pytest.mark.network


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
