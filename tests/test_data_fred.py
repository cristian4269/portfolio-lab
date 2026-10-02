import numpy as np
import pandas as pd
import pytest

from portfolio_lab.data import fred

# Current FRED layout: `observation_date` header, empty field for a missing day.
CSV_NOW = (
    "observation_date,DEXCAUS\n"
    "1971-01-04,1.0109\n"
    "1971-01-05,1.0102\n"
    "1971-02-12,\n"
    "1971-02-16,1.0050\n"
)
# Older layout seen in guides and archives: `DATE` header, `.` for a missing day.
CSV_OLD = "DATE,DEXCAUS\r\n1971-01-04,1.0109\r\n1971-01-05,.\r\n"


def test_parse_current_layout():
    s = fred.parse_fred_csv(CSV_NOW, "DEXCAUS")
    assert s.name == "DEXCAUS"
    assert s.index.name == "date"
    assert s.index[0] == pd.Timestamp("1971-01-04")
    assert s.dtype == np.float64
    assert s.loc["1971-01-04"] == pytest.approx(1.0109)
    assert np.isnan(s.loc["1971-02-12"])
    assert len(s) == 4


def test_parse_old_layout_dot_is_missing():
    s = fred.parse_fred_csv(CSV_OLD, "DEXCAUS")
    assert s.isna().tolist() == [False, True]


def test_parse_rejects_wrong_series_column():
    with pytest.raises(ValueError, match="expected column 'DEXCAUS'"):
        fred.parse_fred_csv("observation_date,DEXUSEU\n2000-01-03,1.0\n", "DEXCAUS")


def test_parse_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="1971-01-05"):
        fred.parse_fred_csv("DATE,DEXCAUS\n1971-01-04,1.0\n1971-01-05,abc\n", "DEXCAUS")


def test_parse_rejects_unsorted_or_duplicate_dates():
    with pytest.raises(ValueError, match="increasing"):
        fred.parse_fred_csv("DATE,DEXCAUS\n1971-01-05,1.0\n1971-01-04,1.0\n", "DEXCAUS")
    with pytest.raises(ValueError, match="increasing"):
        fred.parse_fred_csv("DATE,DEXCAUS\n1971-01-04,1.0\n1971-01-04,1.0\n", "DEXCAUS")


def test_parse_rejects_empty_file():
    with pytest.raises(ValueError, match="no observations"):
        fred.parse_fred_csv("observation_date,DEXCAUS\n", "DEXCAUS")


def test_fetch_uses_keyless_csv_url_and_cache(fake_http):
    url = fred.csv_url("DEXCAUS")
    assert url == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXCAUS"
    fake_http.add(url, CSV_NOW)
    assert len(fred.fetch_fred("DEXCAUS")) == 4
    fred.fetch_fred("DEXCAUS")
    assert len(fake_http.calls) == 1


def test_fetch_refuses_series_not_cleared_in_data_sources(fake_http):
    with pytest.raises(PermissionError, match="DATA_SOURCES.md"):
        fred.fetch_fred("DEXUSEU")
    assert fake_http.calls == []


def test_usdcad_is_cad_per_usd_level(fake_http):
    fake_http.add(fred.csv_url("DEXCAUS"), CSV_NOW)
    s = fred.usdcad()
    assert s.name == "USDCAD"
    assert s.loc["1971-01-05"] == pytest.approx(1.0102)  # a level, not a percent


def test_usdcad_rejects_non_positive_rate(fake_http):
    fake_http.add(fred.csv_url("DEXCAUS"), "DATE,DEXCAUS\n1971-01-04,0\n")
    with pytest.raises(ValueError, match="positive"):
        fred.usdcad()
