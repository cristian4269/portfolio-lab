import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.data import french

FIXTURES = Path(__file__).parent / "fixtures"
VW = french.VALUE_WEIGHTED


def fixture_text(name: str) -> str:
    # The real files use CRLF line endings; the fixtures are stored with LF.
    return (FIXTURES / "french" / name).read_text().replace("\n", "\r\n")


def zipped(name: str, inner: str = "inner.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(inner, fixture_text(name).encode("latin-1"))
    return buf.getvalue()


def test_single_block_file_percent_to_decimal():
    f = french.parse_french_csv(fixture_text("ff5_daily.csv"))
    assert f.vintage == "209912"
    assert list(f.blocks) == [french.MAIN_BLOCK]
    df = f.blocks[french.MAIN_BLOCK]
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert df.index.name == "date"
    assert list(df.index) == list(pd.to_datetime(["1963-07-01", "1963-07-02", "1963-07-03"]))
    # 1.50 percent is 0.015; -0.67 percent is -0.0067.
    assert df.loc["1963-07-03", "Mkt-RF"] == pytest.approx(0.015)
    assert df.loc["1963-07-01", "Mkt-RF"] == pytest.approx(-0.0067)
    assert df.loc["1963-07-03", "RF"] == pytest.approx(0.0002)
    assert (df.dtypes == np.float64).all()


def test_missing_codes_become_nan():
    df = french.parse_french_csv(fixture_text("mom_daily.csv")).blocks[french.MAIN_BLOCK]
    assert list(df.columns) == ["Mom"]
    assert df["Mom"].isna().tolist() == [False, False, True]


def test_titled_blocks_are_split_and_missing_handled():
    f = french.parse_french_csv(fixture_text("industry3_daily.csv"))
    assert set(f.blocks) == {VW, "Average Equal Weighted Returns -- Daily"}
    vw = f.blocks[VW]
    assert list(vw.columns) == ["NoDur", "Enrgy", "Money"]
    assert len(vw) == 3
    assert np.isnan(vw.loc["1963-07-02", "Enrgy"])  # -999
    assert np.isnan(vw.loc["1963-07-03", "Money"])  # -99.99
    assert vw.loc["1963-07-03", "NoDur"] == pytest.approx(0.10)
    ew = f.blocks["Average Equal Weighted Returns -- Daily"]
    assert ew.loc["1963-07-03", "Money"] == pytest.approx(-0.001)


def test_monthly_and_annual_dates_map_to_period_end():
    text = (
        "Header text\r\n\r\n,A\r\n196307,   1.00\r\n196308,   2.00\r\n\r\n"
        " Annual Factors: January-December\r\n,A\r\n1964,   3.00\r\n"
    )
    f = french.parse_french_csv(text)
    monthly = f.blocks[french.MAIN_BLOCK]
    assert list(monthly.index) == list(pd.to_datetime(["1963-07-31", "1963-08-31"]))
    annual = f.blocks["Annual Factors: January-December"]
    assert list(annual.index) == [pd.Timestamp("1964-12-31")]
    assert f.vintage is None


def test_no_data_raises():
    with pytest.raises(ValueError, match="no data blocks"):
        french.parse_french_csv("just prose\r\nno table here\r\n")


def test_bad_row_width_raises():
    with pytest.raises(ValueError, match="columns"):
        french.parse_french_csv(",A,B\r\n19630701, 1.0\r\n")


@pytest.mark.parametrize("dates", [("1963070",), ("19630701", "196307")])
def test_unrecognized_or_mixed_date_widths_raise(dates):
    rows = "".join(f"{d}, 1.0\r\n" for d in dates)
    with pytest.raises(ValueError, match="date format"):
        french.parse_french_csv(",A\r\n" + rows)


def test_duplicate_block_title_raises():
    text = "T\r\n,A\r\n19630701, 1.0\r\n\r\nT\r\n,A\r\n19630702, 1.0\r\n"
    with pytest.raises(ValueError, match="duplicate"):
        french.parse_french_csv(text)


def test_read_zip():
    f = french.read_french_zip(zipped("ff5_daily.csv"))
    assert f.blocks[french.MAIN_BLOCK].shape == (3, 6)


def test_read_zip_needs_exactly_one_csv():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "y")
    with pytest.raises(ValueError, match="one CSV"):
        french.read_french_zip(buf.getvalue())


def test_fetch_uses_library_url_and_cache(fake_http):
    url = french.BASE_URL + french.DATASETS["ind12_daily"]
    fake_http.add(url, zipped("industry3_daily.csv"))
    f = french.fetch_french("ind12_daily")
    assert VW in f.blocks
    french.fetch_french("ind12_daily")
    assert [c["url"] for c in fake_http.calls] == [url]  # second call served from cache


def test_fetch_unknown_dataset():
    with pytest.raises(KeyError, match="unknown French dataset"):
        french.fetch_french("ind7_daily")


def test_industry_returns_selects_value_weighted(fake_http):
    url = french.BASE_URL + french.DATASETS["ind12_daily"]
    fake_http.add(url, zipped("industry3_daily.csv"))
    df = french.industry_returns(12)
    assert df.loc["1963-07-03", "NoDur"] == pytest.approx(0.10)  # VW block, not EW (0.005)


def test_industry_returns_rejects_other_universes():
    with pytest.raises(ValueError, match="12 or 49"):
        french.industry_returns(30)


def test_factors_joins_ff5_and_momentum(fake_http):
    fake_http.add(french.BASE_URL + french.DATASETS["ff5_daily"], zipped("ff5_daily.csv"))
    fake_http.add(french.BASE_URL + french.DATASETS["mom_daily"], zipped("mom_daily.csv"))
    df = french.factors()
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    assert df.loc["1963-07-01", "Mom"] == pytest.approx(0.0055)
    assert len(df) == 3  # FF5 dates drive the index
