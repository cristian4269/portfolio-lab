"""FRED (Federal Reserve Bank of St. Louis): keyless CSV download of approved series.

Terms: each series carries its own copyright line; only "Public Domain:
Citation Requested" series are used, cited as "Source: <original> via FRED".
Missing days are an empty field in the current CSV layout and `.` in older ones.
"""

import io

import pandas as pd

from portfolio_lab.data import _checks, cache

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
# Series -> citation. Add a series only after recording its verdict in DATA_SOURCES.md.
APPROVED = {
    "DEXCAUS": "Source: Board of Governors of the Federal Reserve System (US) via FRED",
}
MISSING = ("", ".")


def csv_url(series: str) -> str:
    return CSV_URL.format(series=series)


def parse_fred_csv(text: str, series: str) -> pd.Series:
    """Parse a fredgraph CSV (date column, then one value column named `series`)."""
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if list(frame.columns[1:]) != [series]:
        raise ValueError(f"expected column {series!r}, got {list(frame.columns[1:])}")
    dates = frame.iloc[:, 0].tolist()
    values = [
        _checks.to_float(raw.strip(), MISSING, f"{series} {d}")
        for d, raw in zip(dates, frame[series], strict=True)
    ]
    index = pd.DatetimeIndex(pd.to_datetime(dates, format="%Y-%m-%d"))
    return _checks.build_series(index, values, series)


def fetch_fred(series: str, *, refresh: bool = False) -> pd.Series:
    _checks.require_approved(series, APPROVED, "FRED")
    content = cache.fetch(csv_url(series), refresh=refresh)
    return parse_fred_csv(content.decode("utf-8-sig"), series)


def usdcad(*, refresh: bool = False) -> pd.Series:
    """USD/CAD as CAD per 1 USD (H.10 noon New York rate), a level, not a return."""
    s = fetch_fred("DEXCAUS", refresh=refresh).rename("USDCAD")
    if (s.dropna() <= 0).any():
        raise ValueError("USDCAD: exchange rates must be positive")
    return s
