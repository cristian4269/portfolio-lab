"""Kenneth R. French Data Library: download and parse the CSV zip files.

Terms (owner decision G1): fetched at runtime into the local cache, never
committed, always cited; only derived analytics are published.

File layout: free-text header (the first line names the CRSP vintage), then
one or more blocks. A block is an optional title line, a header line starting
with a comma, and rows `date, v1, v2, ...` with dates as YYYYMMDD (daily),
YYYYMM (monthly) or YYYY (annual). Values are percent; -99.99 and -999 mean
missing. A copyright line closes the file.
"""

import io
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_lab.data import cache

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DATASETS = {
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "ind12_daily": "12_Industry_Portfolios_daily_CSV.zip",
    "ind49_daily": "49_Industry_Portfolios_daily_CSV.zip",
}
CITATION = (
    "Kenneth R. French Data Library, "
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
)

MAIN_BLOCK = "main"  # name of a block that has no title line
VALUE_WEIGHTED = "Average Value Weighted Returns -- Daily"
MISSING_CODES = (-99.99, -999.0)

_VINTAGE = re.compile(r"\b(\d{6}) CRSP database")
_DATE_FORMATS = {8: "%Y%m%d", 6: "%Y%m", 4: "%Y"}


@dataclass(frozen=True)
class FrenchFile:
    """Parsed file: blocks by title (decimal returns, DatetimeIndex) and CRSP vintage."""

    blocks: dict[str, pd.DataFrame]
    vintage: str | None


def parse_french_csv(text: str) -> FrenchFile:
    """Parse the text of one library CSV into its blocks; raises on anything unexpected."""
    lines = [line.strip() for line in text.splitlines()]
    match = _VINTAGE.search("\n".join(lines[:5]))
    blocks: dict[str, pd.DataFrame] = {}
    for i, line in enumerate(lines):
        if not line.startswith(","):
            continue
        title = lines[i - 1] if i > 0 and lines[i - 1] else MAIN_BLOCK
        if title in blocks:
            raise ValueError(f"duplicate block title {title!r}")
        blocks[title] = _parse_block(title, line, lines[i + 1 :])
    if not blocks:
        raise ValueError("no data blocks found (no header line starting with ',')")
    return FrenchFile(blocks=blocks, vintage=match.group(1) if match else None)


def _parse_block(title: str, header: str, rest: list[str]) -> pd.DataFrame:
    columns = [c.strip() for c in header.split(",")[1:]]
    dates: list[str] = []
    rows: list[list[float]] = []
    for line in rest:
        fields = [f.strip() for f in line.split(",")]
        if not fields[0].isdigit():
            # A block ends at a blank line (or end of file); any other line inside
            # it would otherwise cut the block short without notice.
            if rows and line:
                raise ValueError(f"unexpected line inside block {title!r}: {line[:60]!r}")
            break
        if len(fields) - 1 != len(columns):
            raise ValueError(
                f"row {fields[0]} has {len(fields) - 1} columns, header has {len(columns)}"
            )
        try:
            row = [float(f) for f in fields[1:]]
        except ValueError:
            row = [np.nan]
        if not np.isfinite(row).all():  # unparseable, or the strings "nan" / "inf"
            raise ValueError(f"row {fields[0]}: cannot parse value in {fields[1:]}")
        rows.append(row)
        dates.append(fields[0])
    if not rows:
        raise ValueError(f"no data rows under the header of block {title!r}")

    values = np.array(rows, dtype=np.float64).reshape(len(rows), len(columns))
    values[np.isin(values, MISSING_CODES)] = np.nan
    return pd.DataFrame(values / 100.0, index=_parse_dates(dates), columns=columns)


def _parse_dates(raw: list[str]) -> pd.DatetimeIndex:
    width = len(raw[0]) if raw else 8
    if width not in _DATE_FORMATS or any(len(d) != width for d in raw):
        raise ValueError(f"unrecognized date format in block starting {raw[:1]}")
    index = pd.DatetimeIndex(pd.to_datetime(raw, format=_DATE_FORMATS[width]), name="date")
    if width == 6:
        index = index + pd.offsets.MonthEnd(0)
    elif width == 4:
        index = index + pd.offsets.YearEnd(0)
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(f"dates must be strictly increasing in block starting {raw[:1]}")
    return index


def read_french_zip(content: bytes) -> FrenchFile:
    """Parse a library zip, which must hold exactly one CSV file."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in the zip, found {names}")
        text = zf.read(names[0]).decode("latin-1")
    return parse_french_csv(text)


def fetch_french(dataset: str, *, refresh: bool = False) -> FrenchFile:
    """Download (or read from cache) and parse one dataset named in DATASETS."""
    if dataset not in DATASETS:
        raise KeyError(f"unknown French dataset {dataset!r}; known: {sorted(DATASETS)}")
    return cache.fetch_parsed(BASE_URL + DATASETS[dataset], read_french_zip, refresh=refresh)


def industry_returns(n: int = 12, *, refresh: bool = False) -> pd.DataFrame:
    """Daily value-weighted industry portfolio returns (decimal), 12 or 49 industries."""
    if n not in (12, 49):
        raise ValueError(f"industry universe must be 12 or 49, got {n}")
    return fetch_french(f"ind{n}_daily", refresh=refresh).blocks[VALUE_WEIGHTED]


def factors(*, refresh: bool = False) -> pd.DataFrame:
    """Daily FF5 factors plus momentum, and the US risk-free rate RF (decimal).

    Both files must come from the same CRSP vintage, and momentum must cover
    every FF5 date (momentum starts earlier, in 1926; its extra dates are dropped).
    """
    ff5_file = fetch_french("ff5_daily", refresh=refresh)
    mom_file = fetch_french("mom_daily", refresh=refresh)
    if ff5_file.vintage != mom_file.vintage:
        raise ValueError(
            f"French vintages differ: FF5 {ff5_file.vintage}, momentum {mom_file.vintage}; "
            "refresh both files"
        )
    ff5, mom = ff5_file.blocks[MAIN_BLOCK], mom_file.blocks[MAIN_BLOCK]
    gaps = ff5.index.difference(mom.index)
    if len(gaps):
        raise ValueError(f"momentum is missing {len(gaps)} FF5 dates, first {gaps[0].date()}")
    out = ff5.join(mom)
    return out[[c for c in out.columns if c != "RF"] + ["RF"]]
