"""Statistics Canada Web Data Service: monthly Bank of Canada rates back to before 1971.

Terms: Statistics Canada Open Licence; adapted use is cited as "Adapted from
Statistics Canada, <product>, <reference date>. This does not constitute an
endorsement by Statistics Canada of this product."

Timing: table 10-10-0122-01 values are observed on the last Wednesday of the
month but the service stamps them with the 1st (`refPer` = YYYY-MM-01). Using
the 1st would put a rate up to four weeks before it existed, so each value is
re-stamped on its month's last Wednesday here.
"""

import json
from urllib.parse import urlencode

import pandas as pd

from portfolio_lab.data import _checks, cache

WDS_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorByReferencePeriodRange"
CITATION = (
    "Adapted from Statistics Canada, Table 10-10-0122-01 Financial market statistics, "
    "last Wednesday unless otherwise stated, Bank of Canada (DOI 10.25318/1010012201-eng). "
    "This does not constitute an endorsement by Statistics Canada of this product."
)
# Vector -> description. All are yields in percent per year, monthly, last Wednesday.
APPROVED = {
    "v122531": "Treasury bills: 3 month",
    "v122487": "Government of Canada marketable bonds, average yield: over 10 years",
}
# Wide enough to return every observation the vector has.
FIRST_PERIOD, LAST_PERIOD = "1900-01-01", "2100-12-01"
WEDNESDAY = 2


def data_url(vector: str) -> str:
    query = {
        "vectorIds": f'"{_vector_id(vector)}"',
        "startRefPeriod": FIRST_PERIOD,
        "endReferencePeriod": LAST_PERIOD,
    }
    return f"{WDS_URL}?{urlencode(query)}"


def _vector_id(vector: str) -> int:
    return int(vector.lower().removeprefix("v"))


def last_wednesday(months: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last Wednesday of each date's month."""
    month_end = months + pd.offsets.MonthEnd(0)
    days_back = (month_end.weekday - WEDNESDAY) % 7
    return month_end - pd.to_timedelta(days_back, unit="D")


def parse_wds_json(text: str, vector: str) -> pd.Series:
    """Parse a WDS vector payload into decimal yields stamped on their observation day."""
    (entry,) = json.loads(text)
    if entry.get("status") != "SUCCESS":
        raise ValueError(f"{vector}: Statistics Canada returned status {entry.get('status')!r}")
    obj = entry["object"]
    if obj.get("vectorId") != _vector_id(vector):
        raise ValueError(f"{vector}: payload is for vector {obj.get('vectorId')}")
    points = obj.get("vectorDataPoint") or []
    refs = [p["refPer"] for p in points]
    values = [
        _checks.to_float(p["value"], (None,), f"{vector} {p['refPer']}") / 100.0 for p in points
    ]
    index = last_wednesday(pd.DatetimeIndex(pd.to_datetime(refs, format="%Y-%m-%d")))
    return _checks.build_series(index, values, vector)


def fetch_vector(vector: str, *, refresh: bool = False) -> pd.Series:
    _checks.require_approved(vector, APPROVED, "Statistics Canada")
    content = cache.fetch(data_url(vector), refresh=refresh)
    return parse_wds_json(content.decode("utf-8"), vector)
