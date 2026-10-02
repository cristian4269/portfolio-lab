"""Bank of Canada Valet API: daily series, full history in one request per series.

Terms: attribute the Bank of Canada and say what was changed (values are
converted from percent to decimal here); do not circumvent request limits, so
each series is fetched once and then served from the cache.
"""

import json

import pandas as pd

from portfolio_lab.data import _checks, cache

OBSERVATIONS_URL = "https://www.bankofcanada.ca/valet/observations/{series}/json"
CITATION = "Source: Bank of Canada (Valet API). Converted from percent to decimal by portfolio-lab."
# Series -> description. All are yields in percent per year.
APPROVED = {
    "BD.CDN.10YR.DQ.YLD": "Government of Canada 10-year benchmark bond yield, daily from 2001",
    "TB.CDN.90D.MID": "3-month treasury bill mid yield, daily from 2001 (cross-check only)",
}
MISSING = (None, "")


def observations_url(series: str) -> str:
    return OBSERVATIONS_URL.format(series=series)


def parse_valet_json(text: str, series: str) -> pd.Series:
    """Parse a Valet observations payload into decimal yields (percent / 100)."""
    observations = json.loads(text).get("observations") or []
    dates, values = [], []
    for obs in observations:
        raw = obs.get(series, {}).get("v")
        dates.append(obs["d"])
        values.append(_checks.to_float(raw, MISSING, f"{series} {obs['d']}") / 100.0)
    index = pd.DatetimeIndex(pd.to_datetime(dates, format="%Y-%m-%d"))
    return _checks.build_series(index, values, series)


def fetch_valet(series: str, *, refresh: bool = False) -> pd.Series:
    _checks.require_approved(series, APPROVED, "Bank of Canada Valet")
    content = cache.fetch(observations_url(series), refresh=refresh)
    return parse_valet_json(content.decode("utf-8"), series)
