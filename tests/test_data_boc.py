import json

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.data import boc

S = "BD.CDN.10YR.DQ.YLD"


def valet_json(series: str, rows: list[tuple[str, object]]) -> str:
    """Valet observations payload; a value of None leaves the series key out of the row."""
    obs = [{"d": d} | ({} if v is None else {series: {"v": v}}) for d, v in rows]
    return json.dumps({"seriesDetail": {series: {"label": "x"}}, "observations": obs})


def test_parse_percent_to_decimal():
    s = boc.parse_valet_json(valet_json(S, [("2001-01-02", "5.28"), ("2001-01-03", "5.43")]), S)
    assert s.name == S
    assert s.index.name == "date"
    assert s.index[0] == pd.Timestamp("2001-01-02")
    assert s.dtype == np.float64
    assert s.loc["2001-01-03"] == pytest.approx(0.0543)


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_observations_are_nan(missing):
    s = boc.parse_valet_json(valet_json(S, [("2001-01-02", "5.28"), ("2001-01-03", missing)]), S)
    assert s.isna().tolist() == [False, True]


def test_non_numeric_value_raises_with_date():
    with pytest.raises(ValueError, match="2001-01-03"):
        boc.parse_valet_json(valet_json(S, [("2001-01-02", "5"), ("2001-01-03", "n/a")]), S)


def test_payload_without_observations_raises():
    with pytest.raises(ValueError, match="no observations"):
        boc.parse_valet_json(json.dumps({"message": "Series not found"}), S)


def test_unsorted_dates_raise():
    with pytest.raises(ValueError, match="increasing"):
        boc.parse_valet_json(valet_json(S, [("2001-01-03", "5"), ("2001-01-02", "5")]), S)


def test_fetch_uses_full_history_url_and_cache(fake_http):
    url = boc.observations_url(S)
    assert url == f"https://www.bankofcanada.ca/valet/observations/{S}/json"
    fake_http.add(url, valet_json(S, [("2001-01-02", "5.28")]))
    boc.fetch_valet(S)
    boc.fetch_valet(S)
    assert len(fake_http.calls) == 1  # one request per series, then cache (Valet terms)


def test_fetch_refuses_series_not_cleared(fake_http):
    with pytest.raises(PermissionError, match="DATA_SOURCES.md"):
        boc.fetch_valet("FXUSDCAD")
    assert fake_http.calls == []


def test_unparseable_download_is_not_cached(fake_http):
    url = boc.observations_url(S)
    fake_http.add(url, json.dumps({"message": "Series not found"}))
    with pytest.raises(ValueError):
        boc.fetch_valet(S)
    fake_http.add(url, valet_json(S, [("2001-01-02", "5.28")]))
    assert len(boc.fetch_valet(S)) == 1
