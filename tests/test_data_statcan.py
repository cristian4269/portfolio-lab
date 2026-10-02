import json

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.data import statcan

V = "v122531"


def wds_json(vector_id: int, points: list[tuple[str, object]], status="SUCCESS") -> str:
    data = [{"refPer": ref, "value": value, "decimals": 2} for ref, value in points]
    obj = {"vectorId": vector_id, "vectorDataPoint": data}
    return json.dumps([{"status": status, "object": obj}])


@pytest.mark.parametrize(
    ("ref_per", "last_wednesday"),
    [
        ("2026-01-01", "2026-01-28"),  # month ends on a Saturday
        ("2026-02-01", "2026-02-25"),  # February, ends on a Saturday
        ("2025-12-01", "2025-12-31"),  # month ends on a Wednesday itself
        ("2025-07-01", "2025-07-30"),  # ends Thursday -> the day before
        ("1971-01-01", "1971-01-27"),  # first month of the CAD view
    ],
)
def test_last_wednesday_of_month(ref_per, last_wednesday):
    got = statcan.last_wednesday(pd.DatetimeIndex([ref_per]))
    assert got[0] == pd.Timestamp(last_wednesday)
    assert got[0].day_name() == "Wednesday"


def test_parse_stamps_on_observation_day_and_converts_percent():
    payload = wds_json(122531, [("1971-01-01", 6.25), ("1971-02-01", 5.5)])
    s = statcan.parse_wds_json(payload, V)
    assert s.name == V
    assert s.index.name == "date"
    # Observed on the last Wednesday, not the 1st the source stamps it with.
    assert list(s.index) == list(pd.to_datetime(["1971-01-27", "1971-02-24"]))
    assert s.iloc[0] == pytest.approx(0.0625)
    assert s.dtype == np.float64


def test_null_values_are_nan():
    s = statcan.parse_wds_json(wds_json(122531, [("1971-01-01", None), ("1971-02-01", 5.5)]), V)
    assert s.isna().tolist() == [True, False]


def test_failed_status_raises():
    with pytest.raises(ValueError, match="FAILED"):
        statcan.parse_wds_json(wds_json(122531, [], status="FAILED"), V)


def test_wrong_vector_raises():
    with pytest.raises(ValueError, match="122487"):
        statcan.parse_wds_json(wds_json(122487, [("1971-01-01", 7.0)]), V)


def test_no_points_raises():
    with pytest.raises(ValueError, match="no observations"):
        statcan.parse_wds_json(wds_json(122531, []), V)


def test_fetch_url_and_cache(fake_http):
    url = statcan.data_url(V)
    assert url.startswith(
        "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorByReferencePeriodRange?"
    )
    assert "vectorIds=%22122531%22" in url
    fake_http.add(url, wds_json(122531, [("1971-01-01", 6.25)]))
    statcan.fetch_vector(V)
    statcan.fetch_vector(V)
    assert len(fake_http.calls) == 1


def test_fetch_refuses_vector_not_cleared(fake_http):
    with pytest.raises(PermissionError, match="DATA_SOURCES.md"):
        statcan.fetch_vector("v122543")
    assert fake_http.calls == []
