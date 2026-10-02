"""The public loaders: holdout guard, truncation, and point-in-time daily rates."""

import numpy as np
import pandas as pd
import pytest

from portfolio_lab import data
from portfolio_lab.config import DEV_END
from portfolio_lab.data import boc, fred, french, statcan

DEV = pd.Timestamp(DEV_END)
DAYS = pd.bdate_range("2010-12-20", "2011-01-14", name="date")  # straddles DEV_END


def frame(columns):
    values = np.arange(len(DAYS) * len(columns), dtype=float).reshape(len(DAYS), -1) / 1e4
    return pd.DataFrame(values, index=DAYS, columns=columns)


@pytest.fixture
def sources(monkeypatch):
    """Synthetic stand-ins for every source function; records the arguments they get."""
    calls: list[tuple] = []

    def industries(n, *, refresh=False):
        calls.append(("industries", n, refresh))
        return frame([f"I{i}" for i in range(n)])

    def factors(*, refresh=False):
        calls.append(("factors", refresh))
        return frame(["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"])

    def usdcad(*, refresh=False):
        calls.append(("usdcad", refresh))
        return pd.Series(1.3, index=DAYS, name="USDCAD")

    monthly = pd.DatetimeIndex(["2010-11-24", "2010-12-29", "2011-01-26"], name="date")

    def vector(v, *, refresh=False):
        calls.append(("statcan", v, refresh))
        level = {"v122531": 0.01, "v122487": 0.03}[v]
        return pd.Series([level, level + 0.001, level + 0.002], index=monthly, name=v)

    def valet(series, *, refresh=False):
        calls.append(("valet", series, refresh))
        return pd.Series(0.035, index=DAYS, name=series)

    monkeypatch.setattr(french, "industry_returns", industries)
    monkeypatch.setattr(french, "factors", factors)
    monkeypatch.setattr(fred, "usdcad", usdcad)
    monkeypatch.setattr(statcan, "fetch_vector", vector)
    monkeypatch.setattr(boc, "fetch_valet", valet)
    return calls


LOADERS = [data.load_assets, data.load_factors, data.load_fx, data.load_rates]


@pytest.mark.parametrize("loader", LOADERS)
def test_default_stops_at_dev_end(sources, loader):
    out = loader()
    assert out.index.max() == DEV  # 2010-12-31 is a Friday, so it is in the data
    assert not out.empty


@pytest.mark.parametrize("loader", LOADERS)
def test_earlier_end_truncates_earlier(sources, loader):
    assert loader(end="2010-12-24").index.max() == pd.Timestamp("2010-12-24")


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize(
    "end", ["2011-01-01", "2011-01-14", pd.Timestamp("2010-12-31 00:00:01"), None]
)
def test_holdout_needs_the_flag(sources, loader, end):
    with pytest.raises(data.HoldoutError, match="research/PLAN.md"):
        loader(end=end)
    assert sources == []  # refused before any download


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("end", ["", "NaT", pd.NaT, float("nan")])
def test_not_a_date_end_is_refused(sources, loader, allow, end):
    # Regression: these all became NaT, and `.loc[:NaT]` returns the whole series.
    with pytest.raises(ValueError, match="not a date"):
        loader(end=end, allow_holdout=allow)
    assert sources == []  # refused before any download


@pytest.mark.parametrize("loader", LOADERS)
def test_flag_alone_does_not_move_the_end(sources, loader):
    # Opening the holdout takes both the flag and an explicit later end.
    assert loader(allow_holdout=True).index.max() == DEV


@pytest.mark.parametrize("loader", LOADERS)
def test_flag_with_later_end_reaches_into_holdout(sources, loader):
    assert loader(end="2011-01-07", allow_holdout=True).index.max() == pd.Timestamp("2011-01-07")


@pytest.mark.parametrize("loader", [data.load_assets, data.load_factors, data.load_fx])
def test_end_none_with_flag_returns_everything(sources, loader):
    assert loader(end=None, allow_holdout=True).index.max() == DAYS[-1]


def test_load_assets_universe_and_refresh_pass_through(sources):
    assert data.load_assets(49, refresh=True).shape[1] == 49
    assert sources == [("industries", 49, True)]


def test_load_factors_columns(sources):
    assert list(data.load_factors().columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]


def test_load_fx_is_usdcad_series(sources):
    fx = data.load_fx()
    assert fx.name == "USDCAD"
    assert isinstance(fx, pd.Series)


def test_load_rates_columns_and_sources(sources):
    rates = data.load_rates(refresh=True)
    assert list(rates.columns) == ["CA_TBILL_3M", "CA_GOVT_LONG", "CA_GOVT_10Y"]
    assert ("statcan", "v122531", True) in sources
    assert ("statcan", "v122487", True) in sources
    assert ("valet", "BD.CDN.10YR.DQ.YLD", True) in sources


def test_load_rates_monthly_values_are_point_in_time(sources):
    rates = data.load_rates()
    tbill = rates["CA_TBILL_3M"]
    # December's value is observed Wed 2010-12-29: unusable that day, usable the next.
    assert tbill.loc["2010-12-29"] == pytest.approx(0.010)
    assert tbill.loc["2010-12-30"] == pytest.approx(0.011)
    assert tbill.loc["2010-12-31"] == pytest.approx(0.011)
    assert rates["CA_GOVT_LONG"].loc["2010-12-30"] == pytest.approx(0.031)
    assert rates["CA_GOVT_10Y"].loc["2010-12-30"] == pytest.approx(0.035)


def test_load_rates_with_holdout_includes_later_stamps(sources):
    rates = data.load_rates(end=None, allow_holdout=True)
    # January's value (stamped Wed 2011-01-26) is usable from Thu 2011-01-27.
    assert rates["CA_TBILL_3M"].loc["2011-01-26"] == pytest.approx(0.011)
    assert rates["CA_TBILL_3M"].loc["2011-01-27"] == pytest.approx(0.012)
    assert rates.index.max() == pd.Timestamp("2011-01-27")


def test_load_rates_daily_series_stay_on_their_observation_day(monkeypatch, sources):
    # Convention: row t holds what was observed by the close of t. A daily close
    # stays on its own day (the backtest adds the t+1 trade lag); only the monthly
    # series get the extra one-business-day margin.
    stepped = pd.Series(0.035, index=DAYS, name="BD.CDN.10YR.DQ.YLD")
    stepped.loc["2010-12-29":] = 0.040
    monkeypatch.setattr(boc, "fetch_valet", lambda s, *, refresh=False: stepped)
    rates = data.load_rates()
    assert rates["CA_GOVT_10Y"].loc["2010-12-28"] == pytest.approx(0.035)
    assert rates["CA_GOVT_10Y"].loc["2010-12-29"] == pytest.approx(0.040)


def test_load_rates_end_before_any_data_raises(sources):
    with pytest.raises(ValueError, match="before the first"):
        data.load_rates(end="1900-01-01")


class TestPointInTimeDaily:
    calendar = pd.bdate_range("1971-01-25", "1971-03-02", name="date")
    monthly = pd.Series(
        [0.06, 0.05], index=pd.DatetimeIndex(["1971-01-27", "1971-02-24"]), name="x"
    )

    def test_value_starts_the_business_day_after_its_stamp(self):
        s = data.point_in_time_daily(self.monthly, self.calendar)
        assert s.loc["1971-01-25":"1971-01-27"].isna().all()
        assert s.loc["1971-01-28"] == pytest.approx(0.06)
        assert s.loc["1971-02-24"] == pytest.approx(0.06)
        assert s.loc["1971-02-25"] == pytest.approx(0.05)
        assert s.index.equals(self.calendar)
        assert s.name == "x"

    def test_friday_stamp_is_usable_monday(self):
        monthly = pd.Series([0.07], index=pd.DatetimeIndex(["1971-01-29"]))  # a Friday
        s = data.point_in_time_daily(monthly, self.calendar)
        assert np.isnan(s.loc["1971-01-29"])
        assert s.loc["1971-02-01"] == pytest.approx(0.07)

    def test_next_day_missing_from_calendar_rolls_forward(self):
        calendar = self.calendar.drop(pd.Timestamp("1971-01-28"))  # e.g. a market holiday
        s = data.point_in_time_daily(self.monthly, calendar)
        assert s.loc["1971-01-29"] == pytest.approx(0.06)

    def test_missing_monthly_value_keeps_last_observed_rate(self):
        monthly = pd.Series([0.06, np.nan], index=pd.DatetimeIndex(["1971-01-27", "1971-02-24"]))
        s = data.point_in_time_daily(monthly, self.calendar)
        # A missing month keeps the last observed rate (it is still the latest known value).
        assert s.loc["1971-03-01"] == pytest.approx(0.06)

    def test_non_datetime_index_raises(self):
        # Date strings are not parsed silently: the observation day must be explicit.
        monthly = pd.Series([0.06], index=["1971-01-27"], name="x")
        with pytest.raises(TypeError, match="DatetimeIndex"):
            data.point_in_time_daily(monthly, self.calendar)


def test_truncation_does_not_depend_on_sort_order(monkeypatch, sources):
    # Regression: `.loc[:end]` on an unsorted index keeps rows after `end`.
    unsorted = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2010-12-30", "2011-01-03", "2010-12-31"])
    )
    monkeypatch.setattr(french, "industry_returns", lambda n, *, refresh=False: unsorted)
    assert list(data.load_assets().index) == list(pd.to_datetime(["2010-12-30", "2010-12-31"]))
