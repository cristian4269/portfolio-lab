"""Hypothetical stress shocks priced with Step 5b's partial betas (plans/step-6-plan.md).

Each shock moves one regressor and holds every other at 0. Its P&L is linear: beta x move, with
an SE of |move| x the beta's HAC SE. The betas are those of the current weights held as a
constant mix. FX is Step 2's (S = CAD per USD, fx > 0 when the USD gains), so "CAD +10%" is
fx = 1/1.10 - 1; yield changes are in decimals (+200 bp = 0.02).
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.data.views import View, make_views
from portfolio_lab.risk import factors as fa
from portfolio_lab.stress import hypothetical as hy

ASSETS = ["A", "B"]


def days(n, start="2001-01-02"):
    return pd.bdate_range(start, periods=n, name="date")


def factor_frame(n, seed=0):
    """Synthetic daily factors (FF5 + momentum) and a constant RF, as `load_factors` names them."""
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(rng.normal(0, 0.01, (n, 6)), index=days(n), columns=list(fa.FACTORS))
    f["RF"] = 0.0001
    return f


def beta_table(rows: dict[str, dict[str, tuple[float, float]]]) -> pd.DataFrame:
    """A long table shaped like `factor_betas`' output: (asset, term) -> coef, se and the fit."""
    blocks = []
    for asset, terms in rows.items():
        block = pd.DataFrame(
            {"coef": [c for c, _ in terms.values()], "se": [s for _, s in terms.values()]},
            index=pd.MultiIndex.from_product([[asset], list(terms)], names=["asset", "term"]),
        )
        block["t"] = block["coef"] / block["se"]
        block["nobs"] = 523
        block["start"] = pd.Timestamp("2001-01-03")
        block["end"] = pd.Timestamp("2010-12-31")
        blocks.append(block)
    return pd.concat(blocks)


CAD_TABLE = beta_table(
    {
        "p": {"alpha": (0.0, 0.1), "Mkt-RF": (0.75, 0.03), "Mom": (0.04, 0.02),
              "fx": (0.99, 0.035), "dy": (-1.5, 0.4)},
        "q": {"alpha": (0.0, 0.1), "Mkt-RF": (1.0, 0.01), "Mom": (-0.01, 0.01),
              "fx": (0.97, 0.015), "dy": (-0.1, 0.15)},
    }
)  # fmt: skip
USD_TABLE = beta_table({"p": {"alpha": (0.0, 0.1), "Mkt-RF": (0.72, 0.03), "Mom": (0.02, 0.02)}})


def view(returns: pd.DataFrame, rf: pd.Series) -> View:
    return View(returns, rf)


# --- the registry ---------------------------------------------------------------------------


def test_shocks_are_the_blueprints():
    got = {key: (s.name, s.term, s.move) for key, s in hy.SHOCKS.items()}
    assert got == {
        "market": ("market", "Mkt-RF", -0.20),
        "momentum": ("momentum", "Mom", -0.35),
        "rates": ("rates", "dy", 0.02),
        "cad_up": ("cad_up", "fx", pytest.approx(-1 / 11, abs=1e-16)),
        "cad_down": ("cad_down", "fx", pytest.approx(1 / 9, abs=1e-16)),
    }
    assert all(s.note for s in hy.SHOCKS.values())


def test_shocks_are_read_only():
    with pytest.raises(TypeError):
        hy.SHOCKS["market"] = hy.Shock("market", "Mkt-RF", -0.5, "")
    with pytest.raises(dataclasses.FrozenInstanceError):
        hy.SHOCKS["market"].move = -0.5


def test_a_cad_rise_is_a_fall_in_the_usd_price_of_a_cad():
    # S is CAD per USD: the CAD gaining 10% takes S to S / 1.10, not S x 0.90.
    assert hy.cad_move(0.10) == pytest.approx(-1 / 11, abs=1e-16)
    assert hy.cad_move(-0.10) == pytest.approx(1 / 9, abs=1e-16)
    assert hy.cad_move(0.0) == 0.0
    for pct in (-0.5, -0.1, 0.03, 0.25, 2.0):
        assert (1 + hy.cad_move(pct)) * (1 + pct) == pytest.approx(1, abs=1e-15)


@pytest.mark.parametrize("bad", [-1.0, -1.5, np.nan, np.inf])
def test_cad_move_needs_a_possible_move(bad):
    with pytest.raises(ValueError):
        hy.cad_move(bad)


def test_a_shock_moves_a_known_regressor_by_a_finite_amount():
    with pytest.raises(ValueError, match="Mkt_RF"):
        hy.Shock("typo", "Mkt_RF", -0.2, "")
    with pytest.raises(ValueError, match="alpha"):
        hy.Shock("intercept", "alpha", 0.01, "")
    with pytest.raises(ValueError, match="finite"):
        hy.Shock("nan", "Mkt-RF", np.nan, "")


@pytest.mark.parametrize("bad", [True, np.bool_(False), "-0.2", None])
def test_a_shock_move_is_a_number(bad):
    with pytest.raises(TypeError, match="odd"):
        hy.Shock("odd", "Mkt-RF", bad, "")


# --- apply ----------------------------------------------------------------------------------


def test_apply_by_hand():
    market, rates = hy.SHOCKS["market"], hy.SHOCKS["rates"]
    got = hy.apply(CAD_TABLE, [market, rates])
    assert got.index.names == ["asset", "shock"]
    assert list(got.index) == [("p", "market"), ("p", "rates"), ("q", "market"), ("q", "rates")]
    row = got.loc[("p", "rates")]
    assert (row["term"], row["move"], row["beta"], row["se"]) == ("dy", 0.02, -1.5, 0.4)
    assert row["pnl"] == pytest.approx(-0.03, abs=1e-16)  # -1.5 x +200 bp
    assert row["pnl_se"] == pytest.approx(0.008, abs=1e-16)  # |0.02| x 0.4
    row = got.loc[("q", "market")]
    assert row["pnl"] == pytest.approx(-0.20, abs=1e-16)
    assert row["pnl_se"] == pytest.approx(0.002, abs=1e-16)  # |-0.20| x 0.01: the SE is >= 0
    assert (row["nobs"], row["start"], row["end"]) == (
        523,
        pd.Timestamp("2001-01-03"),
        pd.Timestamp("2010-12-31"),
    )


def test_a_cad_rise_is_a_loss_for_an_unhedged_portfolio():
    got = hy.apply(CAD_TABLE, [hy.SHOCKS["cad_up"], hy.SHOCKS["cad_down"]])
    assert got.loc[("p", "cad_up"), "pnl"] == pytest.approx(0.99 * -1 / 11, abs=1e-16)
    assert got.loc[("p", "cad_down"), "pnl"] == pytest.approx(0.99 / 9, abs=1e-16)
    assert (got.xs("cad_up", level="shock")["pnl"] < 0).all()


def test_a_usd_fit_takes_only_the_market_and_momentum_shocks():
    got = hy.apply(USD_TABLE, [hy.SHOCKS["market"], hy.SHOCKS["momentum"]])
    assert got.loc[("p", "momentum"), "pnl"] == pytest.approx(0.02 * -0.35, abs=1e-16)
    for key in ("rates", "cad_up", "cad_down"):
        with pytest.raises(ValueError, match=hy.SHOCKS[key].term):
            hy.apply(USD_TABLE, [hy.SHOCKS["market"], hy.SHOCKS[key]])


def test_apply_checks_the_terms_of_each_asset():
    mixed = pd.concat([USD_TABLE.rename(index={"p": "u"}, level="asset"), CAD_TABLE])
    with pytest.raises(ValueError, match="'u'"):
        hy.apply(mixed, [hy.SHOCKS["cad_up"]])
    got = hy.apply(mixed, [hy.SHOCKS["market"]])
    assert got.index.get_level_values("asset").tolist() == ["u", "p", "q"]


def test_apply_needs_shocks():
    with pytest.raises(ValueError, match="at least one shock"):
        hy.apply(CAD_TABLE, [])
    with pytest.raises(ValueError, match="unique"):
        hy.apply(CAD_TABLE, [hy.SHOCKS["market"], hy.SHOCKS["market"]])
    with pytest.raises(TypeError, match="Shock"):
        hy.apply(CAD_TABLE, ["market"])


@pytest.mark.parametrize(
    "bad, error",
    [
        (CAD_TABLE.to_numpy(), TypeError),
        (CAD_TABLE.reset_index(), ValueError),
        (CAD_TABLE.drop(columns="se"), ValueError),
        (CAD_TABLE.rename_axis(["portfolio", "term"]), ValueError),
    ],
)
def test_apply_needs_a_beta_table(bad, error):
    with pytest.raises(error, match="factor_betas"):
        hy.apply(bad, [hy.SHOCKS["market"]])


# --- portfolio_excess -----------------------------------------------------------------------


def test_portfolio_excess_is_the_constant_mix_of_excess_returns():
    d = days(4)
    r = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0], "B": [0.02, 0.01, -0.01, 0.04]}, index=d)
    rf = pd.Series([0.001, 0.001, 0.002, 0.0], index=d)
    w = pd.Series([0.25, 0.75], index=ASSETS, name="mix")
    got = hy.portfolio_excess(w, view(r, rf))
    expected = 0.25 * (r["A"] - rf) + 0.75 * (r["B"] - rf)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-16)
    assert got.index.equals(d)
    assert got.name == "mix"


def test_portfolio_excess_leaves_out_assets_at_weight_zero():
    d = days(4)
    r = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0], "C": [np.nan, np.nan, 0.05, 0.01]}, index=d)
    rf = pd.Series(0.0, index=d)
    got = hy.portfolio_excess(pd.Series([1.0, 0.0], index=["A", "C"]), view(r, rf))
    np.testing.assert_allclose(got.to_numpy(), r["A"].to_numpy(), rtol=0, atol=0)
    held = hy.portfolio_excess(pd.Series([0.5, 0.5], index=["A", "C"]), view(r, rf))
    assert held.isna().tolist() == [True, True, False, False]  # 5b drops NaN edges only
    assert held.name == "portfolio"  # unnamed weights


@pytest.mark.parametrize(
    "w, error, match",
    [
        (pd.Series([0.4, 0.6], index=["B", "A"]), ValueError, "same assets"),
        (pd.Series([1.2, -0.2], index=ASSETS), ValueError, ">= 0"),
        (pd.Series([0.5, 0.4], index=ASSETS), ValueError, "sum to 1"),
        ([0.5, 0.5], TypeError, "Series"),
    ],
)
def test_portfolio_excess_checks_the_weights(w, error, match):
    d = days(3)
    r = pd.DataFrame(0.01, index=d, columns=ASSETS)
    with pytest.raises(error, match=match):
        hy.portfolio_excess(w, view(r, pd.Series(0.0, index=d)))


def test_portfolio_excess_needs_a_view():
    with pytest.raises(TypeError, match="View"):
        hy.portfolio_excess(pd.Series([0.5, 0.5], index=ASSETS), pd.DataFrame())


# --- end to end on synthetic data -----------------------------------------------------------

N = 2600  # about ten years of days, ~520 weeks as in the dev sample


def synthetic_cad():
    """CAD-view excess returns linear in the factors, fx and dy, priced in two assets."""
    rng = np.random.default_rng(11)
    f = factor_frame(N, seed=5)
    d = f.index
    fx = pd.Series(rng.normal(0, 0.006, N), index=d)
    dy = pd.Series(rng.normal(0, 0.0005, N), index=d)  # 5 bp a day
    rf = pd.Series(0.00008, index=d)
    excess = pd.DataFrame(
        {
            "A": 0.6 * f["Mkt-RF"] - 0.2 * f["Mom"] + fx - 5 * dy + rng.normal(0, 0.004, N),
            "B": 1.0 * f["Mkt-RF"] - 0.4 * f["Mom"] + fx - 9 * dy + rng.normal(0, 0.004, N),
        }
    )
    return f, fx, dy, view(excess.add(rf, axis=0), rf)


def test_shock_pnl_recovers_the_true_betas_of_the_constant_mix():
    f, fx, dy, cad = synthetic_cad()
    port = hy.portfolio_excess(pd.Series([0.5, 0.5], index=ASSETS, name="mix"), cad)
    betas = fa.macro_betas(port, dy, fx, f, frequency="weekly")
    got = hy.apply(betas, hy.SHOCKS.values()).xs("mix", level="asset")
    truth = {"Mkt-RF": 0.8, "Mom": -0.3, "fx": 1.0, "dy": -7.0}  # the average of A and B
    for shock in hy.SHOCKS.values():
        row = got.loc[shock.name]
        assert abs(row["pnl"] - truth[shock.term] * shock.move) < 4 * row["pnl_se"], shock.name
    assert got.loc["cad_up", "pnl"] == pytest.approx(-1 / 11, abs=4 * got.loc["cad_up", "pnl_se"])


def test_the_cad_view_of_usd_assets_loses_when_the_cad_rises():
    # Built through Step 2's make_views: r_cad = (1 + r_usd)(1 + fx) - 1 with S in CAD per USD.
    rng = np.random.default_rng(3)
    f = factor_frame(N, seed=8)
    d = f.index
    r_usd = pd.DataFrame(
        {a: 0.8 * f["Mkt-RF"] + f["RF"] + rng.normal(0, 0.004, N) for a in ASSETS}, index=d
    )
    usdcad = pd.Series(1.25 * np.cumprod(1 + rng.normal(0, 0.006, N)), index=d)
    ca_tbill = pd.Series(0.02, index=d)
    cad = make_views(r_usd, f["RF"], usdcad, ca_tbill)["CAD"]
    fx = usdcad / usdcad.shift(1) - 1
    dy = pd.Series(rng.normal(0, 0.0005, N), index=d)
    port = hy.portfolio_excess(pd.Series([0.5, 0.5], index=ASSETS, name="usd assets"), cad)
    got = hy.apply(fa.macro_betas(port, dy, fx, f, frequency="weekly"), hy.SHOCKS.values())
    up, down = got.loc[("usd assets", "cad_up")], got.loc[("usd assets", "cad_down")]
    assert up["beta"] == pytest.approx(1.0, abs=4 * up["se"])  # unhedged: the mechanical 1
    assert up["pnl"] < 0 < down["pnl"]
    assert up["pnl"] == pytest.approx(-1 / 11, abs=4 * up["pnl_se"])


def test_usd_factor_betas_take_the_market_and_momentum_shocks():
    f, _, _, cad = synthetic_cad()
    port = hy.portfolio_excess(pd.Series([0.5, 0.5], index=ASSETS, name="mix"), cad)
    usd = fa.factor_betas(port, f, frequency="weekly")  # no fx or dy in the model
    got = hy.apply(usd, [hy.SHOCKS["market"], hy.SHOCKS["momentum"]])
    assert list(got.index) == [("mix", "market"), ("mix", "momentum")]
    with pytest.raises(ValueError, match="dy"):
        hy.apply(usd, [hy.SHOCKS["rates"]])
