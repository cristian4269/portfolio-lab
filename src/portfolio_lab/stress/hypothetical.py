"""Hypothetical stress tests: one-variable shocks priced with Step 5b's partial betas (Step 6).

A shock (`SHOCKS`) moves one regressor of the 5b model and holds every other at 0. Its P&L
(`apply`) is linear, beta x move: an excess return, as a fraction of the portfolio's value. Its
SE is |move| x the beta's HAC SE (the move is fixed, not estimated).

The betas are those of the current weights held as a constant mix (`portfolio_excess`), fitted
weekly and partial over the beta sample (2001-01-03 to DEV_END in Step 6): `macro_betas` (the
six factors, fx and dy) for the CAD view, `factor_betas` for USD. Not the strategy's own return
history, which averages the exposures of its past portfolios.

Units: `dy` is the change in the Canadian 10-year yield in decimals (+200 bp = 0.02). `fx` is
Step 2's USD/CAD return with S in CAD per USD, so the CAD gaining 10% is fx = 1/1.10 - 1
(`cad_move`): a loss for an unhedged CAD investor in USD assets.

Caveats, stated rather than modelled:
- Partial: the market shock holds fx at 0, but the USD has tended to rally against the CAD in
  crises (+16.3% over the 2008 window). That is a natural hedge for a CAD investor, which this
  shock leaves out and the historical replay keeps.
- Linear: the momentum shock applies a weekly beta to a 3-month cumulative move, assuming the
  beta is stable and ignoring compounding and the path.
- One variable at a time: 5b's fits keep no covariance between betas, so a combined shock would
  have no SE. Partial betas add, so single shocks can still be summed by hand.
"""

import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from portfolio_lab._validate import check_finite, portfolio_values
from portfolio_lab.data.views import View, check_view
from portfolio_lab.risk.factors import DY, FACTORS, FX

__all__ = ["SHOCKS", "TERMS", "Shock", "apply", "cad_move", "portfolio_excess"]

TERMS = (*FACTORS, FX, DY)  # the regressors a shock may move
_TABLE_COLUMNS = ("coef", "se", "nobs", "start", "end")


@dataclass(frozen=True)
class Shock:
    """One regressor of the 5b model (`term`) moving by `move`, every other held at 0."""

    name: str
    term: str
    move: float
    note: str

    def __post_init__(self) -> None:
        if self.term not in TERMS:
            raise ValueError(
                f"shock {self.name!r} moves {self.term!r}: a shock moves one of {list(TERMS)}"
            )
        if not isinstance(self.move, numbers.Real) or isinstance(self.move, bool | np.bool_):
            raise TypeError(
                f"the move of shock {self.name!r} must be a number, got {type(self.move).__name__}"
            )
        check_finite(self.move, f"the move of shock {self.name!r}")
        object.__setattr__(self, "move", float(self.move))


def cad_move(pct: float) -> float:
    """The fx return (Step 2's: S = CAD per USD) when the CAD moves by `pct` against the USD.

    The CAD gaining 10% (pct = 0.10) takes S to S / 1.10, so fx = 1/1.10 - 1 = -9.09%, not
    -10%. `pct` must be finite and above -1.
    """
    check_finite(pct, "pct")
    if pct <= -1:
        raise ValueError(f"the CAD cannot lose 100% or more of its value, got pct={pct}")
    return 1 / (1 + pct) - 1


SHOCKS: Mapping[str, Shock] = MappingProxyType(  # read-only: pre-registered (blueprint)
    {
        s.name: s
        for s in (
            Shock("market", "Mkt-RF", -0.20, "The US market's excess return falls 20%."),
            Shock(
                "momentum",
                "Mom",
                -0.35,
                "Momentum crash: UMD falls 35% over 3 months (cf. 2009), as one cumulative move.",
            ),
            Shock("rates", DY, 0.02, "The Canadian 10-year yield rises 200 bp."),
            Shock("cad_up", FX, cad_move(0.10), "The CAD gains 10% on the USD: fx = 1/1.10 - 1."),
            Shock(
                "cad_down", FX, cad_move(-0.10), "The CAD loses 10% on the USD: fx = 1/0.90 - 1."
            ),
        )
    }
)


def portfolio_excess(weights: pd.Series, view: View) -> pd.Series:
    """The excess return of `weights` held as a constant mix: sum_i w_i (r_i - rf), each day.

    `weights`: long-only, summing to 1, labeled by the view's assets in their order. Assets at
    weight 0 are left out first, so one not yet listed (NaN) does not spoil the series; NaN in a
    held asset stays NaN (5b's fits drop NaN only at the edges). The Series is named after
    `weights`, or "portfolio".
    """
    check_view(view)
    w = portfolio_values(weights, view.returns.columns, "weights", "the view's returns")
    held = w > 0
    values = view.excess.to_numpy(dtype=float)[:, held] @ w[held]
    name = weights.name if weights.name is not None else "portfolio"
    return pd.Series(values, index=view.returns.index, name=name)


def apply(betas: pd.DataFrame, shocks: Iterable[Shock]) -> pd.DataFrame:
    """Each shock's P&L for each portfolio of a beta table: one row per (asset, shock).

    `betas`: the long table of `factor_betas` or `macro_betas`, indexed by (asset, term) with
    coef, se, nobs, start and end. Columns: the shock's `term` and `move`; that term's `beta` and
    `se`; `pnl` = beta x move and `pnl_se` = |move| x se; and the fit's nobs, start and end, so
    the sample is visible. A shock that moves a term the table lacks raises: USD fits have no fx
    or dy.
    """
    _check_table(betas)
    chosen = list(shocks)
    if not chosen:
        raise ValueError("apply needs at least one shock")
    for s in chosen:
        if not isinstance(s, Shock):
            raise TypeError(f"shocks must be Shock objects (see SHOCKS), got {type(s).__name__}")
    names = [s.name for s in chosen]
    if len(set(names)) < len(names):
        raise ValueError(f"shock names must be unique (rows are keyed by them): {names}")
    moved = [s.term for s in chosen]
    moves = np.array([s.move for s in chosen])
    fits = {asset: betas.xs(asset, level="asset") for asset in betas.index.unique(level="asset")}
    for asset, fit in fits.items():  # each fit's own terms: a table may mix USD and CAD fits
        terms = list(fit.index)
        for s in chosen:
            if s.term not in terms:
                raise ValueError(
                    f"shock {s.name!r} moves {s.term!r}, a term the betas of {asset!r} lack: "
                    f"{terms}"
                )

    blocks = []
    for asset, all_terms in fits.items():
        fit = all_terms.loc[moved]  # one row per shock, in their order
        beta, se = fit["coef"].to_numpy(dtype=float), fit["se"].to_numpy(dtype=float)
        index = pd.MultiIndex.from_product([[asset], names], names=["asset", "shock"])
        blocks.append(
            pd.DataFrame(
                {
                    "term": moved,
                    "move": moves,
                    "beta": beta,
                    "se": se,
                    "pnl": beta * moves,
                    "pnl_se": np.abs(moves) * se,
                    "nobs": fit["nobs"].to_numpy(),
                    "start": fit["start"].to_numpy(),
                    "end": fit["end"].to_numpy(),
                },
                index=index,
            )
        )
    return pd.concat(blocks)


def _check_table(betas: object) -> None:
    what = "betas must be the table of factor_betas or macro_betas"
    if not isinstance(betas, pd.DataFrame):
        raise TypeError(f"{what}, got {type(betas).__name__}")
    if list(betas.index.names) != ["asset", "term"]:
        raise ValueError(f"{what}, indexed by (asset, term), not {list(betas.index.names)}")
    missing = [col for col in _TABLE_COLUMNS if col not in betas.columns]
    if missing:
        raise ValueError(f"{what}: the columns {missing} are missing")
