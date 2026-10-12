"""The report's figures (blueprint Step 7b; plans/step-7b-plan.md decision 6).

Pure functions from result tables to matplotlib figures, one per figure of research/REPORT.md,
and `write`, which reads research/results and saves each figure twice: `<name>.png` on a light
surface and `<name>-dark.png` on a dark one (the report shows the one that matches the reader's
color scheme).

The drawing follows the dataviz method:
- Each portfolio keeps one categorical slot in every figure: 1/N, minimum variance and ERC take
  slots 1-3, which pass the palette validator on every pair, in both modes.
- The weights stack shows the 7 largest industries in slots 1-7 (validated on adjacent pairs)
  and folds the rest into one neutral band.
- Lines are 2 px, dots 8 px with a 2 px surface ring, grids solid hairlines. Every figure with
  several series has a legend. Text is in ink tokens, never in a series color.
The figures draw portfolio returns, weights and test statistics, never an industry's own returns.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from portfolio_lab.report.palette import INDUSTRY_NAMES, THEMES, Theme
from portfolio_lab.robustness import PRIMARY, role

__all__ = [
    "FIGURES",
    "INDUSTRY_NAMES",
    "ROLES",
    "THEMES",
    "Theme",
    "color",
    "drawdowns",
    "growth",
    "sharpe_differences",
    "stress",
    "weights",
    "write",
]


ROLES = ("1/N", "min variance", "ERC")
LABELS = {"1/N": "1/N", "min variance": "Minimum variance", "ERC": "ERC"}
VIEWS = {"USD": "USD investor", "CAD": "Canadian investor, unhedged"}
VARIANTS = {
    "primary": "Primary (252 days, 5 bp)",
    "window 126d": "Window 126 days",
    "window 504d": "Window 504 days",
    "cost 0bp": "Cost 0 bp",
    "cost 10bp": "Cost 10 bp",
    "cost 20bp": "Cost 20 bp",
    "49 industries": "49 industries",
}
SCENARIOS = {"gfc": "GFC, Sep-Nov 2008", "covid": "COVID, Feb-Mar 2020", "rates": "Rates, 2022"}
FIGURES = ("growth", "drawdowns", "sharpe_differences", "weights", "stress")

PX = 0.75  # points per CSS pixel (1/96 inch): the mark specs below read in pixels
DPI = 150
WIDTH = 7.5  # inches
FONT, LABEL, TITLE = 11 * PX, 12 * PX, 14 * PX
STACKED = 7  # industries with their own band; the rest fold into one
_MINUS = str.maketrans({"-": "−"})


def color(name: str, theme: Theme) -> str:
    """A portfolio role's slot (1/N, minimum variance, ERC: slots 1-3)."""
    return theme.series[ROLES.index(name)]


# --- the figures ------------------------------------------------------------------------------


def growth(monthly: pd.DataFrame, *, theme: str = "light") -> Figure:
    """Growth of $1 over the holdout, net of costs, for the three portfolios (USD and CAD)."""
    t = THEMES[theme]
    fig, axes = _canvas(t, height=3.4, ncols=2, sharey=True)
    for ax, view in zip(axes[0], VIEWS, strict=True):
        for r in ROLES:
            wealth = _wealth(_net(monthly, view, r))
            ax.plot(wealth.index, wealth.to_numpy(), **_line(r, t))
        _dates(ax)
        ax.set_title(VIEWS[view], loc="left", color=t.ink, fontsize=LABEL)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    for ax in axes[0]:  # after every panel, so the shared y-limits are final
        _end_labels(fig, ax, t)
    axes[0, 0].set_ylabel("Growth of $1, net of costs", color=t.secondary, fontsize=FONT)
    _legend(fig, axes[0, 0], t)
    months = pd.DatetimeIndex(monthly.index.get_level_values("month"))
    span = f"{months.min():%B %Y} to {months.max():%B %Y}"  # from the table, not a literal
    _title(fig, t, f"Growth of $1 over the holdout, {span}")
    return fig


def drawdowns(monthly: pd.DataFrame, *, theme: str = "light") -> Figure:
    """The fall from the running peak of each portfolio's month-end wealth (USD and CAD)."""
    t = THEMES[theme]
    fig, axes = _canvas(t, height=3.2, ncols=2, sharey=True)
    for ax, view in zip(axes[0], VIEWS, strict=True):
        for r in ROLES:
            wealth = _wealth(_net(monthly, view, r))
            fall = wealth / wealth.cummax() - 1
            ax.plot(fall.index, fall.to_numpy(), **_line(r, t))
        _dates(ax)
        ax.set_title(VIEWS[view], loc="left", color=t.ink, fontsize=LABEL)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0, 0].set_ylabel("Drawdown, month-end, net", color=t.secondary, fontsize=FONT)
    _legend(fig, axes[0, 0], t)
    _title(fig, t, "Drawdowns over the holdout")
    return fig


def sharpe_differences(tests: pd.DataFrame, *, theme: str = "light") -> Figure:
    """Each robustness variant's Sharpe difference against 1/N with its 95% interval, one
    panel per strategy and view (the robustness `tests` table)."""
    t = THEMES[theme]
    fig, axes = _canvas(t, height=4.4, ncols=2, nrows=2, sharey=True, grid="x")
    axes[0, 1].sharex(axes[0, 0])
    axes[1, 1].sharex(axes[1, 0])
    variants = list(dict.fromkeys(tests.index.get_level_values("variant")))
    variants.sort(key=lambda v: v != PRIMARY)  # the primary test first, whatever the order
    y = np.arange(len(variants))
    for row, r in enumerate(ROLES[1:]):
        for col, view in enumerate(VIEWS):
            ax = axes[row, col]
            sub = tests.xs(view, level="view")
            sub = sub.loc[sub["role"] == r].droplevel("strategy").reindex(variants)
            ax.axvline(0.0, color=t.axis, linewidth=PX)
            ax.hlines(y, sub["ci_low"], sub["ci_high"], color=color(r, t), linewidth=2 * PX)
            ax.scatter(sub["diff"], y, **_dot(r, t))
            ax.annotate(  # the primary test's value; the report's tables carry the rest
                f"{sub['diff'].iloc[0]:+.3f}".translate(_MINUS),
                (sub["ci_high"].iloc[0], 0),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=t.secondary,
                fontsize=FONT,
            )
            ax.set_yticks(y, [VARIANTS.get(v, v) for v in variants])
            ax.set_title(f"{LABELS[r]} vs 1/N, {view}", loc="left", color=t.ink, fontsize=LABEL)
            ax.margins(x=0.12)
        axes[row, 0].get_yticklabels()[0].set_fontweight("bold")
    axes[0, 0].invert_yaxis()
    fig.supxlabel(
        "Annual Sharpe ratio difference vs 1/N, with its 95% interval",
        color=t.secondary,
        fontsize=FONT,
    )
    _title(fig, t, "Sharpe ratio differences on the holdout: the primary test and its robustness")
    return fig


def weights(table: pd.DataFrame, *, view: str = "USD", theme: str = "light") -> Figure:
    """Minimum variance's target weights at each monthly trade of the holdout (the robustness
    `weights` table), the 7 largest industries on average in their own band."""
    t = THEMES[theme]
    sub = table.xs(view, level="view")
    w = sub.xs(_one(sub, "strategy", ROLES[1], view), level="strategy")
    assert isinstance(w, pd.DataFrame)  # one row per trade, one column per industry
    order = list(w.mean().sort_values(ascending=False, kind="stable").index)
    bands = {INDUSTRY_NAMES.get(c, c): w[c] for c in order[:STACKED]}
    rest = order[STACKED:]
    if rest:
        bands[f"Remaining {len(rest)} industries"] = w[rest].sum(axis=1)
    fig, axes = _canvas(t, height=3.6)
    ax = axes[0, 0]
    days = pd.DatetimeIndex(w.index)
    hold = days[-1] - days[-2] if len(days) > 1 else pd.Timedelta(days=30)  # about a month
    x = np.append(days.to_numpy(), (days[-1] + hold).to_datetime64())
    lower = np.zeros(len(x))
    for k, (label, band) in enumerate(bands.items()):
        values = np.append(band.to_numpy(dtype=float), float(band.iloc[-1]))  # hold the last
        upper = lower + values
        fill = t.series[k] if k < STACKED else t.rest
        ax.fill_between(x, lower, upper, step="post", color=fill, linewidth=0, label=label)
        ax.step(x, upper, where="post", color=t.surface, linewidth=2 * PX)  # the 2 px gap
        lower = upper
    ax.set_ylim(0, 1)
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Target weight", color=t.secondary, fontsize=FONT)
    _dates(ax)
    fig.legend(
        loc="outside lower center",
        ncols=4,
        frameon=False,
        labelcolor=t.secondary,
        fontsize=FONT,
        handlelength=1.0,
    )
    _title(fig, t, f"Minimum variance weights at each monthly trade, {VIEWS[view]}")
    return fig


def stress(
    replay: pd.DataFrame, realized: pd.DataFrame, *, view: str = "USD", theme: str = "light"
) -> Figure:
    """The stress scenarios: the replay of the holdout's final holdings (`stress`) next to each
    strategy's realized path (`stress_realized`), one bar per portfolio."""
    t = THEMES[theme]
    fig, axes = _canvas(t, height=3.4, ncols=2, sharey=True, grid="x")
    scenarios = list(dict.fromkeys(replay.index.get_level_values("scenario")))
    height, step = 0.22, 0.25  # a bar under 24 px, and air between neighbours
    panels = [
        (replay, "Replay of the final holdings"),
        (realized, "Realized, as each strategy ran"),
    ]
    for ax, (table, title) in zip(axes[0], panels, strict=True):
        sub = table.xs(view, level="view")
        ax.axvline(0.0, color=t.axis, linewidth=PX)
        for j, r in enumerate(ROLES):
            name = _one(sub, "portfolio", r, view)
            pnl = sub.xs(name, level="portfolio")["pnl"].reindex(scenarios).to_numpy(float)
            bars = np.arange(len(scenarios)) + (j - 1) * step
            ax.barh(bars, pnl, height=height, color=color(r, t), label=LABELS[r])
            for at, v in zip(bars, pnl, strict=True):
                ax.annotate(
                    f"{v:.1%}".translate(_MINUS),
                    (v, at),
                    xytext=(3 if v >= 0 else -3, 0),
                    textcoords="offset points",
                    ha="left" if v >= 0 else "right",
                    va="center",
                    color=t.secondary,
                    fontsize=FONT,
                )
        ax.set_yticks(np.arange(len(scenarios)), [SCENARIOS.get(s, s) for s in scenarios])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(title, loc="left", color=t.ink, fontsize=LABEL)
        ax.margins(x=0.22)
    axes[0, 0].invert_yaxis()
    _legend(fig, axes[0, 0], t)
    _title(fig, t, f"Stress scenarios, {VIEWS[view]}")
    return fig


def write(results: Path, out: Path) -> list[Path]:
    """Every figure from the result tables in `results` (research/results), saved in `out`
    as `<name>.png` and `<name>-dark.png`; returns the paths."""
    rob, hold = results / "robustness", results / "holdout"
    monthly = pd.read_csv(rob / "monthly.csv", index_col=[0, 1, 2], parse_dates=["month"])
    tests = pd.read_csv(rob / "tests.csv", index_col=[0, 1, 2])
    targets = pd.read_csv(rob / "weights.csv", index_col=[0, 1, 2], parse_dates=["traded"])
    replay = pd.read_csv(hold / "stress.csv", index_col=[0, 1, 2])
    realized = pd.read_csv(hold / "stress_realized.csv", index_col=[0, 1, 2])
    makers: dict[str, Callable[[str], Figure]] = {
        "growth": lambda th: growth(monthly, theme=th),
        "drawdowns": lambda th: drawdowns(monthly, theme=th),
        "sharpe_differences": lambda th: sharpe_differences(tests, theme=th),
        "weights": lambda th: weights(targets, theme=th),
        "stress": lambda th: stress(replay, realized, theme=th),
    }
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "axes.unicode_minus": True}):
        for name in FIGURES:
            for th in THEMES:
                fig = makers[name](th)
                path = out / (f"{name}.png" if th == "light" else f"{name}-{th}.png")
                # No Software tag: the bytes depend on the figure, not on who drew it.
                surface = THEMES[th].surface
                fig.savefig(path, dpi=DPI, facecolor=surface, metadata={"Software": None})
                paths.append(path)
    return paths


# --- helpers ----------------------------------------------------------------------------------


def _canvas(
    t: Theme,
    *,
    height: float,
    ncols: int = 1,
    nrows: int = 1,
    sharey: bool = False,
    grid: str = "y",
) -> tuple[Figure, np.ndarray]:
    fig = Figure(figsize=(WIDTH, height), dpi=DPI, facecolor=t.surface, layout="constrained")
    axes = fig.subplots(nrows, ncols, sharey=sharey, squeeze=False)
    for ax in axes.flat:
        _style(ax, t, grid)
    return fig, axes


def _style(ax: Axes, t: Theme, grid: str) -> None:
    ax.set_facecolor(t.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
        ax.spines[side].set_linewidth(PX)
    ax.tick_params(colors=t.axis, labelcolor=t.secondary, labelsize=FONT, width=PX, length=3)
    ax.grid(axis=grid, color=t.grid, linewidth=PX, linestyle="-")  # type: ignore[arg-type]
    ax.set_axisbelow(True)


def _title(fig: Figure, t: Theme, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", color=t.ink, fontsize=TITLE, fontweight="bold")


def _legend(fig: Figure, ax: Axes, t: Theme) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(labels),
        frameon=False,
        labelcolor=t.secondary,
        fontsize=FONT,
        handlelength=1.6,
    )


def _line(r: str, t: Theme) -> dict[str, Any]:
    return {
        "color": color(r, t),
        "linewidth": 2 * PX,
        "label": LABELS[r],
        "solid_capstyle": "round",
        "solid_joinstyle": "round",
    }


def _dot(r: str, t: Theme) -> dict[str, Any]:
    return {
        "s": (8 * PX) ** 2,  # an 8 px dot
        "color": color(r, t),
        "edgecolors": t.surface,  # its 2 px ring
        "linewidths": 2 * PX,
        "zorder": 3,
    }


def _dates(ax: Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _names(table: pd.DataFrame | pd.Series, level: str) -> list[str]:
    return [str(n) for n in dict.fromkeys(table.index.get_level_values(level))]


def _net(monthly: pd.DataFrame, view: str, r: str) -> pd.Series:
    """One portfolio's monthly net returns in one view, found by its role."""
    sub = monthly.xs(view, level="view")
    net: pd.Series = sub.xs(_one(sub, "strategy", r, view), level="strategy")["net"]
    return net


def _one(table: pd.DataFrame | pd.Series, level: str, r: str, view: str) -> str:
    """The name of the one portfolio of role `r` at `level` (a clear error otherwise)."""
    names = [n for n in _names(table, level) if role(n) == r]
    if len(names) != 1:
        raise ValueError(f"the {view} view needs exactly one {r} portfolio, got {names}")
    return names[0]


def _wealth(net: pd.Series) -> pd.Series:
    """$1 at the month end before the first month, compounded by the net returns."""
    days = pd.DatetimeIndex(net.index)
    start = days[0] - pd.offsets.MonthEnd(1)
    wealth = pd.concat([pd.Series([1.0], index=[start]), (1 + net).cumprod()])
    wealth.index = pd.DatetimeIndex(wealth.index)
    return wealth


def _end_labels(fig: Figure, ax: Axes, t: Theme) -> None:
    """An end dot on each line and its value, unless it would sit on a higher one's (the legend
    still names every line)."""
    ends = []
    for line in ax.get_lines():
        x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata(), dtype=float)
        ax.scatter(
            [x[-1]],
            [y[-1]],
            s=(8 * PX) ** 2,
            color=line.get_color(),
            edgecolors=t.surface,
            linewidths=2 * PX,
            zorder=3,
        )
        ends.append((float(y[-1]), x[-1]))
    lo, hi = ax.get_ylim()
    height = ax.get_position().height * fig.get_figheight() * 72  # the panel's height in points
    gap = 1.3 * FONT / height * (hi - lo)  # a label's height, in data units
    placed: list[float] = []
    for y_end, x_end in sorted(ends, key=lambda e: e[0], reverse=True):
        if all(abs(y_end - p) > gap for p in placed):
            ax.annotate(
                f"${y_end:.2f}",
                (x_end, y_end),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=t.secondary,
                fontsize=FONT,
            )
            placed.append(y_end)
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.12 * (right - left))
