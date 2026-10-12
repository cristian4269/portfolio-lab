"""The dashboard's Altair charts, in the palette of `portfolio_lab.report.palette`.

The dataviz skill's specs: 2 px lines with round joins, 8 px end dots with a 2 px surface ring,
solid hairline grids, a legend whenever two or more series are drawn, direct end labels only
where they do not collide, and a crosshair whose tooltip lists every series at that day. Text is
in ink tokens, never in a series color.
"""

import altair as alt
import numpy as np
import pandas as pd

from portfolio_lab import dashboard as db
from portfolio_lab.report.palette import THEMES, Theme

FONT = '"Source Sans 3", "Source Sans Pro", sans-serif'
HEIGHT = 300
LABEL_GAP = 16  # px between two end labels, about one line of 12 px text


def theme(mode: str | None) -> Theme:
    """The palette for Streamlit's current theme (light unless it says dark)."""
    return THEMES["dark" if mode == "dark" else "light"]


def lines(
    frame: pd.DataFrame,
    colors: dict[str, str],
    t: Theme,
    *,
    y_title: str,
    y_format: str,
    log: bool = False,
    x_format: str = "%Y",
) -> alt.LayerChart:
    """One line per column of `frame` (its index the days), colored by `colors`; `x_format`
    labels the days (years by default, "%b %d" for a window of weeks)."""
    names = [str(c) for c in frame.columns]
    keys = {n: f"s{i}" for i, n in enumerate(names)}  # names hold dots, which Vega reads as paths
    wide = frame.set_axis([keys[n] for n in names], axis=1).rename_axis("day").reset_index()
    long = wide.melt("day", var_name="key", value_name="value")
    long["strategy"] = long["key"].map({k: n for n, k in keys.items()})

    color = alt.Color(
        "strategy:N",
        scale=alt.Scale(domain=names, range=[colors[n] for n in names]),
        legend=alt.Legend(title=None, orient="bottom", columns=1, symbolType="stroke"),
    )
    scale = alt.Scale(type="log", nice=False) if log else alt.Scale(zero=False)
    x = alt.X("day:T", title=None, axis=alt.Axis(format=x_format, labelOverlap=True))
    y = alt.Y("value:Q", title=y_title, scale=scale, axis=alt.Axis(format=y_format))
    hover = alt.selection_point(
        fields=["day"], nearest=True, on="pointermove", empty=False, clear="pointerout"
    )

    drawn = alt.Chart(long).mark_line(strokeWidth=2, strokeJoin="round", strokeCap="round")
    drawn = drawn.encode(x=x, y=y, color=color)
    tooltip = [alt.Tooltip("day:T", title="Day", format="%b %d, %Y")]
    tooltip += [alt.Tooltip(f"{keys[n]}:Q", title=n, format=y_format) for n in names]
    rule = (
        alt.Chart(wide)
        .mark_rule(color=t.secondary, strokeWidth=1)
        .encode(
            x="day:T", opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tooltip
        )
        .add_params(hover)
    )
    dot = dict(filled=True, size=64, stroke=t.surface, strokeWidth=2, opacity=1)
    points = alt.Chart(long).mark_point(**dot).encode(x=x, y=y, color=color)
    points = points.transform_filter(hover)

    last = long.loc[long["day"] == long["day"].max()]
    ends = alt.Chart(last).mark_point(**dot).encode(x=x, y=y, color=color)
    labelled = last.loc[last["strategy"].isin(_clear_labels(last, long["value"], log))].copy()
    labelled["text"] = [_format(v, y_format) for v in labelled["value"]]
    labels = alt.Chart(labelled).mark_text(align="left", dx=8, color=t.ink, fontSize=12)
    labels = labels.encode(x=x, y=y, text="text:N")

    chart = alt.layer(drawn, rule, points, ends, labels).properties(height=HEIGHT)
    return _finish(chart, t)


def var_chart(path: pd.DataFrame, color: str, t: Theme, *, label: str) -> alt.LayerChart:
    """A strategy's daily returns (a muted 1 px line) above its rolling loss limit, -VaR (a 2 px
    line in the strategy's color), with each exception, a loss past the VaR, as an ink dot."""
    drawn = db.thin(pd.DataFrame({"r": path["return"], "v": -path["var"]}), "min")
    span = len(path) / len(drawn)  # 1 for days; about 5 when thinned to weeks, 21 to months
    first = "Daily return" if span == 1 else f"Worst day of each {'week' if span < 10 else 'month'}"
    names = {"r": first, "v": label}
    long = drawn.rename_axis("day").reset_index().melt("day", var_name="key", value_name="value")
    long["series"] = long["key"].map(names)
    domain = list(names.values())
    x = alt.X("day:T", title=None, axis=alt.Axis(format="%Y", labelOverlap=True))
    y = alt.Y("value:Q", title="Daily return", axis=alt.Axis(format=".0%"))
    lines_ = (
        alt.Chart(long)
        .mark_line(strokeJoin="round", strokeCap="round")
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=domain, range=[t.muted, color]),
                legend=alt.Legend(title=None, orient="bottom", columns=1, symbolType="stroke"),
            ),
            strokeWidth=alt.StrokeWidth(
                "series:N", scale=alt.Scale(domain=domain, range=[1, 2]), legend=None
            ),
        )
    )
    hits = path.loc[path["exception"]].rename_axis("day").reset_index()
    dots = (
        alt.Chart(hits)
        .mark_point(filled=True, size=64, color=t.ink, stroke=t.surface, strokeWidth=2, opacity=1)
        .encode(
            x="day:T",
            y=alt.Y("return:Q"),
            tooltip=[
                alt.Tooltip("day:T", title="Exception", format="%b %d, %Y"),
                alt.Tooltip("return:Q", title="Return", format=".2%"),
                alt.Tooltip("var:Q", title="VaR", format=".2%"),
            ],
        )
    )
    return _finish(alt.layer(lines_, dots).properties(height=HEIGHT), t)


def stack(frame: pd.DataFrame, colors: list[str], t: Theme) -> alt.LayerChart:
    """Target weights by trade date as stacked bands, each held until the next trade, in the
    columns' order (largest first; slots 1-7 pass the validator on adjacent pairs, which is all
    a stack shows), a 1 px surface gap between bands, and a tooltip with every band."""
    names = [str(c) for c in frame.columns]
    keys = {n: f"s{i}" for i, n in enumerate(names)}
    wide = frame.set_axis([keys[n] for n in names], axis=1).rename_axis("day").reset_index()
    long = wide.melt("day", var_name="key", value_name="value")
    long["industry"] = long["key"].map({k: n for n, k in keys.items()})
    long["order"] = long["key"].map({k: i for i, k in enumerate(keys.values())})
    x = alt.X("day:T", title=None, axis=alt.Axis(format="%Y", labelOverlap=True))
    bands = (
        alt.Chart(long)
        .mark_area(interpolate="step-after", stroke=t.surface, strokeWidth=1)
        .encode(
            x=x,
            y=alt.Y("value:Q", stack="zero", title="Weight", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "industry:N",
                scale=alt.Scale(domain=names, range=colors),
                legend=alt.Legend(title=None, orient="bottom", columns=2, symbolType="square"),
                sort=names,
            ),
            order=alt.Order("order:Q"),
        )
    )
    hover = alt.selection_point(
        fields=["day"], nearest=True, on="pointermove", empty=False, clear="pointerout"
    )
    tooltip = [alt.Tooltip("day:T", title="Traded", format="%b %d, %Y")]
    tooltip += [alt.Tooltip(f"{keys[n]}:Q", title=n, format=".1%") for n in names]
    rule = (
        alt.Chart(wide)
        .mark_rule(color=t.secondary, strokeWidth=1)
        .encode(
            x="day:T", opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tooltip
        )
        .add_params(hover)
    )
    return _finish(alt.layer(bands, rule).properties(height=HEIGHT), t)


def _clear_labels(last: pd.DataFrame, plotted: pd.Series, log: bool) -> list[str]:
    """The series whose end labels sit at least LABEL_GAP apart (an approximate pixel scale
    from the range of every plotted value), keeping the highest first; the rest rely on the
    legend and the tooltip rather than labels nudged away from their lines."""

    def scaled(v: np.ndarray) -> np.ndarray:
        return np.log(v) if log else v

    every = scaled(plotted.to_numpy(float))
    span = float(every.max() - every.min()) or 1.0
    px = scaled(last["value"].to_numpy(float)) / span * HEIGHT * 0.8  # about the plot height
    kept: list[str] = []
    placed: list[float] = []
    for i in np.argsort(-px):
        if all(abs(px[i] - p) >= LABEL_GAP for p in placed):
            kept.append(str(last["strategy"].iloc[i]))
            placed.append(float(px[i]))
    return kept


def _format(value: float, spec: str) -> str:
    """An end label in the axis's style (a true minus sign, as Vega draws one)."""
    if spec.startswith("$"):
        text = f"${value:,.2f}"
    elif spec.endswith("%"):
        text = f"{value:.1%}"
    else:
        text = f"{value:,.2f}"
    return text.replace("-", "−")


def _finish(chart: alt.LayerChart, t: Theme) -> alt.LayerChart:
    return (
        chart.configure(background=t.surface, font=FONT, padding={"right": 64, "left": 4})
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=t.grid,
            gridWidth=1,
            domainColor=t.axis,
            tickColor=t.axis,
            labelColor=t.secondary,
            titleColor=t.secondary,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(labelColor=t.ink, labelFontSize=12, labelLimit=0, symbolStrokeWidth=2)
    )
