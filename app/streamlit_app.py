"""portfolio-lab's dashboard (blueprint Step 8, plans/step-8-plan.md).

Run from the repository root: `uv run --group app streamlit run app/streamlit_app.py`.

It reads the bundle that `portfolio-lab reproduce --dashboard` wrote into app/data/
(`PORTFOLIO_LAB_APP_DATA` points elsewhere, as the tests do) and computes everything else with
the library's pure functions. It never loads the raw data or opens the holdout itself.

The page leads with the registered result, which never changes with the controls, then the
descriptive tabs, which follow them.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import charts
import pandas as pd
import streamlit as st

from portfolio_lab import dashboard as db
from portfolio_lab import study
from portfolio_lab.config import HOLDOUT_START
from portfolio_lab.report.palette import INDUSTRY_NAMES, Theme
from portfolio_lab.risk import var
from portfolio_lab.stress import historical as hs

DATA_ENV = "PORTFOLIO_LAB_APP_DATA"
VIEWS = {"USD": "USD", "CAD": "CAD", "CAD_HEDGED": "CAD hedged"}
VIEW_NAMES = {
    "USD": "USD investor",
    "CAD": "Canadian investor, unhedged",
    "CAD_HEDGED": "Canadian investor, hedged",
}
BENCHMARK_LABELS = {db.MARKET: "Market", study.BENCHMARK: "1/N"}
BENCHMARK_NAMES = {db.MARKET: "Fama-French market", study.BENCHMARK: "1/N"}
CACHE = {"max_entries": 32, "ttl": 3600}  # a bounded cache: the host has little memory
TOP_HOLDINGS = 7  # the palette's adjacent-pair slots; the rest fold into one neutral band
METHODS = {
    "historical": "Historical",
    "normal": "Normal",
    "cornish_fisher": "Cornish-Fisher",
    "fhs": "Filtered historical (EWMA)",
}
SCENARIOS = {
    "gfc": "Financial crisis, Sep-Nov 2008",
    "covid": "COVID crash, Feb-Mar 2020",
    "rates": "Rate shock, 2022",
}
TERMS = {
    "alpha": "Alpha (annual)",
    "Mkt-RF": "Market",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
    "Mom": "Momentum",
    "fx": "USD/CAD",
    "dy": "CAD 10-year yield",
}
METRICS = {  # column: (label, format)
    "cagr": ("CAGR", "percent"),
    "annualized_vol": ("Volatility", "percent"),
    "sharpe_monthly": ("Sharpe (monthly)", "%.2f"),
    "sharpe": ("Sharpe (daily)", "%.2f"),
    "sortino": ("Sortino", "%.2f"),
    "max_drawdown": ("Max drawdown", "percent"),
    "max_drawdown_duration": ("Longest drawdown (days)", "%d"),
    "calmar": ("Calmar", "%.2f"),
    "cvar_95_daily": ("CVaR 95% (daily)", "percent"),
    "tracking_error": ("Tracking error", "percent"),
    "information_ratio": ("Information ratio", "%.2f"),
    "annual_turnover": ("Turnover a year", "percent"),
    "annual_cost": ("Cost a year", "percent"),
    "months": ("Months", "%d"),
}


@dataclass(frozen=True)
class Choice:
    """What the controls selected, shared by every tab."""

    folder: str
    view: str
    charted: list[str]
    start: pd.Timestamp
    end: pd.Timestamp
    cost: float
    benchmark: str
    level: float
    colors: dict[str, str]  # by display name
    theme: Theme


# --- data, cached -----------------------------------------------------------------------------


def data_folder() -> Path:
    return Path(os.environ.get(DATA_ENV) or Path(__file__).parent / "data")


@st.cache_resource(show_spinner="Loading the data")
def bundle(folder: str) -> db.Bundle:
    """The bundle, read once and shared by every session: its frames are read-only, so no
    session can change the data another one sees."""
    return db.read(Path(folder))


def primary(b: db.Bundle) -> list[str]:
    """1/N and the primary pair of the bundle's estimator: the charts' default."""
    names = [study.BENCHMARK, *study.primary(str(b.design["value"]["estimator"]))]
    return [n for n in names if n in b.strategies] or b.strategies[: db.MAX_CHARTED]


@st.cache_data(show_spinner=False, **CACHE)
def nets(folder: str, view: str, cost: float) -> pd.DataFrame:
    return db.net(bundle(folder), view, cost)


@st.cache_data(show_spinner=False, **CACHE)
def performance(
    folder: str,
    view: str,
    names: tuple[str, ...],
    cost: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    benchmark: str,
) -> pd.DataFrame:
    b = bundle(folder)
    return db.performance(
        b, view, list(names), cost_bps=cost, start=start, end=end, benchmark=benchmark
    )


@st.cache_data(show_spinner="Backtesting the VaR", **CACHE)
def var_backtest(
    folder: str,
    view: str,
    name: str,
    cost: float,
    method: str,
    level: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    series = nets(folder, view, cost)[name]
    return db.var_backtest(series, method=method, level=level, start=start, end=end)


@st.cache_data(show_spinner="Estimating the betas", **CACHE)
def betas(
    folder: str,
    view: str,
    names: tuple[str, ...],
    cost: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return db.betas(bundle(folder), view, list(names), cost_bps=cost, start=start, end=end)


@st.cache_data(show_spinner="Estimating the rolling betas", **CACHE)
def rolling(
    folder: str, view: str, name: str, cost: float, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    return db.rolling_betas(bundle(folder), view, name, cost_bps=cost, start=start, end=end)


@st.cache_data(show_spinner=False, **CACHE)
def stress_outcomes(folder: str, view: str, names: tuple[str, ...], cost: float) -> pd.DataFrame:
    return db.realized(bundle(folder), view, list(names), cost_bps=cost)


# --- shared pieces -----------------------------------------------------------------------------


def shown(names: pd.Index | list[str]) -> list[str]:
    """Strategy names as the page shows them."""
    return [db.display_name(str(n)) for n in names]


def numbers(frame: pd.DataFrame, formats: dict[str, str], **kwargs: object) -> None:
    """A table with each listed column in its format."""
    number = st.column_config.NumberColumn
    config = {c: number(c, format=f) for c, f in formats.items() if c in frame.columns}
    st.dataframe(frame, column_config=config, **kwargs)


def metrics_table(table: pd.DataFrame) -> None:
    table = table[list(METRICS)].rename(columns={k: v[0] for k, v in METRICS.items()})
    table.index = pd.Index(shown(table.index), name="Strategy")
    numbers(table, dict(METRICS.values()))


def _signed(value: float) -> str:
    """A signed difference to 3 places, with a true minus sign as the charts draw it."""
    return f"{value:+.3f}".replace("-", "−")


def _day(value: object) -> str:
    """A date as YYYY-MM-DD, whether it comes as a timestamp or a string."""
    return str(pd.Timestamp(str(value)).date())


# --- the registered result ---------------------------------------------------------------------


def registered(b: db.Bundle, c: Choice) -> None:
    """The four pre-registered tests, as they ran once on the holdout: they do not move with
    the controls, and the page says so."""
    tests = b.tables["tests"]
    rows = tests.loc[tests["family"] == "primary"].copy()
    rows["label"] = [
        f"{db.display_name(s).split(' · ')[0]} · {v}"
        for s, v in zip(rows["strategy"], rows["view"], strict=True)
    ]
    rejected = int(rows["reject_primary"].astype(bool).sum())
    holm = rows["holm_primary"]
    holm_text = (
        f"every Holm-adjusted p is {holm.iloc[0]:.2f}"
        if holm.min() == holm.max()
        else f"Holm-adjusted p from {holm.min():.2f} to {holm.max():.2f}"
    )
    zero = (rows["ci_low"] <= 0) & (rows["ci_high"] >= 0)
    with st.container(border=True):
        st.badge("Registered · run once on the holdout · fixed", icon=":material/lock:")
        plot, words = st.columns([3, 2], vertical_alignment="center")
        with plot:
            st.altair_chart(charts.forest(rows, c.theme), theme=None)
        with words:
            verdict = (
                "No registered test rejects."
                if rejected == 0
                else (f"{rejected} of the {len(rows)} registered tests reject.")
            )
            st.markdown(f"**{verdict}**")
            intervals = (
                "every 95% interval contains zero"
                if zero.all()
                else f"{int(zero.sum())} of the {len(rows)} 95% intervals contain zero"
            )
            st.markdown(
                f"Annual Sharpe ratio differences against 1/N run from "
                f"{_signed(rows['diff'].min())} to {_signed(rows['diff'].max())}; {intervals}, "
                f"and {holm_text}."
            )
            st.caption(
                f"Holdout {pd.Timestamp(HOLDOUT_START):%b %Y} to "
                f"{pd.Timestamp(str(b.design['value']['end'])):%b %Y} "
                f"({int(rows['months'].iloc[0])} months), {b.design['value']['cost_bps']} bp per "
                "side, pre-registered in research/PLAN.md."
            )
        if c.view not in study.TESTED_VIEWS:
            st.caption(
                "The hedged view has no test: its excess return is the USD one (research/PLAN.md "
                "§2). Everything about it below is descriptive."
            )
        with st.expander("The registered table"):
            table = pd.DataFrame(
                {
                    "Test": rows["label"],
                    "Sharpe difference": rows["diff"],
                    "95% CI low": rows["ci_low"],
                    "95% CI high": rows["ci_high"],
                    "Block (months)": rows["block"],
                    "p (Ledoit-Wolf)": rows["p"],
                    "Holm p": rows["holm_primary"],
                }
            )
            numbers(
                table,
                {col: "%.3f" for col in table.columns[1:]} | {"Block (months)": "%d"},
                hide_index=True,
            )


# --- the tabs ---------------------------------------------------------------------------------


def performance_tab(b: db.Bundle, c: Choice) -> None:
    all_nets = nets(c.folder, c.view, c.cost)
    days = (all_nets.index >= c.start) & (all_nets.index <= c.end)
    picked = all_nets.loc[days, c.charted]
    picked.columns = shown(picked.columns)
    st.subheader("Growth of $1, net of costs")
    log = st.toggle("Log scale", value=len(picked) / 252 > 25)
    growth = db.thin(db.growth(picked), "last")
    chart = charts.lines(
        growth, c.colors, c.theme, y_title="Value of $1", y_format="$,.2f", log=log
    )
    st.altair_chart(chart, theme=None)
    st.subheader("Drawdowns")
    drawn = db.thin(db.drawdowns(picked), "min")
    chart = charts.lines(drawn, c.colors, c.theme, y_title="Below the running peak", y_format=".0%")
    st.altair_chart(chart, theme=None)
    if len(picked) > db.MAX_POINTS:
        st.caption("The charts plot weekly or monthly points; the metrics use every day.")
    st.subheader("Metrics")
    names = tuple(c.charted)
    metrics_table(performance(c.folder, c.view, names, c.cost, c.start, c.end, c.benchmark))
    if st.toggle(f"Show all {len(b.strategies)} strategies at these settings"):
        every = tuple(b.strategies)
        metrics_table(performance(c.folder, c.view, every, c.cost, c.start, c.end, c.benchmark))
    st.caption(
        f"Daily net returns at {c.cost:g} bp per side. Sharpe (monthly) is the study's measure: "
        "monthly excess returns, annualized by the square root of 12. Tracking error and "
        f"information ratio are against the {BENCHMARK_NAMES[c.benchmark]}."
    )
    holdings_section(b, c)


def holdings_section(b: db.Bundle, c: Choice) -> None:
    st.subheader("Holdings")
    name = st.selectbox("Strategy", c.charted, format_func=db.display_name, key="holdings_strategy")
    held = db.holdings(b, c.view, name, top=TOP_HOLDINGS, start=c.start, end=c.end)
    colors = list(c.theme.series[: min(len(held.columns), TOP_HOLDINGS)])
    if db.OTHER in held.columns:
        colors = colors[: len(held.columns) - 1] + [c.theme.rest]
    held = held.rename(columns=lambda n: INDUSTRY_NAMES.get(str(n), str(n)))
    st.altair_chart(charts.stack(held, colors, c.theme), theme=None)
    st.caption(
        "The target weights set at each monthly rebalance and held until the next one (they "
        f"drift with prices in between). The {TOP_HOLDINGS} largest industries on average are "
        "shown, the rest in one band."
    )


def risk_tab(c: Choice) -> None:
    name = st.selectbox("Strategy", c.charted, format_func=db.display_name, key="risk_strategy")
    method = st.segmented_control(
        "Method",
        list(METHODS),
        format_func=METHODS.__getitem__,
        default="fhs",
        required=True,
        key="method",
    )
    try:
        path, row = var_backtest(c.folder, c.view, name, c.cost, method, c.level, c.start, c.end)
    except ValueError as err:
        st.info(f"Choose later months: {err}.")
        return
    level = f"{c.level:.0%}"
    st.subheader(f"One-day {level} VaR, from the last {study.VAR_WINDOW} days")
    one, two, three, four = st.columns(4)
    one.metric(f"Exceptions ({row['expected']:.0f} expected)", f"{int(row['exceptions'])}")
    two.metric("Kupiec p (exact)", f"{row['kupiec_exact_p']:.2f}")
    three.metric("Independence p", f"{row['independence_p']:.2f}")
    four.metric("Conditional coverage p", f"{row['cc_p']:.2f}")
    color = c.colors[db.display_name(name)]
    st.altair_chart(
        charts.var_chart(path, color, c.theme, label=f"Loss limit (-VaR {level})"), theme=None
    )
    st.caption(
        f"Each dot is an exception: a day whose loss went past that day's {level} VaR. Over "
        f"{int(row['days'])} days, {int(row['exceptions'])} exceptions against "
        f"{row['expected']:.0f} expected. Kupiec tests their number, Christoffersen whether they "
        "cluster (independence), and conditional coverage both. A p below 0.05 rejects the model."
    )
    if row["basel"] is not None and isinstance(row["basel"], str):
        st.caption(f"Basel traffic light over the last 250 days: {row['basel']}.")
    st.subheader("Latest forecast")
    last = path.iloc[-1]
    st.markdown(
        f"On {path.index[-1].date()}, the {level} VaR was **{last['var']:.2%}** of the "
        f"portfolio, and the expected shortfall beyond it **{last['es']:.2%}**."
    )


def factors_tab(c: Choice) -> None:
    try:
        table = betas(c.folder, c.view, tuple(c.charted), c.cost, c.start, c.end)
    except ValueError as err:
        st.info(f"No factor regression for these months: {err}.")
        return
    coef, tstat = db.beta_table(table)
    st.subheader("Factor betas")
    rows = pd.Index(shown(coef.index), name="Strategy")
    numbers(
        coef.rename(columns=TERMS).set_axis(rows),
        {TERMS["alpha"]: "percent"} | {TERMS[t]: "%.2f" for t in coef.columns if t != "alpha"},
    )
    fit = table.xs("alpha", level="term")
    cad = ""
    if c.view == "CAD":
        cad = "The CAD view adds the USD/CAD return" + (
            " and the change in the Canadian 10-year yield. "
            if "dy" in coef.columns
            else "; the daily 10-year yield starts in 2001, so it is left out of this window. "
        )
    r2 = ", ".join(f"{db.display_name(str(n))} {r:.2f}" for n, r in fit["r2"].items())
    st.caption(
        "Weekly excess returns (Wednesday to Wednesday) on the Fama-French five factors and "
        "momentum, all at once, so each beta holds the others fixed (Step 5b). "
        + cad
        + f"R²: {r2}. {int(fit['nobs'].iloc[0])} weeks, {_day(fit['start'].iloc[0])} to "
        f"{_day(fit['end'].iloc[0])}."
    )
    with st.expander("t-statistics (Newey-West)"):
        numbers(
            tstat.rename(columns=TERMS).set_axis(rows),
            {TERMS[t]: "%.1f" for t in tstat.columns},
        )
    st.subheader("Rolling betas, 252 days")
    term = st.selectbox(
        "Factor",
        [t for t in TERMS if t not in ("alpha", "dy")]
        if c.view == "CAD"
        else [t for t in TERMS if t not in ("alpha", "fx", "dy")],
        format_func=TERMS.__getitem__,
    )
    frame = pd.DataFrame(
        {
            db.display_name(n): rolling(c.folder, c.view, n, c.cost, c.start, c.end)[term]
            for n in c.charted
        }
    )
    chart = charts.lines(
        db.thin(frame, "last"), c.colors, c.theme, y_title=f"{TERMS[term]} beta", y_format=".2f"
    )
    st.altair_chart(chart, theme=None)
    st.caption("Daily regressions over each trailing year, a description rather than a forecast.")


def stress_tab(b: db.Bundle, c: Choice) -> None:
    all_nets = nets(c.folder, c.view, c.cost)
    covered = [s for s in SCENARIOS if db.scenario_days(all_nets.index, hs.SCENARIOS[s]).any()]
    if not covered:
        st.info("The data hold none of the registered stress windows.")
        return
    scenario = st.segmented_control(
        "Scenario",
        covered,
        format_func=SCENARIOS.__getitem__,
        default=covered[0],
        required=True,
        key="scenario",
    )
    s = hs.SCENARIOS[scenario]
    st.caption(s.note)
    inside = db.scenario_days(all_nets.index, s)
    st.subheader("What each strategy went through")
    window = all_nets.loc[inside, c.charted]
    window.columns = shown(window.columns)
    chart = charts.lines(
        db.growth(window),
        c.colors,
        c.theme,
        y_title="Value of $1",
        y_format="$,.2f",
        x_format="%b %d",
    )
    st.altair_chart(chart, theme=None)
    realized = stress_outcomes(c.folder, c.view, tuple(c.charted), c.cost).xs(
        scenario, level="scenario"
    )
    realized = realized[["pnl", "worst_day", "worst_date", "max_drawdown", "days"]].rename(
        columns={
            "pnl": "P&L",
            "worst_day": "Worst day",
            "worst_date": "On",
            "max_drawdown": "Max drawdown",
            "days": "Days",
        }
    )
    realized.index = pd.Index(shown(realized.index), name="Strategy")
    numbers(
        realized,
        {"P&L": "percent", "Worst day": "percent", "Max drawdown": "percent", "Days": "%d"},
    )
    st.caption(
        f"Each strategy's own net returns at {c.cost:g} bp, as it traded through the window."
    )
    if c.view not in study.TESTED_VIEWS:
        return
    st.subheader("The registered stress tests")
    replay = b.tables["stress"]
    rows = replay.loc[(replay["view"] == c.view) & (replay["scenario"] == scenario)]
    numbers(
        pd.DataFrame(
            {
                "Portfolio": shown(list(rows["portfolio"])),
                "P&L": rows["pnl"].to_numpy(),
                "Worst day": rows["worst_day"].to_numpy(),
                "Max drawdown": rows["max_drawdown"].to_numpy(),
            }
        ),
        {"P&L": "percent", "Worst day": "percent", "Max drawdown": "percent"},
        hide_index=True,
    )
    st.caption(
        "The weights held on the holdout's last day (August 2026), bought before the window "
        "and held through it (Step 6, registered before the holdout ran; 5 bp). Fixed."
    )
    shocks = b.tables["shocks"]
    rows = shocks.loc[(shocks["view"] == c.view) & (shocks["span"] == "holdout")]
    numbers(
        pd.DataFrame(
            {
                "Portfolio": shown(list(rows["asset"])),
                "Shock": rows["shock"].to_numpy(),
                "Move": rows["move"].to_numpy(),
                "Beta": rows["beta"].to_numpy(),
                "P&L": rows["pnl"].to_numpy(),
                "P&L standard error": rows["pnl_se"].to_numpy(),
            }
        ),
        {"Move": "percent", "Beta": "%.2f", "P&L": "percent", "P&L standard error": "percent"},
        hide_index=True,
    )
    st.caption(
        "Hypothetical shocks: the move in one factor, times the portfolio's weekly partial beta "
        "over the holdout (Step 6). The 10-year shock is in yield units (0.02 = 200 bp). Fixed."
    )


def about_tab(b: db.Bundle) -> None:
    st.subheader("What this is")
    st.markdown(
        'A replication and extension of DeMiguel, Garlappi and Uppal (2009), "Optimal versus '
        'naive diversification", for a Canadian investor. Long-only minimum variance and equal '
        "risk contribution (ERC) portfolios on the 12 Fama-French industries are compared with "
        "1/N after transaction costs, out of sample, for a USD investor and a Canadian investor "
        "who holds the same assets unhedged or hedged.\n\n"
        "- **Pre-registered.** The tests, the estimator rule and the holdout (January 2011 to "
        "August 2026) were fixed in research/PLAN.md, tagged before any holdout number existed. "
        "The holdout ran once.\n"
        "- **Backtest.** Monthly rebalancing on the last trading day, traded at the next close, "
        "5 bp per side by default. Covariances use Ledoit-Wolf shrinkage over 252 days.\n"
        "- **Test.** Ledoit and Wolf's (2008) bootstrap of the Sharpe ratio difference against "
        "1/N, with Holm's correction over the four primary tests.\n"
        "- **Everything outside the registered panel is descriptive.** Other dates, costs and "
        "strategies show what happened, not what the test found."
    )
    st.subheader("Data")
    st.markdown(
        "- Kenneth R. French Data Library: the 12 industry portfolios and the Fama-French "
        "factors, daily (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "data_library.html). Copyright Eugene F. Fama and Kenneth R. French.\n"
        "- USD/CAD: Board of Governors of the Federal Reserve System (US), H.10 release, noon "
        "buying rates in New York (public domain), retrieved via FRED (DEXCAUS) and changed "
        "from levels to daily returns.\n"
        "- Canadian rates: adapted from Statistics Canada, Table 10-10-0122-01, and the Bank of "
        "Canada (Valet API), changed from levels in percent to daily returns and changes. This "
        "does not constitute an endorsement by Statistics Canada of this product."
    )
    design = b.design["value"]
    st.caption(
        f"Data through {design['end']}; {design['strategies']} strategies; primary estimator "
        f"{design['estimator']}. Source code: the portfolio-lab repository. A research "
        "exercise, not investment advice."
    )


# --- the page ---------------------------------------------------------------------------------


def controls(b: db.Bundle, folder: str, t: Theme) -> Choice:
    with st.sidebar:
        st.subheader("Investor")
        view = st.segmented_control(
            "Currency view",
            list(VIEWS),
            format_func=VIEWS.__getitem__,
            default="USD",
            required=True,
            key="view",
            help="USD: a US investor. CAD: a Canadian investor holding the same US assets, "
            "unhedged or with the currency hedged.",
        )
        st.subheader("Strategies")
        charted = st.multiselect(
            f"Up to {db.MAX_CHARTED} on the charts",
            b.strategies,
            default=primary(b),
            format_func=db.display_name,
            max_selections=db.MAX_CHARTED,
            key="charted",
        )
        st.subheader("Period and costs")
        months = list(db.months(b, view))
        first = max(pd.Period(HOLDOUT_START, "M"), months[0])
        chosen = st.select_slider(
            "Months",
            options=months,
            value=(first, months[-1]),
            format_func=lambda p: p.strftime("%b %Y"),
            key=f"months_{view}",
        )
        cost = float(st.slider("Trading cost (bp per side)", 0, 50, int(db.COST_BPS), key="cost"))
        st.subheader("Risk")
        benchmark = st.segmented_control(
            "Tracking error against",
            list(BENCHMARK_LABELS),
            format_func=BENCHMARK_LABELS.__getitem__,
            default=db.MARKET,
            required=True,
            key="benchmark",
        )
        level = st.segmented_control(
            "VaR level",
            var.LEVELS,
            format_func=lambda x: f"{x:.0%}",
            default=var.LEVELS[-1],
            required=True,
            key="level",
        )
    slots = db.slots(charted, st.session_state.get("slots", {}))
    st.session_state["slots"] = slots
    return Choice(
        folder=folder,
        view=str(view),
        charted=charted,
        start=chosen[0].start_time,
        end=chosen[1].end_time,
        cost=cost,
        benchmark=str(benchmark),
        level=float(level),
        colors={db.display_name(n): t.series[s] for n, s in slots.items()},
        theme=t,
    )


def main() -> None:
    st.set_page_config(
        page_title="portfolio-lab", page_icon=":material/donut_small:", layout="wide"
    )
    folder = str(data_folder())
    try:
        b = bundle(folder)
    except (OSError, ValueError) as err:  # a missing or damaged bundle: say so, no traceback
        st.error(f"The dashboard's data could not be read: {err}")
        st.stop()
    c = controls(b, folder, charts.theme(getattr(st.context.theme, "type", None)))

    st.caption("portfolio-lab · a replication of DeMiguel, Garlappi and Uppal (2009)")
    st.title("Optimal versus naive diversification")
    st.markdown(
        f"Do {len(b.strategies) - 1} long-only strategies built from the 12 Fama-French "
        "industries beat an equal split (1/N) after costs, for a US and a Canadian investor? "
        "A research exercise, not investment advice."
    )
    registered(b, c)

    months = (c.end.year - c.start.year) * 12 + c.end.month - c.start.month + 1
    problem = None
    if not c.charted:
        problem = "Choose at least one strategy in the controls."
    elif months < db.MIN_MONTHS:
        problem = f"Choose at least {db.MIN_MONTHS} months: annual figures need a full year."
    st.badge(
        f"Descriptive · {VIEW_NAMES[c.view]} · {c.start:%b %Y} to {c.end:%b %Y} · {c.cost:g} bp",
        icon=":material/tune:",
        color="gray",
    )
    if c.start < pd.Timestamp(HOLDOUT_START):
        st.caption(
            "The months before January 2011 were used to design the study; the registered test "
            "covers January 2011 to August 2026."
        )
    tabs = st.tabs(["Performance", "Risk", "Factors", "Stress", "About"])
    with tabs[4]:
        about_tab(b)
    if problem:
        for tab in tabs[:4]:
            with tab:
                st.info(problem)
        return
    with tabs[0]:
        performance_tab(b, c)
    with tabs[1]:
        risk_tab(c)
    with tabs[2]:
        factors_tab(c)
    with tabs[3]:
        stress_tab(b, c)


main()
