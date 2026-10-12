"""portfolio-lab's dashboard (blueprint Step 8, plans/step-8-plan.md).

Run from the repository root: `uv run --group app streamlit run app/streamlit_app.py`.

It reads the bundle that `portfolio-lab reproduce --dashboard` wrote into app/data/
(`PORTFOLIO_LAB_APP_DATA` points elsewhere, as the tests do) and computes everything else with
the library's pure functions. It never loads the raw data or opens the holdout itself.
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
from portfolio_lab.report.palette import Theme
from portfolio_lab.risk import var
from portfolio_lab.stress import historical as hs

DATA_ENV = "PORTFOLIO_LAB_APP_DATA"
VIEWS = {
    "USD": "USD investor",
    "CAD": "Canadian investor, unhedged",
    "CAD_HEDGED": "Canadian investor, hedged",
}
BENCHMARK_LABELS = {db.MARKET: "Fama-French market", study.BENCHMARK: "1/N"}
PRIMARY = [study.BENCHMARK, *study.primary("lw_identity")]
MIN_MONTHS = 12  # annual figures from less than a year would mislead (and Sharpe needs 6)
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
    colors: dict[str, str]
    theme: Theme


# --- data, cached -----------------------------------------------------------------------------


def data_folder() -> Path:
    return Path(os.environ.get(DATA_ENV) or Path(__file__).parent / "data")


@st.cache_resource(show_spinner="Loading the data")
def bundle(folder: str) -> db.Bundle:
    return db.read(Path(folder))


@st.cache_data(show_spinner=False)
def nets(folder: str, view: str, cost: float) -> pd.DataFrame:
    return db.net(bundle(folder), view, cost)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner="Backtesting the VaR")
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


@st.cache_data(show_spinner="Estimating the betas")
def betas(
    folder: str,
    view: str,
    names: tuple[str, ...],
    cost: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return db.betas(bundle(folder), view, list(names), cost_bps=cost, start=start, end=end)


@st.cache_data(show_spinner="Estimating the rolling betas")
def rolling(
    folder: str, view: str, name: str, cost: float, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    return db.rolling_betas(bundle(folder), view, name, cost_bps=cost, start=start, end=end)


# --- the tabs ---------------------------------------------------------------------------------


def numbers(frame: pd.DataFrame, formats: dict[str, str], **kwargs: object) -> None:
    """A table with each listed column in its format."""
    number = st.column_config.NumberColumn
    config = {c: number(c, format=f) for c, f in formats.items() if c in frame.columns}
    st.dataframe(frame, column_config=config, **kwargs)


def metrics_table(table: pd.DataFrame) -> None:
    shown = table[list(METRICS)].rename(columns={k: v[0] for k, v in METRICS.items()})
    numbers(shown, dict(METRICS.values()))


def performance_tab(b: db.Bundle, c: Choice) -> None:
    all_nets = nets(c.folder, c.view, c.cost)
    days = (all_nets.index >= c.start) & (all_nets.index <= c.end)
    picked = all_nets.loc[days, c.charted]
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
    with st.expander(f"All {len(b.strategies)} strategies at these settings"):
        every = tuple(b.strategies)
        metrics_table(performance(c.folder, c.view, every, c.cost, c.start, c.end, c.benchmark))
    st.caption(
        f"Daily net returns at {c.cost:g} bp per side. Sharpe (monthly) is the study's measure: "
        "monthly excess returns, annualized by the square root of 12. Tracking error and "
        f"information ratio are against the {BENCHMARK_LABELS[c.benchmark]}."
    )
    registered_tests(b, c.view)


def registered_tests(b: db.Bundle, view: str) -> None:
    st.subheader("The registered tests")
    if view not in study.TESTED_VIEWS:
        st.caption(
            "The hedged view has no test: its excess return is the USD one (research/PLAN.md "
            "§2). Its metrics above are descriptive."
        )
        return
    tests = b.tables["tests"]
    rows = tests.loc[(tests["view"] == view) & (tests["family"] == "primary")]
    shown = pd.DataFrame(
        {
            "Strategy": rows["strategy"],
            "Sharpe difference vs 1/N": rows["diff"],
            "95% CI low": rows["ci_low"],
            "95% CI high": rows["ci_high"],
            "Block (months)": rows["block"],
            "p (Ledoit-Wolf)": rows["p"],
            "Holm p": rows["holm_primary"],
        }
    )
    numbers(
        shown,
        {col: "%.3f" for col in shown.columns[1:]} | {"Block (months)": "%d"},
        hide_index=True,
    )
    st.caption(
        "Pre-registered in research/PLAN.md and run once on the holdout (Jan 2011 to Aug 2026, "
        "5 bp, annual Sharpe ratios from monthly returns). They do not change with the "
        "controls: everything else on this page is descriptive."
    )


def risk_tab(c: Choice) -> None:
    name = st.selectbox("Strategy", c.charted, key="risk_strategy")
    method = st.radio(
        "Method", list(METHODS), format_func=METHODS.__getitem__, horizontal=True, index=3
    )
    try:
        path, row = var_backtest(c.folder, c.view, name, c.cost, method, c.level, c.start, c.end)
    except ValueError as err:
        st.info(f"Choose later months: {err}.")
        return
    level = f"{c.level:.0%}"
    st.subheader(f"One-day {level} VaR, from the last 500 days")
    one, two, three, four = st.columns(4)
    one.metric(f"Exceptions ({row['expected']:.0f} expected)", f"{int(row['exceptions'])}")
    two.metric("Kupiec p (exact)", f"{row['kupiec_exact_p']:.2f}")
    three.metric("Independence p", f"{row['independence_p']:.2f}")
    four.metric("Conditional coverage p", f"{row['cc_p']:.2f}")
    st.altair_chart(
        charts.var_chart(path, c.colors[name], c.theme, label=f"Loss limit (-VaR {level})"),
        theme=None,
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
    table = betas(c.folder, c.view, tuple(c.charted), c.cost, c.start, c.end)
    long = table.reset_index()
    coef = long.pivot_table(index="asset", columns="term", values="coef", sort=False)
    coef = coef[[t for t in TERMS if t in coef.columns]]
    coef["alpha"] = coef["alpha"] * 52  # weekly to annual
    tstat = long.pivot_table(index="asset", columns="term", values="t", sort=False)[coef.columns]
    st.subheader("Factor betas")
    numbers(
        coef.rename(columns=TERMS).rename_axis("Strategy"),
        {TERMS["alpha"]: "percent"} | {TERMS[t]: "%.2f" for t in coef.columns if t != "alpha"},
    )
    fit = table.xs("alpha", level="term")[["r2", "nobs"]]
    st.caption(
        "Weekly excess returns (Wednesday to Wednesday) on the Fama-French five factors and "
        "momentum, all at once, so each beta holds the others fixed (Step 5b). "
        + (
            "The CAD view adds the USD/CAD return and the change in the Canadian 10-year yield. "
            if c.view == "CAD"
            else ""
        )
        + f"R²: {', '.join(f'{r:.2f}' for r in fit['r2'])}; {int(fit['nobs'].iloc[0])} weeks."
    )
    with st.expander("t-statistics (Newey-West)"):
        numbers(
            tstat.rename(columns=TERMS).rename_axis("Strategy"),
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
        {n: rolling(c.folder, c.view, n, c.cost, c.start, c.end)[term] for n in c.charted}
    )
    chart = charts.lines(
        db.thin(frame, "last"), c.colors, c.theme, y_title=f"{TERMS[term]} beta", y_format=".2f"
    )
    st.altair_chart(chart, theme=None)
    st.caption("Daily regressions over each trailing year, a description rather than a forecast.")


def stress_tab(b: db.Bundle, c: Choice) -> None:
    all_nets = nets(c.folder, c.view, c.cost)
    covered = [s for s in SCENARIOS if db.scenario_days(all_nets.index, hs.SCENARIOS[s]).any()]
    scenario = st.radio("Scenario", covered, format_func=SCENARIOS.__getitem__, horizontal=True)
    s = hs.SCENARIOS[scenario]
    st.caption(s.note)
    inside = db.scenario_days(all_nets.index, s)
    st.subheader("What each strategy went through")
    growth = db.growth(all_nets.loc[inside, c.charted])
    chart = charts.lines(growth, c.colors, c.theme, y_title="Value of $1", y_format="$,.2f")
    st.altair_chart(chart, theme=None)
    realized = db.realized(b, c.view, c.charted, cost_bps=c.cost).xs(scenario, level="scenario")
    numbers(
        realized[["pnl", "worst_day", "worst_date", "max_drawdown", "days"]]
        .rename(
            columns={
                "pnl": "P&L",
                "worst_day": "Worst day",
                "worst_date": "On",
                "max_drawdown": "Max drawdown",
                "days": "Days",
            }
        )
        .rename_axis("Strategy"),
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
        rows[["portfolio", "pnl", "worst_day", "max_drawdown"]].rename(
            columns={
                "portfolio": "Portfolio",
                "pnl": "P&L",
                "worst_day": "Worst day",
                "max_drawdown": "Max drawdown",
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
        rows[["asset", "shock", "move", "beta", "pnl", "pnl_se"]].rename(
            columns={
                "asset": "Portfolio",
                "shock": "Shock",
                "move": "Move",
                "beta": "Beta",
                "pnl": "P&L",
                "pnl_se": "P&L standard error",
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
        "1/N, with Holm's correction over the four primary tests. None rejects.\n"
        "- **Everything else here is descriptive.** Other dates, costs and strategies show what "
        "happened, not what the test found."
    )
    st.subheader("Data")
    st.markdown(
        "- Kenneth R. French Data Library: the 12 industry portfolios and the Fama-French "
        "factors, daily (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "data_library.html). Copyright Eugene F. Fama and Kenneth R. French.\n"
        "- USD/CAD: Board of Governors of the Federal Reserve System (US) via FRED (DEXCAUS).\n"
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
        st.header("Controls")
        view = st.radio("Investor", list(VIEWS), format_func=VIEWS.__getitem__, key="view")
        charted = st.multiselect(
            f"Strategies on the charts (up to {db.MAX_CHARTED})",
            b.strategies,
            default=PRIMARY,
            max_selections=db.MAX_CHARTED,
            key="charted",
        )
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
        benchmark = st.radio(
            "Tracking error against",
            list(BENCHMARK_LABELS),
            format_func=BENCHMARK_LABELS.__getitem__,
            horizontal=True,
        )
        level = st.radio(
            "VaR level", var.LEVELS, index=1, format_func=lambda x: f"{x:.0%}", horizontal=True
        )
    slots = db.slots(charted, st.session_state.get("slots", {}))
    st.session_state["slots"] = slots
    return Choice(
        folder=folder,
        view=view,
        charted=charted,
        start=chosen[0].start_time,
        end=chosen[1].end_time,
        cost=cost,
        benchmark=benchmark,
        level=float(level),
        colors={n: t.series[s] for n, s in slots.items()},
        theme=t,
    )


def main() -> None:
    st.set_page_config(
        page_title="portfolio-lab", page_icon=":material/donut_small:", layout="wide"
    )
    folder = str(data_folder())
    b = bundle(folder)
    c = controls(b, folder, charts.theme(getattr(st.context.theme, "type", None)))

    st.title("Optimal versus naive diversification")
    st.caption(
        "A replication and extension of DeMiguel, Garlappi and Uppal (2009) for a Canadian "
        "investor: 17 long-only strategies on the 12 Fama-French industries, monthly "
        "rebalancing, costs, and a pre-registered test. A research exercise, not investment "
        "advice."
    )
    if c.start < pd.Timestamp(HOLDOUT_START):
        st.caption(
            "The months before January 2011 were used to design the study. The pre-registered "
            "test covers January 2011 to August 2026."
        )
    tabs = st.tabs(["Performance", "Risk", "Factors", "Stress", "About"])
    with tabs[4]:
        about_tab(b)
    months = (c.end.year - c.start.year) * 12 + c.end.month - c.start.month + 1
    problem = None
    if not c.charted:
        problem = "Choose at least one strategy in the controls."
    elif months < MIN_MONTHS:
        problem = f"Choose at least {MIN_MONTHS} months: annual figures need a full year."
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
