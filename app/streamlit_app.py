"""portfolio-lab's dashboard (blueprint Step 8, plans/step-8-plan.md).

Run from the repository root: `uv run --group app streamlit run app/streamlit_app.py`.

It reads the bundle that `portfolio-lab reproduce --dashboard` wrote into app/data/
(`PORTFOLIO_LAB_APP_DATA` points elsewhere, as the tests do) and computes everything else with
the library's pure functions. It never loads the raw data or opens the holdout itself.
"""

import os
from pathlib import Path

import charts
import pandas as pd
import streamlit as st

from portfolio_lab import dashboard as db
from portfolio_lab import study
from portfolio_lab.config import HOLDOUT_START

DATA_ENV = "PORTFOLIO_LAB_APP_DATA"
VIEWS = {
    "USD": "USD investor",
    "CAD": "Canadian investor, unhedged",
    "CAD_HEDGED": "Canadian investor, hedged",
}
BENCHMARK_LABELS = {db.MARKET: "Fama-French market", study.BENCHMARK: "1/N"}
PRIMARY = [study.BENCHMARK, *study.primary("lw_identity")]
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


def metrics_table(table: pd.DataFrame) -> None:
    shown = table[list(METRICS)].rename(columns={k: v[0] for k, v in METRICS.items()})
    config = {
        label: st.column_config.NumberColumn(label, format=fmt) for label, fmt in METRICS.values()
    }
    st.dataframe(shown, column_config=config)


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
    number = st.column_config.NumberColumn
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            c: number(c, format="%.3f") for c in shown.columns if c not in ("Strategy",)
        },
    )
    st.caption(
        "Pre-registered in research/PLAN.md and run once on the holdout (Jan 2011 to Aug 2026, "
        "5 bp, annual Sharpe ratios from monthly returns). These numbers do not change with the "
        "controls: the tables and charts above are descriptive."
    )


def main() -> None:
    st.set_page_config(
        page_title="portfolio-lab", page_icon=":material/donut_small:", layout="wide"
    )
    folder = str(data_folder())
    b = bundle(folder)
    t = charts.theme(getattr(st.context.theme, "type", None))

    with st.sidebar:
        st.header("Controls")
        view = st.radio("Investor", list(VIEWS), format_func=VIEWS.__getitem__)
        charted = st.multiselect(
            f"Strategies on the charts (up to {db.MAX_CHARTED})",
            b.strategies,
            default=PRIMARY,
            max_selections=db.MAX_CHARTED,
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
        cost = float(st.slider("Trading cost (bp per side)", 0, 50, int(db.COST_BPS)))
        benchmark = st.radio(
            "Tracking error against",
            list(BENCHMARK_LABELS),
            format_func=BENCHMARK_LABELS.__getitem__,
            horizontal=True,
        )

    start, end = chosen[0].start_time, chosen[1].end_time
    st.title("Optimal versus naive diversification")
    st.caption(
        "A replication and extension of DeMiguel, Garlappi and Uppal (2009) for a Canadian "
        "investor: 17 long-only strategies on the 12 Fama-French industries, monthly "
        "rebalancing, costs, and a pre-registered test. A research exercise, not investment "
        "advice."
    )
    if chosen[0] < pd.Period(HOLDOUT_START, "M"):
        st.caption(
            "The months before January 2011 were used to design the study. The pre-registered "
            "test covers January 2011 to August 2026."
        )

    performance_tab, risk_tab, factors_tab, stress_tab, about_tab = st.tabs(
        ["Performance", "Risk", "Factors", "Stress", "About"]
    )
    with performance_tab:
        if not charted:
            st.info("Choose at least one strategy in the controls.")
        else:
            slots = db.slots(charted, st.session_state.get("slots", {}))
            st.session_state["slots"] = slots
            colors = {n: t.series[s] for n, s in slots.items()}
            all_nets = nets(folder, view, cost)
            days = (all_nets.index >= start) & (all_nets.index <= end)
            picked = all_nets.loc[days, charted]
            years = len(picked) / 252
            st.subheader("Growth of $1, net of costs")
            log = st.toggle("Log scale", value=years > 25)
            growth = db.thin(db.growth(picked), "last")
            st.altair_chart(
                charts.lines(growth, colors, t, y_title="Value of $1", y_format="$,.2f", log=log),
                theme=None,
            )
            st.subheader("Drawdowns")
            st.altair_chart(
                charts.lines(
                    db.thin(db.drawdowns(picked), "min"),
                    colors,
                    t,
                    y_title="Below the running peak",
                    y_format=".0%",
                ),
                theme=None,
            )
            if len(picked) > db.MAX_POINTS:
                st.caption("The charts plot weekly or monthly points; the metrics use every day.")
            st.subheader("Metrics")
            table = performance(folder, view, tuple(charted), cost, start, end, benchmark)
            metrics_table(table)
            with st.expander(f"All {len(b.strategies)} strategies at these settings"):
                every = performance(folder, view, tuple(b.strategies), cost, start, end, benchmark)
                metrics_table(every)
            st.caption(
                f"Daily net returns at {cost:g} bp per side. Sharpe (monthly) is the study's "
                "measure: monthly excess returns, annualized by the square root of 12. Tracking "
                f"error and information ratio are against the {BENCHMARK_LABELS[benchmark]}."
            )
        registered_tests(b, view)

    for tab in (risk_tab, factors_tab, stress_tab):
        with tab:
            st.info("This tab is coming in the next session.")
    with about_tab:
        st.subheader("Data")
        st.markdown(
            "- Kenneth R. French Data Library: the 12 industry portfolios and the Fama-French "
            "factors, daily (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "data_library.html).\n"
            "- USD/CAD: Board of Governors of the Federal Reserve System (US) via FRED "
            "(DEXCAUS).\n"
            "- Canadian rates: adapted from Statistics Canada, Table 10-10-0122-01, and the "
            "Bank of Canada (Valet API). This does not constitute an endorsement by Statistics "
            "Canada of this product.\n"
        )
        st.caption("Research exercise, not investment advice.")


main()
