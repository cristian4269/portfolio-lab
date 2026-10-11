"""The report's figures (Step 7b, plans/step-7b-plan.md decision 6), on synthetic tables."""

import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from portfolio_lab import cli, robustness
from portfolio_lab.report import figures

MV = "min variance (lw_identity, 252d)"
ERC = "ERC (lw_identity, 252d)"
NAMES = {"1/N": "1/N", "min variance": MV, "ERC": ERC}
INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq"]
INDUSTRIES += ["Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]


def monthly_table(months=24, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2011-01-31", periods=months, freq="ME")
    parts = {}
    for view in ("USD", "CAD", "CAD_HEDGED"):
        for name in NAMES.values():
            net = rng.normal(0.008, 0.04, months)
            frame = pd.DataFrame({"net": net, "excess": net - 0.001}, index=index)
            frame.index.name = "month"
            parts[(view, name)] = frame
    return pd.concat(parts, names=["view", "strategy"])


def robustness_tests():
    variants = ["primary", "window 126d", "window 504d", "cost 0bp", "cost 10bp", "cost 20bp"]
    variants.append("49 industries")
    rows = []
    for k, variant in enumerate(variants):
        for view in ("USD", "CAD"):
            for role in ("min variance", "ERC"):
                diff = 0.01 * k - 0.02
                rows.append(
                    {
                        "variant": variant,
                        "view": view,
                        "strategy": NAMES[role],
                        "role": role,
                        "diff": diff,
                        "ci_low": diff - 0.1,
                        "ci_high": diff + 0.1,
                    }
                )
    return pd.DataFrame(rows).set_index(["variant", "view", "strategy"])


def weights_table(trades=12, seed=1):
    rng = np.random.default_rng(seed)
    traded = pd.bdate_range("2011-01-03", periods=trades * 21)[::21]
    raw = rng.dirichlet(np.linspace(6, 0.5, len(INDUSTRIES)), trades)
    parts = {}
    for view in ("USD", "CAD"):
        for name in (MV, ERC):
            frame = pd.DataFrame(raw, index=pd.DatetimeIndex(traded, name="traded"))
            frame.columns = INDUSTRIES
            parts[(view, name)] = frame
    return pd.concat(parts, names=["view", "strategy"])


def stress_table(shift=0.0):
    rows = []
    for k, scenario in enumerate(("gfc", "covid", "rates")):
        for view in ("USD", "CAD"):
            for j, name in enumerate(NAMES.values()):
                rows.append(
                    {
                        "scenario": scenario,
                        "view": view,
                        "portfolio": name,
                        "pnl": -0.3 + 0.1 * k + 0.02 * j + shift,
                    }
                )
    return pd.DataFrame(rows).set_index(["scenario", "view", "portfolio"])


def write_results(root):
    rob, hold = root / "robustness", root / "holdout"
    rob.mkdir(parents=True)
    hold.mkdir(parents=True)
    monthly_table().to_csv(rob / "monthly.csv")
    robustness_tests().to_csv(rob / "tests.csv")
    weights_table().to_csv(rob / "weights.csv")
    stress_table().to_csv(hold / "stress.csv")
    stress_table(shift=0.01).to_csv(hold / "stress_realized.csv")


def lines_by_label(ax):
    return {line.get_label(): line for line in ax.get_lines()}


# --- the palette ------------------------------------------------------------------------------


def test_the_themes_are_the_validated_palette():
    light, dark = figures.THEMES["light"], figures.THEMES["dark"]
    assert light.series[:3] == ("#2a78d6", "#eb6834", "#1baf7a")
    assert dark.series[:3] == ("#3987e5", "#d95926", "#199e70")
    assert len(light.series) == len(dark.series) == 7
    assert (light.surface, dark.surface) == ("#fcfcfb", "#1a1a19")


def test_each_portfolio_keeps_its_slot():
    for theme in figures.THEMES.values():
        colors = [figures.color(role, theme) for role in figures.ROLES]
        assert colors == list(theme.series[:3])


def test_role_names_the_portfolio():
    assert robustness.role("1/N") == "1/N"
    assert robustness.role(MV) == "min variance"
    assert robustness.role(ERC) == "ERC"


# --- the figures ------------------------------------------------------------------------------


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_growth_compounds_net_returns_from_one(theme):
    monthly = monthly_table()
    fig = figures.growth(monthly, theme=theme)
    assert isinstance(fig, Figure)
    axes = [ax for ax in fig.axes if ax.get_lines()]
    assert len(axes) == 2
    usd = lines_by_label(axes[0])
    net = monthly.xs(("USD", MV), level=["view", "strategy"])["net"]
    y = usd["Minimum variance"].get_ydata()
    assert y[0] == 1.0 and len(y) == len(net) + 1
    assert y[-1] == pytest.approx(np.prod(1 + net.to_numpy()))
    assert fig.get_facecolor()[:3] == pytest.approx(_rgb(figures.THEMES[theme].surface))
    assert fig.legends, "a legend for the three portfolios"


def test_drawdowns_are_from_the_running_peak():
    monthly = monthly_table()
    fig = figures.drawdowns(monthly)
    cad = lines_by_label([ax for ax in fig.axes if ax.get_lines()][1])
    net = monthly.xs(("CAD", "1/N"), level=["view", "strategy"])["net"].to_numpy()
    wealth = np.concatenate([[1.0], np.cumprod(1 + net)])
    expected = wealth / np.maximum.accumulate(wealth) - 1
    np.testing.assert_allclose(cad["1/N"].get_ydata(), expected)
    assert max(cad["1/N"].get_ydata()) == 0.0


def test_sharpe_differences_show_every_variant_with_its_interval():
    tests = robustness_tests()
    fig = figures.sharpe_differences(tests)
    panels = [ax for ax in fig.axes if ax.get_lines() or ax.collections]
    assert len(panels) == 4
    labels = [t.get_text() for t in panels[0].get_yticklabels()]
    assert labels[0].startswith("Primary") and labels[-1] == "49 industries"
    assert len(labels) == 7
    # every interval is drawn: one segment per variant in each panel
    segments = [len(c.get_segments()) for c in panels[0].collections if hasattr(c, "get_segments")]
    assert 7 in segments


def test_weights_fold_the_tail_into_the_rest():
    weights = weights_table()
    fig = figures.weights(weights, view="USD")
    legend = fig.axes[0].get_legend() or fig.legends[0]
    texts = [t.get_text() for t in legend.get_texts()]
    assert len(texts) == 8
    assert texts[-1].startswith("Remaining 5")
    top = fig.axes[0].get_lines()[-1]  # the gap drawn on the stack's top edge
    np.testing.assert_allclose(top.get_ydata(), 1.0)  # the bands add up to the whole portfolio
    mv = weights.xs(("USD", MV), level=["view", "strategy"])
    top = mv.mean().sort_values(ascending=False).index[:7]
    assert figures.INDUSTRY_NAMES[top[0]] in texts[0]


def test_weights_pick_the_minimum_variance_portfolio_of_the_view():
    weights = weights_table().sort_index()
    weights.loc[("CAD", MV)] = 0.0
    weights.loc[("CAD", MV), "Utils"] = 1.0
    fig = figures.weights(weights, view="CAD")
    texts = [t.get_text() for t in (fig.axes[0].get_legend() or fig.legends[0]).get_texts()]
    assert texts[0] == figures.INDUSTRY_NAMES["Utils"]


def test_stress_bars_are_the_pnl():
    replay, realized = stress_table(), stress_table(shift=0.01)
    fig = figures.stress(replay, realized, view="USD")
    panels = [ax for ax in fig.axes if ax.patches]
    assert len(panels) == 2
    widths = sorted(round(p.get_width(), 12) for p in panels[0].patches)
    expected = sorted(round(v, 12) for v in replay.xs("USD", level="view")["pnl"])
    assert widths == expected


def test_write_saves_a_light_and_a_dark_png_of_each_figure(tmp_path):
    write_results(tmp_path / "results")
    paths = figures.write(tmp_path / "results", tmp_path / "figures")
    names = sorted(p.name for p in paths)
    expected = [f"{n}{s}.png" for n in figures.FIGURES for s in ("", "-dark")]
    assert names == sorted(expected)
    for p in paths:
        content = p.read_bytes()
        assert content[:8] == b"\x89PNG\r\n\x1a\n"
        assert b"Software" not in content and b"matplotlib" not in content.lower()


def test_write_needs_the_result_tables(tmp_path):
    with pytest.raises(FileNotFoundError):
        figures.write(tmp_path / "nothing", tmp_path / "figures")


# --- the command ------------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "PLAN.md").write_text("# Plan\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_figures_command_writes_into_research_figures(project, capsys):
    write_results(project / "research" / "results")
    assert cli.main(["figures"]) == 0
    assert (project / "research" / "figures" / "growth.png").exists()
    assert (project / "research" / "figures" / "growth-dark.png").exists()
    assert "wrote 10 figures" in capsys.readouterr().out


def test_figures_command_takes_other_folders(project):
    write_results(project / "elsewhere")
    assert cli.main(["figures", "--results", "elsewhere", "--out", "pictures"]) == 0
    assert (project / "pictures" / "stress.png").exists()


def test_figures_command_without_results_is_a_clean_error(project, capsys):
    assert cli.main(["figures"]) == 2
    assert "portfolio-lab figures:" in capsys.readouterr().err


def test_figures_command_without_matplotlib_names_the_extra(project, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # as if the extra were not installed
    monkeypatch.delitem(sys.modules, "portfolio_lab.report.figures")
    assert cli.main(["figures"]) == 2
    assert "--extra report" in capsys.readouterr().err


def test_figures_command_reraises_a_missing_core_module(project, monkeypatch):
    monkeypatch.setitem(sys.modules, "portfolio_lab.robustness", None)
    monkeypatch.delitem(sys.modules, "portfolio_lab.report.figures")
    with pytest.raises(ModuleNotFoundError, match="portfolio_lab.robustness"):
        cli.main(["figures"])


@pytest.mark.parametrize("missing", ["PIL", "kiwisolver"])
def test_a_missing_matplotlib_dependency_names_the_extra(project, monkeypatch, capsys, missing):
    def absent(name):
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(cli, "import_module", absent)
    assert cli.main(["figures"]) == 2
    assert "--extra report" in capsys.readouterr().err


def test_weights_need_the_views_minimum_variance_portfolio():
    weights = weights_table().drop(index=MV, level="strategy")
    with pytest.raises(ValueError, match="exactly one min variance"):
        figures.weights(weights, view="USD")


def test_weights_draw_a_single_trade():
    fig = figures.weights(weights_table(trades=1), view="USD")
    assert fig.legends


def test_growth_titles_the_span_of_its_table():
    fig = figures.growth(monthly_table(months=24))
    assert "January 2011 to December 2012" in fig.get_suptitle()


def test_sharpe_differences_put_the_primary_test_first_whatever_the_order():
    tests = robustness_tests()
    tests = pd.concat([tests.drop(index="primary", level="variant"), tests.loc[["primary"]]])
    fig = figures.sharpe_differences(tests)
    ax = [a for a in fig.axes if a.get_lines() or a.collections][0]
    assert ax.get_yticklabels()[0].get_text().startswith("Primary")
    primary = tests.loc[("primary", "USD", MV), "diff"]
    assert [t.get_text() for t in ax.texts] == [f"{primary:+.3f}".replace("-", "−")]


def _rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def test_weights_with_seven_industries_need_no_remaining_band():
    weights = weights_table()[INDUSTRIES[:7]]
    fig = figures.weights(weights, view="USD")
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert len(texts) == 7 and not any(t.startswith("Remaining") for t in texts)


def test_a_view_without_one_portfolio_of_a_role_is_refused():
    monthly = monthly_table().drop(index="1/N", level="strategy")
    with pytest.raises(ValueError, match="exactly one 1/N"):
        figures.growth(monthly)


def test_end_labels_skip_a_value_that_would_sit_on_another():
    monthly = monthly_table()
    same = monthly.xs("1/N", level="strategy", drop_level=False)
    for name in (MV, ERC):
        copy = same.rename(index={"1/N": name}, level="strategy")
        monthly.loc[copy.index, "net"] = copy["net"].to_numpy()  # all three end together
    fig = figures.growth(monthly)
    usd = [ax for ax in fig.axes if ax.get_lines()][0]
    assert len(usd.texts) == 1  # one label for the three coinciding ends
