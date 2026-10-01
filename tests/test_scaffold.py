import importlib
from pathlib import Path

import pytest

import portfolio_lab
from portfolio_lab import cli, config

SUBPACKAGES = ["data", "construct", "backtest", "risk", "stress", "report"]


def test_version():
    assert portfolio_lab.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    assert importlib.import_module(f"portfolio_lab.{name}").__doc__


def test_research_split_is_fixed():
    # Changing these moves the holdout; it must also change research/PLAN.md.
    assert config.DEV_END == "2010-12-31"
    assert config.HOLDOUT_START == "2011-01-01"


def test_cache_dir_default_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv(config.CACHE_ENV_VAR, raising=False)
    assert config.cache_dir() == Path.home() / ".cache" / "portfolio-lab"
    monkeypatch.setenv(config.CACHE_ENV_VAR, str(tmp_path))
    assert config.cache_dir() == tmp_path


def test_cli_help_without_command(capsys):
    assert cli.main([]) == 0
    assert "portfolio-lab" in capsys.readouterr().out


def test_cli_unbuilt_command(capsys):
    assert cli.main(["reproduce"]) == 1
    assert "not implemented yet" in capsys.readouterr().out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
