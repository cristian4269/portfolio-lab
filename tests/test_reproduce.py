"""`portfolio-lab reproduce` (Step 7a): manifest, drift warning, the holdout lock, the wiring."""

import io
import json
import subprocess
import zipfile

import pandas as pd
import pytest

from portfolio_lab import cli, config, manifest, study
from portfolio_lab.data import HoldoutError, cache


def french_zip(first_line="This file was created using the 202608 CRSP database."):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        info = zipfile.ZipInfo("F.csv", date_time=(2026, 1, 1, 0, 0, 0))  # same bytes every time
        z.writestr(info, f"{first_line}\r\nmore\r\n")
    return buf.getvalue()


def cache_every_source(content_by_name=None):
    content_by_name = content_by_name or {}
    for name, url in manifest.SOURCES.items():
        default = french_zip() if url.endswith(".zip") else f"{name} header\nrow\n".encode()
        cache._store(url, content_by_name.get(name, default))


# --- the manifest -----------------------------------------------------------------------------


def test_sources_are_the_seven_files_the_study_reads():
    assert set(manifest.SOURCES) == {
        "ff5_daily",
        "mom_daily",
        "ind12_daily",
        "usdcad",
        "ca_tbill_3m",
        "ca_govt_long",
        "ca_govt_10y",
    }
    assert manifest.SOURCES["usdcad"].endswith("id=DEXCAUS")


def test_header_line_of_zips_and_plain_files():
    assert manifest.header_line(french_zip("Line one")) == "Line one"
    assert manifest.header_line(b"\xef\xbb\xbfDATE,DEXCAUS\r\n1971-01-04,1.01\n") == "DATE,DEXCAUS"
    assert len(manifest.header_line(b"x" * 1000)) == manifest.HEADER_CHARS


def test_build_reads_the_cache_sidecars():
    cache_every_source()
    m = manifest.build(period="dev", command="portfolio-lab reproduce", estimator="sample")
    src = m["sources"]["ff5_daily"]
    meta = cache.cache_info(manifest.SOURCES["ff5_daily"])
    assert src["sha256"] == meta["sha256"] and src["bytes"] == meta["bytes"]
    assert src["retrieved"] == meta["retrieved"]
    assert src["header"].startswith("This file was created using the 202608 CRSP")
    run = m["runs"]["dev"]
    assert run["estimator"] == "sample" and run["command"] == "portfolio-lab reproduce"
    assert m["seed"] == study.SEED and m["dev_end"] == config.DEV_END
    assert m["bit_generator"] == "PCG64"
    assert {"python", "numpy", "pandas", "scipy", "statsmodels", "portfolio-lab"} <= set(
        m["packages"]
    )


def test_build_refuses_an_uncached_source():
    with pytest.raises(FileNotFoundError, match="ff5_daily"):
        manifest.build(period="dev", command="c", estimator="sample")


def test_drift_names_the_changed_files():
    old = {"sources": {"a": {"sha256": "1"}, "b": {"sha256": "2"}}}
    new = {"sources": {"a": {"sha256": "1"}, "b": {"sha256": "3"}, "c": {"sha256": "4"}}}
    assert manifest.drift(old, new) == ["b", "c"]
    assert manifest.drift(old, old) == []


def test_merge_keeps_the_other_periods_runs():
    old = {"runs": {"dev": {"created": "x"}}, "sources": {}}
    new = {"runs": {"holdout": {"created": "y"}}, "sources": {"a": {}}}
    merged = manifest.merge(old, new)
    assert merged["runs"] == {"dev": {"created": "x"}, "holdout": {"created": "y"}}
    assert merged["sources"] == {"a": {}}


# --- the holdout lock -------------------------------------------------------------------------


def git(root, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "research").mkdir(exist_ok=True)
    (tmp_path / "research" / "PLAN.md").write_bytes(b"# Plan\n\nFROZEN\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "freeze")
    git(tmp_path, "tag", cli.PREREG_TAG)
    monkeypatch.setattr(config, "HOLDOUT_END", "2026-08-31")
    monkeypatch.setattr(study, "PRIMARY_ESTIMATOR", "lw_identity")
    return tmp_path


def freeze(repo):
    """The freeze: a dev run records the vintage, then the manifest is committed and tagged."""
    assert cli.main(["reproduce"]) == 0
    git(repo, "add", "research/results/manifest.json")
    git(repo, "commit", "-qm", "freeze with the manifest")
    git(repo, "tag", "-f", cli.PREREG_TAG)


def test_lock_opens_with_the_tag_and_an_unchanged_plan(repo):
    cli.check_prereg(repo)


def test_lock_refuses_an_edited_plan(repo):
    (repo / "research" / "PLAN.md").write_bytes(b"# Plan\n\nFROZEN, then edited\n")
    with pytest.raises(HoldoutError, match="PLAN.md"):
        cli.check_prereg(repo)


def test_lock_ignores_line_endings_only(repo):
    (repo / "research" / "PLAN.md").write_bytes(b"# Plan\r\n\r\nFROZEN\r\n")
    cli.check_prereg(repo)


def test_lock_refuses_without_the_tag(repo):
    git(repo, "tag", "-d", cli.PREREG_TAG)
    with pytest.raises(HoldoutError, match="prereg-v1"):
        cli.check_prereg(repo)


def test_lock_refuses_without_holdout_end_or_estimator(repo, monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_END", None)
    with pytest.raises(HoldoutError, match="HOLDOUT_END"):
        cli.check_prereg(repo)
    monkeypatch.setattr(config, "HOLDOUT_END", "2026-08-31")
    monkeypatch.setattr(study, "PRIMARY_ESTIMATOR", None)
    with pytest.raises(HoldoutError, match="estimator"):
        cli.check_prereg(repo)


def test_lock_refuses_outside_a_git_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_END", "2026-08-31")
    monkeypatch.setattr(study, "PRIMARY_ESTIMATOR", "lw_identity")
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "PLAN.md").write_text("x")
    with pytest.raises(HoldoutError, match="prereg-v1"):
        cli.check_prereg(tmp_path)


# --- the command ------------------------------------------------------------------------------


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Replace the data and the study by stubs that record their arguments."""
    calls = {}

    def fake_load(**kw):
        calls["load"] = kw
        return "inputs"

    def fake_run(inputs, **kw):
        calls["run"] = kw
        design = pd.DataFrame({"value": {"estimator": "sample"}})
        return {"design": design, "tests": pd.DataFrame({"p": [0.1234567890123456]})}

    monkeypatch.setattr(study, "load", fake_load)
    monkeypatch.setattr(study, "run", fake_run)
    (tmp_path / "research").mkdir(exist_ok=True)
    (tmp_path / "research" / "PLAN.md").write_bytes(b"# Plan\n\nFROZEN\n")
    monkeypatch.chdir(tmp_path)
    cache_every_source()
    return calls


def test_parser_has_the_reproduce_options():
    args = cli.build_parser().parse_args(["reproduce", "--holdout", "--refresh"])
    assert args.holdout and args.refresh
    assert cli.build_parser().parse_args(["reproduce"]).out == cli.RESULTS


def test_reproduce_dev_writes_tables_and_manifest(stubbed, tmp_path):
    assert cli.main(["reproduce"]) == 0
    assert stubbed["load"] == {"end": config.DEV_END, "allow_holdout": False, "refresh": False}
    assert stubbed["run"]["period"] == "dev" and stubbed["run"]["allow_holdout"] is False
    out = tmp_path / "research" / "results"
    tests = pd.read_csv(out / "dev" / "tests.csv", index_col=0)
    assert tests["p"].iloc[0] == 0.1234567890123456  # full precision
    m = json.loads((out / "manifest.json").read_text())
    assert m["runs"]["dev"]["estimator"] == "sample"
    assert set(m["sources"]) == set(manifest.SOURCES)


def test_reproduce_dev_passes_the_holdout_length_once_frozen(stubbed, monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_END", "2026-08-31")
    cli.main(["reproduce"])
    assert stubbed["run"]["holdout_months"] == 188
    monkeypatch.setattr(config, "HOLDOUT_END", None)
    cli.main(["reproduce"])
    assert stubbed["run"]["holdout_months"] is None


def test_reproduce_warns_on_a_new_vintage(stubbed):
    cli.main(["reproduce"])
    cache_every_source({"usdcad": b"DATE,DEXCAUS\nrevised\n"})
    with pytest.warns(UserWarning, match="usdcad"):
        cli.main(["reproduce"])


def test_reproduce_holdout_refuses_without_the_lock(stubbed, capsys, monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_END", None)
    assert cli.main(["reproduce", "--holdout"]) == 2
    assert "run" not in stubbed and "load" not in stubbed
    assert "HOLDOUT_END" in capsys.readouterr().err


def test_reproduce_holdout_opens_with_the_lock(stubbed, repo, monkeypatch):
    monkeypatch.chdir(repo)
    freeze(repo)
    assert cli.main(["reproduce", "--holdout"]) == 0
    assert stubbed["load"] == {"end": "2026-08-31", "allow_holdout": True, "refresh": False}
    assert stubbed["run"]["period"] == "holdout"
    assert stubbed["run"]["estimator"] == "lw_identity"
    assert stubbed["run"]["allow_holdout"] is True
    assert (repo / "research" / "results" / "holdout" / "tests.csv").exists()


def test_main_without_a_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "reproduce" in capsys.readouterr().out


def test_reproduce_holdout_refuses_with_holdout_end_but_no_tag(stubbed, capsys, monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_END", "2026-08-31")
    monkeypatch.setattr(study, "PRIMARY_ESTIMATOR", "lw_identity")
    assert cli.main(["reproduce", "--holdout"]) == 2  # tmp_path is no git repository
    assert "run" not in stubbed and "load" not in stubbed
    assert "prereg-v1" in capsys.readouterr().err


def test_check_prereg_returns_the_tag_commit(repo):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert cli.check_prereg(repo) == head


def test_lock_refuses_a_tag_that_is_not_an_ancestor_of_head(repo):
    git(repo, "checkout", "-q", "--orphan", "elsewhere")
    git(repo, "commit", "-qm", "another history")  # same PLAN.md, unrelated history
    with pytest.raises(HoldoutError, match="ancestor"):
        cli.check_prereg(repo)


def test_holdout_refuses_refresh(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    assert cli.main(["reproduce", "--holdout", "--refresh"]) == 2
    assert "load" not in stubbed
    assert "refresh" in capsys.readouterr().err


def test_holdout_needs_the_frozen_vintage(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    assert cli.main(["reproduce", "--holdout"]) == 2  # no manifest.json at the tag
    assert "load" not in stubbed
    assert "manifest" in capsys.readouterr().err


def test_holdout_refuses_drift_unless_accepted(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    freeze(repo)
    stubbed.clear()
    cache_every_source({"usdcad": b"DATE,DEXCAUS\nrevised\n"})
    assert cli.main(["reproduce", "--holdout"]) == 2
    assert "load" not in stubbed and "usdcad" in capsys.readouterr().err
    with pytest.warns(UserWarning, match="usdcad"):
        assert cli.main(["reproduce", "--holdout", "--accept-drift"]) == 0


def test_holdout_runs_once_unless_rerun(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    freeze(repo)
    assert cli.main(["reproduce", "--holdout"]) == 0
    stubbed.clear()
    assert cli.main(["reproduce", "--holdout"]) == 2
    assert "load" not in stubbed and "already" in capsys.readouterr().err
    assert cli.main(["reproduce", "--holdout", "--rerun"]) == 0
    m = json.loads((repo / "research" / "results" / "manifest.json").read_text())
    assert m["runs"]["holdout"]["git"]["tag_commit"] == cli.check_prereg(repo)


def test_reproduce_finds_the_repository_root_from_a_subfolder(stubbed, repo, monkeypatch):
    (repo / "src").mkdir()
    monkeypatch.chdir(repo / "src")
    assert cli.main(["reproduce"]) == 0
    assert (repo / "research" / "results" / "dev" / "tests.csv").exists()


def test_reproduce_needs_the_plan(stubbed, tmp_path, capsys):
    (tmp_path / "research" / "PLAN.md").unlink()
    assert cli.main(["reproduce"]) == 2
    assert "PLAN.md" in capsys.readouterr().err


def test_reproduce_checks_the_cache_before_running(stubbed, capsys):
    cache.evict(manifest.SOURCES["ca_govt_10y"])
    assert cli.main(["reproduce"]) == 2
    assert "run" not in stubbed and "load" not in stubbed  # refused before any data is read
    assert "ca_govt_10y" in capsys.readouterr().err


def test_holdout_refuses_a_cache_that_changes_while_loading(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    freeze(repo)
    real = manifest.sources

    def changing():
        out = real()
        if "load" in stubbed:  # after the data were loaded
            out["usdcad"] = {**out["usdcad"], "sha256": "changed"}
        return out

    stubbed.clear()
    monkeypatch.setattr(manifest, "sources", changing)
    assert cli.main(["reproduce", "--holdout"]) == 2
    assert "run" not in stubbed and "changed while loading" in capsys.readouterr().err


def test_holdout_compares_with_the_manifest_at_the_tag(stubbed, repo, monkeypatch, capsys):
    """A dev run after the freeze rewrites the working manifest with the new vintage; the
    holdout still compares with the vintage frozen at the tag."""
    monkeypatch.chdir(repo)
    freeze(repo)
    cache_every_source({"usdcad": b"DATE,DEXCAUS\nrevised\n"})
    with pytest.warns(UserWarning, match="usdcad"):
        assert cli.main(["reproduce"]) == 0
    stubbed.clear()
    assert cli.main(["reproduce", "--holdout"]) == 2
    assert "load" not in stubbed and "usdcad" in capsys.readouterr().err


def test_one_shot_record_does_not_depend_on_out(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    freeze(repo)
    assert cli.main(["reproduce", "--holdout"]) == 0
    stubbed.clear()
    assert cli.main(["reproduce", "--holdout", "--out", str(repo / "elsewhere")]) == 2
    assert "already" in capsys.readouterr().err


def test_each_run_records_its_own_sources(stubbed, repo, monkeypatch):
    monkeypatch.chdir(repo)
    freeze(repo)
    cache_every_source({"usdcad": b"DATE,DEXCAUS\nrevised\n"})
    with pytest.warns(UserWarning):
        cli.main(["reproduce", "--holdout", "--accept-drift"])
    m = json.loads((repo / "research" / "results" / "manifest.json").read_text())
    dev, hold = m["runs"]["dev"]["sources"], m["runs"]["holdout"]["sources"]
    assert dev["usdcad"]["sha256"] != hold["usdcad"]["sha256"]
    assert dev["ff5_daily"]["sha256"] == hold["ff5_daily"]["sha256"]
