"""`portfolio-lab reproduce` (Step 7a): manifest, drift warning, the holdout lock, the wiring;
and Step 7b's robustness run and rerun record."""

import hashlib
import io
import json
import subprocess
import zipfile

import pandas as pd
import pytest

from portfolio_lab import cli, config, manifest, robustness, study
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


def test_holdout_refuses_another_results_folder(stubbed, repo, monkeypatch, capsys):
    """A first holdout to another folder would leave no record where the one-shot check looks."""
    monkeypatch.chdir(repo)
    freeze(repo)
    stubbed.clear()  # the freeze's own dev run loaded data
    assert cli.main(["reproduce", "--holdout", "--out", str(repo / "elsewhere")]) == 2
    assert "load" not in stubbed and "--out" in capsys.readouterr().err
    assert cli.main(["reproduce", "--holdout"]) == 0
    stubbed.clear()
    assert cli.main(["reproduce", "--holdout"]) == 2  # recorded where the check looks


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


def test_holdout_refuses_uncommitted_source_changes(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    (repo / "src").mkdir()
    (repo / "src" / "procedure.py").write_text("RESAMPLES = 4999\n")
    git(repo, "add", "src")
    git(repo, "commit", "-qm", "code")
    freeze(repo)
    (repo / "src" / "procedure.py").write_text("RESAMPLES = 99\n")
    stubbed.clear()
    assert cli.main(["reproduce", "--holdout"]) == 2
    assert "load" not in stubbed and "uncommitted" in capsys.readouterr().err


def test_holdout_records_and_warns_about_code_changed_since_the_tag(stubbed, repo, monkeypatch):
    monkeypatch.chdir(repo)
    (repo / "src").mkdir()
    (repo / "src" / "procedure.py").write_text("RESAMPLES = 4999\n")
    git(repo, "add", "src")
    git(repo, "commit", "-qm", "code")
    freeze(repo)
    (repo / "src" / "procedure.py").write_text("RESAMPLES = 99\n")
    git(repo, "commit", "-qam", "changed after the tag")
    with pytest.warns(UserWarning, match="procedure.py"):
        assert cli.main(["reproduce", "--holdout"]) == 0
    m = json.loads((repo / "research" / "results" / "manifest.json").read_text())
    assert m["runs"]["holdout"]["git"]["changed_since_tag"] == ["src/procedure.py"]


def test_dirty_ignores_the_results_manifest(repo):
    (repo / "research" / "results").mkdir(parents=True)
    (repo / "research" / "results" / "manifest.json").write_text("{}")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "manifest")
    (repo / "research" / "results" / "manifest.json").write_text('{"runs": {}}')
    assert manifest._git_state(repo)["dirty"] is False
    (repo / "research" / "PLAN.md").write_bytes(b"edited")
    assert manifest._git_state(repo)["dirty"] is True


def test_a_corrupt_manifest_is_a_clean_error(stubbed, tmp_path, capsys):
    out = tmp_path / "research" / "results"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text("{ truncated")
    assert cli.main(["reproduce"]) == 2
    assert "manifest.json" in capsys.readouterr().err


def test_dev_refresh_loads_first_then_records_the_new_cache(stubbed):
    cache.evict(manifest.SOURCES["ca_govt_10y"])  # a refresh would download it again

    def refreshing_load(**kw):
        stubbed["load"] = kw
        cache_every_source()  # what the loaders' downloads leave in the cache
        return "inputs"

    study_load = study.load
    try:
        study.load = refreshing_load
        assert cli.main(["reproduce", "--refresh"]) == 0
    finally:
        study.load = study_load
    assert stubbed["load"]["refresh"] is True


# --- step 7b: the robustness run and the rerun record -----------------------------------------


def test_robustness_sources_are_the_49_industries_only():
    assert set(manifest.ROBUSTNESS_SOURCES) == {"ind49_daily"}
    url = manifest.ROBUSTNESS_SOURCES["ind49_daily"]
    assert url.endswith("49_Industry_Portfolios_daily_CSV.zip")
    assert not set(manifest.ROBUSTNESS_SOURCES) & set(manifest.SOURCES)  # the vintage stays 7


def test_sources_reads_another_table():
    url = manifest.ROBUSTNESS_SOURCES["ind49_daily"]
    cache._store(url, french_zip("49 industries"))
    got = manifest.sources(manifest.ROBUSTNESS_SOURCES)
    assert set(got) == {"ind49_daily"}
    assert got["ind49_daily"]["sha256"] == cache.cache_info(url)["sha256"]


def test_build_records_extra_sources_in_the_run_only():
    cache_every_source()
    extra = {"ind49_daily": {"sha256": "abc"}}
    m = manifest.build(period="robustness", command="c", estimator="e", extra_sources=extra)
    assert m["runs"]["robustness"]["extra_sources"] == extra
    assert "ind49_daily" not in m["sources"]


@pytest.fixture
def robust(stubbed, repo, monkeypatch):
    """A frozen repository whose one-shot holdout ran, the 49-industry file cached with the
    plan's hash, and stubs for the robustness module."""
    monkeypatch.chdir(repo)
    freeze(repo)
    assert cli.main(["reproduce", "--holdout"]) == 0
    tests = pd.DataFrame({"view": ["USD"], "strategy": ["s"], "block": [2], "p": [0.5]})
    tests.to_csv(repo / "research" / "results" / "holdout" / "tests.csv", index=False)
    content = french_zip("This file was created using the 202608 CRSP database (49).")
    cache._store(manifest.ROBUSTNESS_SOURCES["ind49_daily"], content)
    monkeypatch.setattr(robustness, "IND49_SHA256", hashlib.sha256(content).hexdigest())

    def fake_industries(**kw):
        stubbed["industries"] = kw
        return "industries"

    def fake_robustness(inputs, holdout_tests, **kw):
        stubbed["robustness"] = {"inputs": inputs, "holdout_tests": holdout_tests, **kw}
        design = pd.DataFrame({"value": {"estimator": "lw_identity"}})
        return {"design": design, "tests": pd.DataFrame({"p": [0.25]})}

    monkeypatch.setattr(robustness, "load_industries", fake_industries)
    monkeypatch.setattr(robustness, "run", fake_robustness)
    stubbed.clear()
    return stubbed


def manifest_of(repo):
    return json.loads((repo / "research" / "results" / "manifest.json").read_text())


def test_parser_has_the_robustness_option():
    assert cli.build_parser().parse_args(["reproduce", "--robustness"]).robustness
    assert not cli.build_parser().parse_args(["reproduce"]).robustness


def test_robustness_runs_after_the_holdout_on_its_tests(robust, repo):
    assert cli.main(["reproduce", "--robustness"]) == 0
    assert robust["load"] == {"end": "2026-08-31", "allow_holdout": True, "refresh": False}
    assert robust["industries"] == {"end": "2026-08-31", "allow_holdout": True}
    run = robust["robustness"]
    assert run["inputs"] == "inputs" and run["industries"] == "industries"
    assert run["end"] == "2026-08-31" and run["estimator"] == "lw_identity"
    assert run["allow_holdout"] is True
    held = run["holdout_tests"]
    assert list(held.index.names) == ["view", "strategy"] and held.loc[("USD", "s"), "block"] == 2
    assert (repo / "research" / "results" / "robustness" / "tests.csv").exists()
    m = manifest_of(repo)
    rec = m["runs"]["robustness"]
    assert rec["command"] == "portfolio-lab reproduce --robustness"
    assert rec["git"]["tag_commit"] == cli.check_prereg(repo)
    assert rec["extra_sources"]["ind49_daily"]["sha256"] == robustness.IND49_SHA256
    assert "holdout" in m["runs"] and "ind49_daily" not in m["sources"]


def test_robustness_refuses_before_the_one_shot_holdout(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    freeze(repo)
    stubbed.clear()
    assert cli.main(["reproduce", "--robustness"]) == 2
    assert "load" not in stubbed and "--holdout" in capsys.readouterr().err


def test_robustness_needs_the_holdout_tests_table(robust, repo, capsys):
    (repo / "research" / "results" / "holdout" / "tests.csv").unlink()
    assert cli.main(["reproduce", "--robustness"]) == 2
    assert "load" not in robust and "--rerun" in capsys.readouterr().err


def test_robustness_refuses_another_49_industry_file_unless_accepted(robust, repo, capsys):
    cache._store(manifest.ROBUSTNESS_SOURCES["ind49_daily"], french_zip("a newer file"))
    assert cli.main(["reproduce", "--robustness"]) == 2
    assert "load" not in robust and "step-7b-plan.md" in capsys.readouterr().err
    with pytest.warns(UserWarning, match="49-industry"):
        assert cli.main(["reproduce", "--robustness", "--accept-drift"]) == 0


def test_robustness_needs_the_49_industry_file_cached(robust, capsys):
    cache.evict(manifest.ROBUSTNESS_SOURCES["ind49_daily"])
    assert cli.main(["reproduce", "--robustness"]) == 2
    assert "load" not in robust and "ind49_daily" in capsys.readouterr().err


def test_robustness_keeps_the_holdout_lock(robust, repo, capsys):
    (repo / "research" / "PLAN.md").write_bytes(b"# Plan\n\nedited after the holdout\n")
    assert cli.main(["reproduce", "--robustness"]) == 2
    assert "load" not in robust and "PLAN.md" in capsys.readouterr().err


def test_robustness_refuses_the_options_of_other_runs(robust, repo, capsys):
    for extra in (["--holdout"], ["--refresh"], ["--out", str(repo / "elsewhere")]):
        assert cli.main(["reproduce", "--robustness", *extra]) == 2, extra
    assert "load" not in robust
    err = capsys.readouterr().err
    assert "--holdout" in err and "--refresh" in err and "--out" in err


def test_robustness_refuses_a_49_industry_file_that_changes_while_loading(
    robust, monkeypatch, capsys
):
    def changing(**kw):
        robust["industries"] = kw
        cache._store(manifest.ROBUSTNESS_SOURCES["ind49_daily"], french_zip("swapped"))
        return "industries"

    monkeypatch.setattr(robustness, "load_industries", changing)
    assert cli.main(["reproduce", "--robustness"]) == 2
    assert "robustness" not in robust and "changed while loading" in capsys.readouterr().err


def test_a_rerun_gets_its_own_record_and_keeps_the_one_shot(robust, repo):
    first = manifest_of(repo)["runs"]["holdout"]
    assert cli.main(["reproduce", "--holdout", "--rerun"]) == 0
    runs = manifest_of(repo)["runs"]
    assert runs["holdout"] == first
    assert runs["holdout_rerun"]["command"] == "portfolio-lab reproduce --holdout --rerun"
    assert runs["holdout_rerun"]["git"]["tag_commit"] == first["git"]["tag_commit"]


# --- review fixes -----------------------------------------------------------------------------


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_each_run_records_the_sha256_of_every_table_it_wrote(robust, repo):
    held = repo / "research" / "results" / "holdout"
    first = manifest_of(repo)["runs"]["holdout"]["tables"]
    assert set(first) == {"design.csv", "tests.csv"}  # the stub's tables
    assert first["design.csv"] == sha(held / "design.csv")
    # the fixture replaced tests.csv after the one-shot: the record still names the original
    assert first["tests.csv"] != sha(held / "tests.csv")
    assert cli.main(["reproduce", "--robustness"]) == 0
    rec = manifest_of(repo)["runs"]["robustness"]
    assert rec["holdout_tests"] == sha(held / "tests.csv")  # the table it extended
    written = repo / "research" / "results" / "robustness" / "tests.csv"
    assert rec["tables"]["tests.csv"] == sha(written)
    assert cli.main(["reproduce", "--holdout", "--rerun"]) == 0
    runs = manifest_of(repo)["runs"]
    assert runs["holdout"]["tables"] == first
    assert runs["holdout_rerun"]["tables"]["tests.csv"] == sha(held / "tests.csv")


def test_each_run_records_its_own_package_versions(stubbed, tmp_path):
    assert cli.main(["reproduce"]) == 0
    m = json.loads((tmp_path / "research" / "results" / "manifest.json").read_text())
    assert m["runs"]["dev"]["packages"] == m["packages"]
    assert m["runs"]["dev"]["packages"]["numpy"]


def test_rerun_is_only_for_the_holdout(stubbed, capsys):
    assert cli.main(["reproduce", "--rerun"]) == 2
    assert cli.main(["reproduce", "--robustness", "--rerun"]) == 2
    assert "load" not in stubbed and "--rerun" in capsys.readouterr().err


def test_a_manifest_that_is_not_an_object_is_a_clean_error(stubbed, tmp_path, capsys):
    out = tmp_path / "research" / "results"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text("[1, 2]")
    assert cli.main(["reproduce"]) == 2
    assert "manifest.json" in capsys.readouterr().err


def test_robustness_records_the_code_changed_since_the_tag(robust, repo):
    assert cli.main(["reproduce", "--robustness"]) == 0
    assert manifest_of(repo)["runs"]["robustness"]["git"]["changed_since_tag"] == []


def test_robustness_refuses_frozen_data_that_change_while_loading(robust, monkeypatch, capsys):
    real = manifest.sources

    def changing(table=manifest.SOURCES):
        out = real(table)
        if "industries" in robust and table is manifest.SOURCES:
            out["usdcad"] = {**out["usdcad"], "sha256": "changed"}
        return out

    monkeypatch.setattr(manifest, "sources", changing)
    assert cli.main(["reproduce", "--robustness"]) == 2
    assert "robustness" not in robust and "changed while loading" in capsys.readouterr().err


def test_accept_drift_says_it_covers_the_49_industry_file(capsys):
    with pytest.raises(SystemExit):
        cli.main(["reproduce", "--help"])
    assert "49-industry" in capsys.readouterr().out


def test_a_first_rerun_is_refused(stubbed, repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    freeze(repo)
    stubbed.clear()
    assert cli.main(["reproduce", "--holdout", "--rerun"]) == 2
    assert "load" not in stubbed and "no holdout is recorded" in capsys.readouterr().err
