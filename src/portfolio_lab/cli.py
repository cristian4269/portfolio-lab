"""Command-line entry point: `uv run portfolio-lab <command>`.

`reproduce` regenerates the pre-registered study's tables (research/PLAN.md):
- by default the dev period, written to research/results/dev/;
- with `--holdout`, the one-shot holdout, written to research/results/holdout/ (a `--rerun`
  is recorded apart, as `holdout_rerun`, so the one-shot's own record is never overwritten);
- with `--robustness`, after the holdout, the registered robustness checks (`robustness`),
  written to research/results/robustness/, under the same lock, and only on the 49-industry
  file whose SHA-256 plans/step-7b-plan.md fixed.
Each run updates research/results/manifest.json (the only result file in git), and warns when
a source's SHA-256 differs from the one recorded there (a newer data vintage). The project
root is the nearest folder, from the current one up, that holds research/PLAN.md.

This is the only file in src/ that may open the holdout (tests/test_holdout_scan.py), and it
does so only after these checks, all before any data is read:
- `check_prereg`: HOLDOUT_END and the primary estimator are frozen, the tag prereg-v1 exists
  and is an ancestor of HEAD, and research/PLAN.md is identical to its tagged copy (line
  endings aside);
- the frozen vintage: the cached sources must match the SHA-256s of the manifest committed at
  the tag (`--accept-drift` runs on newer data anyway, with a warning), and `--refresh` is
  refused;
- one shot: a holdout already recorded in research/results/manifest.json runs again only with
  `--rerun`, and a holdout always writes there (`--out` is refused);
- traceable code: no uncommitted change under src/; source files changed since the tag are
  named in a warning and recorded in the manifest.
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_lab import __version__, config, manifest, robustness, study
from portfolio_lab.data import HoldoutError

RESULTS = Path("research/results")
PLAN = Path("research/PLAN.md")
PREREG_TAG = config.PREREG_TAG


class ReproduceError(RuntimeError):
    """A reproduce run that cannot start as asked (no plan, a refused option)."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portfolio-lab",
        description="Build portfolios, backtest them honestly, and measure their risk.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command")
    rep = sub.add_parser("reproduce", help="regenerate every table of the study (research/)")
    rep.add_argument(
        "--holdout",
        action="store_true",
        help=f"run the one-shot holdout (needs the tag {PREREG_TAG} on research/PLAN.md)",
    )
    rep.add_argument(
        "--robustness",
        action="store_true",
        help="run the registered robustness checks on the recorded holdout (Step 7b)",
    )
    rep.add_argument("--refresh", action="store_true", help="download every source again (dev)")
    rep.add_argument(
        "--accept-drift",
        action="store_true",
        help="run the holdout on data that differ from the frozen vintage (warns)",
    )
    rep.add_argument("--rerun", action="store_true", help="run a holdout already recorded again")
    rep.add_argument(
        "--out", type=Path, default=RESULTS, help="results folder (default %(default)s)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return _reproduce(args)
    except (HoldoutError, ReproduceError, FileNotFoundError) as err:
        print(f"portfolio-lab reproduce: {err}", file=sys.stderr)
        return 2


def check_prereg(root: Path) -> str:
    """The tag's commit, or HoldoutError unless the pre-registration is frozen and tagged."""
    if config.HOLDOUT_END is None:
        raise HoldoutError(
            "HOLDOUT_END is not frozen in config.py (research/PLAN.md §13): the holdout is closed"
        )
    if study.PRIMARY_ESTIMATOR is None:
        raise HoldoutError("the primary estimator is not frozen in study.py (research/PLAN.md §13)")
    ref = f"refs/tags/{PREREG_TAG}"  # a branch of the same name cannot stand in for the tag
    try:
        tagged = manifest.git_output(root, "show", f"{ref}:{PLAN.as_posix()}")
    except (OSError, subprocess.CalledProcessError) as err:
        raise HoldoutError(
            f"no tag {PREREG_TAG} holding {PLAN.as_posix()} in {root}: the pre-registration "
            "must be tagged before the holdout runs"
        ) from err
    if _lf(tagged) != _lf((root / PLAN).read_bytes()):
        raise HoldoutError(
            f"{PLAN.as_posix()} differs from its copy at {PREREG_TAG}: the holdout runs only "
            "on the plan as frozen"
        )
    try:
        manifest.git_output(root, "merge-base", "--is-ancestor", ref, "HEAD")
    except (OSError, subprocess.CalledProcessError) as err:
        raise HoldoutError(
            f"the tag {PREREG_TAG} is not an ancestor of HEAD: run from the history it froze"
        ) from err
    return manifest.git_output(root, "rev-parse", f"{ref}^{{commit}}").decode().strip()


def _reproduce(args: argparse.Namespace) -> int:
    root = _root()
    out = args.out if args.out.is_absolute() else root / args.out
    path = out / "manifest.json"
    old = _read_manifest(path)
    recorded = old is not None and "holdout" in old.get("runs", {})
    tag_commit, changed, extra = None, None, None
    if args.robustness:
        if args.holdout:
            raise ReproduceError(
                "--robustness and --holdout are separate runs: the holdout first, then "
                "--robustness on its results"
            )
        tag_commit, current = _locked(args, root, out, "--robustness")
        if not recorded:
            raise HoldoutError(
                f"robustness extends the one-shot holdout, which {path} does not record: run "
                "`reproduce --holdout` first"
            )
        held_path = out / "holdout" / "tests.csv"
        if not held_path.exists():
            raise ReproduceError(
                f"no {held_path} (result tables are not committed): run `reproduce --holdout "
                "--rerun` to regenerate the holdout's tables first"
            )
        extra = _industries_vintage(accept_drift=args.accept_drift)
        changed = _changed_since_tag(root)
        period, key, end = "robustness", "robustness", config.HOLDOUT_END
        assert end is not None  # checked by check_prereg
        inputs = study.load(end=end, allow_holdout=True, refresh=False)
        industries = robustness.load_industries(end=end, allow_holdout=True)
        if manifest.sources() != current or (
            manifest.sources(manifest.ROBUSTNESS_SOURCES) != extra
        ):
            raise HoldoutError("the cached data changed while loading: the vintage is not frozen")
        assert study.PRIMARY_ESTIMATOR is not None  # checked by check_prereg
        tables = robustness.run(
            inputs,
            pd.read_csv(held_path, index_col=["view", "strategy"]),
            industries=industries,
            end=end,
            estimator=study.PRIMARY_ESTIMATOR,
            allow_holdout=True,
        )
        command = "portfolio-lab reproduce --robustness"
    elif args.holdout:
        tag_commit, current = _locked(args, root, out, "--holdout")
        if recorded and not args.rerun:
            raise HoldoutError(
                f"the holdout has already been run once (see {path}): pass --rerun to reproduce it"
            )
        changed = _changed_since_tag(root)
        period, end = "holdout", config.HOLDOUT_END
        key = "holdout_rerun" if recorded else "holdout"  # never over the one-shot's record
        assert end is not None  # checked by check_prereg
        inputs = study.load(end=end, allow_holdout=True, refresh=False)
        if manifest.sources() != current:
            raise HoldoutError("the cached data changed while loading: the vintage is not frozen")
        tables = study.run(
            inputs,
            period=period,
            end=end,
            estimator=study.PRIMARY_ESTIMATOR,
            allow_holdout=True,
            settings=study.Settings(workers=_workers()),
        )
        command = "portfolio-lab reproduce --holdout" + (" --rerun" if recorded else "")
    else:
        period, key, end = "dev", "dev", config.DEV_END
        if not args.refresh:
            manifest.sources()  # every source cached and intact before any data is read
        inputs = study.load(end=end, allow_holdout=False, refresh=args.refresh)
        current = manifest.sources()
        _warn_drift(old, current, str(path))
        tables = study.run(
            inputs,
            period=period,
            end=end,
            allow_holdout=False,
            holdout_months=_holdout_months(),
            settings=study.Settings(workers=_workers()),
        )
        command = "portfolio-lab reproduce"
    folder = out / period
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", lineterminator="\n")
    new = manifest.build(
        period=key,
        command=command,
        estimator=str(tables["design"].loc["estimator", "value"]),
        source_table=current,
        root=root,
        tag_commit=tag_commit,
        changed_since_tag=changed,
        extra_sources=extra,
    )
    merged = new if old is None else manifest.merge(old, new)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(tables)} tables to {folder} and the manifest to {path}")
    return 0


def _root() -> Path:
    """The nearest folder, from the current one up, that holds research/PLAN.md."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / PLAN).exists():
            return folder
    raise ReproduceError(
        f"no {PLAN.as_posix()} in {here} or above: run reproduce inside the portfolio-lab "
        "repository"
    )


def _locked(
    args: argparse.Namespace, root: Path, out: Path, flag: str
) -> tuple[str, dict[str, dict[str, Any]]]:
    """The holdout's lock, shared by `--holdout` and `--robustness`, all before any data is
    read: the tag's commit and the frozen vintage of the cached sources."""
    if args.refresh:
        raise ReproduceError(
            f"{flag} runs on the frozen vintage, so --refresh is refused: refresh in a dev run, "
            "which reports the drift"
        )
    if out.resolve() != (root / RESULTS).resolve():
        raise ReproduceError(
            f"{flag} writes to {RESULTS.as_posix()} only, where the one-shot holdout is "
            "recorded: drop --out"
        )
    tag_commit = check_prereg(root)
    _no_uncommitted_source(root)
    return tag_commit, _frozen_vintage(root, accept_drift=args.accept_drift)


def _industries_vintage(*, accept_drift: bool) -> dict[str, dict[str, Any]]:
    """The cached 49-industry file, which must be the one plans/step-7b-plan.md fixed before
    the holdout ran (`robustness.IND49_SHA256`), unless the drift is accepted."""
    extra = manifest.sources(manifest.ROBUSTNESS_SOURCES)
    sha = extra["ind49_daily"]["sha256"]
    if sha != robustness.IND49_SHA256:
        message = (
            f"the cached 49-industry file ({sha[:12]}) is not the one fixed in "
            f"plans/step-7b-plan.md ({robustness.IND49_SHA256[:12]})"
        )
        if not accept_drift:
            raise HoldoutError(f"{message}: pass --accept-drift to run on it anyway")
        warnings.warn(f"{message}, so its rows may drift from the recorded ones", stacklevel=3)
    return extra


def _frozen_vintage(root: Path, *, accept_drift: bool) -> dict[str, dict[str, Any]]:
    """The cached sources, which must be the vintage of the manifest committed at the tag (a
    later dev run rewrites the working copy), unless the drift is accepted."""
    frozen_path = f"refs/tags/{PREREG_TAG}:{(RESULTS / 'manifest.json').as_posix()}"
    try:
        frozen = json.loads(manifest.git_output(root, "show", frozen_path))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as err:
        raise HoldoutError(
            f"no manifest.json at {PREREG_TAG} records the frozen vintage: commit it with the "
            "freeze"
        ) from err
    current = manifest.sources()
    changed = manifest.drift(frozen, {"sources": current})
    if changed and not accept_drift:
        raise HoldoutError(
            f"the cached data differ from the vintage frozen at {PREREG_TAG}: "
            f"{', '.join(changed)}; pass --accept-drift to run on them anyway"
        )
    _warn_drift(frozen, current, frozen_path)
    return current


def _warn_drift(old: dict[str, Any] | None, current: dict[str, Any], where: str) -> None:
    changed = [] if old is None else manifest.drift(old, {"sources": current})
    if changed:
        warnings.warn(
            f"the data differ from {where}: {', '.join(changed)} have a new SHA-256 (a newer "
            "vintage), so the results may drift from the recorded ones",
            stacklevel=3,
        )


def _read_manifest(path: Path) -> dict[str, Any] | None:
    """The manifest at `path`, None if absent; a file that is not JSON is a clean error."""
    if not path.exists():
        return None
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ReproduceError(
            f"{path} is not valid JSON ({err}): restore it from git before reproducing"
        ) from err
    return loaded


def _no_uncommitted_source(root: Path) -> None:
    status = manifest.git_output(root, "status", "--porcelain", "--untracked-files=no", "--", "src")
    if status.strip():
        raise HoldoutError(
            "uncommitted changes under src/: commit them first, so the holdout's code is a "
            f"commit ({status.decode().strip()})"
        )


def _changed_since_tag(root: Path) -> list[str]:
    """The source files that differ between the tag and HEAD, named in a warning."""
    names = manifest.git_output(
        root, "diff", "--name-only", f"refs/tags/{PREREG_TAG}", "HEAD", "--", "src"
    )
    changed = names.decode().split()
    if changed:
        warnings.warn(
            f"the source changed since {PREREG_TAG}: {', '.join(changed)} (recorded in the "
            "manifest)",
            stacklevel=3,
        )
    return changed


def _workers() -> int:
    """Processes for the block calibration: up to 4, never more than the machine has."""
    return max(1, min(4, os.cpu_count() or 1))


def _holdout_months() -> int | None:
    """The number of holdout months once HOLDOUT_END is frozen (for the dev MDE), else None."""
    if config.HOLDOUT_END is None:
        return None
    start, end = pd.Timestamp(config.HOLDOUT_START), pd.Timestamp(config.HOLDOUT_END)
    return (end.year - start.year) * 12 + end.month - start.month + 1


def _lf(content: bytes) -> bytes:
    return content.replace(b"\r\n", b"\n")


if __name__ == "__main__":
    sys.exit(main())
