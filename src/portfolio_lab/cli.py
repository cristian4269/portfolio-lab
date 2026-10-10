"""Command-line entry point: `uv run portfolio-lab <command>`.

`reproduce` regenerates the pre-registered study's tables (research/PLAN.md):
- by default the dev period, written to research/results/dev/;
- with `--holdout`, the one-shot holdout, written to research/results/holdout/.
Each run updates research/results/manifest.json (the only result file in git), and warns when
a source's SHA-256 differs from the one recorded there (a newer data vintage). The project
root is the nearest folder, from the current one up, that holds research/PLAN.md.

This is the only file in src/ that may open the holdout (tests/test_holdout_scan.py), and it
does so only after these checks, all before any data is read:
- `check_prereg`: HOLDOUT_END and the primary estimator are frozen, the tag prereg-v1 exists
  and is an ancestor of HEAD, and research/PLAN.md is identical to its tagged copy (line
  endings aside);
- the frozen vintage: the cached sources must match the manifest's SHA-256s (`--accept-drift`
  runs on newer data anyway, with a warning), and `--refresh` is refused;
- one shot: a holdout already recorded in the manifest runs again only with `--rerun`.
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_lab import __version__, config, manifest, study
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
        tagged = _git(root, "show", f"{ref}:{PLAN.as_posix()}")
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
        _git(root, "merge-base", "--is-ancestor", ref, "HEAD")
    except (OSError, subprocess.CalledProcessError) as err:
        raise HoldoutError(
            f"the tag {PREREG_TAG} is not an ancestor of HEAD: run from the history it froze"
        ) from err
    return _git(root, "rev-parse", f"{ref}^{{commit}}").decode().strip()


def _reproduce(args: argparse.Namespace) -> int:
    root = _root()
    out = args.out if args.out.is_absolute() else root / args.out
    path = out / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    tag_commit = None
    if args.holdout:
        if args.refresh:
            raise ReproduceError(
                "--holdout runs on the frozen vintage, so --refresh is refused: refresh in a dev "
                "run, which reports the drift"
            )
        tag_commit = check_prereg(root)
        current = _frozen_vintage(old, path, accept_drift=args.accept_drift)
        if old is not None and "holdout" in old.get("runs", {}) and not args.rerun:
            raise HoldoutError(
                f"the holdout has already been run once (see {path}): pass --rerun to reproduce it"
            )
        period, end = "holdout", config.HOLDOUT_END
        assert end is not None  # checked by check_prereg
        inputs = study.load(end=end, allow_holdout=True, refresh=False)
        if manifest.sources() != current:
            raise HoldoutError("the cached data changed while loading: the vintage is not frozen")
        tables = study.run(
            inputs, period=period, end=end, estimator=study.PRIMARY_ESTIMATOR, allow_holdout=True
        )
        command = "portfolio-lab reproduce --holdout"
    else:
        period, end = "dev", config.DEV_END
        inputs = study.load(end=end, allow_holdout=False, refresh=args.refresh)
        current = manifest.sources()
        _warn_drift(old, current, path)
        tables = study.run(
            inputs, period=period, end=end, allow_holdout=False, holdout_months=_holdout_months()
        )
        command = "portfolio-lab reproduce"
    folder = out / period
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", lineterminator="\n")
    new = manifest.build(
        period=period,
        command=command,
        estimator=str(tables["design"].loc["estimator", "value"]),
        source_table=current,
        root=root,
        tag_commit=tag_commit,
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


def _frozen_vintage(
    old: dict[str, Any] | None, path: Path, *, accept_drift: bool
) -> dict[str, dict[str, Any]]:
    """The cached sources, which must be the vintage the manifest froze (or accepted drift)."""
    if old is None:
        raise HoldoutError(f"no {path} records the frozen vintage: run the dev analysis first")
    current = manifest.sources()
    changed = manifest.drift(old, {"sources": current})
    if changed and not accept_drift:
        raise HoldoutError(
            f"the cached data differ from the frozen vintage in {path}: {', '.join(changed)}; "
            "pass --accept-drift to run on them anyway"
        )
    _warn_drift(old, current, path)
    return current


def _warn_drift(old: dict[str, Any] | None, current: dict[str, Any], path: Path) -> None:
    changed = [] if old is None else manifest.drift(old, {"sources": current})
    if changed:
        warnings.warn(
            f"the data differ from {path}: {', '.join(changed)} have a new SHA-256 (a newer "
            "vintage), so the results may drift from the recorded ones",
            stacklevel=3,
        )


def _holdout_months() -> int | None:
    """The number of holdout months once HOLDOUT_END is frozen (for the dev MDE), else None."""
    if config.HOLDOUT_END is None:
        return None
    start, end = pd.Timestamp(config.HOLDOUT_START), pd.Timestamp(config.HOLDOUT_END)
    return (end.year - start.year) * 12 + end.month - start.month + 1


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def _lf(content: bytes) -> bytes:
    return content.replace(b"\r\n", b"\n")


if __name__ == "__main__":
    sys.exit(main())
