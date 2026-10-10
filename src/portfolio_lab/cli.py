"""Command-line entry point: `uv run portfolio-lab <command>`.

`reproduce` regenerates the pre-registered study's tables (research/PLAN.md):
- by default the dev period, written to research/results/dev/;
- with `--holdout`, the one-shot holdout, written to research/results/holdout/.
Each run updates research/results/manifest.json (the only result file in git), and warns when
a source's SHA-256 differs from the one recorded there (a newer data vintage).

This is the only file in src/ that may open the holdout (tests/test_holdout_scan.py), and it
does so only after `check_prereg`: HOLDOUT_END and the primary estimator are frozen, the tag
prereg-v1 exists, and research/PLAN.md is identical to its tagged copy (line endings aside).
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

from portfolio_lab import __version__, config, manifest, study
from portfolio_lab.data import HoldoutError

RESULTS = Path("research/results")
PLAN = Path("research/PLAN.md")
PREREG_TAG = config.PREREG_TAG


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
    rep.add_argument("--refresh", action="store_true", help="download every source again")
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
    except HoldoutError as err:
        print(f"portfolio-lab reproduce: {err}", file=sys.stderr)
        return 2


def check_prereg(root: Path) -> None:
    """Raise HoldoutError unless the pre-registration is frozen and tagged (module doc)."""
    if config.HOLDOUT_END is None:
        raise HoldoutError(
            "HOLDOUT_END is not frozen in config.py (research/PLAN.md §13): the holdout is closed"
        )
    if study.PRIMARY_ESTIMATOR is None:
        raise HoldoutError("the primary estimator is not frozen in study.py (research/PLAN.md §13)")
    try:
        tagged = subprocess.run(
            ["git", "show", f"{PREREG_TAG}:{PLAN.as_posix()}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
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


def _reproduce(args: argparse.Namespace) -> int:
    root = Path.cwd()
    out = args.out if args.out.is_absolute() else root / args.out
    if args.holdout:
        check_prereg(root)
        period, end = "holdout", config.HOLDOUT_END
        assert end is not None  # checked by check_prereg
        inputs = study.load(end=end, allow_holdout=True, refresh=args.refresh)
        tables = study.run(
            inputs, period=period, end=end, estimator=study.PRIMARY_ESTIMATOR, allow_holdout=True
        )
        command = "portfolio-lab reproduce --holdout"
    else:
        period, end = "dev", config.DEV_END
        inputs = study.load(end=end, allow_holdout=False, refresh=args.refresh)
        tables = study.run(
            inputs, period=period, end=end, allow_holdout=False, holdout_months=_holdout_months()
        )
        command = "portfolio-lab reproduce"
    folder = out / period
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", lineterminator="\n")
    estimator = str(tables["design"].loc["estimator", "value"])
    new = manifest.build(period=period, command=command, estimator=estimator, root=root)
    path = out / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        changed = manifest.drift(old, new)
        if changed:
            warnings.warn(
                f"the data differ from {path}: {', '.join(changed)} have a new SHA-256 (a newer "
                "vintage), so the results may drift from the recorded ones",
                stacklevel=2,
            )
        new = manifest.merge(old, new)
    path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(tables)} tables to {folder} and the manifest to {path}")
    return 0


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
