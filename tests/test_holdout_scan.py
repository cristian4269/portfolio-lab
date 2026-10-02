"""The holdout may only be opened from the `reproduce` command and research/ (blueprint)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDLE = "allow_holdout=True"
# tests/ is exempt too: guard tests must call the flag, and tests cannot leak the
# holdout into results (owner decision, 2026-10-02).
ALLOWED_FILES = {ROOT / "src" / "portfolio_lab" / "cli.py"}
SKIPPED_DIRS = {"tests", "research", "build", "dist", "node_modules"}


def python_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIPPED_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def test_scan_covers_the_package():
    found = set(python_files(ROOT))
    assert ROOT / "src" / "portfolio_lab" / "config.py" in found
    assert not any("tests" in p.relative_to(ROOT).parts for p in found)


def test_holdout_flag_only_in_allowed_places():
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}"
        for path in python_files(ROOT)
        if path not in ALLOWED_FILES
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if NEEDLE in line
    ]
    assert offenders == [], f"{NEEDLE} outside cli.py/research/: {offenders}"
