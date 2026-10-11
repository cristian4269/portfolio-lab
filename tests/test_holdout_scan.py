"""The holdout may only be opened from the `reproduce` command and research/ (blueprint)."""

import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDLE = "allow_holdout=True"
FLAG = "allow_holdout"
# tests/ is exempt too: guard tests must call the flag, and tests cannot leak the
# holdout into results (owner decision, 2026-10-02).
ALLOWED_FILES = {ROOT / "src" / "portfolio_lab" / "cli.py"}
SKIPPED_AT_ROOT = {"tests", "research", "build", "dist", "node_modules"}  # top level only


def python_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        top = Path(dirpath) == root
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d != "__pycache__" and not (top and d in SKIPPED_AT_ROOT)
        ]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def flag_set_true(source: str) -> list[int]:
    """Lines that set allow_holdout to the constant True, however it is spelled: a keyword
    argument, an assignment, or a dict key (as in **{"allow_holdout": True})."""

    def is_true(node: ast.AST | None) -> bool:
        return isinstance(node, ast.Constant) and node.value is True

    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == FLAG and is_true(node.value):
            lines.append(node.value.lineno)
        elif isinstance(node, ast.Assign) and is_true(node.value):
            if any(isinstance(t, ast.Name) and t.id == FLAG for t in node.targets):
                lines.append(node.lineno)
        elif isinstance(node, ast.AnnAssign) and is_true(node.value):
            if isinstance(node.target, ast.Name) and node.target.id == FLAG:
                lines.append(node.lineno)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == FLAG and is_true(value):
                    lines.append(key.lineno)
    return sorted(lines)


def test_scan_covers_the_package():
    found = set(python_files(ROOT))
    assert ROOT / "src" / "portfolio_lab" / "config.py" in found
    assert ROOT / "src" / "portfolio_lab" / "report" / "inference.py" in found
    assert not any(p.relative_to(ROOT).parts[0] in SKIPPED_AT_ROOT for p in found)


def test_the_detector_sees_every_spelling():
    assert flag_set_true("f(x, allow_holdout=True)") == [1]
    assert flag_set_true("allow_holdout = True") == [1]
    assert flag_set_true("allow_holdout: bool = True") == [1]
    assert flag_set_true('f(**{"allow_holdout": True})') == [1]
    assert flag_set_true("f(allow_holdout=allow_holdout)\nallow_holdout = False") == []


def test_holdout_flag_only_in_allowed_places():
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}"
        for path in python_files(ROOT)
        if path not in ALLOWED_FILES
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if NEEDLE in line
    ]
    assert offenders == [], f"{NEEDLE} outside cli.py/research/: {offenders}"


def test_no_other_spelling_sets_the_flag_outside_cli():
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}"
        for path in python_files(ROOT)
        if path not in ALLOWED_FILES
        for lineno in flag_set_true(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"allow_holdout set to True outside cli.py: {offenders}"
