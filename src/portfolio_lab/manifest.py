"""The results manifest (research/results/manifest.json): where every number came from.

- For each source file the study reads (`SOURCES`): its URL, its header line, and the SHA-256,
  size and retrieval time from the download cache's sidecar.
- For each run (dev, holdout, holdout_rerun, robustness): the command, the time, the primary
  estimator, and the git commit with a dirty flag. The robustness run also records the
  49-industry file it reads (`ROBUSTNESS_SOURCES`), which is not part of the frozen vintage.
- The Python and package versions, NumPy's bit generator, the seed, DEV_END and HOLDOUT_END.

`drift` names the files whose SHA-256 differs from an earlier manifest (a newer vintage), and
`merge` keeps the other runs when one is run again.
"""

import io
import platform
import subprocess
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from portfolio_lab import config, study
from portfolio_lab.data import boc, cache, fred, french, statcan

__all__ = [
    "HEADER_CHARS",
    "RESULTS_MANIFEST",
    "ROBUSTNESS_SOURCES",
    "SOURCES",
    "build",
    "drift",
    "git_output",
    "header_line",
    "merge",
    "sources",
]

SOURCES = {
    "ff5_daily": french.BASE_URL + french.DATASETS["ff5_daily"],
    "mom_daily": french.BASE_URL + french.DATASETS["mom_daily"],
    "ind12_daily": french.BASE_URL + french.DATASETS["ind12_daily"],
    "usdcad": fred.csv_url("DEXCAUS"),
    "ca_tbill_3m": statcan.data_url("v122531"),
    "ca_govt_long": statcan.data_url("v122487"),
    "ca_govt_10y": boc.observations_url("BD.CDN.10YR.DQ.YLD"),
}
# Read by the robustness run only (Step 7b), so recorded in that run and never part of the
# vintage frozen at the tag; robustness.IND49_SHA256 fixes its content.
ROBUSTNESS_SOURCES = {"ind49_daily": french.BASE_URL + french.DATASETS["ind49_daily"]}
HEADER_CHARS = 200
RESULTS_MANIFEST = "research/results/manifest.json"  # tracked, and rewritten by every run
PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "portfolio-lab")


def header_line(content: bytes) -> str:
    """The first line of a file (of the first member of a zip), at most HEADER_CHARS long."""
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            content = z.read(z.namelist()[0])
    lines = content.decode("utf-8-sig", errors="replace").splitlines()
    return lines[0].strip()[:HEADER_CHARS] if lines else ""


def sources(table: Mapping[str, str] = SOURCES) -> dict[str, dict[str, Any]]:
    """The cached copy of every source of `table` (name -> URL; the study's by default): URL,
    header line, SHA-256, size, retrieval time.

    Reads the cache only (never downloads); a source missing or failing its checksum raises.
    """
    out = {}
    for name, url in table.items():
        meta = cache.cache_info(url)
        content = cache.read_cached(url)
        if meta is None or content is None:
            raise FileNotFoundError(
                f"{name} is not in the download cache, or fails its checksum ({url}): "
                "run `portfolio-lab reproduce --refresh`"
            )
        out[name] = {
            "url": url,
            "header": header_line(content),
            "sha256": meta["sha256"],
            "bytes": meta["bytes"],
            "retrieved": meta["retrieved"],
        }
    return out


def build(
    *,
    period: str,
    command: str,
    estimator: str,
    source_table: dict[str, dict[str, Any]] | None = None,
    root: Path | None = None,
    tag_commit: str | None = None,
    changed_since_tag: list[str] | None = None,
    extra_sources: dict[str, dict[str, Any]] | None = None,
    tables: dict[str, str] | None = None,
    holdout_tests: str | None = None,
) -> dict[str, Any]:
    """The manifest of one run, keyed `period`: `source_table` (default: `sources()` now) and
    the run, with the git state (for a holdout: the tag's commit and the source files changed
    since it). The run's record also keeps its package versions, the SHA-256 of each table it
    wrote (`tables`), files read outside the study's vintage (`extra_sources`), and for
    robustness the SHA-256 of the holdout tests table it extended (`holdout_tests`).
    """
    table = sources() if source_table is None else source_table
    git = _git_state(Path.cwd() if root is None else root)
    git["tag_commit"] = tag_commit
    git["changed_since_tag"] = changed_since_tag
    versions = {"python": platform.python_version()}
    versions |= {p: metadata.version(p) for p in PACKAGES}
    run: dict[str, Any] = {
        "command": command,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "estimator": estimator,
        "git": git,
        "sources": table,  # each run's own vintage (the top level is the latest run's)
        "packages": versions,  # and its own environment, which later runs do not overwrite
    }
    optional = {"extra_sources": extra_sources, "tables": tables, "holdout_tests": holdout_tests}
    run |= {k: v for k, v in optional.items() if v is not None}
    return {
        "schema": 1,
        "seed": study.SEED,
        "bit_generator": "PCG64",
        "dev_end": config.DEV_END,
        "holdout_start": config.HOLDOUT_START,
        "holdout_end": config.HOLDOUT_END,
        "packages": versions,
        "sources": table,
        "runs": {period: run},
    }


def drift(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """The sources of `new` whose SHA-256 is not the one in `old`, sorted."""
    before = old.get("sources", {})
    return sorted(
        name
        for name, src in new["sources"].items()
        if before.get(name, {}).get("sha256") != src["sha256"]
    )


def merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """`new`, keeping the runs of `old` for the other periods."""
    return {**new, "runs": {**old.get("runs", {}), **new["runs"]}}


def git_output(root: Path, *args: str) -> bytes:
    """The output of `git <args>` run in `root`; raises CalledProcessError on failure."""
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = git_output(root, "rev-parse", "HEAD").decode().strip()
        # Tracked changes only, and not the results manifest that every run rewrites: the
        # flag says whether the code differs from the commit.
        status = git_output(
            root,
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            f":(exclude){RESULTS_MANIFEST}",
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}
