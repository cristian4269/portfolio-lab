"""Project-wide settings. The research split is fixed here and in research/PLAN.md.

Development uses data up to DEV_END. The holdout (HOLDOUT_START on) stays
unseen until the pre-registration is committed (blueprint step 7a); the data
loaders truncate at DEV_END unless their `allow_holdout` flag is set, which only
the `reproduce` command and research/ may do (enforced by tests/test_holdout_scan.py).
"""

import os
from pathlib import Path

DEV_END = "2010-12-31"
HOLDOUT_START = "2011-01-01"

CACHE_ENV_VAR = "PORTFOLIO_LAB_CACHE"


def cache_dir() -> Path:
    """Where downloaded source files are cached (outside the repo by default)."""
    override = os.environ.get(CACHE_ENV_VAR)
    return Path(override) if override else Path.home() / ".cache" / "portfolio-lab"
