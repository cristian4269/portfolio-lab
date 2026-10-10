"""On-disk download cache, keyed by URL.

Each download is stored as `<key>.bin` with a `<key>.meta.json` sidecar holding
the URL, UTC retrieval time, SHA-256 and size. The metadata feeds the results
manifest (blueprint step 7a), so every number can be traced to the exact file
it came from. Files live outside the repo (`config.cache_dir()`), never in git.

Requests to the same host are spaced at least MIN_INTERVAL_SECONDS apart; the
Bank of Canada and FRED terms both forbid hammering their services.
"""

import hashlib
import json
import os
import time
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from portfolio_lab import __version__, config

USER_AGENT = f"portfolio-lab/{__version__} (+https://github.com/cristian4269/portfolio-lab)"
TIMEOUT_SECONDS = 60
MIN_INTERVAL_SECONDS = 1.0

# Host -> time.monotonic() of the last request to it.
_last_request: dict[str, float] = {}


def cache_key(url: str) -> str:
    """Stable file stem for a URL (the full URL, query string included)."""
    return hashlib.sha256(url.encode()).hexdigest()[:32]


def _paths(url: str) -> tuple[Path, Path]:
    stem = config.cache_dir() / cache_key(url)
    return stem.with_suffix(".bin"), stem.with_suffix(".meta.json")


def cache_info(url: str) -> dict[str, Any] | None:
    """Metadata of the cached copy of `url`, or None if it isn't cached or is unreadable."""
    _, meta_path = _paths(url)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        meta = None
    if (
        not isinstance(meta, dict)
        or not isinstance(meta.get("sha256"), str)
        or meta.get("url") != url
    ):
        warnings.warn(f"unreadable cache metadata for {url}; treating as not cached", stacklevel=2)
        return None
    return meta


def evict(url: str) -> None:
    """Delete the cached copy of `url`, if any."""
    for path in _paths(url):
        path.unlink(missing_ok=True)


def fetch_parsed[T](url: str, parse: Callable[[bytes], T], *, refresh: bool = False) -> T:
    """Fetch `url` and parse it; a body that fails to parse is evicted, then the error raised.

    Without this, a 200 response carrying an error page would be cached and
    served on every later call.
    """
    content = fetch(url, refresh=refresh)
    try:
        return parse(content)
    except Exception:
        evict(url)
        raise


def read_cached(url: str) -> bytes | None:
    """The cached body of `url` if its checksum holds, else None: never downloads."""
    data_path, _ = _paths(url)
    meta = cache_info(url)
    if meta is None or not data_path.exists():
        return None
    content = data_path.read_bytes()
    return content if hashlib.sha256(content).hexdigest() == meta["sha256"] else None


def fetch(url: str, *, refresh: bool = False) -> bytes:
    """Return the body at `url`, from the cache unless `refresh` or the copy is bad."""
    if not refresh:
        content = read_cached(url)
        if content is not None:
            return content
        data_path, _ = _paths(url)
        if cache_info(url) is not None and data_path.exists():
            warnings.warn(
                f"cached copy of {url} failed its checksum; downloading again", stacklevel=2
            )
    return _download(url)


def _download(url: str) -> bytes:
    _wait_for_host(urlsplit(url).netloc)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.content
    _store(url, content)
    return content


def _wait_for_host(host: str) -> None:
    last = _last_request.get(host)
    if last is not None:
        remaining = MIN_INTERVAL_SECONDS - (time.monotonic() - last)
        if remaining > 0:
            time.sleep(remaining)
    _last_request[host] = time.monotonic()


def _store(url: str, content: bytes) -> None:
    data_path, meta_path = _paths(url)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "url": url,
        "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }
    # Write data before metadata, each atomically: a crash leaves at worst a
    # data file without metadata, which counts as a cache miss.
    _atomic_write(data_path, content)
    _atomic_write(meta_path, json.dumps(meta, indent=2).encode())


def _atomic_write(path: Path, content: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)
