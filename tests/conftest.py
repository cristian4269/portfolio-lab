"""Shared fixtures. Tests never touch the network or the real cache (except `-m network`)."""

from pathlib import Path

import pytest
import requests

from portfolio_lab import config
from portfolio_lab.data import cache

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_cache(request, monkeypatch, tmp_path):
    """Point the download cache at a temporary directory and reset the rate limiter.

    Live (`network`) tests keep the configured cache and the real rate limiter, so
    a full live run downloads each file at most once (the sources' terms ask us
    not to hammer them).
    """
    if request.node.get_closest_marker("network"):
        return config.cache_dir()
    path = tmp_path / "cache"
    monkeypatch.setenv(config.CACHE_ENV_VAR, str(path))
    monkeypatch.setattr(cache, "_last_request", {})
    return path


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeHTTP:
    """Stands in for `requests.get`: serves registered bodies and records every call."""

    def __init__(self):
        self.routes: dict[str, FakeResponse] = {}
        self.calls: list[dict] = []

    def add(self, url: str, content: bytes | str, status_code: int = 200):
        body = content.encode() if isinstance(content, str) else content
        self.routes[url] = FakeResponse(body, status_code)

    def __call__(self, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "headers": headers or {}, "timeout": timeout})
        if url not in self.routes:
            return FakeResponse(b"not found", 404)
        return self.routes[url]


@pytest.fixture
def fake_http(monkeypatch):
    """Replace HTTP GET in the cache module (the only place that downloads)."""
    http = FakeHTTP()
    monkeypatch.setattr(cache.requests, "get", http)
    monkeypatch.setattr(cache.time, "sleep", lambda s: None)
    return http
