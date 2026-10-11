import hashlib
import json

import pytest
import requests

from portfolio_lab.data import cache

URL = "https://example.org/files/a.zip"


def test_miss_downloads_and_writes_file_and_metadata(fake_http, isolated_cache):
    fake_http.add(URL, b"payload")
    assert cache.fetch(URL) == b"payload"
    assert len(fake_http.calls) == 1

    meta = cache.cache_info(URL)
    assert meta["url"] == URL
    assert meta["sha256"] == hashlib.sha256(b"payload").hexdigest()
    assert meta["bytes"] == 7
    assert meta["retrieved"].endswith("+00:00")  # UTC, ISO 8601
    assert (isolated_cache / f"{cache.cache_key(URL)}.bin").read_bytes() == b"payload"
    stored = json.loads((isolated_cache / f"{cache.cache_key(URL)}.meta.json").read_text())
    assert stored == meta


def test_hit_does_not_download(fake_http):
    fake_http.add(URL, b"payload")
    cache.fetch(URL)
    assert cache.fetch(URL) == b"payload"
    assert len(fake_http.calls) == 1


def test_refresh_downloads_again(fake_http):
    fake_http.add(URL, b"old")
    cache.fetch(URL)
    fake_http.add(URL, b"new")
    assert cache.fetch(URL, refresh=True) == b"new"
    assert len(fake_http.calls) == 2
    assert cache.cache_info(URL)["sha256"] == hashlib.sha256(b"new").hexdigest()


def test_corrupted_file_is_downloaded_again_with_warning(fake_http, isolated_cache):
    fake_http.add(URL, b"payload")
    cache.fetch(URL)
    (isolated_cache / f"{cache.cache_key(URL)}.bin").write_bytes(b"tampered")
    with pytest.warns(UserWarning, match="checksum"):
        assert cache.fetch(URL) == b"payload"
    assert len(fake_http.calls) == 2


def test_missing_metadata_counts_as_miss(fake_http, isolated_cache):
    fake_http.add(URL, b"payload")
    cache.fetch(URL)
    (isolated_cache / f"{cache.cache_key(URL)}.meta.json").unlink()
    assert cache.cache_info(URL) is None
    cache.fetch(URL)
    assert len(fake_http.calls) == 2


@pytest.mark.parametrize("bad_meta", ["{not json", '{"url": "x"}', "[]"])
def test_unreadable_metadata_counts_as_miss(fake_http, isolated_cache, bad_meta):
    fake_http.add(URL, b"payload")
    cache.fetch(URL)
    (isolated_cache / f"{cache.cache_key(URL)}.meta.json").write_text(bad_meta)
    with pytest.warns(UserWarning, match="metadata"):
        assert cache.cache_info(URL) is None
    with pytest.warns(UserWarning, match="metadata"):
        assert cache.fetch(URL) == b"payload"
    assert len(fake_http.calls) == 2
    assert cache.cache_info(URL)["sha256"] == hashlib.sha256(b"payload").hexdigest()


def test_http_error_raises_and_caches_nothing(fake_http):
    fake_http.add(URL, b"server error", status_code=500)
    with pytest.raises(requests.HTTPError):
        cache.fetch(URL)
    assert cache.cache_info(URL) is None


def test_identifies_itself_and_sets_timeout(fake_http):
    fake_http.add(URL, b"x")
    cache.fetch(URL)
    call = fake_http.calls[0]
    assert call["headers"]["User-Agent"].startswith("portfolio-lab/")
    assert call["timeout"] == cache.TIMEOUT_SECONDS


def test_key_depends_on_full_url_including_query():
    assert cache.cache_key(URL) != cache.cache_key(URL + "?id=1")
    assert cache.cache_key(URL) == cache.cache_key(URL)


def test_polite_delay_between_requests_to_same_host(monkeypatch, fake_http):
    clock = {"now": 100.0}
    sleeps: list[float] = []
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(cache.time, "sleep", sleeps.append)

    fake_http.add("https://a.org/1", b"1")
    fake_http.add("https://a.org/2", b"2")
    fake_http.add("https://b.org/1", b"3")

    cache.fetch("https://a.org/1")
    clock["now"] += 0.25
    cache.fetch("https://a.org/2")  # same host, 0.25 s later -> waits the rest
    cache.fetch("https://b.org/1")  # other host -> no wait

    assert sleeps == [pytest.approx(cache.MIN_INTERVAL_SECONDS - 0.25)]


def test_no_delay_after_interval_has_passed(monkeypatch, fake_http):
    clock = {"now": 0.0}
    sleeps: list[float] = []
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(cache.time, "sleep", sleeps.append)
    fake_http.add("https://a.org/1", b"1")
    fake_http.add("https://a.org/2", b"2")

    cache.fetch("https://a.org/1")
    clock["now"] += cache.MIN_INTERVAL_SECONDS + 1
    cache.fetch("https://a.org/2")
    assert sleeps == []


def test_metadata_for_another_url_counts_as_miss(fake_http, isolated_cache):
    fake_http.add(URL, b"payload")
    cache.fetch(URL)
    meta_path = isolated_cache / f"{cache.cache_key(URL)}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps(meta | {"url": "https://example.org/other"}))
    with pytest.warns(UserWarning, match="metadata"):
        assert cache.cache_info(URL) is None


def test_fetch_parsed_returns_parsed_value_and_caches(fake_http):
    fake_http.add(URL, b"42")
    assert cache.fetch_parsed(URL, lambda b: int(b)) == 42
    assert cache.fetch_parsed(URL, lambda b: int(b)) == 42
    assert len(fake_http.calls) == 1


def test_fetch_parsed_evicts_a_body_that_does_not_parse(fake_http):
    # A 200 response with an error page must not be cached and served forever.
    fake_http.add(URL, b"<html>maintenance</html>")
    with pytest.raises(ValueError):
        cache.fetch_parsed(URL, lambda b: int(b))
    assert cache.cache_info(URL) is None
    fake_http.add(URL, b"42")
    assert cache.fetch_parsed(URL, lambda b: int(b)) == 42
    assert len(fake_http.calls) == 2


def test_fetch_parsed_evicts_a_stale_cached_copy_that_no_longer_parses(fake_http):
    fake_http.add(URL, b"old format")
    cache.fetch(URL)  # cached by an earlier version, without validation
    with pytest.raises(ValueError):
        cache.fetch_parsed(URL, lambda b: int(b))
    assert cache.cache_info(URL) is None


def test_evict_missing_entry_is_a_no_op():
    cache.evict(URL)
    assert cache.cache_info(URL) is None


def test_read_cached_never_downloads(fake_http, isolated_cache):
    assert cache.read_cached(URL) is None  # a miss is None, not a download
    fake_http.add(URL, b"payload")
    cache.fetch(URL)
    assert cache.read_cached(URL) == b"payload"
    (isolated_cache / f"{cache.cache_key(URL)}.bin").write_bytes(b"tampered")
    assert cache.read_cached(URL) is None  # a bad checksum is None too
    assert len(fake_http.calls) == 1


def test_unreadable_metadata_warns_once_per_fetch(fake_http, isolated_cache):
    fake_http.add(URL, b"payload")
    cache.fetch(URL)
    (isolated_cache / f"{cache.cache_key(URL)}.meta.json").write_text("{not json")
    with pytest.warns(UserWarning) as record:
        cache.fetch(URL)
    assert len([w for w in record if "metadata" in str(w.message)]) == 1
