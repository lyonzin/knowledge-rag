"""Cache snapshots preserve mutable results, key boundaries, and LRU/TTL rules."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import mcp_server.server as server


@pytest.mark.parametrize(("rows", "fields"), [(0, 0), (1, 1), (5, 9), (10, 0), (10, 9)])
def test_flat_rows_preserve_full_isolation_for_varied_payload_shapes(rows, fields):
    values = ("text", 3, 1.25, True, b"bytes", None, 2j, range(3), "last")
    original = [{f"field_{column}": values[column] for column in range(fields)} for _ in range(rows)]
    expected = deepcopy(original)
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, original)

    for row in original:
        row.clear()
    original.append({"new": "not cached"})
    cached = cache.get("query", 5, None, 0.3)
    assert cached == expected
    assert len({id(row) for row in cached}) == rows

    for row in cached:
        row["new"] = "local mutation"
    cached.append({"new": "local result"})
    assert cache.get("query", 5, None, 0.3) == expected


def test_repeated_flat_rows_preserve_aliases_inside_each_isolated_result():
    shared = {"content": "full document", "score": 1.0}
    original = [shared] * 10
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, original)
    shared["content"] = "input mutation"

    cached = cache.get("query", 5, None, 0.3)
    assert cached == [{"content": "full document", "score": 1.0}] * 10
    assert all(row is cached[0] for row in cached)
    cached[0]["content"] = "output mutation"
    fresh = cache.get("query", 5, None, 0.3)
    assert all(row is fresh[0] for row in fresh)
    assert fresh[0]["content"] == "full document"


def test_dictionary_subclass_rows_preserve_type_and_mutable_attributes():
    class Row(dict):
        pass

    row = Row(content="document")
    row.labels = ["alpha"]
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, [row])
    row.labels.append("input mutation")

    cached = cache.get("query", 5, None, 0.3)
    assert type(cached[0]) is Row
    assert cached[0].labels == ["alpha"]
    cached[0].labels.clear()
    assert cache.get("query", 5, None, 0.3)[0].labels == ["alpha"]


def test_atomic_subclasses_in_keys_and_values_keep_their_mutable_state_isolated():
    class Text(str):
        pass

    key, value = Text("key"), Text("value")
    key.labels, value.labels = ["key label"], ["value label"]
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, [{key: value}])
    key.labels.clear()
    value.labels.clear()

    cached = cache.get("query", 5, None, 0.3)[0]
    cached_key = next(iter(cached))
    assert type(cached_key) is Text and type(cached[cached_key]) is Text
    assert cached_key.labels == ["key label"]
    assert cached[cached_key].labels == ["value label"]
    cached_key.labels.clear()
    cached[cached_key].labels.clear()
    fresh = cache.get("query", 5, None, 0.3)[0]
    assert next(iter(fresh)).labels == ["key label"]
    assert fresh["key"].labels == ["value label"]


def test_put_snapshots_nested_containers_and_shares_immutable_text():
    content = "documentation " * 2_000
    original = [{"content": content, "metadata": {"sections": [{"keywords": ["alpha"]}]}}]
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, original)

    original[0]["metadata"]["sections"][0]["keywords"].append("not cached")
    original[0]["metadata"]["sections"].append({"keywords": ["not cached"]})
    original.append({"content": "not cached"})
    cached = cache.get("query", 5, None, 0.3)

    assert cached == [{"content": content, "metadata": {"sections": [{"keywords": ["alpha"]}]}}]
    assert cached[0]["content"] is content


def test_each_get_owns_all_nested_mutable_containers():
    original = [{"content": "full document", "metadata": {"sections": [{"keywords": ["alpha"]}]}}]
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, original)
    first = cache.get("query", 5, None, 0.3)
    second = cache.get("query", 5, None, 0.3)

    first[0]["content"] = "snippet"
    first[0]["metadata"]["sections"][0]["keywords"].clear()
    first[0]["metadata"]["sections"].clear()
    first.append({"content": "local result"})

    assert second == original
    assert cache.get("query", 5, None, 0.3) == original


def test_snapshot_preserves_aliases_and_cycles_without_sharing_with_callers():
    shared = {"keywords": ["alpha"]}
    original = [shared, shared]
    shared["results"] = original
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, original)

    shared["keywords"].append("not cached")
    cached = cache.get("query", 5, None, 0.3)
    assert cached[0] is cached[1]
    assert cached[0]["results"] is cached
    assert cached[0]["keywords"] == ["alpha"]

    cached[0]["keywords"].clear()
    assert cache.get("query", 5, None, 0.3)[0]["keywords"] == ["alpha"]


def test_uncommon_values_keep_deepcopy_semantics_and_aliases():
    class ResultList(list):
        pass

    shared = {"keywords": ["alpha"]}
    wrapped = ResultList([shared])
    original = {"wrapped": wrapped, "metadata": SimpleNamespace(shared=shared), "tuple": (shared,)}
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, original)
    shared["keywords"].append("not cached")

    cached = cache.get("query", 5, None, 0.3)
    assert type(cached["wrapped"]) is ResultList
    assert cached["wrapped"][0] is cached["metadata"].shared is cached["tuple"][0]
    assert cached["wrapped"][0]["keywords"] == ["alpha"]
    cached["metadata"].shared["keywords"].clear()
    assert cache.get("query", 5, None, 0.3)["tuple"][0]["keywords"] == ["alpha"]


def test_mutable_dictionary_keys_are_copied_with_the_same_memo():
    class Key:
        def __init__(self):
            self.labels = ["alpha"]

    key = Key()
    original = {key: {"key": key}}
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, original)
    key.labels.append("not cached")

    cached = cache.get("query", 5, None, 0.3)
    cached_key = next(iter(cached))
    assert cached_key is cached[cached_key]["key"]
    assert cached_key.labels == ["alpha"]
    cached_key.labels.clear()
    assert next(iter(cache.get("query", 5, None, 0.3))).labels == ["alpha"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("query", 5, None, 0.3), ("query", 5, "None", 0.3)),
        (("x|5|category", 5, "z", 0.3), ("x", 5, "category|5|z", 0.3)),
        (("query", 5, "category", 0.3), ("other", 5, "category", 0.3)),
        (("query", 5, "category", 0.3), ("query", 10, "category", 0.3)),
        (("query", 5, "category", 0.3), ("query", 5, "other", 0.3)),
        (("query", 5, "category", 0.3), ("query", 5, "category", 0.7)),
    ],
)
def test_key_parameters_keep_independent_entries(first, second):
    cache = server.QueryCache()
    cache.put(*first, ["first"])
    cache.put(*second, ["second"])

    assert cache.get(*first) == ["first"]
    assert cache.get(*second) == ["second"]


def test_default_method_shares_auto_entry_and_other_methods_stay_separate():
    cache = server.QueryCache()
    cache.put("query", 5, None, 0.3, ["auto"])
    cache.put("query", 5, None, 0.3, ["hybrid"], search_method="hybrid")
    cache.put("query", 5, None, 0.3, ["fts5"], search_method="fts5")

    assert cache.get("query", 5, None, 0.3, "auto") == ["auto"]
    assert cache.get("query", 5, None, 0.3, "hybrid") == ["hybrid"]
    assert cache.get("query", 5, None, 0.3, "fts5") == ["fts5"]


def test_entry_expires_at_ttl_and_updates_counters(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(server.time, "time", lambda: now[0])
    cache = server.QueryCache(ttl_seconds=5)
    cache.put("query", 5, None, 0.3, ["cached"])

    now[0] = 104.99
    assert cache.get("query", 5, None, 0.3) == ["cached"]
    now[0] = 105.0
    assert cache.get("query", 5, None, 0.3) is None
    assert cache.stats() == {
        "size": 0,
        "max_size": 100,
        "ttl_seconds": 5,
        "hits": 1,
        "misses": 1,
        "hit_rate": "50.0%",
    }


def test_hit_and_replacement_refresh_lru_without_displacing_another_key():
    cache = server.QueryCache(max_size=2)
    cache.put("a", 5, None, 0.3, ["a"])
    cache.put("b", 5, None, 0.3, ["b"])
    assert cache.get("a", 5, None, 0.3) == ["a"]
    cache.put("c", 5, None, 0.3, ["c"])
    assert cache.get("b", 5, None, 0.3) is None

    cache.put("a", 5, None, 0.3, ["updated a"])
    assert cache.stats()["size"] == 2
    cache.put("d", 5, None, 0.3, ["d"])
    assert cache.get("c", 5, None, 0.3) is None
    assert cache.get("a", 5, None, 0.3) == ["updated a"]
    assert cache.get("d", 5, None, 0.3) == ["d"]


def test_replacement_refreshes_ttl(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(server.time, "time", lambda: now[0])
    cache = server.QueryCache(ttl_seconds=5)
    cache.put("query", 5, None, 0.3, ["old"])
    now[0] = 103.0
    cache.put("query", 5, None, 0.3, ["new"])

    now[0] = 106.0
    assert cache.get("query", 5, None, 0.3) == ["new"]
    now[0] = 108.0
    assert cache.get("query", 5, None, 0.3) is None


@pytest.mark.parametrize("max_size", [0, -1])
def test_disabled_cache_does_not_store_results(max_size):
    cache = server.QueryCache(max_size=max_size)
    cache.put("query", 5, None, 0.3, [{"content": "document"}])

    assert cache.get("query", 5, None, 0.3) is None
    assert cache.stats()["size"] == 0
