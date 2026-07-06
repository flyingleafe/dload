import json
from typing import Any

import pytest

from dload.errors import PackFormatError
from dload.manifest import FORMAT_VERSION, Manifest, ShardInfo


def make_manifest(**overrides: Any) -> Manifest:
    defaults: dict[str, Any] = dict(
        name="my-dataset",
        created="2026-07-03T12:00:00Z",
        shards=(
            ShardInfo(digest="a" * 64, size=1000, num_samples=10),
            ShardInfo(digest="b" * 64, size=2000, num_samples=20),
        ),
        meta={"source": "test"},
        recipe=None,
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_num_samples_and_total_bytes():
    m = make_manifest()
    assert m.num_samples == 30
    assert m.total_bytes == 3000


def test_num_samples_empty_shards():
    m = make_manifest(shards=())
    assert m.num_samples == 0
    assert m.total_bytes == 0


def test_version_is_stable_for_same_content():
    m1 = make_manifest()
    m2 = make_manifest()
    assert m1.version == m2.version


def test_version_field_order_irrelevant():
    m1 = Manifest(
        name="ds",
        created="2026-07-03T12:00:00Z",
        shards=(ShardInfo(digest="c" * 64, size=1, num_samples=1),),
        meta={"a": 1, "b": 2},
    )
    m2 = Manifest(
        name="ds",
        created="2026-07-03T12:00:00Z",
        shards=(ShardInfo(digest="c" * 64, size=1, num_samples=1),),
        meta={"b": 2, "a": 1},
    )
    assert m1.version == m2.version


def test_version_changes_with_content():
    m1 = make_manifest()
    m2 = make_manifest(name="other-dataset")
    assert m1.version != m2.version

    m3 = make_manifest(meta={"source": "different"})
    assert m1.version != m3.version

    m4 = make_manifest(shards=(ShardInfo(digest="a" * 64, size=1000, num_samples=10),))
    assert m1.version != m4.version


def test_version_is_sha256_hex():
    m = make_manifest()
    v = m.version
    assert len(v) == 64
    assert all(c in "0123456789abcdef" for c in v)


def test_version_excludes_itself_from_digest():
    # calling .version repeatedly (or via to_json which embeds it) must not
    # change the digest -- version is never part of its own input.
    m = make_manifest()
    v1 = m.version
    _ = m.to_json()
    v2 = m.version
    assert v1 == v2


def test_to_json_includes_version_and_is_valid_json():
    m = make_manifest()
    text = m.to_json()
    data = json.loads(text)
    assert data["version"] == m.version
    assert data["name"] == "my-dataset"
    assert data["format"] == FORMAT_VERSION


def test_to_json_is_indented():
    m = make_manifest()
    text = m.to_json()
    assert "\n" in text


def test_from_json_round_trip():
    m = make_manifest()
    text = m.to_json()
    m2 = Manifest.from_json(text)
    assert m2 == m


def test_from_json_ignores_version_key():
    m = make_manifest()
    data = json.loads(m.to_json())
    assert "version" in data
    data["version"] = "0" * 64  # bogus, must be ignored
    m2 = Manifest.from_json(json.dumps(data))
    assert m2.version == m.version
    assert m2 == m


def test_from_json_missing_version_key_still_works():
    m = make_manifest()
    data = json.loads(m.to_json())
    del data["version"]
    m2 = Manifest.from_json(json.dumps(data))
    assert m2 == m


def test_from_json_unsupported_format_raises():
    m = make_manifest()
    data = json.loads(m.to_json())
    data["format"] = FORMAT_VERSION + 1
    with pytest.raises(PackFormatError):
        Manifest.from_json(json.dumps(data))


def test_shard_info_round_trip_via_manifest():
    m = make_manifest()
    m2 = Manifest.from_json(m.to_json())
    assert m2.shards == m.shards
    for s in m2.shards:
        assert isinstance(s, ShardInfo)


def test_recipe_and_defaults():
    m = Manifest(
        name="ds",
        created="2026-07-03T12:00:00Z",
        shards=(),
        recipe="print('hello')",
    )
    assert m.recipe == "print('hello')"
    assert m.meta == {}
    assert m.format == FORMAT_VERSION
    m2 = Manifest.from_json(m.to_json())
    assert m2 == m


def test_manifest_is_frozen():
    m = make_manifest()
    with pytest.raises(Exception):
        m.name = "changed"  # pyright: ignore[reportAttributeAccessIssue]


def test_shard_info_is_frozen():
    s = ShardInfo(digest="a" * 64, size=1, num_samples=1)
    with pytest.raises(Exception):
        s.size = 2  # pyright: ignore[reportAttributeAccessIssue]


def test_canonical_json_matches_spec():
    m = make_manifest()
    text = m.to_json()
    data = json.loads(text)
    del data["version"]
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    import hashlib

    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == m.version
