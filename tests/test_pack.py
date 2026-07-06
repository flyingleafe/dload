import hashlib
import struct

import pytest

from dload.errors import PackFormatError
from dload.pack import MAGIC, PackReader, PackWriter


def test_round_trip_basic(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("a", {"audio": b"hello", "meta": b"{}"})
    w.add("b", {"audio": b"world!!"})
    digest, size, n = w.finish()

    assert n == 2
    assert size == p.stat().st_size
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()

    with PackReader(p) as r:
        assert len(r) == 2
        assert r.keys == ["a", "b"]
        key, fields = r.read(0)
        assert key == "a"
        assert fields == {"audio": b"hello", "meta": b"{}"}
        key, fields = r.read(1)
        assert key == "b"
        assert fields == {"audio": b"world!!"}


def test_empty_fields_dict(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("empty", {})
    w.add("nonempty", {"x": b"1"})
    w.finish()

    with PackReader(p) as r:
        key, fields = r.read(0)
        assert key == "empty"
        assert fields == {}
        key, fields = r.read(1)
        assert key == "nonempty"
        assert fields == {"x": b"1"}


def test_unicode_keys(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    keys = ["a", "éè", "日本語", "\U0001f600"]
    for k in keys:
        w.add(k, {"data": k.encode("utf-8")})
    w.finish()

    with PackReader(p) as r:
        assert r.keys == keys
        for k in keys:
            key, fields = r.read_key(k)
            assert key == k
            assert fields["data"] == k.encode("utf-8")


def test_many_samples(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    n = 500
    for i in range(n):
        w.add(f"sample-{i:04d}", {"payload": str(i).encode(), "tag": b"x" * (i % 7)})
    digest, size, count = w.finish()
    assert count == n

    with PackReader(p) as r:
        assert len(r) == n
        for i in range(n):
            key, fields = r.read(i)
            assert key == f"sample-{i:04d}"
            assert fields["payload"] == str(i).encode()
            assert fields["tag"] == b"x" * (i % 7)


def test_read_by_index_and_key_match(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    for i in range(10):
        w.add(f"k{i}", {"v": bytes([i])})
    w.finish()

    with PackReader(p) as r:
        for i in range(10):
            assert r.read(i) == r.read_key(f"k{i}")


def test_read_key_missing_raises_keyerror(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("only", {"v": b"1"})
    w.finish()

    with PackReader(p) as r:
        with pytest.raises(KeyError):
            r.read_key("missing")


def test_iteration_order(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    keys = [f"item-{i}" for i in range(20)]
    for k in keys:
        w.add(k, {"v": k.encode()})
    w.finish()

    with PackReader(p) as r:
        seen = [key for key, _ in r]
        assert seen == keys
        seen_fields = [fields["v"] for _, fields in r]
        assert seen_fields == [k.encode() for k in keys]


def test_tell_tracks_payload_bytes(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    assert w.tell() == 0
    w.add("a", {"x": b"12345"})
    assert w.tell() == 5
    w.add("b", {"x": b"12", "y": b"345"})
    assert w.tell() == 10
    assert w.num_samples == 2
    w.abort()


def test_tell_based_shard_cutting(tmp_path):
    target = 50
    samples = [(f"s{i}", {"d": b"x" * 10}) for i in range(20)]
    shard_paths = []
    idx = 0
    w = PackWriter(tmp_path / f"shard-{idx}.pack")
    for key, fields in samples:
        w.add(key, fields)
        if w.tell() >= target:
            w.finish()
            shard_paths.append(tmp_path / f"shard-{idx}.pack")
            idx += 1
            w = PackWriter(tmp_path / f"shard-{idx}.pack")
    if w.num_samples:
        w.finish()
        shard_paths.append(tmp_path / f"shard-{idx}.pack")
    else:
        w.abort()

    assert len(shard_paths) > 1  # sanity: cutting actually happened
    all_keys = []
    for p in shard_paths:
        with PackReader(p) as r:
            all_keys.extend(r.keys)
    assert all_keys == [k for k, _ in samples]


def test_abort_deletes_partial_file(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("a", {"x": b"1"})
    assert p.exists()
    w.abort()
    assert not p.exists()


def test_abort_then_add_raises(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.abort()
    with pytest.raises(ValueError):
        w.add("a", {"x": b"1"})


def test_finish_then_add_raises(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("a", {"x": b"1"})
    w.finish()
    with pytest.raises(ValueError):
        w.add("b", {"x": b"1"})


def test_truncated_file_raises(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("a", {"x": b"hello"})
    w.finish()

    data = p.read_bytes()
    truncated = tmp_path / "truncated.pack"
    truncated.write_bytes(data[: len(data) // 2])

    with pytest.raises(PackFormatError):
        PackReader(truncated)


def test_too_small_file_raises(tmp_path):
    p = tmp_path / "tiny.pack"
    p.write_bytes(b"short")
    with pytest.raises(PackFormatError):
        PackReader(p)


def test_bad_magic_raises(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("a", {"x": b"hello"})
    w.finish()

    data = bytearray(p.read_bytes())
    data[-1] = ord("X")
    corrupt = tmp_path / "badmagic.pack"
    corrupt.write_bytes(bytes(data))

    with pytest.raises(PackFormatError, match="magic"):
        PackReader(corrupt)


def test_corrupt_index_length_raises(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("a", {"x": b"hello"})
    w.finish()

    data = bytearray(p.read_bytes())
    # overwrite the u64 index length with an absurdly large value
    huge = struct.pack("<Q", 1 << 40)
    data[-16:-8] = huge
    corrupt = tmp_path / "badlen.pack"
    corrupt.write_bytes(bytes(data))

    with pytest.raises(PackFormatError):
        PackReader(corrupt)


def test_corrupt_msgpack_index_raises(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("a", {"x": b"hello"})
    digest, size, n = w.finish()

    data = bytearray(p.read_bytes())
    # index region is [size-16-index_len, size-16); scramble a byte in it
    index_len = struct.unpack("<Q", bytes(data[-16:-8]))[0]
    index_start = size - 16 - index_len
    data[index_start] ^= 0xFF
    corrupt = tmp_path / "badindex.pack"
    corrupt.write_bytes(bytes(data))

    with pytest.raises(PackFormatError):
        PackReader(corrupt)


def test_digest_matches_sha256_of_file(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    for i in range(5):
        w.add(f"k{i}", {"d": bytes([i]) * 100})
    digest, size, n = w.finish()
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()
    assert size == p.stat().st_size


def test_empty_pack_file(tmp_path):
    p = tmp_path / "empty.pack"
    w = PackWriter(p)
    digest, size, n = w.finish()
    assert n == 0
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()

    with PackReader(p) as r:
        assert len(r) == 0
        assert r.keys == []
        assert list(r) == []


def test_reader_context_manager_closes(tmp_path):
    p = tmp_path / "shard.pack"
    w = PackWriter(p)
    w.add("a", {"x": b"1"})
    w.finish()

    with PackReader(p) as r:
        assert len(r) == 1
    # closed file should raise on further use
    with pytest.raises(ValueError):
        r.read(0)


def test_magic_constant_matches_docstring():
    assert MAGIC == b"DLOADPK1"
    assert len(MAGIC) == 8
