import os
import uuid
from pathlib import Path

import pytest

from dload.errors import NotFoundError
from dload.remote import LocalRemote, S3Remote

# -- LocalRemote ----------------------------------------------------------------


def remote(tmp_path: Path) -> LocalRemote:
    return LocalRemote(tmp_path / "remote")


def test_put_get_bytes_round_trip(tmp_path):
    r = remote(tmp_path)
    r.put_bytes("a/b.txt", b"hello")
    assert r.get_bytes("a/b.txt") == b"hello"


def test_get_bytes_missing_raises_not_found(tmp_path):
    r = remote(tmp_path)
    with pytest.raises(NotFoundError):
        r.get_bytes("nope")


def test_exists(tmp_path):
    r = remote(tmp_path)
    assert not r.exists("k")
    r.put_bytes("k", b"x")
    assert r.exists("k")


def test_size(tmp_path):
    r = remote(tmp_path)
    r.put_bytes("k", b"12345")
    assert r.size("k") == 5


def test_size_missing_raises_not_found(tmp_path):
    r = remote(tmp_path)
    with pytest.raises(NotFoundError):
        r.size("nope")


def test_delete_is_idempotent(tmp_path):
    r = remote(tmp_path)
    r.put_bytes("k", b"x")
    r.delete("k")
    assert not r.exists("k")
    r.delete("k")  # no error


def test_list_with_prefix(tmp_path):
    r = remote(tmp_path)
    r.put_bytes("shards/aa/1", b"1")
    r.put_bytes("shards/aa/2", b"2")
    r.put_bytes("shards/bb/3", b"3")
    r.put_bytes("other/4", b"4")

    assert sorted(r.list("shards/aa/")) == ["shards/aa/1", "shards/aa/2"]
    assert sorted(r.list("shards/")) == ["shards/aa/1", "shards/aa/2", "shards/bb/3"]
    assert sorted(r.list()) == [
        "other/4",
        "shards/aa/1",
        "shards/aa/2",
        "shards/bb/3",
    ]


def test_list_empty_prefix_matches_all_no_arg(tmp_path):
    r = remote(tmp_path)
    assert list(r.list()) == []
    r.put_bytes("k", b"x")
    assert list(r.list()) == ["k"]


def test_put_get_file(tmp_path):
    r = remote(tmp_path)
    src = tmp_path / "src.bin"
    src.write_bytes(b"file contents")
    r.put_file("f", src)

    dest = tmp_path / "dest.bin"
    r.get_to_file("f", dest)
    assert dest.read_bytes() == b"file contents"


def test_get_to_file_missing_raises_not_found(tmp_path):
    r = remote(tmp_path)
    with pytest.raises(NotFoundError):
        r.get_to_file("nope", tmp_path / "dest")


def test_get_to_file_creates_parent_dirs(tmp_path):
    r = remote(tmp_path)
    r.put_bytes("k", b"data")
    dest = tmp_path / "nested" / "deep" / "dest.bin"
    r.get_to_file("k", dest)
    assert dest.read_bytes() == b"data"


def test_get_to_file_is_atomic_no_tmp_left(tmp_path):
    r = remote(tmp_path)
    r.put_bytes("k", b"data")
    dest_dir = tmp_path / "dest_dir"
    dest_dir.mkdir()
    dest = dest_dir / "dest.bin"
    r.get_to_file("k", dest)
    leftovers = [p for p in dest_dir.iterdir() if p.name != dest.name]
    assert leftovers == []


def test_put_bytes_overwrites_atomically(tmp_path):
    r = remote(tmp_path)
    r.put_bytes("k", b"first")
    r.put_bytes("k", b"second")
    assert r.get_bytes("k") == b"second"


def test_op_counts_tracks_calls_by_method_name(tmp_path):
    r = remote(tmp_path)
    r.put_bytes("k", b"x")
    r.put_bytes("k2", b"y")
    r.get_bytes("k")
    r.exists("k")
    r.exists("missing")
    r.size("k")
    list(r.list())
    r.delete("k")

    assert r.op_counts["put_bytes"] == 2
    assert r.op_counts["get_bytes"] == 1
    assert r.op_counts["exists"] == 2
    assert r.op_counts["size"] == 1
    assert r.op_counts["list"] == 1
    assert r.op_counts["delete"] == 1
    assert "put_file" not in r.op_counts


def test_op_counts_incremented_even_on_not_found(tmp_path):
    r = remote(tmp_path)
    with pytest.raises(NotFoundError):
        r.get_bytes("nope")
    assert r.op_counts["get_bytes"] == 1


def test_constructor_creates_root_dir(tmp_path):
    root = tmp_path / "does" / "not" / "exist"
    LocalRemote(root)
    assert root.is_dir()


# -- S3Remote integration tests (real Cloudflare R2) -----------------------------

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


@pytest.fixture(scope="module")
def r2_env():
    if not ENV_PATH.is_file():
        pytest.skip(f".env not found at {ENV_PATH}, skipping R2 integration tests")

    env = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")

    required = [
        "R2_ACCOUNT_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]
    missing = [k for k in required if k not in env]
    if missing:
        pytest.skip(f".env missing keys: {missing}")

    old = {k: os.environ.get(k) for k in required}
    os.environ.update(env)
    try:
        yield env
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _TrackedS3Remote(S3Remote):
    """An S3Remote that remembers keys it created, so the fixture can clean
    them up afterward."""

    def __init__(self, endpoint_url: str, bucket: str, prefix: str = "") -> None:
        super().__init__(endpoint_url, bucket, prefix=prefix)
        self.created_keys: list[str] = []


@pytest.fixture
def s3_remote(r2_env):
    endpoint_url = f"https://{r2_env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    r = _TrackedS3Remote(endpoint_url, "dload-test", prefix="citest/")
    yield r
    for key in r.created_keys:
        r.delete(key)


def _tracked_put_bytes(r: _TrackedS3Remote, key: str, data: bytes) -> None:
    r.put_bytes(key, data)
    r.created_keys.append(key)


def _tracked_put_file(r: _TrackedS3Remote, key: str, src: Path) -> None:
    r.put_file(key, src)
    r.created_keys.append(key)


@pytest.mark.integration
def test_s3_put_get_bytes_round_trip(s3_remote):
    key = f"round-trip-{uuid.uuid4().hex}"
    _tracked_put_bytes(s3_remote, key, b"hello r2")
    assert s3_remote.get_bytes(key) == b"hello r2"


@pytest.mark.integration
def test_s3_exists_and_size(s3_remote):
    key = f"exists-{uuid.uuid4().hex}"
    assert not s3_remote.exists(key)
    _tracked_put_bytes(s3_remote, key, b"12345")
    assert s3_remote.exists(key)
    assert s3_remote.size(key) == 5


@pytest.mark.integration
def test_s3_get_bytes_missing_raises_not_found(s3_remote):
    with pytest.raises(NotFoundError):
        s3_remote.get_bytes(f"missing-{uuid.uuid4().hex}")


@pytest.mark.integration
def test_s3_size_missing_raises_not_found(s3_remote):
    with pytest.raises(NotFoundError):
        s3_remote.size(f"missing-{uuid.uuid4().hex}")


@pytest.mark.integration
def test_s3_delete_is_idempotent(s3_remote):
    key = f"delete-{uuid.uuid4().hex}"
    _tracked_put_bytes(s3_remote, key, b"x")
    s3_remote.delete(key)
    assert not s3_remote.exists(key)
    s3_remote.delete(key)  # no error


@pytest.mark.integration
def test_s3_list_with_prefix(s3_remote):
    run_id = uuid.uuid4().hex
    keys = [f"list-{run_id}/a", f"list-{run_id}/b", f"list-{run_id}/sub/c"]
    for key in keys:
        _tracked_put_bytes(s3_remote, key, b"x")

    listed = sorted(s3_remote.list(f"list-{run_id}/"))
    assert listed == sorted(keys)


@pytest.mark.integration
def test_s3_put_get_file_large(s3_remote, tmp_path):
    key = f"large-{uuid.uuid4().hex}"
    src = tmp_path / "big.bin"
    size = 10 * 1024 * 1024
    src.write_bytes(os.urandom(size))

    _tracked_put_file(s3_remote, key, src)
    assert s3_remote.size(key) == size

    dest = tmp_path / "downloaded.bin"
    s3_remote.get_to_file(key, dest)
    assert dest.stat().st_size == size
    assert dest.read_bytes() == src.read_bytes()
